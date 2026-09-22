"""Grouped, block-target Vecchia engine used by the corridor models.

It builds user-sized grid-cell groups, orders them by max-min ordering of their
centroids, and evaluates block conditional likelihoods in batches on the input
tensor's device.  Corridor strategies subclass this engine and provide their
own conditioning geometry.

Default conditioning budget for a target cluster:
  A at t:     6 previous spatial neighbor clusters
  B at t-1:  same cluster + 3 local clusters + 1 shifted fresh cluster
  C at t-2:  same cluster + 2 local clusters + 1 shifted fresh cluster

Conditioning-set logic, in block units:

  1. Build native-grid target clusters.
     With block_shape=(3, 3), each cluster is a 3-by-3 regular-grid block.
     The block is the multivariate target in the Vecchia conditional.  The
     coordinates used to build blocks are regular-grid coordinates
     (grid_coords), while covariance is evaluated on the coordinates stored in
     input_map.  This is deliberate: keep the geometry stable even when
     input_map contains real source-location offsets.

  2. Order clusters by max-min ordering of block centroids.
     Vecchia conditioning must only use variables that are earlier in the
     ordering at the same time.  Therefore the "A" set is not simply the six
     closest blocks in space; it is the six nearest centroid neighbors among
     clusters with index < current cluster index after max-min ordering.

  3. For a target block i at time t, use:
       A(t)   = nearest previous same-time blocks, cap n_neighbor_blocks_t.
       B(t-1) = same block i
                + first lag1_local_blocks blocks from A(t)
                + lag1_shifted_blocks fresh blocks around the advection-shifted
                  block center.
       C(t-s) = same block i
                + first lag2_local_blocks blocks from A(t)
                + lag2_shifted_blocks fresh blocks around the two-step lag
                  center, where s is ``second_lag_stride``. The two longitude
                  offsets are independently configurable.

  4. Shifted fresh blocks are chosen at the cluster level.
     The block centroid is shifted eastward in longitude by lag*_lon_offset,
     then mapped to the nearest cluster centroid.  If that block has already
     been included by same/local conditioning, the code falls back to nearest
     neighbors of the shifted block so the requested fresh block count can still
     add new information when possible.

  5. Point counts are derived from block counts.
     For a 3-by-3 grid block, the default A6 + B(same1+L3+F1)
     + C(same1+L2+F1) gives at most 15 conditioning blocks, or 135 conditioning
     points, before adding the target block itself to the Cholesky system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.neighbors import BallTree

from GEMS_TCO import orderings

from ._base import GroupedVecchiaBase, _integer_option
from ._native_covariance import native_covariance, native_covariance_available

logger = logging.getLogger(__name__)


@dataclass
class _ClusterBatch:
    label: str
    max_cond_points: int
    target_size: int
    coordinates: torch.Tensor
    response: torch.Tensor
    design: torch.Tensor
    indices: torch.Tensor
    is_dummy: torch.Tensor


class GroupedBatchedVecchia(GroupedVecchiaBase):
    """Vecchia engine with grouped targets and batched conditional systems.

    Parameters
    ----------
    input_map
        Dict of hourly tensors with columns
        [lat, lon, centered_o3, time, hour dummies...].
    grid_coords
        Optional ``(n_points, 2)`` array of regular-grid [lat, lon] coordinates
        matching the local point order of ``input_map``.  If ``input_map`` was
        built with ``use_source_coordinates=True``, pass grid coordinates here so clusters are
        made on the grid but covariance uses source coordinates.
    block_shape
        Grid cells per target cluster, as ``(n_lat_cells, n_lon_cells)``.
    covariance_backend
        ``"auto"`` uses the optional fused CPU or CUDA float64 kernel matching
        the input tensors for the smoothness-0.5 closed-form Matérn model;
        ``"torch"`` forces the portable reference path and ``"native"``
        requires the compiled kernel.
    """

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, torch.Tensor],
        grid_coords: Optional[np.ndarray] = None,
        block_shape: Tuple[int, int] = (3, 3),
        n_neighbor_blocks_t: int = 6,
        lag1_same_block: bool = True,
        lag1_local_blocks: int = 3,
        lag1_shifted_blocks: int = 1,
        lag2_same_block: bool = True,
        lag2_local_blocks: int = 2,
        lag2_shifted_blocks: int = 1,
        second_lag_stride: int = 2,
        lag1_lon_offset: float = 0.063,
        lag2_lon_offset: Optional[float] = None,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        covariance_backend: str = "auto",
    ):
        if not np.isfinite(smooth) or smooth not in (0.5, 1.5):
            raise ValueError(f"smooth must be 0.5 or 1.5, got {smooth}")

        try:
            block_shape = tuple(block_shape)
        except TypeError as exc:
            raise ValueError("block_shape must contain exactly two dimensions") from exc
        if len(block_shape) != 2:
            raise ValueError("block_shape must contain exactly two dimensions")
        normalized_block_shape = tuple(
            _integer_option(f"block_shape[{index}]", value, minimum=1)
            for index, value in enumerate(block_shape)
        )

        block_counts = {
            "n_neighbor_blocks_t": n_neighbor_blocks_t,
            "lag1_local_blocks": lag1_local_blocks,
            "lag1_shifted_blocks": lag1_shifted_blocks,
            "lag2_local_blocks": lag2_local_blocks,
            "lag2_shifted_blocks": lag2_shifted_blocks,
        }
        block_counts = {
            name: _integer_option(name, value, minimum=0) for name, value in block_counts.items()
        }
        target_chunk_size = _integer_option("target_chunk_size", target_chunk_size, minimum=1)
        min_target_points = _integer_option("min_target_points", min_target_points, minimum=1)
        if max_neighbor_search is not None:
            max_neighbor_search = _integer_option(
                "max_neighbor_search", max_neighbor_search, minimum=1
            )
        if not np.isfinite(lag1_lon_offset) or (
            lag2_lon_offset is not None and not np.isfinite(lag2_lon_offset)
        ):
            raise ValueError("longitude offsets must be finite")
        covariance_backend = str(covariance_backend).lower()
        if covariance_backend not in {"auto", "native", "torch"}:
            raise ValueError("covariance_backend must be 'auto', 'native', or 'torch'")

        super().__init__(
            smooth=smooth,
            input_map=input_map,
            second_lag_stride=second_lag_stride,
        )

        self.grid_coords = (
            None if grid_coords is None else np.asarray(grid_coords, dtype=np.float64)
        )
        self.block_shape = normalized_block_shape
        self.n_neighbor_blocks_t = block_counts["n_neighbor_blocks_t"]
        self.lag1_same_block = bool(lag1_same_block)
        self.lag1_local_blocks = block_counts["lag1_local_blocks"]
        self.lag1_shifted_blocks = block_counts["lag1_shifted_blocks"]
        self.lag2_same_block = bool(lag2_same_block)
        self.lag2_local_blocks = block_counts["lag2_local_blocks"]
        self.lag2_shifted_blocks = block_counts["lag2_shifted_blocks"]
        self.lag1_lon_offset = float(abs(lag1_lon_offset))
        self.lag2_lon_offset = (
            float(abs(lag2_lon_offset))
            if lag2_lon_offset is not None
            else 2.0 * self.lag1_lon_offset
        )
        self.target_chunk_size = target_chunk_size
        self.min_target_points = min_target_points
        self.max_neighbor_search = max_neighbor_search
        self.covariance_backend = covariance_backend

        self._cluster_batches: List[_ClusterBatch] = []
        self.cluster_points: List[np.ndarray] = []
        self.cluster_centroids: Optional[np.ndarray] = None
        self.cluster_nns: Optional[np.ndarray] = None
        self.shift_lookup_lag1: Optional[np.ndarray] = None
        self.shift_lookup_lag2: Optional[np.ndarray] = None
        self.n_clusters: int = 0
        self.max_points_per_cluster: int = 0
        self.n_target_blocks: int = 0
        self.n_target_points: int = 0

    # ------------------------------------------------------------------
    # Covariance; subclasses replace only the correlation shape
    # ------------------------------------------------------------------

    def _correlation(self, scaled_distance: torch.Tensor) -> torch.Tensor:
        """Evaluate the standard-range closed-form Matérn correlation.

        ``scaled_distance`` is geometric distance divided by the reported
        range.  The Bessel argument is therefore ``sqrt(2 * smooth)`` times
        this value, matching the arbitrary-smoothness spline implementation
        and the package's spatial Matérn models.
        """
        if self.smooth == 0.5:
            return torch.exp(-scaled_distance)
        sqrt_three_distance = np.sqrt(3.0) * scaled_distance
        return (1.0 + sqrt_three_distance) * torch.exp(-sqrt_three_distance)

    def batched_covariance(self, params: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        """Return covariance matrices for batches of joint conditioning sets.

        ``x_batch`` has shape ``(batch, points, 3)`` and stores latitude,
        longitude, and time.  Each matrix is a covariance of one point set with
        itself, so the modeled nugget and a ``1e-6`` numerical Cholesky
        stabilization are added to its diagonal.
        """
        params = self._validated_params(params)
        self._validate_batched_coordinates(x_batch, params.device)
        return self._batched_covariance(params, x_batch)

    def _batched_covariance(
        self,
        params: torch.Tensor,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked covariance kernel for likelihood hot loops."""
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = self._nugget_from_params(params)
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        d = self._batched_anisotropic_distance_unchecked(dist_params, x_batch)
        scaled_d = d * phi2
        cov = (phi1 / phi2) * self._correlation(scaled_d)
        cov.diagonal(dim1=-2, dim2=-1).add_(nugget + 1e-6)
        return cov

    def _supports_native_covariance(self) -> bool:
        """Return whether this model has the exact fused-kernel covariance."""

        return (
            self.smooth == 0.5
            and self.covariance_parameter_count == 7
            and type(self)._correlation is GroupedBatchedVecchia._correlation
        )

    def resolved_covariance_backend(self) -> str:
        """Return the backend that will be used for likelihood covariance chunks."""

        first_value = next(iter(self.input_map.values()))
        native_ready = (
            self._supports_native_covariance()
            and isinstance(first_value, torch.Tensor)
            and first_value.device.type in {"cpu", "cuda"}
            and first_value.dtype == torch.float64
            and native_covariance_available(first_value.device)
        )
        if self.covariance_backend == "native":
            return "native" if native_ready else "unavailable"
        if self.covariance_backend == "auto" and native_ready:
            return "native"
        return "torch"

    def _batched_covariance_with_dummy(
        self,
        params: torch.Tensor,
        coordinates: torch.Tensor,
        is_dummy: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble one covariance chunk using the matching native device path."""

        if self.covariance_backend != "torch" and self._supports_native_covariance():
            if (
                self.covariance_backend == "native"
                or self.resolved_covariance_backend() == "native"
            ):
                return native_covariance(
                    params,
                    coordinates,
                    is_dummy,
                    smooth=self.smooth,
                    backend=self.covariance_backend,
                )
        elif self.covariance_backend == "native":
            raise RuntimeError(
                "native covariance is available only for the seven-parameter "
                "smooth=0.5 closed-form Matern model"
            )

        covariance = self._batched_covariance(params, coordinates)
        return self._decouple_dummy_covariance(covariance, is_dummy)

    def point_covariance(
        self,
        params: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return pointwise covariance between the rows of ``x`` and ``y``.

        Rows use ``[latitude, longitude, response, time, ...]``; response and
        later mean-design columns are ignored.  The result has shape
        ``(len(x), len(y))``.  The modeled measurement nugget is added at every
        pairwise exact latitude/longitude/time match, including permuted or
        partially overlapping point sets.  The ``1e-8`` numerical jitter is
        restricted to the diagonal when ``x`` and ``y`` identify the same ordered point set.
        The API assumes exact space-time coordinates
        uniquely identify observations.
        """
        params = self._validated_params(params)
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = self._nugget_from_params(params)
        advec_lat = params[4]
        advec_lon = params[5]
        sigmasq = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance = self.pairwise_anisotropic_distance(dist_params, x, y)
        scaled_d = distance * phi2

        cov = sigmasq * self._correlation(scaled_d)

        x_coordinates = x[:, [0, 1, 3]]
        y_coordinates = y[:, [0, 1, 3]]
        same_observation = torch.all(
            x_coordinates[:, None, :] == y_coordinates[None, :, :],
            dim=2,
        )
        cov = cov + same_observation.to(cov.dtype) * nugget
        if self._same_ordered_points(x, y):
            eye = torch.eye(x.shape[0], device=cov.device, dtype=cov.dtype)
            cov = cov + eye * 1e-8
        return cov

    # ------------------------------------------------------------------
    # Cluster construction
    # ------------------------------------------------------------------

    def _grid_coords_np(self, n_points: int) -> np.ndarray:
        if self.grid_coords is not None:
            coords = np.asarray(self.grid_coords, dtype=np.float64)
        else:
            first = next(iter(self.input_map.values()))
            if isinstance(first, torch.Tensor):
                coords = first[:n_points, :2].detach().cpu().numpy().astype(np.float64)
            else:
                coords = np.asarray(first[:n_points, :2], dtype=np.float64)
        if coords.shape != (n_points, 2):
            raise ValueError(f"grid_coords must have shape ({n_points}, 2), got {coords.shape}")
        if not np.isfinite(coords).all():
            source = (
                "explicit grid_coords" if self.grid_coords is not None else "inferred coordinates"
            )
            raise ValueError(
                f"{source} must be finite; pass regular grid_coords when missing "
                "responses also have missing source coordinates"
            )
        return coords

    @staticmethod
    def _unique_inverse(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rounded = np.round(values.astype(np.float64), 10)
        unique = np.unique(rounded)
        lookup = {v: i for i, v in enumerate(unique)}
        inverse = np.array([lookup[v] for v in rounded], dtype=np.int64)
        return unique, inverse

    @staticmethod
    def _as_zero_based_order(order: np.ndarray, n: int) -> np.ndarray:
        order = np.asarray(order, dtype=np.int64)
        if order.size != n:
            raise ValueError(f"max-min order length {order.size} != n_clusters {n}")
        if order.min() == 1 and order.max() == n:
            order = order - 1
        if order.min() < 0 or order.max() >= n:
            raise ValueError("max-min order has indices outside [0, n_clusters)")
        if np.unique(order).size != n:
            raise ValueError("max-min order must be a permutation without duplicates")
        return order

    def _build_clusters(self, n_points: int):
        coords = self._grid_coords_np(n_points)
        _, row_idx = self._unique_inverse(coords[:, 0])
        _, col_idx = self._unique_inverse(coords[:, 1])

        # Convert native grid row/column indices into rectangular block IDs.
        # For block_shape=(3, 3), every full interior block contains 9 grid
        # points.  Edge blocks can be smaller if the grid dimensions are not
        # exact multiples of 3.
        br = self.block_shape[0]
        bc = self.block_shape[1]
        block_keys = list(zip((row_idx // br).tolist(), (col_idx // bc).tolist()))
        raw: Dict[Tuple[int, int], List[int]] = {}
        for local_idx, key in enumerate(block_keys):
            raw.setdefault(key, []).append(local_idx)

        raw_keys = sorted(raw)
        raw_points = [np.array(raw[k], dtype=np.int64) for k in raw_keys]
        raw_centroids = np.vstack([coords[p].mean(axis=0) for p in raw_points])

        # The Vecchia ordering is on cluster centroids, not individual points.
        # After this permutation, every cluster index is also its ordering rank.
        order = self._as_zero_based_order(orderings.maxmin_order(raw_centroids), len(raw_points))
        self.cluster_points = [raw_points[i] for i in order]
        self.cluster_centroids = raw_centroids[order]
        self.n_clusters = len(self.cluster_points)
        self.max_points_per_cluster = max(len(p) for p in self.cluster_points)

        # cluster_nns[k] stores nearest neighbor cluster indices that precede k
        # in the max-min order.  It is reused both for the same-time A set and
        # for local lagged blocks.
        max_blocks = self.max_neighbor_search
        if max_blocks is None:
            max_blocks = (
                max(
                    self.n_neighbor_blocks_t,
                    self.lag1_local_blocks,
                    self.lag2_local_blocks,
                    self.lag1_shifted_blocks,
                    self.lag2_shifted_blocks,
                    1,
                )
                + 8
            )
        self.cluster_nns = orderings.predecessor_neighbors(
            self.cluster_centroids,
            max_neighbors=int(max_blocks),
        )
        self.shift_lookup_lag1 = self._build_shift_lookup(lon_offset=self.lag1_lon_offset)
        self.shift_lookup_lag2 = self._build_shift_lookup(lon_offset=self.lag2_lon_offset)

    def _build_shift_lookup(self, lon_offset: float) -> np.ndarray:
        if self.cluster_centroids is None:
            raise RuntimeError("Clusters must be built before shift lookup")
        coords = self.cluster_centroids
        tree = BallTree(np.radians(coords), metric="haversine")
        lats = coords[:, 0]
        lons = coords[:, 1]
        lon_min = float(np.nanmin(lons))
        lon_max = float(np.nanmax(lons))
        base_ids = np.arange(coords.shape[0], dtype=np.int64)

        # Shift each cluster centroid in longitude and snap it back to the
        # nearest cluster centroid.  This is the block-level analog of the
        # pointwise hybrid shifted-center neighbor lookup.
        target_lons = lons + float(lon_offset)
        outside = (target_lons < lon_min) | (target_lons > lon_max)
        query = np.column_stack([np.radians(lats), np.radians(target_lons)])
        _, idx = tree.query(query, k=1)
        lookup = idx.flatten().astype(np.int64)

        # Near the domain boundary, do not create an invalid shifted block.
        # Falling back to the original block keeps the precompute deterministic;
        # later duplicate filtering may still replace it with nearby fresh
        # candidates when requested.
        lookup[outside] = base_ids[outside]
        return lookup

    def _fresh_block_candidates(self, center_block: int, count: int) -> List[int]:
        if count <= 0:
            return []
        candidates = [int(center_block)]
        if self.cluster_nns is not None and 0 <= center_block < len(self.cluster_nns):
            candidates.extend(int(v) for v in self.cluster_nns[center_block] if int(v) >= 0)
        out: List[int] = []
        for cand in candidates:
            if cand in out:
                continue
            out.append(cand)
        # Return more than "count" candidates on purpose.  The caller applies
        # the cap after checking duplicates against same/local lag blocks.  This
        # fallback matters because a 3-by-3 block plus a small shift can map
        # back to a block already included by same/local conditioning.
        return out

    def _conditioning_blocks(
        self, block_idx: int, time_idx: int
    ) -> Tuple[str, int, List[Tuple[int, int]]]:
        # local_prev is the same-time spatial neighbor list for this target
        # block, restricted to clusters that are earlier in the max-min Vecchia
        # ordering.  The lagged "local" blocks are prefixes of this same list,
        # so B/C inherit the current-time spatial context at a smaller budget.
        local_prev = [
            int(v) for v in self.cluster_nns[block_idx] if int(v) >= 0 and int(v) < block_idx
        ]

        cond: List[Tuple[int, int]] = []

        # A set: current time t, previous blocks only.
        # This is the block version of pointwise same-time NN conditioning.
        self._append_unique_block_times(cond, time_idx, local_prev, self.n_neighbor_blocks_t)
        label = "A"
        max_cond_blocks = self.n_neighbor_blocks_t

        if time_idx > 0:
            label = "AB"
            prev_time = time_idx - 1

            # B same: the same spatial block at t-1.  For advection models this
            # captures persistence before adding shifted upstream information.
            if self.lag1_same_block:
                self._append_unique_block_times(cond, prev_time, [block_idx], 1)

            # B local: reuse the nearest blocks from the current-time A list,
            # but at t-1.  Default is 3 blocks.
            self._append_unique_block_times(cond, prev_time, local_prev, self.lag1_local_blocks)

            # B fresh: shift the current block centroid by lag1_lon_offset and
            # add a fresh block around that shifted center.  If it overlaps with
            # same/local, _append_unique_block_times skips it and uses the next
            # candidate returned by _fresh_block_candidates.
            center = int(self.shift_lookup_lag1[block_idx])
            fresh = self._fresh_block_candidates(center, self.lag1_shifted_blocks)
            self._append_unique_block_times(cond, prev_time, fresh, self.lag1_shifted_blocks)
            max_cond_blocks += (
                int(self.lag1_same_block) + self.lag1_local_blocks + self.lag1_shifted_blocks
            )

        if time_idx >= self.second_lag_stride:
            label = "ABC"
            prev2_time = time_idx - self.second_lag_stride

            # C same: same spatial block at the configured second-lag slice.
            # With second_lag_stride=2 this is the t-2 conditioning layer.
            if self.lag2_same_block:
                self._append_unique_block_times(cond, prev2_time, [block_idx], 1)

            # C local: a smaller prefix of the same A neighbor list at t-2.
            # Default is 2 blocks, intentionally lighter than B local.
            self._append_unique_block_times(cond, prev2_time, local_prev, self.lag2_local_blocks)

            # C fresh: block-level shifted-center fresh conditioning at t-2.
            # This candidate version allows lag2_lon_offset to be set equal to
            # lag1_lon_offset, rather than forcing a 2x multiplier.
            center = int(self.shift_lookup_lag2[block_idx])
            fresh = self._fresh_block_candidates(center, self.lag2_shifted_blocks)
            self._append_unique_block_times(cond, prev2_time, fresh, self.lag2_shifted_blocks)
            max_cond_blocks += (
                int(self.lag2_same_block) + self.lag2_local_blocks + self.lag2_shifted_blocks
            )

        # The batching layer pads conditioning rows to the maximum possible
        # number of points for the label.  If edge blocks are smaller or missing
        # values remove points, dummies are prepended later so all rows in the
        # batch share the same tensor shape.
        return label, max_cond_blocks * self.max_points_per_cluster, cond

    @staticmethod
    def _append_unique_block_times(
        out: List[Tuple[int, int]],
        time_idx: int,
        block_candidates: Sequence[int],
        cap: int,
    ):
        added = 0
        seen = set(out)
        for block_idx in block_candidates:
            if added >= cap:
                break
            key = (int(time_idx), int(block_idx))
            if key in seen:
                continue
            out.append(key)
            seen.add(key)
            added += 1

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _prepared_time_slices(self) -> List[torch.Tensor]:
        """Validate time slices and normalize numerical work to float64."""
        prepared: List[torch.Tensor] = []
        expected_columns: Optional[int] = None
        time_ranges: List[Tuple[float, float]] = []

        for key, value in self.input_map.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                tensor = value
            else:
                raise TypeError(f"input_map[{key!r}] must be a NumPy array or torch.Tensor")
            if tensor.ndim != 2:
                raise ValueError(
                    f"input_map[{key!r}] must be two-dimensional, got {tuple(tensor.shape)}"
                )
            if tensor.shape[0] == 0:
                raise ValueError(f"input_map[{key!r}] must contain at least one grid point")
            if tensor.shape[1] < 4:
                raise ValueError(
                    f"input_map[{key!r}] needs at least [lat, lon, response, time] columns"
                )
            if 4 < tensor.shape[1] < 11:
                raise ValueError(
                    f"input_map[{key!r}] must have either 4 columns or at least "
                    "11 columns including seven time indicators"
                )
            if expected_columns is None:
                expected_columns = int(tensor.shape[1])
            elif tensor.shape[1] != expected_columns:
                raise ValueError("all input_map time slices must have the same number of columns")

            tensor = tensor.to(self.device, dtype=torch.float64)
            if not torch.isfinite(tensor[:, 3]).all():
                raise ValueError(f"input_map[{key!r}] contains non-finite time values")
            valid_response = torch.isfinite(tensor[:, 2])
            if valid_response.any() and not torch.isfinite(tensor[valid_response][:, [0, 1]]).all():
                raise ValueError(
                    f"input_map[{key!r}] observed responses require finite "
                    "latitude and longitude"
                )
            if tensor.shape[1] >= 11 and valid_response.any():
                if not torch.isfinite(tensor[valid_response, 4:11]).all():
                    raise ValueError(f"input_map[{key!r}] contains non-finite mean-design values")

            time_values = tensor[:, 3]
            if torch.unique(time_values).numel() != 1:
                raise ValueError(
                    f"input_map[{key!r}] must represent exactly one time value; "
                    "split mixed-time rows into separate slices"
                )
            time_ranges.append((float(time_values.min().item()), float(time_values.max().item())))
            prepared.append(tensor)

        for previous, current in zip(time_ranges, time_ranges[1:]):
            if previous[1] >= current[0]:
                raise ValueError(
                    "input_map insertion order must be strictly chronological with "
                    "non-overlapping time ranges"
                )
        return prepared

    def _precompute_message(self) -> str:
        return (
            "Pre-computing grouped-batch Vecchia "
            f"(smooth={self.smooth}, block={self.block_shape}, "
            f"A={self.n_neighbor_blocks_t}, "
            f"B=same{int(self.lag1_same_block)}+local{self.lag1_local_blocks}+fresh{self.lag1_shifted_blocks}, "
            f"C=same{int(self.lag2_same_block)}+local{self.lag2_local_blocks}+fresh{self.lag2_shifted_blocks}, "
            f"offsets={self.lag1_lon_offset:.4f}/{self.lag2_lon_offset:.4f})..."
        )

    def precompute_conditioning_sets(self):
        """Build and batch every valid grouped Vecchia conditional."""

        logger.info(self._precompute_message())

        all_data_list = self._prepared_time_slices()
        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if len(set(day_lengths)) != 1:
            raise ValueError(
                "GroupedBatchedVecchia requires equal grid length per time, " f"got {day_lengths}"
            )

        n_grid = day_lengths[0]
        self._build_clusters(n_grid)

        real_data = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = real_data.shape
        is_missing_real = ~torch.isfinite(real_data[:, 2])
        is_missing_np = is_missing_real.detach().cpu().numpy()

        valid_lats = real_data[~is_missing_real, 0]
        if valid_lats.numel() == 0:
            raise ValueError("no valid target observations remain after applying the response mask")
        self.lat_mean_val = valid_lats.mean().item()

        valid_data = real_data[~is_missing_real]
        valid_design = torch.cat(
            [
                torch.ones_like(valid_data[:, :1]),
                valid_data[:, :1] - self.lat_mean_val,
                (
                    valid_data[:, 4:11]
                    if valid_data.shape[1] >= 11
                    else valid_data.new_zeros((valid_data.shape[0], 7))
                ),
            ],
            dim=1,
        )
        active_mask = torch.any(valid_design != 0, dim=0)
        active_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
        active_design = valid_design[:, active_indices]
        design_rank = int(torch.linalg.matrix_rank(active_design).item())
        if design_rank != active_design.shape[1]:
            raise ValueError(
                "mean design is rank deficient after removing structurally zero columns"
            )
        self._active_feature_indices = active_indices

        max_cond_blocks = (
            self.n_neighbor_blocks_t
            + int(self.lag1_same_block)
            + self.lag1_local_blocks
            + self.lag1_shifted_blocks
            + int(self.lag2_same_block)
            + self.lag2_local_blocks
            + self.lag2_shifted_blocks
        )
        n_dummies = max(1, max_cond_blocks * self.max_points_per_cluster)
        dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float64)
        for k in range(n_dummies):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        full_data = torch.cat([real_data, dummy_block], dim=0).contiguous()
        dummy_start = n_real

        cumulative_len = np.cumsum([0] + day_lengths)
        batch_rows: Dict[Tuple[str, int, int], List[List[int]]] = {}

        self.n_target_blocks = 0
        self.n_target_points = 0

        for time_idx in range(len(all_data_list)):
            offset = int(cumulative_len[time_idx])
            for block_idx, point_locals in enumerate(self.cluster_points):
                target = [
                    offset + int(p) for p in point_locals if not is_missing_np[offset + int(p)]
                ]
                if len(target) < self.min_target_points:
                    continue

                label, max_cond_points, cond_block_refs = self._conditioning_blocks(
                    block_idx, time_idx
                )
                cond_points: List[int] = []
                seen_points = set()
                for cond_time, cond_block in cond_block_refs:
                    cond_offset = int(cumulative_len[cond_time])
                    for p in self.cluster_points[cond_block]:
                        g = cond_offset + int(p)
                        if g in seen_points or is_missing_np[g]:
                            continue
                        cond_points.append(g)
                        seen_points.add(g)

                if len(cond_points) < max_cond_points:
                    padded = [
                        dummy_start + k for k in range(max_cond_points - len(cond_points))
                    ] + cond_points
                else:
                    padded = cond_points[-max_cond_points:]

                row = padded + target
                key = (label, int(max_cond_points), int(len(target)))
                batch_rows.setdefault(key, []).append(row)
                self.n_target_blocks += 1
                self.n_target_points += len(target)

        if self.n_target_points == 0:
            raise ValueError(
                "no valid target observations remain after applying the response mask "
                "and min_target_points"
            )

        self._cluster_batches = []

        def build_batch(key: Tuple[str, int, int], rows: List[List[int]]) -> _ClusterBatch:
            label, max_cond_points, target_size = key
            indices = torch.tensor(rows, device=self.device, dtype=torch.long)
            joint_data = full_data[indices]
            coordinates = joint_data[..., [0, 1, 3]].contiguous().to(torch.float64)
            response = joint_data[..., 2].unsqueeze(-1).contiguous().to(torch.float64)
            ones = torch.ones_like(joint_data[..., 0]).unsqueeze(-1)
            latitude = (joint_data[..., 0] - self.lat_mean_val).unsqueeze(-1)
            indicators = (
                joint_data[..., 4:11]
                if joint_data.shape[-1] >= 11
                else joint_data.new_zeros((*joint_data.shape[:-1], 7))
            )
            design = torch.cat([ones, latitude, indicators], dim=-1).contiguous().to(torch.float64)
            is_dummy = (indices >= dummy_start).unsqueeze(-1)
            design = design.masked_fill(is_dummy, 0.0)
            response = response.masked_fill(is_dummy, 0.0)
            return _ClusterBatch(
                label=label,
                max_cond_points=max_cond_points,
                target_size=target_size,
                coordinates=coordinates,
                response=response,
                design=design,
                indices=indices,
                is_dummy=is_dummy,
            )

        for key in sorted(batch_rows, key=lambda x: (x[0], x[1], x[2])):
            self._cluster_batches.append(build_batch(key, batch_rows[key]))

        self.is_precomputed = True

        batch_summary = ", ".join(
            f"{b.label}:m{b.max_cond_points}:b{b.target_size}x{b.coordinates.shape[0]}"
            for b in self._cluster_batches[:10]
        )
        more = (
            ""
            if len(self._cluster_batches) <= 10
            else f", ... ({len(self._cluster_batches)} batches)"
        )
        logger.info(
            f"Done. clusters={self.n_clusters}, max_points/block={self.max_points_per_cluster}, "
            f"target_blocks={self.n_target_blocks}, target_points={self.n_target_points}, "
            f"batches=[{batch_summary}{more}]"
        )
        return self

    # ------------------------------------------------------------------
    # Block conditional GLS accumulation
    # ------------------------------------------------------------------

    @staticmethod
    def _decouple_dummy_covariance(
        covariance: torch.Tensor,
        is_dummy: torch.Tensor,
    ) -> torch.Tensor:
        """Make padding rows exactly independent of all real observations.

        Coordinate sentinels alone are insufficient for polynomial-tail
        correlations (and for a spline evaluated only up to a finite radius).
        Dummy rows carry zero response/design, so an identity block removes
        them from the target conditional without changing its covariance.
        """
        dummy = is_dummy.squeeze(-1)
        touches_dummy = dummy.unsqueeze(2) | dummy.unsqueeze(1)
        covariance = covariance.masked_fill(touches_dummy, 0.0)
        covariance.diagonal(dim1=-2, dim2=-1).add_(dummy.to(covariance.dtype))
        return covariance

    def _accumulate_gls_stats(self, params, include_y_quad=True, catch_cholesky=False):
        self._check_precomputed()

        n_active_features = int(self._active_feature_indices.numel())
        xt_sinv_x = torch.zeros(
            (n_active_features, n_active_features), device=self.device, dtype=torch.float64
        )
        xt_sinv_y = torch.zeros((n_active_features, 1), device=self.device, dtype=torch.float64)
        yt_sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        chunk_size = max(1, int(self.target_chunk_size))
        for batch in self._cluster_batches:
            if batch.coordinates.shape[0] == 0:
                continue
            target_slice = slice(batch.max_cond_points, batch.max_cond_points + batch.target_size)
            for start in range(0, batch.coordinates.shape[0], chunk_size):
                end = min(start + chunk_size, batch.coordinates.shape[0])
                cov_chunk = self._batched_covariance_with_dummy(
                    params,
                    batch.coordinates[start:end],
                    batch.is_dummy[start:end],
                )
                try:
                    L_chunk = torch.linalg.cholesky(cov_chunk)
                except torch.linalg.LinAlgError:
                    if catch_cholesky:
                        self._log_cholesky_failure(params, f"Cluster tails {batch.label}")
                        return None
                    raise

                design_chunk = batch.design[start:end]
                if n_active_features != self.n_features:
                    design_chunk = design_chunk.index_select(
                        2,
                        self._active_feature_indices,
                    )
                whitened_system = torch.linalg.solve_triangular(
                    L_chunk,
                    torch.cat([design_chunk, batch.response[start:end]], dim=2),
                    upper=False,
                )

                target_design = whitened_system[:, target_slice, :n_active_features].reshape(
                    -1,
                    n_active_features,
                )
                target_response = whitened_system[:, target_slice, -1:].reshape(-1, 1)
                diag_L = torch.diagonal(L_chunk, dim1=1, dim2=2)[:, target_slice]

                log_det += 2.0 * torch.sum(torch.log(diag_L))
                xt_sinv_x += target_design.T @ target_design
                xt_sinv_y += target_design.T @ target_response
                if include_y_quad:
                    yt_sinv_y += (target_response.T @ target_response).squeeze()

        total_n = int(self.n_target_points)
        return xt_sinv_x, xt_sinv_y, yt_sinv_y, log_det, total_n

    def cluster_summary(self) -> Dict[str, float | str]:
        """Return geometry and batching counts after precomputation."""

        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first")
        return {
            "n_clusters": self.n_clusters,
            "block_shape_lat": self.block_shape[0],
            "block_shape_lon": self.block_shape[1],
            "max_points_per_cluster": self.max_points_per_cluster,
            "n_target_blocks": self.n_target_blocks,
            "n_target_points": self.n_target_points,
            "n_batches": len(self._cluster_batches),
            "target_chunk_size": self.target_chunk_size,
            "covariance_backend": self.resolved_covariance_backend(),
        }


__all__ = ["GroupedBatchedVecchia"]
