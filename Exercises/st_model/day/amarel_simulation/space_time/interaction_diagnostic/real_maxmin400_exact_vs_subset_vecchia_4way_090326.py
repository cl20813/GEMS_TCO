#!/usr/bin/env python3
"""Four-way exact/Vecchia and full-eigen/Lanczos diagnostic on 8 x 400 data.

This experiment separates two errors that must not be conflated:

    E1  exact K, full eigendecomposition
    E2  exact Omega=K^-1, precision-operator Lanczos/SLQ
    V1  subset-specific Vecchia Omega~=B.T B, full eigendecomposition
    V2  the same Omega~, sparse-operator Lanczos/SLQ

All four methods use the same 400 common-valid locations, eight times, stored
adapted lag-4/3/2 parameters, and full-data GLS residual.  Locations can be a
global max-min sample or a contiguous native-grid rectangle.  The Vecchia
problem is rebuilt using only the selected observations and nonempty subset
blocks; it is not a principal submatrix of the full-data precision.  Native-grid
block IDs, centroids, footprints, and width are retained so that the 4x4
corridor construction is not distorted by compressing max-min gaps.

The E1--V1 comparison is performed by covariance-mode rank, because the two
operators have different eigenvalue thresholds.  Band-subspace distances and
the expected V1 band energy when residuals are generated from exact K quantify
Vecchia-induced spectral distortion independently of the observed residual.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import scipy.sparse
import torch
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
VECCHIA_APPROX = HERE.parent / "vecchia_approximation"
for path in (HERE, SRC, VECCHIA_APPROX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import vecchia_adapted_fixed_lag643_core as core  # noqa: E402
from GEMS_TCO import orderings  # noqa: E402
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (  # noqa: E402
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
)
from real_full_vecchia_precision_lanczos_090326 import (  # noqa: E402
    ascending_rank_thresholds,
    evaluate_precision_curves,
)
from real_maxmin400_full_eigen_vs_lanczos_090326 import (  # noqa: E402
    LanczosRun,
    atomic_csv,
    atomic_json,
    build_dense_covariance,
    fitted_physical,
    lanczos_tridiagonal,
    load_fitted_record,
    loader_args,
    parse_date,
    parse_steps,
    quadrature_atoms,
    recompute_gls_beta,
    select_common_maxmin,
    subset_residual_data,
)
from vecchia_sparse_precision_operator_090326 import (  # noqa: E402
    build_sparse_vecchia_precision,
)


DTYPE = torch.float64


class NativeGridSubsetDirectionalLag432(
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit
):
    """Subset-only neighbors with block geometry inherited from the native grid.

    Passing only max-min points to the stock class would compress missing grid
    rows/columns before forming 4x4 blocks.  Passing the full grid with missing
    responses would instead leave mostly empty blocks in the fixed neighbor
    lists.  This override uses only nonempty subset blocks for ordering and
    neighbors, while deriving each block's centroid and footprint from all
    native-grid cells belonging to that original block.
    """

    def __init__(self, *args: Any, full_grid_coords: np.ndarray, selected: np.ndarray, **kwargs: Any):
        full = np.asarray(full_grid_coords, dtype=np.float64)
        chosen = np.asarray(selected, dtype=np.int64)
        unique_lat, row_index = self._unique_inverse(full[:, 0])
        unique_lon, col_index = self._unique_inverse(full[:, 1])
        block_rows = np.floor_divide(row_index, 4)
        block_cols = np.floor_divide(col_index, 4)
        full_keys = list(zip(block_rows.tolist(), block_cols.tolist()))
        geometry: dict[tuple[int, int], dict[str, float | np.ndarray]] = {}
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, key in enumerate(full_keys):
            grouped.setdefault(key, []).append(index)
        for key, indices in grouped.items():
            points = full[np.asarray(indices, dtype=np.int64)]
            geometry[key] = {
                "centroid": points.mean(axis=0),
                "lat_min": float(points[:, 0].min()),
                "lat_max": float(points[:, 0].max()),
                "lon_min": float(points[:, 1].min()),
                "lon_max": float(points[:, 1].max()),
            }
        self._subset_native_block_keys = [full_keys[int(index)] for index in chosen]
        self._native_block_geometry = geometry
        self._native_grid_lon_step = float(np.median(np.diff(unique_lon)))
        self._native_grid_lat_levels = int(len(unique_lat))
        self._native_grid_lon_levels = int(len(unique_lon))
        super().__init__(*args, **kwargs)

    def _build_clusters(self, n_points: int) -> None:
        coords = self._grid_coords_np(n_points)
        if n_points != len(self._subset_native_block_keys):
            raise RuntimeError("Subset native-block key count does not match input rows")
        raw: dict[tuple[int, int], list[int]] = {}
        for local_index, key in enumerate(self._subset_native_block_keys):
            raw.setdefault(key, []).append(local_index)
        raw_keys = sorted(raw)
        raw_points = [np.asarray(raw[key], dtype=np.int64) for key in raw_keys]
        raw_centroids = np.vstack(
            [self._native_block_geometry[key]["centroid"] for key in raw_keys]
        ).astype(np.float64)
        raw_lat_min = np.asarray(
            [self._native_block_geometry[key]["lat_min"] for key in raw_keys],
            dtype=np.float64,
        )
        raw_lat_max = np.asarray(
            [self._native_block_geometry[key]["lat_max"] for key in raw_keys],
            dtype=np.float64,
        )
        raw_lon_min = np.asarray(
            [self._native_block_geometry[key]["lon_min"] for key in raw_keys],
            dtype=np.float64,
        )
        raw_lon_max = np.asarray(
            [self._native_block_geometry[key]["lon_max"] for key in raw_keys],
            dtype=np.float64,
        )
        order = self._as_zero_based_order(
            orderings.maxmin_cpp(raw_centroids), len(raw_points)
        )
        self.cluster_points = [raw_points[index] for index in order]
        self.cluster_centroids = raw_centroids[order]
        self.cluster_lat_min = raw_lat_min[order]
        self.cluster_lat_max = raw_lat_max[order]
        self.cluster_lon_min = raw_lon_min[order]
        self.cluster_lon_max = raw_lon_max[order]
        self.n_clusters = len(self.cluster_points)
        self.max_points_per_cluster = max(len(points) for points in self.cluster_points)
        self.grid_lon_step = self._native_grid_lon_step
        self.corridor_block_lon_width = float(self.block_shape[1] * self.grid_lon_step)

        max_blocks = self.max_neighbor_search
        if max_blocks is None:
            max_blocks = max(
                self.lag0_block_count,
                self.lag1_max_blocks,
                self.lag2_max_blocks,
                1,
            ) + 8
        self.cluster_nns = orderings.find_nns_l2(
            self.cluster_centroids, max_nn=int(max_blocks)
        )
        self.shift_lookup_lag1 = self._build_shift_lookup(
            lon_offset=self.lag1_lon_offset
        )
        self.shift_lookup_lag2 = self._build_shift_lookup(
            lon_offset=self.lag2_lon_offset
        )

        n_clusters = int(self.cluster_centroids.shape[0])
        k = min(n_clusters, max(1, self.all_neighbor_search + 1))
        tree = cKDTree(np.asarray(self.cluster_centroids, dtype=np.float64))
        self.cluster_all_tree = tree
        _, neighbor_index = tree.query(self.cluster_centroids, k=k)
        neighbor_index = np.asarray(neighbor_index, dtype=np.int64)
        if neighbor_index.ndim == 1:
            neighbor_index = neighbor_index[:, None]
        rows = []
        for cluster_index in range(n_clusters):
            row = [
                int(value)
                for value in neighbor_index[cluster_index]
                if int(value) != cluster_index and int(value) >= 0
            ]
            rows.append(np.asarray(row, dtype=np.int64))
        max_len = max((len(row) for row in rows), default=0)
        all_neighbors = -np.ones((n_clusters, max_len), dtype=np.int64)
        for cluster_index, row in enumerate(rows):
            all_neighbors[cluster_index, : len(row)] = row
        self.cluster_all_nns = all_neighbors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2024-07-03")
    parser.add_argument(
        "--real-data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA"),
    )
    parser.add_argument(
        "--fit-results",
        type=Path,
        default=HERE / "batch_benchmark_results_090326" / "batch_benchmark_results.json",
    )
    parser.add_argument(
        "--fit-initializer",
        type=Path,
        default=HERE / "batch_benchmark_results_090326" / "initializer.json",
    )
    parser.add_argument("--n-spatial", type=int, default=400)
    parser.add_argument(
        "--selection",
        choices=["global-maxmin", "contiguous"],
        default="global-maxmin",
    )
    parser.add_argument("--contiguous-rows", type=int, default=20)
    parser.add_argument("--contiguous-cols", type=int, default=20)
    parser.add_argument("--hours", type=int, default=8, choices=[8])
    parser.add_argument("--smooth", type=float, default=0.5, choices=[0.5])
    parser.add_argument("--target-chunk-size", type=int, default=64, choices=[64])
    parser.add_argument(
        "--lanczos-steps", nargs="+", type=int, default=[64, 128, 256, 512]
    )
    parser.add_argument("--slq-probes", type=int, default=32)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--simulation-replicates", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=20260903)
    parser.add_argument("--vecchia-diagonal-jitter", type=float, default=1e-6)
    parser.add_argument(
        "--reorthogonalization",
        choices=["full", "none"],
        default="full",
    )
    parser.add_argument(
        "--also-no-reorthogonalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "real_maxmin400_exact_vs_subset_vecchia_4way_20240703_090326",
    )
    return parser


def select_common_contiguous(
    asset: core.DayAsset,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Choose a central all-common-valid native-grid rectangular window."""
    started = time.perf_counter()
    arrays = [tensor.detach().cpu().numpy() for tensor in asset.source_map.values()]
    common = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        common &= np.all(np.isfinite(array), axis=1)
    grid = np.asarray(asset.grid_coords, dtype=np.float64)
    rounded_lat = np.round(grid[:, 0], 10)
    rounded_lon = np.round(grid[:, 1], 10)
    unique_lat = np.unique(rounded_lat)
    unique_lon = np.unique(rounded_lon)
    if int(n_rows) > len(unique_lat) or int(n_cols) > len(unique_lon):
        raise ValueError("Requested contiguous window exceeds the native grid")
    lat_lookup = {value: index for index, value in enumerate(unique_lat)}
    lon_lookup = {value: index for index, value in enumerate(unique_lon)}
    grid_index = -np.ones((len(unique_lat), len(unique_lon)), dtype=np.int64)
    for original_index, (lat, lon) in enumerate(zip(rounded_lat, rounded_lon)):
        grid_index[lat_lookup[lat], lon_lookup[lon]] = original_index
    domain_center = np.asarray(
        [(len(unique_lat) - 1) / 2.0, (len(unique_lon) - 1) / 2.0]
    )
    candidates: list[tuple[float, int, int]] = []
    best_score = -1
    for row_start in range(len(unique_lat) - int(n_rows) + 1):
        for col_start in range(len(unique_lon) - int(n_cols) + 1):
            window = grid_index[
                row_start : row_start + int(n_rows),
                col_start : col_start + int(n_cols),
            ]
            present = window >= 0
            valid = present & common[np.maximum(window, 0)]
            score = int(valid.sum())
            best_score = max(best_score, score)
            if score != int(n_rows) * int(n_cols):
                continue
            center = np.asarray(
                [
                    row_start + (int(n_rows) - 1) / 2.0,
                    col_start + (int(n_cols) - 1) / 2.0,
                ]
            )
            normalized = (center - domain_center) / np.asarray(
                [len(unique_lat), len(unique_lon)], dtype=np.float64
            )
            candidates.append((float(normalized @ normalized), row_start, col_start))
    if not candidates:
        raise RuntimeError(
            f"No fully common-valid {n_rows}x{n_cols} window; best valid count={best_score}"
        )
    _, row_start, col_start = min(candidates)
    selected_matrix = grid_index[
        row_start : row_start + int(n_rows),
        col_start : col_start + int(n_cols),
    ]
    selected = selected_matrix.reshape(-1)
    if np.any(selected < 0) or not np.all(common[selected]):
        raise RuntimeError("Chosen contiguous window is not fully common-valid")
    return selected.astype(np.int64), np.arange(len(selected), dtype=np.int64), {
        "selection": "central fully common-valid native-grid rectangle",
        "n_grid": int(len(grid)),
        "n_common_valid": int(common.sum()),
        "n_spatial_selected": int(len(selected)),
        "native_grid_rows": int(len(unique_lat)),
        "native_grid_cols": int(len(unique_lon)),
        "window_rows": int(n_rows),
        "window_cols": int(n_cols),
        "window_row_start": int(row_start),
        "window_col_start": int(col_start),
        "window_lat_min": float(unique_lat[row_start]),
        "window_lat_max": float(unique_lat[row_start + int(n_rows) - 1]),
        "window_lon_min": float(unique_lon[col_start]),
        "window_lon_max": float(unique_lon[col_start + int(n_cols) - 1]),
        "fully_valid_window_count": int(len(candidates)),
        "selection_s": time.perf_counter() - started,
        "coordinate_order": "native grid row-major",
    }


def selected_subset_input_map(
    asset: core.DayAsset,
    selected: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Return only selected common-valid observations in max-min order."""
    keep = torch.as_tensor(np.asarray(selected, dtype=np.int64), dtype=torch.long)
    out: dict[str, torch.Tensor] = {}
    for key, value in asset.source_map.items():
        source = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        out[key] = source[keep].detach().clone().to(device="cpu", dtype=DTYPE).contiguous()
    return out


def build_subset_vecchia(
    asset: core.DayAsset,
    selected: np.ndarray,
    fitted: dict[str, float],
    seed: dict[str, Any],
    beta_global: np.ndarray,
    lat_mean_global: float,
    args: argparse.Namespace,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    subset_map = selected_subset_input_map(asset, selected)
    model = NativeGridSubsetDirectionalLag432(
        smooth=float(args.smooth),
        input_map=subset_map,
        grid_coords=np.asarray(asset.grid_coords, dtype=np.float64)[selected],
        full_grid_coords=asset.grid_coords,
        selected=selected,
        reference_advec_lat=float(seed["seed_lat"]),
        reference_advec_lon=float(seed["seed_lon"]),
        daily_stride=2,
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=1,
    )
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - started
    subset_lat_mean = float(model.lat_mean_val)
    beta_subset = np.asarray(beta_global, dtype=np.float64).copy()
    beta_subset[0] += beta_subset[1] * (subset_lat_mean - float(lat_mean_global))
    params = torch.as_tensor(core.physical_to_raw(fitted), dtype=DTYPE)
    precision = build_sparse_vecchia_precision(
        model,
        params,
        beta_subset,
        chunk_size=int(args.target_chunk_size),
        coefficient_drop_tolerance=0.0,
    )
    n_spatial = int(len(selected))
    desired_globals = np.arange(n_spatial * len(asset.keys), dtype=np.int64)
    desired_compact = precision.global_to_compact[desired_globals]
    if np.any(desired_compact < 0) or np.unique(desired_compact).size != len(desired_compact):
        raise RuntimeError("Subset Vecchia compact ordering does not cover the requested observations")
    # Permuting rows is immaterial to B.T B, but doing both rows and columns
    # makes the stored square factor follow the exact time-major/max-min order.
    whitener = precision.whitener[desired_compact, :][:, desired_compact].tocsr()
    residual = precision.residual[desired_compact]
    metadata = {
        "construction": (
            "selected observations only; nonempty subset blocks reordered by max-min; "
            "block IDs, centroids, footprints, and width inherited from native grid"
        ),
        "uses_full_precision_submatrix": False,
        "uses_observations_outside_subset": False,
        "n_nonempty_subset_blocks": int(model.n_clusters),
        "native_grid_lat_levels": int(model._native_grid_lat_levels),
        "native_grid_lon_levels": int(model._native_grid_lon_levels),
        "precompute_s": precompute_s,
        "subset_lat_mean": subset_lat_mean,
        "global_lat_mean": float(lat_mean_global),
        "beta_intercept_adjustment": float(beta_subset[0] - beta_global[0]),
        "model_summary": model.cluster_summary(),
        "operator": precision.metadata,
        "reordered_whitener_nnz": int(whitener.nnz),
        "reordered_whitener_storage_bytes": int(
            whitener.data.nbytes + whitener.indices.nbytes + whitener.indptr.nbytes
        ),
    }
    del model, precision, subset_map, params
    gc.collect()
    return whitener, residual, metadata


def exact_eigen_curve(
    eigenvalues_precision_ascending: np.ndarray,
    eigenvectors: np.ndarray,
    residual: np.ndarray,
    label: str,
) -> pd.DataFrame:
    mu = np.asarray(eigenvalues_precision_ascending, dtype=np.float64)
    coefficients = eigenvectors.T @ residual
    energy = mu * coefficients**2
    cumulative = np.cumsum(energy)
    n = len(mu)
    return pd.DataFrame(
        {
            "operator": label,
            "rank": np.arange(1, n + 1),
            "rank_fraction": np.arange(1, n + 1) / n,
            "precision_eigenvalue": mu,
            "implied_covariance_eigenvalue": 1.0 / mu,
            "raw_projection": coefficients,
            "standardized_energy": energy,
            "cumulative_energy": cumulative,
            "cumulative_energy_per_n": cumulative / n,
        }
    )


def run_with_probes(
    matvec: Callable[[np.ndarray], np.ndarray],
    residual: np.ndarray,
    probes: np.ndarray,
    max_steps: int,
    reorthogonalization: str,
    label: str,
) -> tuple[LanczosRun, list[LanczosRun], dict[str, float]]:
    print(
        f"{label}: residual + {probes.shape[1]} probes, m={max_steps}, "
        f"reorth={reorthogonalization}",
        flush=True,
    )
    residual_run = lanczos_tridiagonal(
        matvec,
        residual,
        max_steps=max_steps,
        reorthogonalization=reorthogonalization,
    )
    runs: list[LanczosRun] = []
    started = time.perf_counter()
    for index in range(probes.shape[1]):
        run = lanczos_tridiagonal(
            matvec,
            probes[:, index],
            max_steps=max_steps,
            reorthogonalization=reorthogonalization,
        )
        runs.append(run)
        if (index + 1) % 4 == 0 or index + 1 == probes.shape[1]:
            print(
                f"  {label}: probe {index + 1}/{probes.shape[1]}, "
                f"last={run.total_s:.2f}s",
                flush=True,
            )
    return residual_run, runs, {
        "residual_s": residual_run.total_s,
        "probe_s": time.perf_counter() - started,
        "matvec_s": residual_run.matvec_s + sum(run.matvec_s for run in runs),
    }


def numerical_comparison(
    curves: pd.DataFrame,
    exact_curve: pd.DataFrame,
    levels: list[int],
    operator: str,
    reorthogonalization: str,
    n_probes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(exact_curve)
    exact_energy = exact_curve["cumulative_energy"].to_numpy()
    exact_total = float(exact_energy[-1])
    exact_count = np.arange(1, n + 1, dtype=np.float64)
    exact_bridge = (
        exact_energy - exact_count / n * exact_total
    ) / math.sqrt(2.0 * n)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for steps in levels:
        group = curves[curves["lanczos_steps"].eq(steps)].copy()
        group["operator"] = operator
        group["reorthogonalization"] = reorthogonalization
        group["mode_count_exact"] = exact_count
        group["mode_fraction_exact"] = exact_count / n
        group["cumulative_energy_exact"] = exact_energy
        group["cumulative_energy_exact_per_n"] = exact_energy / n
        group["energy_error_per_n"] = (
            group["cumulative_energy_lanczos"] - exact_energy
        ) / n
        group["mode_fraction_error"] = group["mode_fraction_slq"] - exact_count / n
        group["shape_bridge_exact"] = exact_bridge
        frames.append(group)
        summaries.append(
            {
                "operator": operator,
                "reorthogonalization": reorthogonalization,
                "lanczos_steps": int(steps),
                "n_slq_probes": int(n_probes),
                "mode_fraction_rmse": float(
                    np.sqrt(np.mean(group["mode_fraction_error"] ** 2))
                ),
                "mode_fraction_max_abs_error": float(
                    np.max(np.abs(group["mode_fraction_error"]))
                ),
                "energy_per_n_rmse": float(
                    np.sqrt(np.mean(group["energy_error_per_n"] ** 2))
                ),
                "energy_per_n_max_abs_error": float(
                    np.max(np.abs(group["energy_error_per_n"]))
                ),
                "endpoint_energy_relative_error": float(
                    abs(group["cumulative_energy_lanczos"].iloc[-1] / exact_total - 1.0)
                ),
                "shape_D_exact": float(np.max(np.abs(exact_bridge))),
                "shape_D_lanczos_slq": float(np.max(np.abs(group["shape_bridge"]))),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def exact_equal_rank_bands(curve: pd.DataFrame, n_bands: int) -> pd.DataFrame:
    n = len(curve)
    edges = np.linspace(0, n, int(n_bands) + 1).round().astype(int)
    edges[0], edges[-1] = 0, n
    energy = np.r_[0.0, curve["cumulative_energy"].to_numpy()]
    values = curve["implied_covariance_eigenvalue"].to_numpy()
    rows = []
    for band, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        rows.append(
            {
                "operator": str(curve["operator"].iloc[0]),
                "band": band,
                "rank_start": int(lo + 1),
                "rank_end": int(hi),
                "rank_fraction_lo": lo / n,
                "rank_fraction_hi": hi / n,
                "mode_count": int(hi - lo),
                "covariance_eigenvalue_geometric_mid": float(
                    np.sqrt(values[lo] * values[hi - 1])
                ),
                "energy": float(energy[hi] - energy[lo]),
                "energy_per_mode": float((energy[hi] - energy[lo]) / (hi - lo)),
            }
        )
    return pd.DataFrame(rows)


def numerical_band_accuracy(
    numerical_curves: pd.DataFrame,
    exact_curves: pd.DataFrame,
    n_bands: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare hard-band Lanczos/SLQ estimates with operator-specific full eigen."""
    reference_name = {
        "exact_precision": "exact_covariance",
        "subset_vecchia_precision": "subset_vecchia_covariance",
    }
    rows: list[dict[str, Any]] = []
    for (operator, reorth, steps), group in numerical_curves.groupby(
        ["operator", "reorthogonalization", "lanczos_steps"], sort=True
    ):
        reference = exact_curves[
            exact_curves["operator"].eq(reference_name[str(operator)])
        ].sort_values("rank")
        group = group.sort_values("threshold_index")
        n = len(reference)
        edges = np.linspace(0, n, int(n_bands) + 1).round().astype(int)
        edges[0], edges[-1] = 0, n
        exact_energy = np.r_[0.0, reference["cumulative_energy"].to_numpy()]
        approximate_energy = np.r_[
            0.0, group["cumulative_energy_lanczos"].to_numpy()
        ]
        approximate_count = np.r_[0.0, group["mode_count_slq"].to_numpy()]
        for band, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            exact_modes = hi - lo
            estimated_modes = float(approximate_count[hi] - approximate_count[lo])
            exact_value = float(
                (exact_energy[hi] - exact_energy[lo]) / exact_modes
            )
            estimated_value = float(
                (approximate_energy[hi] - approximate_energy[lo]) / estimated_modes
            ) if estimated_modes > 1e-12 else float("nan")
            rows.append(
                {
                    "operator": str(operator),
                    "reorthogonalization": str(reorth),
                    "lanczos_steps": int(steps),
                    "band": band,
                    "rank_start": int(lo + 1),
                    "rank_end": int(hi),
                    "exact_mode_count": int(exact_modes),
                    "estimated_mode_count": estimated_modes,
                    "exact_energy_per_mode": exact_value,
                    "matrix_free_energy_per_mode": estimated_value,
                    "energy_per_mode_error": estimated_value - exact_value,
                }
            )
    bands = pd.DataFrame(rows)
    summary_rows = []
    for (operator, reorth, steps), group in bands.groupby(
        ["operator", "reorthogonalization", "lanczos_steps"], sort=True
    ):
        errors = group["energy_per_mode_error"].dropna().to_numpy()
        summary_rows.append(
            {
                "operator": str(operator),
                "reorthogonalization": str(reorth),
                "lanczos_steps": int(steps),
                "n_bands": int(n_bands),
                "band_energy_per_mode_rmse": float(
                    np.sqrt(np.mean(errors**2))
                ),
                "band_energy_per_mode_max_abs_error": float(
                    np.max(np.abs(errors))
                ),
            }
        )
    return bands, pd.DataFrame(summary_rows)


def approximation_metrics(
    exact_curve: pd.DataFrame,
    vecchia_curve: pd.DataFrame,
    q_exact: np.ndarray,
    q_vecchia: np.ndarray,
    n_bands: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], np.ndarray]:
    n = len(exact_curve)
    lambda_exact = exact_curve["implied_covariance_eigenvalue"].to_numpy()
    lambda_vecchia = vecchia_curve["implied_covariance_eigenvalue"].to_numpy()
    mu_exact = 1.0 / lambda_exact
    mu_vecchia = 1.0 / lambda_vecchia
    cross = q_exact.T @ q_vecchia
    cross_sq = cross**2
    expected_vecchia_energy = mu_vecchia * np.sum(
        lambda_exact[:, None] * cross_sq, axis=0
    )

    exact_bands = exact_equal_rank_bands(exact_curve, n_bands)
    vecchia_bands = exact_equal_rank_bands(vecchia_curve, n_bands)
    band_table = exact_bands.merge(
        vecchia_bands,
        on=["band", "rank_start", "rank_end", "rank_fraction_lo", "rank_fraction_hi", "mode_count"],
        suffixes=("_exact", "_vecchia"),
    )
    edges = np.linspace(0, n, int(n_bands) + 1).round().astype(int)
    expected_rows = []
    subspace_rows = []
    for band, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        singular_values = scipy.linalg.svdvals(cross[lo:hi, lo:hi])
        k = hi - lo
        normalized_projector_distance = math.sqrt(
            max(0.0, 1.0 - float(np.sum(singular_values**2)) / k)
        )
        expected_rows.append(
            {
                "band": band,
                "rank_start": int(lo + 1),
                "rank_end": int(hi),
                "rank_fraction_lo": lo / n,
                "rank_fraction_hi": hi / n,
                "expected_exact_energy_per_mode_under_exact_K": 1.0,
                "expected_vecchia_energy_per_mode_under_exact_K": float(
                    np.mean(expected_vecchia_energy[lo:hi])
                ),
            }
        )
        clipped = np.clip(singular_values, 0.0, 1.0)
        angles = np.degrees(np.arccos(clipped))
        subspace_rows.append(
            {
                "band": band,
                "rank_start": int(lo + 1),
                "rank_end": int(hi),
                "normalized_projector_frobenius_distance": normalized_projector_distance,
                "mean_squared_canonical_correlation": float(np.mean(singular_values**2)),
                "principal_angle_median_deg": float(np.median(angles)),
                "principal_angle_max_deg": float(np.max(angles)),
            }
        )
    expected_bands = pd.DataFrame(expected_rows)
    subspaces = pd.DataFrame(subspace_rows)

    exact_cumulative = exact_curve["cumulative_energy_per_n"].to_numpy()
    vecchia_cumulative = vecchia_curve["cumulative_energy_per_n"].to_numpy()
    band_difference = (
        band_table["energy_per_mode_vecchia"] - band_table["energy_per_mode_exact"]
    ).to_numpy()
    trace_k_kv = float(
        np.sum(lambda_exact[:, None] * lambda_vecchia[None, :] * cross_sq)
    )
    frobenius_difference_sq = float(
        np.sum(lambda_exact**2) + np.sum(lambda_vecchia**2) - 2.0 * trace_k_kv
    )
    trace_omega_v_k = float(np.sum(expected_vecchia_energy))
    trace_omega_e_kv = float(
        np.sum(mu_exact[:, None] * lambda_vecchia[None, :] * cross_sq)
    )
    logdet_exact = float(np.sum(np.log(lambda_exact)))
    logdet_vecchia = float(np.sum(np.log(lambda_vecchia)))
    metrics = {
        "real_residual_curve_rmse_per_n": float(
            np.sqrt(np.mean((vecchia_cumulative - exact_cumulative) ** 2))
        ),
        "real_residual_curve_max_abs_per_n": float(
            np.max(np.abs(vecchia_cumulative - exact_cumulative))
        ),
        "real_residual_endpoint_difference_per_n": float(
            vecchia_cumulative[-1] - exact_cumulative[-1]
        ),
        "real_residual_band_energy_per_mode_rmse": float(
            np.sqrt(np.mean(band_difference**2))
        ),
        "real_residual_band_energy_per_mode_max_abs": float(
            np.max(np.abs(band_difference))
        ),
        "covariance_eigenvalue_log_ratio_rmse": float(
            np.sqrt(np.mean(np.log(lambda_vecchia / lambda_exact) ** 2))
        ),
        "covariance_eigenvalue_log_ratio_max_abs": float(
            np.max(np.abs(np.log(lambda_vecchia / lambda_exact)))
        ),
        "covariance_relative_frobenius_error": float(
            math.sqrt(max(0.0, frobenius_difference_sq))
            / np.linalg.norm(lambda_exact)
        ),
        "kl_exact_to_vecchia_per_observation": float(
            0.5 * (trace_omega_v_k - n + logdet_vecchia - logdet_exact) / n
        ),
        "kl_vecchia_to_exact_per_observation": float(
            0.5 * (trace_omega_e_kv - n + logdet_exact - logdet_vecchia) / n
        ),
        "mean_expected_vecchia_energy_under_exact_K": float(
            np.mean(expected_vecchia_energy)
        ),
        "max_abs_expected_vecchia_band_energy_minus_one": float(
            np.max(
                np.abs(
                    expected_bands["expected_vecchia_energy_per_mode_under_exact_K"] - 1.0
                )
            )
        ),
    }
    return band_table, expected_bands, subspaces, metrics, cross


def simulate_exact_residual_distortion(
    lambda_exact: np.ndarray,
    mu_vecchia: np.ndarray,
    cross_exact_to_vecchia: np.ndarray,
    n_bands: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(lambda_exact)
    z = rng.standard_normal((n, int(n_replicates)))
    exact_energy = z**2
    vecchia_projection = cross_exact_to_vecchia.T @ (
        np.sqrt(lambda_exact)[:, None] * z
    )
    vecchia_energy = mu_vecchia[:, None] * vecchia_projection**2
    exact_cumulative = np.cumsum(exact_energy, axis=0)
    vecchia_cumulative = np.cumsum(vecchia_energy, axis=0)
    edges = np.linspace(0, n, int(n_bands) + 1).round().astype(int)
    rows = []
    for replicate in range(int(n_replicates)):
        band_differences = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            band_differences.append(
                float(
                    np.mean(vecchia_energy[lo:hi, replicate])
                    - np.mean(exact_energy[lo:hi, replicate])
                )
            )
        difference = (vecchia_cumulative[:, replicate] - exact_cumulative[:, replicate]) / n
        rows.append(
            {
                "replicate": replicate + 1,
                "curve_rmse_per_n": float(np.sqrt(np.mean(difference**2))),
                "curve_max_abs_per_n": float(np.max(np.abs(difference))),
                "endpoint_difference_per_n": float(difference[-1]),
                "band_energy_per_mode_rmse": float(
                    np.sqrt(np.mean(np.asarray(band_differences) ** 2))
                ),
                "band_energy_per_mode_max_abs": float(
                    np.max(np.abs(band_differences))
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_four_way(
    exact_curve: pd.DataFrame,
    vecchia_curve: pd.DataFrame,
    numerical_curves: pd.DataFrame,
    band_table: pd.DataFrame,
    expected_bands: pd.DataFrame,
    subspaces: pd.DataFrame,
    levels: list[int],
    output_dir: Path,
) -> Path:
    max_steps = max(levels)
    chosen = numerical_curves[
        numerical_curves["lanczos_steps"].eq(max_steps)
        & numerical_curves["reorthogonalization"].eq("full")
    ]
    exact_lanczos = chosen[chosen["operator"].eq("exact_precision")]
    vecchia_lanczos = chosen[chosen["operator"].eq("subset_vecchia_precision")]
    x = exact_curve["rank_fraction"].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.2), constrained_layout=True)
    axes[0, 0].plot(x, exact_curve["cumulative_energy_per_n"], color="black", lw=2.1, label="E1 exact K full eigen")
    axes[0, 0].plot(x, vecchia_curve["cumulative_energy_per_n"], color="#d95f02", lw=2.0, label="V1 subset Vecchia full eigen")
    axes[0, 0].plot(exact_lanczos["mode_fraction_slq"], exact_lanczos["cumulative_energy_per_n"], color="#1b9e77", lw=1.0, label=f"E2 exact precision Lanczos m={max_steps}")
    axes[0, 0].plot(vecchia_lanczos["mode_fraction_slq"], vecchia_lanczos["cumulative_energy_per_n"], color="#7570b3", lw=1.0, label=f"V2 Vecchia precision Lanczos m={max_steps}")
    axes[0, 0].set_title("Four-way residual eigen diagnostic")
    axes[0, 0].set_xlabel("covariance-mode rank fraction")
    axes[0, 0].set_ylabel("cumulative standardized energy / n")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].plot(x, vecchia_curve["cumulative_energy_per_n"].to_numpy() - exact_curve["cumulative_energy_per_n"].to_numpy(), color="#d95f02", lw=1.5, label="V1 - E1 (Vecchia)")
    axes[0, 1].plot(x, exact_lanczos["energy_error_per_n"], color="#1b9e77", lw=1.0, label="E2 - E1 (Lanczos)")
    axes[0, 1].plot(x, vecchia_lanczos["energy_error_per_n"], color="#7570b3", lw=1.0, label="V2 - V1 (Lanczos)")
    axes[0, 1].axhline(0.0, color="0.4", lw=0.8)
    axes[0, 1].set_title("Separated cumulative-curve errors")
    axes[0, 1].set_xlabel("operator-specific exact rank fraction")
    axes[0, 1].set_ylabel("difference / n")
    axes[0, 1].legend(fontsize=7)

    axes[0, 2].plot(x, exact_curve["implied_covariance_eigenvalue"], color="black", lw=1.4, label="exact K")
    axes[0, 2].plot(x, vecchia_curve["implied_covariance_eigenvalue"], color="#d95f02", lw=1.3, label="Vecchia implied K")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Covariance eigenvalue quantiles")
    axes[0, 2].set_xlabel("rank fraction")
    axes[0, 2].set_ylabel("covariance eigenvalue")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(band_table["band"], band_table["energy_per_mode_exact"], color="black", marker="o", label="E1 real residual")
    axes[1, 0].plot(band_table["band"], band_table["energy_per_mode_vecchia"], color="#d95f02", marker="s", label="V1 real residual")
    axes[1, 0].axhline(1.0, color="0.45", ls="--", lw=0.9)
    axes[1, 0].set_title("Observed 20-band energy")
    axes[1, 0].set_xlabel("equal-rank band: large to small covariance eigenvalue")
    axes[1, 0].set_ylabel("standardized energy per mode")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(expected_bands["band"], expected_bands["expected_vecchia_energy_per_mode_under_exact_K"], color="#e7298a", marker="o")
    axes[1, 1].axhline(1.0, color="0.45", ls="--", lw=0.9)
    axes[1, 1].set_title("Expected Vecchia energy if exact K is true")
    axes[1, 1].set_xlabel("Vecchia equal-rank band")
    axes[1, 1].set_ylabel("expected standardized energy per mode")

    axes[1, 2].plot(subspaces["band"], subspaces["normalized_projector_frobenius_distance"], color="#66a61e", marker="o")
    axes[1, 2].set_ylim(0.0, 1.02)
    axes[1, 2].set_title("Exact vs Vecchia band subspaces")
    axes[1, 2].set_xlabel("corresponding equal-rank band")
    axes[1, 2].set_ylabel("normalized projector distance")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    path = output_dir / "exact_vecchia_four_way_diagnostic.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def write_results(
    output_dir: Path,
    approximation: dict[str, float],
    numerical: pd.DataFrame,
    numerical_band_summary: pd.DataFrame,
    simulation: pd.DataFrame,
    expected_bands: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    largest = numerical["lanczos_steps"].max()
    rows = numerical[
        numerical["lanczos_steps"].eq(largest)
        & numerical["reorthogonalization"].eq("full")
    ].set_index("operator")
    band_rows = numerical_band_summary[
        numerical_band_summary["lanczos_steps"].eq(largest)
        & numerical_band_summary["reorthogonalization"].eq("full")
    ].set_index("operator")
    sim_quantiles = simulation.quantile([0.5, 0.9, 0.95], numeric_only=True)
    expected_deviation = expected_bands[
        "expected_vecchia_energy_per_mode_under_exact_K"
    ] - 1.0
    text = f"""# Exact covariance versus subset-specific Vecchia: four-way validation

## Construction

- Observations: `{metadata['subset']['n_observations']:,}` = {metadata['subset']['n_spatial_selected']} locations x 8 times
- Spatial selection: `{metadata['subset']['selection']}`
- Parameters: stored full-data adapted lag-4/3/2, batch-64 fit; no refit
- Residual: identical full-data GLS residual in E1, E2, V1, and V2
- Vecchia subset: rebuilt from selected observations and nonempty native-grid blocks; no principal precision submatrix
- Lanczos: precision operator, `m={largest}`, `{metadata['settings']['slq_probes']}` shared Rademacher probes

## Numerical error (Lanczos only)

- E2 versus E1 cumulative energy/n RMSE: `{rows.loc['exact_precision', 'energy_per_n_rmse']:.6g}`
- E2 versus E1 spectral-CDF RMSE: `{rows.loc['exact_precision', 'mode_fraction_rmse']:.6g}`
- V2 versus V1 cumulative energy/n RMSE: `{rows.loc['subset_vecchia_precision', 'energy_per_n_rmse']:.6g}`
- V2 versus V1 spectral-CDF RMSE: `{rows.loc['subset_vecchia_precision', 'mode_fraction_rmse']:.6g}`
- E2 versus E1 hard 20-band energy/mode RMSE: `{band_rows.loc['exact_precision', 'band_energy_per_mode_rmse']:.6g}`
- V2 versus V1 hard 20-band energy/mode RMSE: `{band_rows.loc['subset_vecchia_precision', 'band_energy_per_mode_rmse']:.6g}`

## Vecchia approximation error (full eigen, no Lanczos)

- Real-residual cumulative curve/n RMSE: `{approximation['real_residual_curve_rmse_per_n']:.6g}`
- Real-residual 20-band energy/mode RMSE: `{approximation['real_residual_band_energy_per_mode_rmse']:.6g}`
- Covariance relative Frobenius error: `{approximation['covariance_relative_frobenius_error']:.6g}`
- KL exact-to-Vecchia per observation: `{approximation['kl_exact_to_vecchia_per_observation']:.6g}`
- Mean expected Vecchia energy if exact K is true: `{approximation['mean_expected_vecchia_energy_under_exact_K']:.6g}`
- Largest absolute expected 20-band deviation from one: `{np.max(np.abs(expected_deviation)):.6g}`

## Exact-K simulation robustness

Across `{len(simulation)}` independent residuals generated from exact K:

- Median / 95% curve RMSE per n: `{sim_quantiles.loc[0.5, 'curve_rmse_per_n']:.6g}` / `{sim_quantiles.loc[0.95, 'curve_rmse_per_n']:.6g}`
- Median / 95% band-energy RMSE: `{sim_quantiles.loc[0.5, 'band_energy_per_mode_rmse']:.6g}` / `{sim_quantiles.loc[0.95, 'band_energy_per_mode_rmse']:.6g}`

This experiment separates Lanczos truncation/SLQ error from Vecchia spectral
approximation error.  Hard-band error remains larger than the approximately
`sqrt(2/7018)=0.0169` known-parameter fluctuation of a 5% band at full-data
size, so the current full-data hard bands remain exploratory.  Smooth filters
and probe-count convergence are required before calibration.  A final
goodness-of-fit test also requires simulation with beta and covariance
refitting.
"""
    (output_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    levels = parse_steps(args.lanczos_steps)
    if min(
        int(args.n_spatial),
        int(args.slq_probes),
        int(args.bands),
        int(args.simulation_replicates),
    ) < 1:
        raise ValueError("Counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow_started = time.perf_counter()
    rng = np.random.default_rng(int(args.random_seed))

    year, month, day = parse_date(args.date)
    record = load_fitted_record(Path(args.fit_results))
    fitted = fitted_physical(record)
    seed = json.loads(Path(args.fit_initializer).read_text(encoding="utf-8"))
    spec = {
        "dataset_id": f"real_{year}{month:02d}{day:02d}",
        "year": year,
        "month": month,
        "day": day,
        "date": args.date,
    }
    print(f"Loading real data {args.date}", flush=True)
    asset = core.load_real_asset(spec, loader_args(args))
    print("Recomputing full-data GLS beta at the stored parameter vector", flush=True)
    beta, lat_mean, gls_meta = recompute_gls_beta(asset, fitted, seed, args)
    if str(args.selection) == "contiguous":
        expected_spatial = int(args.contiguous_rows) * int(args.contiguous_cols)
        if int(args.n_spatial) != expected_spatial:
            raise ValueError(
                "For contiguous selection, n-spatial must equal contiguous-rows * contiguous-cols"
            )
        selected, _, selection_meta = select_common_contiguous(
            asset, int(args.contiguous_rows), int(args.contiguous_cols)
        )
    else:
        selected, _, selection_meta = select_common_maxmin(
            asset, int(args.n_spatial)
        )
        selection_meta["selection"] = "global max-min among common-valid locations"
    coords, residual, subset_rows = subset_residual_data(
        asset, selected, beta, lat_mean
    )
    subset_rows["selection"] = str(args.selection)
    subset_rows["selection_rank"] = subset_rows["maxmin_rank"]
    n = len(residual)
    if n != int(args.hours) * int(args.n_spatial):
        raise RuntimeError(f"Unexpected subset size {n}")
    atomic_csv(args.output_dir / "selected_subset_points.csv", subset_rows)

    exact_parameters = dict(fitted)
    exact_parameters["nugget"] += float(args.vecchia_diagonal_jitter)
    print(f"Building exact K for n={n:,}", flush=True)
    started = time.perf_counter()
    covariance = build_dense_covariance(coords, exact_parameters, block_rows=256)
    covariance_build_s = time.perf_counter() - started
    print("E1: full eigendecomposition of exact K", flush=True)
    started = time.perf_counter()
    lambda_exact_ascending, q_exact_ascending = scipy.linalg.eigh(
        covariance,
        lower=True,
        check_finite=False,
        driver="evd",
    )
    exact_eigen_s = time.perf_counter() - started
    if lambda_exact_ascending[0] <= 0.0:
        raise RuntimeError("Exact K is not SPD")
    lambda_exact = lambda_exact_ascending[::-1].copy()
    q_exact = q_exact_ascending[:, ::-1].copy()
    mu_exact = 1.0 / lambda_exact
    exact_curve = exact_eigen_curve(mu_exact, q_exact, residual, "exact_covariance")

    print("Building subset-specific adapted-432 Vecchia B", flush=True)
    whitener, vecchia_residual, vecchia_meta = build_subset_vecchia(
        asset,
        selected,
        fitted,
        seed,
        beta,
        lat_mean,
        args,
    )
    residual_max_abs_difference = float(np.max(np.abs(vecchia_residual - residual)))
    # The cluster precompute intentionally casts source data to float32.  The
    # returned residual is used only to verify row alignment; all four actual
    # diagnostics below use the same double-precision ``residual``.
    if residual_max_abs_difference > 1e-5:
        raise RuntimeError(
            f"Exact and Vecchia residuals differ: {residual_max_abs_difference:.3e}"
        )
    scipy.sparse.save_npz(
        args.output_dir / "subset_vecchia_whitener_B.npz", whitener, compressed=False
    )
    print("V1: full eigendecomposition of subset Vecchia precision", flush=True)
    started = time.perf_counter()
    omega_vecchia = (whitener.T @ whitener).toarray()
    mu_vecchia, q_vecchia = scipy.linalg.eigh(
        omega_vecchia,
        lower=True,
        check_finite=False,
        driver="evd",
    )
    vecchia_eigen_s = time.perf_counter() - started
    if mu_vecchia[0] <= 0.0:
        raise RuntimeError("Subset Vecchia precision is not SPD")
    vecchia_curve = exact_eigen_curve(
        mu_vecchia, q_vecchia, residual, "subset_vecchia_covariance"
    )
    exact_curves = pd.concat([exact_curve, vecchia_curve], ignore_index=True)
    atomic_csv(args.output_dir / "E1_V1_full_eigen_curves.csv", exact_curves)

    print("Materializing exact precision K^-1 for the subset benchmark", flush=True)
    started = time.perf_counter()
    factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
    omega_exact = scipy.linalg.cho_solve(
        factor, np.eye(n, dtype=np.float64), check_finite=False
    )
    omega_exact = 0.5 * (omega_exact + omega_exact.T)
    exact_precision_s = time.perf_counter() - started
    precision_eigen_relative_error = float(
        np.max(np.abs(scipy.linalg.eigvalsh(omega_exact) - mu_exact) / mu_exact)
    )

    probes = rng.choice(
        np.asarray([-1.0, 1.0]), size=(n, int(args.slq_probes)), replace=True
    )
    operators = {
        "exact_precision": omega_exact,
        "subset_vecchia_precision": omega_vecchia,
    }
    references = {
        "exact_precision": exact_curve,
        "subset_vecchia_precision": vecchia_curve,
    }
    reorth_modes = [str(args.reorthogonalization)]
    if bool(args.also_no_reorthogonalization) and "none" not in reorth_modes:
        reorth_modes.append("none")
    all_numerical_curves = []
    all_numerical_summaries = []
    lanczos_timings: dict[str, Any] = {}
    for reorth in reorth_modes:
        for operator_name, matrix in operators.items():
            matvec = lambda vector, matrix=matrix: matrix @ vector
            residual_run, probe_runs, timing = run_with_probes(
                matvec,
                residual,
                probes,
                max_steps=max(levels),
                reorthogonalization=reorth,
                label=f"{operator_name}/{reorth}",
            )
            thresholds = ascending_rank_thresholds(
                references[operator_name]["precision_eigenvalue"].to_numpy()
            )
            raw_curves = evaluate_precision_curves(
                residual_run,
                probe_runs,
                levels,
                thresholds,
                n,
            )
            curves, summary = numerical_comparison(
                raw_curves,
                references[operator_name],
                levels,
                operator_name,
                reorth,
                int(args.slq_probes),
            )
            all_numerical_curves.append(curves)
            all_numerical_summaries.append(summary)
            lanczos_timings[f"{operator_name}_{reorth}"] = timing
    numerical_curves = pd.concat(all_numerical_curves, ignore_index=True)
    numerical_summary = pd.concat(all_numerical_summaries, ignore_index=True)
    atomic_csv(args.output_dir / "E2_V2_lanczos_curves.csv", numerical_curves)
    atomic_csv(args.output_dir / "E2_V2_lanczos_accuracy.csv", numerical_summary)
    numerical_bands, numerical_band_summary = numerical_band_accuracy(
        numerical_curves, exact_curves, int(args.bands)
    )
    atomic_csv(args.output_dir / "E2_V2_hard_band_comparison.csv", numerical_bands)
    atomic_csv(
        args.output_dir / "E2_V2_hard_band_accuracy.csv", numerical_band_summary
    )

    print("Computing exact-vs-Vecchia spectral approximation metrics", flush=True)
    band_table, expected_bands, subspaces, approximation, cross = approximation_metrics(
        exact_curve,
        vecchia_curve,
        q_exact,
        q_vecchia,
        int(args.bands),
    )
    atomic_csv(args.output_dir / "E1_V1_real_residual_band_comparison.csv", band_table)
    atomic_csv(args.output_dir / "vecchia_expected_bands_under_exact_K.csv", expected_bands)
    atomic_csv(args.output_dir / "exact_vecchia_band_subspace_comparison.csv", subspaces)

    print(
        f"Simulating {int(args.simulation_replicates)} residuals from exact K",
        flush=True,
    )
    simulation = simulate_exact_residual_distortion(
        lambda_exact,
        mu_vecchia,
        cross,
        int(args.bands),
        int(args.simulation_replicates),
        rng,
    )
    atomic_csv(args.output_dir / "exact_K_simulated_vecchia_distortion.csv", simulation)

    figure = plot_four_way(
        exact_curve,
        vecchia_curve,
        numerical_curves,
        band_table,
        expected_bands,
        subspaces,
        levels,
        args.output_dir,
    )
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "date": args.date,
        "data_source": asset.source_path,
        "fit_reused_without_refit": True,
        "fit": {
            "geometry": "adapted",
            "lag_pattern": "4/3/2",
            "target_chunk_size": 64,
            "fitted_parameters": fitted,
            "full_data_gls_beta": beta,
            "full_data_gls_lat_mean": lat_mean,
            "stored_native_nll": float(record["final_native_nll"]),
        },
        "subset": {
            **selection_meta,
            "hours": int(args.hours),
            "n_observations": n,
            "ordering_for_comparison": (
                "time-major, then global max-min spatial rank"
                if str(args.selection) == "global-maxmin"
                else "time-major, then native-grid row-major within rectangle"
            ),
            "same_raw_residual_all_four_methods": True,
            "residual_max_abs_difference": residual_max_abs_difference,
        },
        "exact_covariance": {
            "diagonal_jitter_matching_vecchia_batched_covariance": float(
                args.vecchia_diagonal_jitter
            ),
            "min_eigenvalue": float(lambda_exact[-1]),
            "max_eigenvalue": float(lambda_exact[0]),
            "condition_number": float(lambda_exact[0] / lambda_exact[-1]),
            "precision_eigenvalue_max_relative_error": precision_eigen_relative_error,
        },
        "subset_vecchia": vecchia_meta,
        "approximation_metrics": approximation,
        "settings": {
            "lanczos_steps": levels,
            "slq_probes": int(args.slq_probes),
            "shared_probes_between_operators": True,
            "reorthogonalization_modes": reorth_modes,
            "bands": int(args.bands),
            "simulation_replicates": int(args.simulation_replicates),
            "random_seed": int(args.random_seed),
        },
        "timings_s": {
            "full_data_vecchia_precompute_for_beta": gls_meta["vecchia_precompute_s"],
            "full_data_gls_beta": gls_meta["gls_beta_s"],
            "exact_covariance_build": covariance_build_s,
            "exact_covariance_full_eigen": exact_eigen_s,
            "subset_vecchia_full_precision_and_eigen": vecchia_eigen_s,
            "exact_precision_materialization": exact_precision_s,
            "lanczos": lanczos_timings,
            "workflow_total": time.perf_counter() - workflow_started,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "figure": figure,
    }
    atomic_json(args.output_dir / "run_metadata.json", metadata)
    write_results(
        args.output_dir,
        approximation,
        numerical_summary,
        numerical_band_summary,
        simulation,
        expected_bands,
        metadata,
    )
    print("Numerical accuracy", flush=True)
    print(numerical_summary.to_string(index=False), flush=True)
    print("Vecchia approximation metrics", flush=True)
    print(json.dumps(approximation, indent=2), flush=True)
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
