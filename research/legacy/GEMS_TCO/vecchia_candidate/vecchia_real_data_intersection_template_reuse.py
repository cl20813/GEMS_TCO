"""
vecchia_real_data_intersection_template_reuse.py

Standalone missing-aware regular-grid reverse-L column Vecchia implementation.

This file is deliberately independent.  It does not import or subclass
`kernels_vecchia_grid_col_template_reuse.py`, so it can be kept, moved, or deleted without
affecting the older column kernels.

Main design:

1. Raw GEMS grid rows are checked as east-to-west within a latitude row, then
   north-to-south across rows.
2. For a day with 8 hourly maps, the default support is the intersection of
   grid locations observed at all 8 hours.  With `compact_intersection_grid`,
   those intersection observations are ordered like raw GEMS scan order
   (north row, east-to-west, then south) and packed onto a hole-free regular
   grid before reverse-L conditioning.  This is the intended template-reuse
   diagnostic.
3. Template reuse is based on regular-grid relative displacement vectors
   (delta latitude, delta longitude, delta time).  If source/original locations
   are used for covariance distances, those relative displacement vectors change
   by target/hour, so the template-reuse assumption no longer holds.  That is a
   different, target-wise batched experiment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import gamma


DELTA_LAT_DEFAULT = 0.044
DELTA_LON_DEFAULT = 0.063


@dataclass
class DayIntersectionResult:
    input_map: Dict[str, torch.Tensor]
    grid_coords: np.ndarray
    kept_locals: np.ndarray
    late_locals: np.ndarray
    intersection_mask: np.ndarray
    row_subset_mask: np.ndarray
    scan_order_all: np.ndarray
    diagnostics: Dict[str, Any]


def _as_frame_list(hourly_frames: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame]):
    if isinstance(hourly_frames, Mapping):
        keys = sorted(hourly_frames.keys(), key=str)
        frames = [hourly_frames[k] for k in keys]
    else:
        frames = list(hourly_frames)
        keys = [f"h{i}" for i in range(len(frames))]
    return keys, frames


def build_grid_index(
    coords: np.ndarray,
    lat_round_decimals: int = 6,
    lon_round_decimals: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    """Build regular-grid row/column maps from [lat, lon] coordinates."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have columns [lat, lon]")

    lat_key = np.round(coords[:, 0], int(lat_round_decimals))
    lon_key = np.round(coords[:, 1], int(lon_round_decimals))
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_to_row = {v: i for i, v in enumerate(lats)}
    lon_to_col = {v: i for i, v in enumerate(lons)}

    local_to_row = np.empty(coords.shape[0], dtype=np.int64)
    local_to_col = np.empty(coords.shape[0], dtype=np.int64)
    row_col_to_local: Dict[Tuple[int, int], int] = {}
    for i, (la, lo) in enumerate(zip(lat_key, lon_key)):
        r = int(lat_to_row[la])
        c = int(lon_to_col[lo])
        local_to_row[i] = r
        local_to_col[i] = c
        row_col_to_local[(r, c)] = int(i)

    return lats, lons, local_to_row, local_to_col, row_col_to_local


def column_scan_order(local_to_row: np.ndarray, local_to_col: np.ndarray) -> np.ndarray:
    """Scan like raw GEMS: north row first, east-to-west, then one row south."""
    # Rows/lons are ascending, so larger row is north and larger col is east.
    # np.lexsort uses the last key as primary: row descending, then col descending.
    return np.lexsort((-local_to_col, -local_to_row)).astype(np.int64)


def _median_positive_step(values: np.ndarray, default: float) -> float:
    vals = np.sort(np.unique(np.asarray(values, dtype=np.float64)))
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else float(default)


def pack_scan_order_coords(
    n_points: int,
    n_cols: int,
    north_lat: float,
    east_lon: float,
    delta_lat: float = DELTA_LAT_DEFAULT,
    delta_lon: float = DELTA_LON_DEFAULT,
    order: str = "row",
    n_rows: Optional[int] = None,
) -> np.ndarray:
    """Place observations on a compact regular grid in scan order.

    The first observation is assigned to the north-east corner.  With
    ``order="row"``, subsequent observations move west within a row; after
    ``n_cols`` points the next row starts one latitude step south.  With
    ``order="column"``, subsequent observations move south within a column;
    after ``n_rows`` points the next longitude column starts one step west.
    This intentionally removes holes from the intersection support so reverse-L
    templates repeat.
    """
    n_points = int(n_points)
    order = str(order).lower()
    seq = np.arange(n_points, dtype=np.int64)
    if order == "row":
        n_cols = max(1, int(n_cols))
        row_from_top = seq // n_cols
        col_from_east = seq % n_cols
    elif order == "column":
        if n_rows is None:
            raise ValueError("n_rows is required when order='column'")
        n_rows = max(1, int(n_rows))
        row_from_top = seq % n_rows
        col_from_east = seq // n_rows
    else:
        raise ValueError("order must be 'row' or 'column'")
    lat = float(north_lat) - row_from_top.astype(np.float64) * float(delta_lat)
    lon = float(east_lon) - col_from_east.astype(np.float64) * float(delta_lon)
    return np.column_stack([lat, lon]).astype(np.float64)


def diagnose_scan_order(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    big_jump: float = 5.0,
) -> Dict[str, Any]:
    """Check whether raw dataframe order is east-to-west, then north-to-south."""
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[finite]
    lon = lon[finite]
    if len(lat) < 3:
        return {"ok": False, "reason": "fewer than 3 finite coordinates"}

    dlon = np.diff(lon)
    row_starts = np.r_[0, np.where(dlon > float(big_jump))[0] + 1]
    row_ends = np.r_[row_starts[1:], len(lon)]
    row_lengths = row_ends - row_starts

    within_row_westward = []
    row_latitudes = []
    for s, e in zip(row_starts, row_ends):
        if e - s <= 1:
            continue
        within_row_westward.append(bool(np.nanmedian(np.diff(lon[s:e])) < 0.0))
        row_latitudes.append(float(np.nanmedian(lat[s:e])))

    lat_steps = np.diff(row_latitudes) if len(row_latitudes) > 1 else np.array([])
    north_to_south = bool(len(lat_steps) == 0 or np.nanmedian(lat_steps) < 0.0)

    return {
        "ok": bool(all(within_row_westward) and north_to_south),
        "n_rows_detected": int(len(row_starts)),
        "row_length_median": float(np.median(row_lengths)) if len(row_lengths) else np.nan,
        "row_length_min": int(np.min(row_lengths)) if len(row_lengths) else 0,
        "row_length_max": int(np.max(row_lengths)) if len(row_lengths) else 0,
        "within_row_westward": bool(all(within_row_westward)) if within_row_westward else False,
        "north_to_south": north_to_south,
        "first_coords": list(zip(lat[:8].tolist(), lon[:8].tolist())),
        "first_reset_index": int(row_starts[1]) if len(row_starts) > 1 else None,
    }


def _hour_time_value(df: pd.DataFrame, hours_col: str, time_origin: float) -> float:
    vals = pd.to_numeric(df[hours_col], errors="coerce")
    if vals.dropna().empty:
        return 0.0
    return float(np.round(vals.dropna().median() - float(time_origin)))


def make_day_intersection_input_map(
    hourly_frames: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    monthly_mean: float = 0.0,
    value_col: str = "ColumnAmountO3",
    grid_lat_col: str = "Latitude",
    grid_lon_col: str = "Longitude",
    source_lat_col: str = "Source_Latitude",
    source_lon_col: str = "Source_Longitude",
    hours_col: str = "Hours_elapsed",
    time_origin: float = 477700.0,
    coord_mode: str = "grid",
    require_source_coordinates: bool = True,
    truncate_to_intersection: bool = True,
    compact_intersection_grid: bool = False,
    compact_pack_order: str = "row",
    packed_grid_n_lon: Optional[int] = None,
    packed_grid_n_lat: Optional[int] = None,
    lat_stride: int = 1,
    lat_stride_offset: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    lat_round_decimals: int = 6,
    lon_round_decimals: int = 6,
) -> DayIntersectionResult:
    """Create equal-support hourly tensors for a single day.

    Returned tensor columns:
      [Lat, Lon, centered O3, centered Hours_elapsed, D1, ..., D7].

    `coord_mode="grid"` is the mode compatible with template reuse.
    `coord_mode="source"` is available only for diagnostics; fitting source
    coordinates with template reuse is not recommended because the relative
    displacement vectors are no longer shared by template.

    `lat_stride=2` keeps every second latitude row, starting from the northern
    scan edge when `lat_stride_offset=0`.  This reduces latitude resolution
    before the nonmissing intersection and compact-grid packing are applied.
    """
    if coord_mode not in ("grid", "source"):
        raise ValueError("coord_mode must be 'grid' or 'source'")
    if compact_intersection_grid and coord_mode != "grid":
        raise ValueError("compact_intersection_grid=True is only compatible with coord_mode='grid'")
    compact_pack_order = str(compact_pack_order).lower()
    if compact_pack_order not in ("row", "column"):
        raise ValueError("compact_pack_order must be 'row' or 'column'")
    lat_stride = int(lat_stride)
    lat_stride_offset = int(lat_stride_offset)
    if lat_stride < 1:
        raise ValueError("lat_stride must be >= 1")
    if not (0 <= lat_stride_offset < lat_stride):
        raise ValueError("lat_stride_offset must satisfy 0 <= offset < lat_stride")

    keys, frames = _as_frame_list(hourly_frames)
    if len(frames) == 0:
        raise ValueError("hourly_frames is empty")

    n = len(frames[0])
    for k, df in zip(keys, frames):
        if len(df) != n:
            raise ValueError(f"All hourly frames must have the same grid row count; {k} has {len(df)} != {n}")

    first = frames[0]
    grid_coords = first[[grid_lat_col, grid_lon_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lats, lons, local_to_row, local_to_col, _ = build_grid_index(
        grid_coords,
        lat_round_decimals=lat_round_decimals,
        lon_round_decimals=lon_round_decimals,
    )
    scan_all = column_scan_order(local_to_row, local_to_col)
    north_row = int(local_to_row.max()) if len(local_to_row) else 0
    row_subset_mask = ((north_row - local_to_row) % lat_stride) == lat_stride_offset

    observed_masks = []
    observed_masks_full_grid = []
    per_hour_valid = []
    per_hour_valid_full_grid = []
    for df in frames:
        y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
        grid_lat = pd.to_numeric(df[grid_lat_col], errors="coerce").to_numpy(dtype=float)
        grid_lon = pd.to_numeric(df[grid_lon_col], errors="coerce").to_numpy(dtype=float)
        obs = np.isfinite(y) & np.isfinite(grid_lat) & np.isfinite(grid_lon)
        if require_source_coordinates:
            src_lat = pd.to_numeric(df[source_lat_col], errors="coerce").to_numpy(dtype=float)
            src_lon = pd.to_numeric(df[source_lon_col], errors="coerce").to_numpy(dtype=float)
            obs &= np.isfinite(src_lat) & np.isfinite(src_lon)
        observed_masks_full_grid.append(obs.copy())
        per_hour_valid_full_grid.append(int(obs.sum()))
        obs &= row_subset_mask
        observed_masks.append(obs)
        per_hour_valid.append(int(obs.sum()))

    full_grid_intersection_mask = np.logical_and.reduce(observed_masks_full_grid)
    intersection_mask = np.logical_and.reduce(observed_masks)
    core_order = scan_all[intersection_mask[scan_all]]
    late_order = scan_all[~intersection_mask[scan_all]]
    kept_locals = core_order if truncate_to_intersection else np.r_[core_order, late_order]

    packed_coords = None
    packed_n_cols = None
    packed_n_rows = None
    packed_last_row_count = None
    packed_last_col_count = None
    if compact_intersection_grid:
        if not truncate_to_intersection:
            raise ValueError("compact_intersection_grid=True requires truncate_to_intersection=True")
        kept_lats = grid_coords[row_subset_mask, 0]
        dlat = _median_positive_step(kept_lats, DELTA_LAT_DEFAULT)
        dlon = _median_positive_step(lons, DELTA_LON_DEFAULT)
        if compact_pack_order == "row":
            packed_n_cols = int(packed_grid_n_lon) if packed_grid_n_lon is not None else int(len(lons))
            packed_n_cols = max(1, packed_n_cols)
            packed_n_rows = int(np.ceil(len(kept_locals) / packed_n_cols)) if len(kept_locals) else 0
            packed_last_row_count = int(len(kept_locals) - (packed_n_rows - 1) * packed_n_cols) if packed_n_rows else 0
        else:
            n_lat_after_stride = int(np.unique(local_to_row[row_subset_mask]).size)
            packed_n_rows = int(packed_grid_n_lat) if packed_grid_n_lat is not None else n_lat_after_stride
            packed_n_rows = max(1, packed_n_rows)
            packed_n_cols = int(np.ceil(len(kept_locals) / packed_n_rows)) if len(kept_locals) else 0
            packed_last_col_count = int(len(kept_locals) - (packed_n_cols - 1) * packed_n_rows) if packed_n_cols else 0
        packed_coords = pack_scan_order_coords(
            len(kept_locals),
            n_cols=packed_n_cols,
            north_lat=float(np.nanmax(lats)),
            east_lon=float(np.nanmax(lons)),
            delta_lat=dlat,
            delta_lon=dlon,
            order=compact_pack_order,
            n_rows=packed_n_rows,
        )

    input_map: Dict[str, torch.Tensor] = {}
    for t_idx, (key, df) in enumerate(zip(keys, frames)):
        if coord_mode == "grid":
            if compact_intersection_grid:
                lat = packed_coords[:, 0]
                lon = packed_coords[:, 1]
            else:
                lat = pd.to_numeric(df[grid_lat_col], errors="coerce").to_numpy(dtype=float)[kept_locals]
                lon = pd.to_numeric(df[grid_lon_col], errors="coerce").to_numpy(dtype=float)[kept_locals]
        else:
            lat = pd.to_numeric(df[source_lat_col], errors="coerce").to_numpy(dtype=float)[kept_locals]
            lon = pd.to_numeric(df[source_lon_col], errors="coerce").to_numpy(dtype=float)[kept_locals]

        y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)[kept_locals] - float(monthly_mean)
        h = np.full(len(kept_locals), _hour_time_value(df, hours_col=hours_col, time_origin=time_origin), dtype=float)
        base = np.column_stack([lat, lon, y, h])
        dummies = np.zeros((len(kept_locals), 7), dtype=float)
        if 1 <= t_idx <= 7:
            dummies[:, t_idx - 1] = 1.0
        input_map[str(key)] = torch.as_tensor(np.concatenate([base, dummies], axis=1), dtype=dtype, device=device)

    diagnostics = {
        "n_hours": int(len(frames)),
        "n_grid_total": int(n),
        "n_full_grid_intersection": int(full_grid_intersection_mask.sum()),
        "n_intersection": int(intersection_mask.sum()),
        "n_late_or_dropped": int((~intersection_mask).sum()),
        "n_missing_or_source_dropped_after_lat_stride": int(row_subset_mask.sum() - intersection_mask.sum()),
        "full_grid_intersection_fraction": float(full_grid_intersection_mask.mean()),
        "intersection_fraction": float(intersection_mask.mean()),
        "intersection_fraction_after_lat_stride": float(intersection_mask.sum() / max(1, row_subset_mask.sum())),
        "per_hour_valid_full_grid": per_hour_valid_full_grid,
        "per_hour_valid": per_hour_valid,
        "coord_mode": coord_mode,
        "truncate_to_intersection": bool(truncate_to_intersection),
        "compact_intersection_grid": bool(compact_intersection_grid),
        "compact_pack_order": compact_pack_order,
        "lat_stride": int(lat_stride),
        "lat_stride_offset": int(lat_stride_offset),
        "n_lat_stride_kept": int(row_subset_mask.sum()),
        "n_lat_stride_dropped": int((~row_subset_mask).sum()),
        "n_lat_after_stride": int(np.unique(local_to_row[row_subset_mask]).size),
        "packed_grid_n_lon": int(packed_n_cols) if packed_n_cols is not None else None,
        "packed_grid_n_rows": int(packed_n_rows) if packed_n_rows is not None else None,
        "packed_last_row_count": int(packed_last_row_count) if packed_last_row_count is not None else None,
        "packed_last_col_count": int(packed_last_col_count) if packed_last_col_count is not None else None,
        "scan_order": "north_row_east_to_west_then_south",
        "n_lat": int(len(lats)),
        "n_lon": int(len(lons)),
    }

    return DayIntersectionResult(
        input_map=input_map,
        grid_coords=(packed_coords if compact_intersection_grid else grid_coords[kept_locals]).astype(np.float64),
        kept_locals=kept_locals.astype(np.int64),
        late_locals=late_order.astype(np.int64),
        intersection_mask=intersection_mask,
        row_subset_mask=row_subset_mask,
        scan_order_all=scan_all,
        diagnostics=diagnostics,
    )


class ReverseLIntersectionColumnVecchiaFit:
    """Template-reuse reverse-L column Vecchia on an all-hours intersection grid."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        mm_cond_number: int = 300,
        nheads: int = 0,
        grid_coords: Optional[np.ndarray] = None,
        head_right_cols: int = 3,
        above_count: int = 2,
        right_col_count: int = 3,
        per_lag_conditioning_count: int = 8,
        lag_count: int = 2,
        include_lag_self: bool = False,
        lat_round_decimals: int = 6,
        lon_round_decimals: int = 6,
        target_chunk_size: int = 2048,
    ):
        if smooth not in (0.5, 1.5):
            raise ValueError(f"smooth must be 0.5 or 1.5, got {smooth}")
        self.smooth = float(smooth)
        self.input_map = input_map
        self.mm_cond_number = int(mm_cond_number)
        self.nheads = int(nheads)
        self.grid_coords = grid_coords
        self.head_right_cols = int(head_right_cols)
        self.above_count = int(above_count)
        self.right_col_count = int(right_col_count)
        self.per_lag_conditioning_count = int(per_lag_conditioning_count)
        self.lag_count = int(lag_count)
        self.include_lag_self = bool(include_lag_self)
        self.lat_round_decimals = int(lat_round_decimals)
        self.lon_round_decimals = int(lon_round_decimals)
        self.target_chunk_size = int(target_chunk_size)

        first_val = next(iter(input_map.values()))
        self.device = first_val.device if isinstance(first_val, torch.Tensor) else torch.device("cpu")

        self.n_features = 9
        self.lat_mean_val = 0.0
        self.is_precomputed = False

        self.Full_Data_Grid = None
        self.Heads_data = None
        self.Grouped_Batches: List[Dict[str, Any]] = []
        self.conditioning_summary: Dict[str, Any] = {}
        self.n_tails = 0
        self._n_real = 0
        self._n_grid = 0
        self._n_time = 0
        self._n_lat = 0
        self._n_lon = 0
        self._local_to_row = None
        self._local_to_col = None
        self._row_col_to_local: Dict[Tuple[int, int], int] = {}
        self._stencil_cache: Dict[Tuple[int, int], List[int]] = {}

        gamma_val = torch.tensor(gamma(self.smooth), dtype=torch.float64)
        self.matern_const = (2 ** (1 - self.smooth)) / gamma_val

    def _regular_coords_np(self, n_points: int) -> np.ndarray:
        if self.grid_coords is not None:
            coords = np.asarray(self.grid_coords[:n_points], dtype=np.float64)
        else:
            first = next(iter(self.input_map.values()))
            coords = first[:n_points, :2].detach().cpu().numpy().astype(np.float64) if isinstance(first, torch.Tensor) else np.asarray(first[:n_points, :2], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("grid_coords must have columns [lat, lon]")
        return coords

    def _get_local(self, row: int, col: int) -> Optional[int]:
        return self._row_col_to_local.get((int(row), int(col)))

    def _spatial_stencil_locals(self, row: int, col: int) -> List[int]:
        key = (int(row), int(col))
        if key in self._stencil_cache:
            return self._stencil_cache[key]

        cap = max(0, int(self.per_lag_conditioning_count))
        out: List[int] = []
        seen = set()

        def add(local_idx: Optional[int]) -> bool:
            if local_idx is None or local_idx in seen:
                return False
            out.append(int(local_idx))
            seen.add(int(local_idx))
            return len(out) >= cap

        # Above/north same column first.
        for k in range(1, self.above_count + 1):
            if add(self._get_local(row + k, col)):
                self._stencil_cache[key] = out
                return out

        # Right/east 3-column block: same row first, then previously scanned
        # north rows.  With GEMS row-major scan, south/lower rows are future
        # targets and must not be conditioning neighbors.
        for up in range(0, self._n_lat - row):
            r2 = row + up
            for dc in range(1, self.right_col_count + 1):
                c2 = col + dc
                if c2 >= self._n_lon:
                    continue
                if add(self._get_local(r2, c2)):
                    self._stencil_cache[key] = out
                    return out

        self._stencil_cache[key] = out
        return out

    def _raw_params(self, params: torch.Tensor):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        sigmasq = phi1 / phi2
        return phi1, phi2, phi3, phi4, params[4], params[5], nugget, sigmasq

    def _cov_from_deltas(self, d_lat, d_lon, d_t, params: torch.Tensor):
        _, phi2, phi3, phi4, advec_lat, advec_lon, _, sigmasq = self._raw_params(params)
        u_lat = d_lat - advec_lat * d_t
        u_lon = d_lon - advec_lon * d_t
        dist = torch.sqrt(d_lat.new_tensor(1e-8) + u_lat.pow(2) * phi3 + u_lon.pow(2) + d_t.pow(2) * phi4)
        scaled = dist * phi2
        if self.smooth == 0.5:
            return sigmasq * torch.exp(-scaled)
        return sigmasq * (1.0 + scaled) * torch.exp(-scaled)

    def _stencil_cov_matrix(self, deltas: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = deltas.unsqueeze(1) - deltas.unsqueeze(0)
        cov = self._cov_from_deltas(diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], params)
        nugget = torch.exp(params[6])
        cov = cov.clone()
        cov.diagonal().add_(nugget + 1e-6)
        return cov

    def _cross_cov_vector(self, deltas: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self._cov_from_deltas(deltas[:, 0], deltas[:, 1], deltas[:, 2], params)

    def _full_cov(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        cov = self._cov_from_deltas(diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], params)
        nugget = torch.exp(params[6])
        cov = cov.clone()
        cov.diagonal().add_(nugget + 1e-8)
        return cov

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        lat = (flat[:, 0:1] - self.lat_mean_val).to(torch.float64)
        dums = flat[:, 4:11].to(torch.float64)
        if dums.shape[1] < 7:
            pad = torch.zeros((flat.shape[0], 7 - dums.shape[1]), device=self.device, dtype=torch.float64)
            dums = torch.cat([dums, pad], dim=1)
        X = torch.cat([ones, lat, dums], dim=1)
        return X.reshape(*orig_shape, self.n_features)

    def precompute_conditioning_sets(self):
        print(
            "Pre-computing standalone ReverseLIntersectionColumnVecchia "
            f"[heads_right={self.head_right_cols}, above={self.above_count}, "
            f"right_cols={self.right_col_count}, per_lag={self.per_lag_conditioning_count}, "
            f"lags={self.lag_count}]...",
            end=" ",
        )
        t0 = time.time()

        all_data_list = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d for d in self.input_map.values()]
        all_data_list = [d.to(self.device, dtype=torch.float64) for d in all_data_list]
        self.Full_Data_Grid = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = self.Full_Data_Grid.shape
        self._n_real = n_real
        self._n_time = len(all_data_list)
        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if len(set(day_lengths)) != 1:
            raise ValueError(f"All hours must have equal support length, got {day_lengths}")
        self._n_grid = day_lengths[0]
        cumulative_len = np.cumsum([0] + day_lengths)

        coords_np = self._regular_coords_np(self._n_grid)
        lats, lons, local_to_row, local_to_col, row_col_to_local = build_grid_index(
            coords_np,
            lat_round_decimals=self.lat_round_decimals,
            lon_round_decimals=self.lon_round_decimals,
        )
        self._n_lat = len(lats)
        self._n_lon = len(lons)
        self._local_to_row = local_to_row
        self._local_to_col = local_to_col
        self._row_col_to_local = row_col_to_local
        self._stencil_cache = {}

        y = self.Full_Data_Grid[:, 2]
        valid_y_np = (~torch.isnan(y)).detach().cpu().numpy()
        valid_lats = self.Full_Data_Grid[~torch.isnan(y), 0]
        self.lat_mean_val = float(valid_lats.mean().item()) if valid_lats.numel() else float(self.Full_Data_Grid[:, 0].mean().item())

        time_values = []
        for d in all_data_list:
            good = ~torch.isnan(d[:, 3])
            time_values.append(float(d[good, 3].median().item()) if good.any() else 0.0)

        head_locals = set()
        for c in range(max(0, self._n_lon - self.head_right_cols), self._n_lon):
            for r in range(self._n_lat):
                nb = self._get_local(r, c)
                if nb is not None:
                    head_locals.add(nb)

        heads_indices: List[int] = []
        groups: Dict[Tuple[Tuple[float, float, float], ...], Dict[str, Any]] = {}
        m_sizes = []

        row_order = range(self._n_lat - 1, -1, -1)
        col_order = range(self._n_lon - 1, -1, -1)

        lag_count_rows: List[Tuple[int, ...]] = []

        def add_group(
            target_global: int,
            neigh_globals: List[int],
            deltas: List[Tuple[float, float, float]],
            lag_counts: Sequence[int],
        ):
            if len(neigh_globals) == 0:
                key = tuple()
                delta_tensor = torch.empty((0, 3), device=self.device, dtype=torch.float64)
            else:
                key = tuple((round(a, 8), round(b, 8), round(c, 8)) for a, b, c in deltas)
                delta_tensor = torch.tensor(deltas, device=self.device, dtype=torch.float64)
            if key not in groups:
                groups[key] = {"deltas": delta_tensor, "batch_idx": [], "target_idx": []}
            groups[key]["batch_idx"].append(neigh_globals)
            groups[key]["target_idx"].append(target_global)
            m_sizes.append(len(neigh_globals))
            lag_count_rows.append(tuple(int(x) for x in lag_counts))

        for time_idx in range(self._n_time):
            time_start = int(cumulative_len[time_idx])
            for row in row_order:
                for col in col_order:
                    local_idx = self._get_local(row, col)
                    if local_idx is None:
                        continue
                    target_global = time_start + local_idx
                    if not valid_y_np[target_global]:
                        continue
                    if local_idx in head_locals:
                        heads_indices.append(target_global)
                        continue

                    target_lat = float(coords_np[local_idx, 0])
                    target_lon = float(coords_np[local_idx, 1])
                    neigh_globals: List[int] = []
                    delta_list: List[Tuple[float, float, float]] = []
                    lag_counts = [0] * (self.lag_count + 1)
                    seen = set()
                    spatial_locals = self._spatial_stencil_locals(row, col)

                    for lag in range(self.lag_count + 1):
                        neigh_time_idx = time_idx - lag
                        if neigh_time_idx < 0:
                            continue
                        neigh_time_start = int(cumulative_len[neigh_time_idx])
                        dt = float(time_values[time_idx] - time_values[neigh_time_idx])

                        if lag > 0 and self.include_lag_self:
                            g = neigh_time_start + local_idx
                            if g not in seen and valid_y_np[g]:
                                neigh_globals.append(g)
                                delta_list.append((0.0, 0.0, dt))
                                lag_counts[lag] += 1
                                seen.add(g)

                        for nb_local in spatial_locals:
                            g = neigh_time_start + int(nb_local)
                            if g in seen or not valid_y_np[g]:
                                continue
                            nb_lat = float(coords_np[nb_local, 0])
                            nb_lon = float(coords_np[nb_local, 1])
                            neigh_globals.append(g)
                            delta_list.append((target_lat - nb_lat, target_lon - nb_lon, dt))
                            lag_counts[lag] += 1
                            seen.add(g)

                    add_group(target_global, neigh_globals, delta_list, lag_counts)

        self.Heads_data = self.Full_Data_Grid[torch.tensor(heads_indices, device=self.device, dtype=torch.long)].contiguous() if heads_indices else torch.empty((0, num_cols), device=self.device, dtype=torch.float64)

        self.Grouped_Batches = []
        for val in groups.values():
            t_idx = torch.tensor(val["target_idx"], device=self.device, dtype=torch.long)
            if val["deltas"].shape[0] == 0:
                b_idx = torch.empty((len(val["target_idx"]), 0), device=self.device, dtype=torch.long)
            else:
                b_idx = torch.tensor(val["batch_idx"], device=self.device, dtype=torch.long)
            self.Grouped_Batches.append({"deltas": val["deltas"], "batch_idx": b_idx, "target_idx": t_idx})

        self.n_tails = int(sum(len(g["target_idx"]) for g in self.Grouped_Batches))
        self.is_precomputed = True
        self.conditioning_summary = {
            "template_count": int(len(self.Grouped_Batches)),
            "per_lag_conditioning_count": int(self.per_lag_conditioning_count),
            "lag_count": int(self.lag_count),
            "include_lag_self": bool(self.include_lag_self),
            "conditioning_total_cap": int(self.per_lag_conditioning_count * (self.lag_count + 1)),
        }
        if self.Grouped_Batches:
            template_batch_sizes = np.asarray([len(g["target_idx"]) for g in self.Grouped_Batches])
            self.conditioning_summary.update(
                {
                    "template_targets_mean": float(template_batch_sizes.mean()),
                    "template_targets_median": float(np.median(template_batch_sizes)),
                    "template_targets_max": int(template_batch_sizes.max()),
                }
            )
            template_reuse_msg = (
                f", targets/template mean/med/max={template_batch_sizes.mean():.1f}/"
                f"{np.median(template_batch_sizes):.0f}/{template_batch_sizes.max()}"
            )
        else:
            self.conditioning_summary.update(
                {
                    "template_targets_mean": 0.0,
                    "template_targets_median": 0.0,
                    "template_targets_max": 0,
                }
            )
            template_reuse_msg = ""
        if m_sizes:
            m_arr = np.asarray(m_sizes)
            self.conditioning_summary.update(
                {
                    "conditioning_m_mean": float(m_arr.mean()),
                    "conditioning_m_median": float(np.median(m_arr)),
                    "conditioning_m_max": int(m_arr.max()),
                }
            )
            m_msg = f"m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max()}"
        else:
            self.conditioning_summary.update(
                {
                    "conditioning_m_mean": 0.0,
                    "conditioning_m_median": 0.0,
                    "conditioning_m_max": 0,
                }
            )
            m_msg = "m empty"
        if lag_count_rows:
            lag_arr = np.asarray(lag_count_rows, dtype=np.int64)
            lag_labels = ["t"] + [f"t-{lag}" for lag in range(1, self.lag_count + 1)]
            lag_msg_parts = []
            for lag, label in enumerate(lag_labels):
                vals = lag_arr[:, lag]
                self.conditioning_summary[f"conditioning_{label}_mean"] = float(vals.mean())
                self.conditioning_summary[f"conditioning_{label}_median"] = float(np.median(vals))
                self.conditioning_summary[f"conditioning_{label}_max"] = int(vals.max())
                lag_msg_parts.append(f"{label} med/max={np.median(vals):.0f}/{vals.max()}")
            lag_msg = "; " + ", ".join(lag_msg_parts)
        else:
            lag_msg = ""
        print(
            f"Done in {time.time() - t0:.1f}s. grid={self._n_lat}x{self._n_lon}, "
            f"heads={len(heads_indices)}, tails={self.n_tails}, templates={len(self.Grouped_Batches)}, "
            f"{m_msg}{lag_msg}{template_reuse_msg}"
        )
        return self

    def _check_precomputed(self):
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first")

    def _gls_jitter(self):
        return torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-6

    def _accumulate_gls_stats(self, params: torch.Tensor, include_y_quad: bool = True, catch_cholesky: bool = False):
        self._check_precomputed()
        XT_Sinv_X = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        if self.Heads_data.shape[0] > 0:
            coords = self.Heads_data[:, [0, 1, 3]].contiguous()
            X_h = self._design_from_rows(self.Heads_data)
            y_h = self.Heads_data[:, 2:3]
            try:
                K_h = self._full_cov(coords, params)
                L_h = torch.linalg.cholesky(K_h)
            except torch.linalg.LinAlgError:
                if catch_cholesky:
                    return None
                raise
            Z_X = torch.linalg.solve_triangular(L_h, X_h, upper=False)
            Z_y = torch.linalg.solve_triangular(L_h, y_h, upper=False)
            XT_Sinv_X += Z_X.T @ Z_X
            XT_Sinv_y += Z_X.T @ Z_y
            if include_y_quad:
                yT_Sinv_y += (Z_y.T @ Z_y).squeeze()
            log_det += 2.0 * torch.log(torch.diagonal(L_h)).sum()

        _, _, _, _, _, _, nugget, sigmasq = self._raw_params(params)
        total_sill = sigmasq + nugget

        for group in self.Grouped_Batches:
            deltas = group["deltas"]
            target_idx = group["target_idx"]
            b_idx = group["batch_idx"]
            B_total = int(target_idx.shape[0])
            m = int(deltas.shape[0])

            if m == 0:
                inv_s = 1.0 / total_sill
                log_s = torch.log(total_sill)
                for start in range(0, B_total, self.target_chunk_size):
                    end = min(start + self.target_chunk_size, B_total)
                    rows_t = self.Full_Data_Grid[target_idx[start:end]]
                    X_t = self._design_from_rows(rows_t)
                    y_t = rows_t[:, 2:3]
                    XT_Sinv_X += (X_t.T @ X_t) * inv_s
                    XT_Sinv_y += (X_t.T @ y_t) * inv_s
                    if include_y_quad:
                        yT_Sinv_y += ((y_t.T @ y_t).squeeze() * inv_s).squeeze()
                    log_det += (end - start) * log_s
                continue

            try:
                K = self._stencil_cov_matrix(deltas, params)
                k = self._cross_cov_vector(deltas, params)
                L = torch.linalg.cholesky(K)
                z = torch.linalg.solve_triangular(L, k.unsqueeze(1), upper=False)
                w = torch.linalg.solve_triangular(L.T, z, upper=True).flatten()
            except torch.linalg.LinAlgError:
                if catch_cholesky:
                    return None
                raise

            sigma_cond = total_sill - torch.dot(z.flatten(), z.flatten())
            if torch.any(sigma_cond <= 1e-10) or torch.isnan(sigma_cond):
                if catch_cholesky:
                    return None
                raise torch.linalg.LinAlgError("non-positive conditional variance")
            inv_s = 1.0 / sigma_cond
            log_s = torch.log(sigma_cond)

            for start in range(0, B_total, self.target_chunk_size):
                end = min(start + self.target_chunk_size, B_total)
                rows_t = self.Full_Data_Grid[target_idx[start:end]]
                X_t = self._design_from_rows(rows_t)
                y_t = rows_t[:, 2:3]

                b_chunk = b_idx[start:end]
                flat_b = b_chunk.reshape(-1)
                rows_n = self.Full_Data_Grid[flat_b].reshape(end - start, m, -1)
                y_n = rows_n[:, :, 2]
                X_n = self._design_from_rows(rows_n)

                y_eff = y_t - (y_n @ w).unsqueeze(1)
                X_eff = X_t - torch.einsum("bmf,m->bf", X_n, w)

                XT_Sinv_X += (X_eff.T @ X_eff) * inv_s
                XT_Sinv_y += (X_eff.T @ y_eff) * inv_s
                if include_y_quad:
                    yT_Sinv_y += ((y_eff.T @ y_eff).squeeze() * inv_s).squeeze()
                log_det += (end - start) * log_s

        total_N = int(self.Heads_data.shape[0]) + int(self.n_tails)
        return XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, total_N

    def vecchia_structured_likelihood(self, params: torch.Tensor) -> torch.Tensor:
        stats = self._accumulate_gls_stats(params, include_y_quad=True, catch_cholesky=True)
        if stats is None:
            return torch.tensor(float("inf"), device=self.device, dtype=torch.float64)
        XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, total_N = stats
        try:
            beta = torch.linalg.solve(XT_Sinv_X + self._gls_jitter(), XT_Sinv_y)
        except torch.linalg.LinAlgError:
            return torch.tensor(float("inf"), device=self.device, dtype=torch.float64)
        quad = yT_Sinv_y - 2.0 * (beta.T @ XT_Sinv_y).squeeze() + (beta.T @ XT_Sinv_X @ beta).squeeze()
        return 0.5 * (log_det + quad) / total_N

    def get_gls_beta(self, params: torch.Tensor) -> torch.Tensor:
        XT_Sinv_X, XT_Sinv_y, _, _, _ = self._accumulate_gls_stats(params, include_y_quad=False, catch_cholesky=False)
        return torch.linalg.solve(XT_Sinv_X + self._gls_jitter(), XT_Sinv_y)

    def set_optimizer(
        self,
        param_groups,
        lr=1.0,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=10,
    ):
        return torch.optim.LBFGS(
            param_groups,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        phi1, phi2, phi3, phi4 = np.exp(raw[0]), np.exp(raw[1]), np.exp(raw[2]), np.exp(raw[3])
        return {
            "sigmasq": phi1 / phi2,
            "range_lon": 1.0 / phi2,
            "range_lat": 1.0 / (phi2 * np.sqrt(phi3)),
            "range_time": 1.0 / (phi2 * np.sqrt(phi4)),
            "advec_lat": raw[4],
            "advec_lon": raw[5],
            "nugget": np.exp(raw[6]),
        }

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor], optimizer: torch.optim.LBFGS, max_steps: int = 50, grad_tol: float = 1e-5):
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print("--- Starting standalone Reverse-L intersection column Vecchia L-BFGS ---")

        def closure():
            optimizer.zero_grad()
            params = torch.stack([p.reshape(()) for p in params_list])
            val = self.vecchia_structured_likelihood(params)
            val.backward()
            return val

        loss = None
        last_iter = 0
        for i in range(max_steps):
            last_iter = i
            loss = optimizer.step(closure)
            with torch.no_grad():
                grads = [abs(float(p.grad.detach().item())) for p in params_list if p.grad is not None]
                max_grad = max(grads) if grads else 0.0
                print(f"--- Step {i + 1}/{max_steps} / Loss: {float(loss.detach().item()):.6f} / Max Grad: {max_grad:.2e} ---")
                if max_grad < grad_tol:
                    print(f"Converged: max_grad {max_grad:.2e} < {grad_tol:.2e}")
                    break

        raw = [float(p.detach().cpu().item()) for p in params_list]
        final_loss = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float("nan")
        print("Final Interpretable Params:", self._convert_params(raw))
        return raw + [final_loss], last_iter


def make_st_initial_params(
    sigmasq: float = 10.0,
    range_lat: float = 0.3,
    range_lon: float = 0.4,
    range_time: float = 2.0,
    advec_lat: float = 0.08,
    advec_lon: float = -0.16,
    nugget: float = 2.5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> List[torch.Tensor]:
    """Initial raw parameters for the ST column parametrization."""
    phi2 = 1.0 / float(range_lon)
    phi3 = (float(range_lon) / float(range_lat)) ** 2
    phi4 = (float(range_lon) / float(range_time)) ** 2
    phi1 = float(sigmasq) * phi2
    vals = [
        np.log(phi1),
        np.log(phi2),
        np.log(phi3),
        np.log(phi4),
        float(advec_lat),
        float(advec_lon),
        np.log(float(nugget)),
    ]
    return [torch.tensor(v, device=device, dtype=dtype, requires_grad=True) for v in vals]


__all__ = [
    "DayIntersectionResult",
    "build_grid_index",
    "column_scan_order",
    "pack_scan_order_coords",
    "diagnose_scan_order",
    "make_day_intersection_input_map",
    "make_st_initial_params",
    "ReverseLIntersectionColumnVecchiaFit",
]
