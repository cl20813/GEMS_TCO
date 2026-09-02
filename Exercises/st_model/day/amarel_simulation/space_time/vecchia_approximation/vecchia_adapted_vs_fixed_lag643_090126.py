#!/usr/bin/env python3
"""Compare three lag-643 conditioning geometries and their exact union.

The experiment uses the same M3 masked-FFT + safeguarded Q3 advection seed for
every fit.  Only the fixed conditioning graph changes:

``adapted``
    4x4 target blocks and 6/4/3 conditioning-block budgets.  Past conditioning
    corridors use the calibrated signed 2-D vector: t-1 spans
    target - [0.5, 1.5]*v_hat and t-2 spans target - [0, 2]*v_hat.  The minus
    sign is required by the covariance convention d(h - v*tau): for a current
    target, correlated past locations are in the -v direction.

``shifted``
    The same 6/4/3 budgets, with four nearest blocks centered at target-v_hat
    for t-1 and three nearest blocks centered at target-2*v_hat for t-2.

``fixed``
    The same target blocks and 6/4/3 budgets, but both past neighborhoods stay
    centered at the target block (zero displacement).

``union``
    The exact set union of adapted, shifted, and fixed conditioning blocks: six
    same-time blocks, up to 12 t-1 blocks, and up to nine t-2 blocks after
    deduplication.

Each data set produces four native fits, a 4x4 cross-objective matrix (every
fitted parameter vector evaluated on every graph), and a four-curve
conditional-eigen diagnostic.  Synthetic fits also retain every truth,
estimate, absolute/log error, advection error, and a combined parameter error.

The conditional-eigen diagnostic follows
``vecchia_conditional_eigen_sort_common_engine_061926.py``: conditional target
covariances are eigendecomposed, fitted mean columns are projected out, and
squared residual scores are accumulated in decreasing conditional-eigenvalue
order.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
LOCAL_REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
LOCAL_SRC = LOCAL_REPO / "src"
AMAREL_SRC = Path("/home/jl2815/tco")
SRC = AMAREL_SRC if (AMAREL_SRC / "GEMS_TCO").exists() else LOCAL_SRC
for path in (SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from GEMS_TCO.data_loader import load_data_dynamic_processed  # noqa: E402
from GEMS_TCO.vecchia_cluster import StrategyClusterVecchiaFit  # noqa: E402
from GEMS_TCO.vecchia_realdata_adapted_corridor_width_4x4_lag643 import (  # noqa: E402
    AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit,
)
from GEMS_TCO.vecchia_realdata_calibrated_shifted_center_4x4_lag643 import (  # noqa: E402
    CalibratedShiftedCenter4x4Lag643VecchiaFit,
)


DTYPE = torch.double
BLOCK_SHAPE = (4, 4)
LAG_COUNTS = (6, 4, 3)
BASE_GEOMETRIES = ("adapted", "shifted", "fixed")
GEOMETRIES = (*BASE_GEOMETRIES, "union")
EIGEN_GEOMETRIES = GEOMETRIES
GEOMETRY_COLORS = {
    "adapted": "#1f77b4",
    "shifted": "#ff7f0e",
    "fixed": "#d62728",
    "union": "#2ca02c",
}
PARAMETERS = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)
POSITIVE_PARAMETERS = ("sigmasq", "range_lat", "range_lon", "range_time", "nugget")
BROWN_BRIDGE_Q95 = 1.3581015157406195
EPS = 1e-12

DEFAULT_REAL_INIT = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0218,
    "advec_lon": -0.1689,
    "nugget": 0.247,
}


@dataclass
class DayAsset:
    dataset_id: str
    data_kind: str
    year: int
    month: int
    day: int
    date: str
    keys: list[str]
    source_map: dict[str, torch.Tensor]
    grid_coords: np.ndarray
    center_value: float
    n_valid: int
    n_total: int
    truth: dict[str, float] | None
    source_path: str


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True), encoding="utf-8")


def parse_pair(text: str, cast=float) -> list[Any]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got {text!r}")
    return [cast(parts[0]), cast(parts[1])]


def count_valid(source_map: dict[str, torch.Tensor]) -> tuple[int, int]:
    total = sum(int(tensor.shape[0]) for tensor in source_map.values())
    valid = sum(int(torch.isfinite(tensor[:, 2]).sum().item()) for tensor in source_map.values())
    return valid, total


def date_from_key(key: str) -> str | None:
    match = re.search(r"y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})", str(key))
    if match is None:
        return None
    return f"20{match.group('yy')}-{match.group('mm')}-{match.group('dd')}"


def assert_grid_order(frames: dict[str, pd.DataFrame], keys: Sequence[str], base: np.ndarray) -> None:
    for key in keys:
        coords = frames[key][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        if coords.shape != base.shape or not np.allclose(coords, base, equal_nan=True):
            raise RuntimeError(f"Regular-grid coordinate order differs at {key}")


def load_selection(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    rows = [dict(row, data_kind="real") for row in selection["real"]]
    rows.extend(dict(row, data_kind="synthetic") for row in selection["synthetic"])
    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 selected data sets, found {len(rows)}")
    if len({row["dataset_id"] for row in rows}) != len(rows):
        raise RuntimeError("Selection contains duplicate dataset_id values")
    return selection, rows


def load_real_asset(spec: dict[str, Any], args: argparse.Namespace) -> DayAsset:
    loader = load_data_dynamic_processed(str(args.real_data_root))
    frames, _, _, monthly_mean = loader.load_maxmin_ordered_data_bymonthyear(
        lat_lon_resolution=[1, 1],
        mm_cond_number=1,
        years_=[str(spec["year"])],
        months_=[int(spec["month"])],
        lat_range=parse_pair(args.lat_range, float),
        lon_range=parse_pair(args.lon_range, float),
        is_whittle=True,
    )
    if not frames:
        raise RuntimeError(f"No real data loaded for {spec['date']} from {args.real_data_root}")
    all_keys = sorted(frames)
    day_keys = [key for key in all_keys if date_from_key(key) == spec["date"]]
    if len(day_keys) != int(args.hours_per_day):
        raise RuntimeError(f"Expected {args.hours_per_day} real slots on {spec['date']}, got {len(day_keys)}")
    base = frames[day_keys[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    assert_grid_order(frames, day_keys, base)
    start = all_keys.index(day_keys[0])
    if all_keys[start : start + len(day_keys)] != day_keys:
        raise RuntimeError(f"Selected real keys are not contiguous for {spec['date']}")
    source_map, _ = loader.load_working_data(
        frames,
        monthly_mean=float(monthly_mean),
        idx_for_datamap=[start, start + len(day_keys)],
        ord_mm=None,
        dtype=DTYPE,
        keep_ori=bool(args.keep_exact_loc),
    )
    if sorted(source_map) != day_keys:
        raise RuntimeError("Real-data loader returned unexpected hourly keys")
    valid, total = count_valid(source_map)
    return DayAsset(
        dataset_id=str(spec["dataset_id"]),
        data_kind="real",
        year=int(spec["year"]),
        month=int(spec["month"]),
        day=int(spec["day"]),
        date=str(spec["date"]),
        keys=day_keys,
        source_map={key: value.contiguous() for key, value in source_map.items()},
        grid_coords=base,
        center_value=float(monthly_mean),
        n_valid=valid,
        n_total=total,
        truth=None,
        source_path=str(Path(args.real_data_root) / f"pickle_{spec['year']}" / f"tco_grid_{str(spec['year'])[2:]}_07.pkl"),
    )


def build_synthetic_tensor(
    frame: pd.DataFrame,
    local_time: int,
    center: float,
    keep_exact_loc: bool,
) -> torch.Tensor:
    grid_lat = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(dtype=np.float64)
    grid_lon = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(dtype=np.float64)
    if keep_exact_loc and {"Source_Latitude", "Source_Longitude"}.issubset(frame.columns):
        lat = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(dtype=np.float64)
        lon = pd.to_numeric(frame["Source_Longitude"], errors="coerce").to_numpy(dtype=np.float64)
        lat = np.where(np.isfinite(lat), lat, grid_lat)
        lon = np.where(np.isfinite(lon), lon, grid_lon)
    else:
        lat, lon = grid_lat, grid_lon
    y = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(dtype=np.float64) - center
    base = torch.from_numpy(
        np.column_stack([lat, lon, y, np.full(len(frame), float(local_time))])
    ).to(dtype=DTYPE)
    dummies = F.one_hot(torch.tensor([local_time]), num_classes=8).repeat(len(frame), 1)[:, 1:]
    return torch.cat([base, dummies.to(dtype=DTYPE)], dim=1).contiguous()


def load_synthetic_asset(spec: dict[str, Any], args: argparse.Namespace) -> DayAsset:
    year_dir = Path(args.synthetic_data_root) / f"{spec['year']}_july_st_circulant"
    data_path = year_dir / f"sim_july{spec['year']}_st_circulant_gridded.pkl"
    truth_path = year_dir / f"sim_july{spec['year']}_st_circulant_truth.json"
    if not data_path.exists() or not truth_path.exists():
        raise FileNotFoundError(f"Synthetic inputs missing: {data_path} or {truth_path}")
    obj = pd.read_pickle(data_path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict pickle at {data_path}, got {type(obj)}")
    all_keys = sorted(obj)
    day_keys = [key for key in all_keys if date_from_key(key) == spec["date"]]
    if len(day_keys) != int(args.hours_per_day):
        raise RuntimeError(f"Expected {args.hours_per_day} synthetic slots on {spec['date']}, got {len(day_keys)}")
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    frames: dict[str, pd.DataFrame] = {}
    for key in day_keys:
        frame = obj[key]
        mask = frame["Latitude"].between(*lat_range) & frame["Longitude"].between(*lon_range)
        frames[key] = frame.loc[mask].reset_index(drop=True)
    base = frames[day_keys[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    assert_grid_order(frames, day_keys, base)
    values = [pd.to_numeric(frames[key]["ColumnAmountO3"], errors="coerce").to_numpy() for key in day_keys]
    center = float(np.nanmean(np.concatenate(values)))
    source_map = {
        key: build_synthetic_tensor(frames[key], local_time, center, bool(args.keep_exact_loc))
        for local_time, key in enumerate(day_keys)
    }
    truth_raw = json.loads(truth_path.read_text(encoding="utf-8"))
    truth = {name: float(truth_raw[name]) for name in PARAMETERS}
    if not math.isclose(float(truth_raw.get("smooth", np.nan)), 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Synthetic truth is not smooth=0.5: {truth_path}")
    valid, total = count_valid(source_map)
    del obj
    gc.collect()
    return DayAsset(
        dataset_id=str(spec["dataset_id"]),
        data_kind="synthetic",
        year=int(spec["year"]),
        month=int(spec["month"]),
        day=int(spec["day"]),
        date=str(spec["date"]),
        keys=day_keys,
        source_map=source_map,
        grid_coords=base,
        center_value=center,
        n_valid=valid,
        n_total=total,
        truth=truth,
        source_path=str(data_path),
    )


class FixedCenterLag643VecchiaFit(StrategyClusterVecchiaFit):
    """Lag-643 graph whose past conditioning centers remain at the target."""

    def __init__(
        self,
        smooth: float,
        input_map: dict[str, torch.Tensor],
        grid_coords: np.ndarray,
        daily_stride: int,
        target_chunk_size: int,
        min_target_points: int,
    ):
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=BLOCK_SHAPE,
            strategy="center_tapered",
            lag0_block_count=LAG_COUNTS[0],
            lag1_block_count=LAG_COUNTS[1],
            lag2_block_count=LAG_COUNTS[2],
            daily_stride=daily_stride,
            lag1_lon_offset=1e-12,
            lag2_lon_offset=2e-12,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
        )
        self.temporal_basis = "fixed_target_center"

    def cluster_summary(self) -> dict[str, Any]:
        out = super().cluster_summary()
        out.update(
            {
                "spec_name": "fixed_target_center_4x4_lag643",
                "geometry": "fixed",
                "geometry_definition": "past centers fixed at target for t-1 and t-2",
                "conditioning_mode": "fixed_target_center",
                "reference_advec_lat": 0.0,
                "reference_advec_lon": 0.0,
                "past_step_lat": 0.0,
                "past_step_lon": 0.0,
            }
        )
        return out


class UnionLag643VecchiaFit(AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit):
    """Exact block-set union of corridor, shifted, and fixed neighbors."""

    def __init__(
        self,
        smooth: float,
        input_map: dict[str, torch.Tensor],
        grid_coords: np.ndarray,
        reference_advec_lat: float,
        reference_advec_lon: float,
        daily_stride: int,
        target_chunk_size: int,
        min_target_points: int,
    ):
        self.component_lag_counts = {1: LAG_COUNTS[1], 2: LAG_COUNTS[2]}
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lat=reference_advec_lat,
            reference_advec_lon=reference_advec_lon,
            daily_stride=daily_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
        )
        # The parent is the 6/4/3 corridor component.  The union also includes
        # a shifted-center and a fixed-center 4/3 component.  These fields all
        # participate in precomputation/dummy allocation and therefore must be
        # changed together before precompute_conditioning_sets() is called.
        self.lag1_block_count = 3 * LAG_COUNTS[1]
        self.lag2_block_count = 3 * LAG_COUNTS[2]
        self.lag1_max_blocks = self.lag1_block_count
        self.lag2_max_blocks = self.lag2_block_count
        self.lag1_local_blocks = self.lag1_max_blocks
        self.lag2_local_blocks = self.lag2_max_blocks
        self.all_neighbor_search = max(
            self.all_neighbor_search,
            self.lag1_max_blocks,
            self.lag2_max_blocks,
        ) + 8
        self.temporal_basis = "union_corridor_shifted_and_fixed"

    def _lag_candidates(self, block_idx: int, lag: int) -> list[int]:
        component_count = self.component_lag_counts[int(lag)]
        multipliers = (
            self.lag1_corridor_multipliers
            if lag == 1
            else self.lag2_corridor_multipliers
        )
        adapted = self._cluster_candidates_from_vector_corridor(
            block_idx,
            multipliers,
            component_count,
        )
        shifted = (
            CalibratedShiftedCenter4x4Lag643VecchiaFit
            ._cluster_candidates_from_shifted_center(
                self,
                int(block_idx),
                1.0 if int(lag) == 1 else 2.0,
                component_count,
            )
        )
        fixed = self._cluster_candidates_from_center(int(block_idx), component_count)
        out: list[int] = []
        self._append_unique_int(out, adapted)
        self._append_unique_int(out, shifted)
        self._append_unique_int(out, fixed)
        return out

    def _precompute_message(self) -> str:
        return (
            "Pre-computing union corridor+shifted+fixed lag643 "
            f"(seed=({self.reference_advec_lat:.6f},{self.reference_advec_lon:.6f}), "
            f"corridor_multipliers={self.lag1_corridor_multipliers}/"
            f"{self.lag2_corridor_multipliers}, "
            "shifted_centers=1v/2v, budgets=6/up-to-12/up-to-9)..."
        )

    def cluster_summary(self) -> dict[str, Any]:
        out = super().cluster_summary()
        out.update(
            {
                "spec_name": "union_corridor_shifted_fixed_4x4_lag643",
                "geometry": "union",
                "geometry_definition": (
                    "exact deduplicated union of calibrated corridor, calibrated "
                    "shifted-center, and fixed-center block sets"
                ),
                "conditioning_mode": "union_corridor_shifted_and_fixed",
                "reference_advec_lat": self.reference_advec_lat,
                "reference_advec_lon": self.reference_advec_lon,
                "past_step_lat": float(self.past_offset_vector[0]),
                "past_step_lon": float(self.past_offset_vector[1]),
                "component_lag1_block_count": LAG_COUNTS[1],
                "component_lag2_block_count": LAG_COUNTS[2],
                "union_component_count": 3,
            }
        )
        return out


def build_geometry_model(
    geometry: str,
    asset: DayAsset,
    seed: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> StrategyClusterVecchiaFit:
    mapped = {
        key: tensor.to(device=device, dtype=DTYPE, non_blocking=True).contiguous()
        for key, tensor in asset.source_map.items()
    }
    common = dict(
        smooth=float(args.smooth),
        input_map=mapped,
        grid_coords=asset.grid_coords,
        daily_stride=int(args.daily_stride),
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=int(args.min_target_points),
    )
    if geometry == "adapted":
        return AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit(
            reference_advec_lat=float(seed["seed_lat"]),
            reference_advec_lon=float(seed["seed_lon"]),
            **common,
        )
    if geometry == "shifted":
        return CalibratedShiftedCenter4x4Lag643VecchiaFit(
            reference_advec_lat=float(seed["seed_lat"]),
            reference_advec_lon=float(seed["seed_lon"]),
            **common,
        )
    if geometry == "fixed":
        return FixedCenterLag643VecchiaFit(**common)
    if geometry == "union":
        return UnionLag643VecchiaFit(
            reference_advec_lat=float(seed["seed_lat"]),
            reference_advec_lon=float(seed["seed_lon"]),
            **common,
        )
    raise ValueError(f"Unknown geometry {geometry!r}")


def conditioning_block_summary(model: StrategyClusterVecchiaFit, n_times: int) -> dict[str, float]:
    total_counts: list[int] = []
    lag1_counts: list[int] = []
    lag2_counts: list[int] = []
    mature_counts: list[int] = []
    for time_idx in range(int(n_times)):
        for block_idx in range(int(model.n_clusters)):
            _, _, refs = model._conditioning_blocks(block_idx, time_idx)
            total_counts.append(len(refs))
            lag1_counts.append(sum(int(t == time_idx - 1) for t, _ in refs) if time_idx > 0 else 0)
            lag2_counts.append(
                sum(int(t == time_idx - int(model.daily_stride)) for t, _ in refs)
                if time_idx >= int(model.daily_stride)
                else 0
            )
            if time_idx >= int(model.daily_stride):
                mature_counts.append(len(refs))
    return {
        "mean_condition_blocks": float(np.mean(total_counts)),
        "max_condition_blocks": int(np.max(total_counts)),
        "mean_lag1_blocks": float(np.mean(lag1_counts)),
        "max_lag1_blocks": int(np.max(lag1_counts)),
        "mean_lag2_blocks": float(np.mean(lag2_counts)),
        "max_lag2_blocks": int(np.max(lag2_counts)),
        "mean_mature_condition_blocks": float(np.mean(mature_counts)) if mature_counts else np.nan,
    }


def make_hourly_grids(asset: DayAsset) -> tuple[list[np.ndarray], float, float]:
    lat_key = np.round(asset.grid_coords[:, 0], 6)
    lon_key = np.round(asset.grid_coords[:, 1], 6)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_lookup = {float(value): idx for idx, value in enumerate(lats)}
    lon_lookup = {float(value): idx for idx, value in enumerate(lons)}
    rows = np.asarray([lat_lookup[float(value)] for value in lat_key], dtype=np.int64)
    cols = np.asarray([lon_lookup[float(value)] for value in lon_key], dtype=np.int64)
    grids: list[np.ndarray] = []
    for key in sorted(asset.source_map):
        values = asset.source_map[key][:, 2].detach().cpu().numpy()
        grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float64)
        grid[rows, cols] = values
        grid -= float(np.nanmean(grid))
        grids.append(grid)
    if len(lats) < 2 or len(lons) < 2:
        raise RuntimeError(f"M3 requires at least a 2x2 grid, got {len(lats)}x{len(lons)}")
    return grids, float(np.median(np.diff(lats))), float(np.median(np.diff(lons)))


def crop_full_correlation(
    full: np.ndarray,
    shape: tuple[int, int],
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
) -> np.ndarray:
    rows = int(shape[0] - 1) + offsets_lat.astype(np.int64)
    cols = int(shape[1] - 1) + offsets_lon.astype(np.int64)
    return np.asarray(full[np.ix_(rows, cols)], dtype=np.float64)


def fft_pair_squared_difference(
    current: np.ndarray,
    following: np.ndarray,
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_a = np.isfinite(current)
    valid_b = np.isfinite(following)
    mask_a = valid_a.astype(np.float64)
    mask_b = valid_b.astype(np.float64)
    a = np.where(valid_a, current, 0.0)
    b = np.where(valid_b, following, 0.0)
    count_full = correlate(mask_b, mask_a, mode="full", method="fft")
    a2_full = correlate(mask_b, a * a, mode="full", method="fft")
    b2_full = correlate(b * b, mask_a, mode="full", method="fft")
    cross_full = correlate(b, a, mode="full", method="fft")
    sumsq_full = a2_full + b2_full - 2.0 * cross_full
    counts = crop_full_correlation(count_full, current.shape, offsets_lat, offsets_lon)
    sumsq = crop_full_correlation(sumsq_full, current.shape, offsets_lat, offsets_lon)
    return np.maximum(sumsq, 0.0), np.rint(np.maximum(counts, 0.0)).astype(np.int64)


def nan_gaussian_filter(array: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    finite = np.isfinite(array)
    values = gaussian_filter(np.where(finite, array, 0.0), sigma=sigma, mode="nearest")
    weights = gaussian_filter(finite.astype(np.float64), sigma=sigma, mode="nearest")
    out = values / np.maximum(weights, EPS)
    out[weights <= EPS] = np.nan
    return out


def surface_minimum(
    surface: np.ndarray,
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
) -> tuple[int, int, float, float]:
    if not np.isfinite(surface).any():
        raise RuntimeError("M3 surface has no finite cells")
    row, col = np.unravel_index(np.nanargmin(surface), surface.shape)
    return (
        int(row),
        int(col),
        float(offsets_lat[row] * lat_step),
        float(offsets_lon[col] * lon_step),
    )


def safeguarded_q3(
    surface: np.ndarray,
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    max_condition_number: float,
) -> dict[str, Any]:
    row, col, grid_lat, grid_lon = surface_minimum(
        surface, offsets_lat, offsets_lon, lat_step, lon_step
    )
    result: dict[str, Any] = {
        "seed_lat": grid_lat,
        "seed_lon": grid_lon,
        "grid_seed_lat": grid_lat,
        "grid_seed_lon": grid_lon,
        "subgrid_used": False,
        "selection_reason": "fallback",
        "delta_lat_cells": 0.0,
        "delta_lon_cells": 0.0,
        "hessian_condition_cells": np.nan,
        "hessian_min_eigenvalue_cells": np.nan,
        "quadratic_residual_rmse": np.nan,
    }
    if row < 1 or col < 1 or row >= surface.shape[0] - 1 or col >= surface.shape[1] - 1:
        result["selection_reason"] = "boundary"
        return result
    patch = np.asarray(surface[row - 1 : row + 2, col - 1 : col + 2], dtype=np.float64)
    if not np.isfinite(patch).all():
        result["selection_reason"] = "nonfinite_patch"
        return result
    design: list[list[float]] = []
    values: list[float] = []
    for local_row, u in enumerate((-1.0, 0.0, 1.0)):
        for local_col, v in enumerate((-1.0, 0.0, 1.0)):
            design.append([1.0, u, v, 0.5 * u * u, u * v, 0.5 * v * v])
            values.append(float(patch[local_row, local_col]))
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    gradient = coefficients[1:3]
    hessian = np.asarray(
        [[coefficients[3], coefficients[4]], [coefficients[4], coefficients[5]]],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    result["hessian_min_eigenvalue_cells"] = float(eigenvalues[0])
    result["quadratic_residual_rmse"] = float(np.sqrt(np.mean(np.square(y - x @ coefficients))))
    if not np.all(np.isfinite(eigenvalues)) or not np.all(eigenvalues > 0.0):
        result["selection_reason"] = "non_positive_hessian"
        return result
    condition = float(np.linalg.cond(hessian))
    result["hessian_condition_cells"] = condition
    if not np.isfinite(condition) or condition > float(max_condition_number):
        result["selection_reason"] = "ill_conditioned_hessian"
        return result
    delta = -np.linalg.solve(hessian, gradient)
    if not np.all(np.isfinite(delta)):
        result["selection_reason"] = "nonfinite_vertex"
        return result
    if abs(float(delta[0])) > 0.5 or abs(float(delta[1])) > 0.5:
        result["selection_reason"] = "delta_outside_half_cell"
        return result
    result.update(
        {
            "seed_lat": float(grid_lat + delta[0] * lat_step),
            "seed_lon": float(grid_lon + delta[1] * lon_step),
            "subgrid_used": True,
            "selection_reason": "accepted",
            "delta_lat_cells": float(delta[0]),
            "delta_lon_cells": float(delta[1]),
        }
    )
    return result


def m3_q3_seed(asset: DayAsset, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    grids, lat_step, lon_step = make_hourly_grids(asset)
    offsets_lat = np.arange(-int(args.empirical_max_lat_offset), int(args.empirical_max_lat_offset) + 1)
    offsets_lon = np.arange(-int(args.empirical_max_lon_offset), int(args.empirical_max_lon_offset) + 1)
    sumsq = np.zeros((len(offsets_lat), len(offsets_lon)), dtype=np.float64)
    counts = np.zeros_like(sumsq, dtype=np.int64)
    for hour in range(len(grids) - 1):
        pair_sumsq, pair_counts = fft_pair_squared_difference(
            grids[hour], grids[hour + 1], offsets_lat, offsets_lon
        )
        sumsq += pair_sumsq
        counts += pair_counts
    gamma = np.full_like(sumsq, np.nan)
    valid = counts >= int(args.empirical_min_pair_count)
    gamma[valid] = 0.5 * sumsq[valid] / counts[valid]
    smoothed = nan_gaussian_filter(
        gamma,
        (
            float(args.empirical_smooth_bandwidth_deg) / abs(lat_step),
            float(args.empirical_smooth_bandwidth_deg) / abs(lon_step),
        ),
    )
    seed = safeguarded_q3(
        smoothed,
        offsets_lat,
        offsets_lon,
        lat_step,
        lon_step,
        float(args.subgrid_max_condition_number),
    )
    seed.update(
        {
            "method": "M3_masked_fft_plus_safeguarded_Q3",
            "lat_step": lat_step,
            "lon_step": lon_step,
            "min_pair_count": int(args.empirical_min_pair_count),
            "max_pair_count": int(np.max(counts)),
            "surface_min": float(np.nanmin(smoothed)),
            "initializer_s": float(time.perf_counter() - started),
        }
    )
    if asset.truth is not None:
        seed["advection_error"] = float(
            np.hypot(seed["seed_lat"] - asset.truth["advec_lat"], seed["seed_lon"] - asset.truth["advec_lon"])
        )
        seed["advection_error_grid_cells"] = float(
            np.hypot(
                (seed["seed_lat"] - asset.truth["advec_lat"]) / abs(lat_step),
                (seed["seed_lon"] - asset.truth["advec_lon"]) / abs(lon_step),
            )
        )
    return seed


def physical_to_raw(params: dict[str, float]) -> list[float]:
    range_lon = float(params["range_lon"])
    phi2 = 1.0 / range_lon
    return [
        float(np.log(float(params["sigmasq"]) * phi2)),
        float(np.log(phi2)),
        float(np.log((range_lon / float(params["range_lat"])) ** 2)),
        float(np.log((range_lon / float(params["range_time"])) ** 2)),
        float(params["advec_lat"]),
        float(params["advec_lon"]),
        float(np.log(float(params["nugget"]))),
    ]


def raw_to_physical(raw: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in raw[:7]]
    phi2 = float(np.exp(values[1]))
    range_lon = 1.0 / phi2
    return {
        "sigmasq": float(np.exp(values[0]) / phi2),
        "range_lat": float(range_lon / np.sqrt(np.exp(values[2]))),
        "range_lon": range_lon,
        "range_time": float(range_lon / np.sqrt(np.exp(values[3]))),
        "advec_lat": values[4],
        "advec_lon": values[5],
        "nugget": float(np.exp(values[6])),
    }


def parameter_error_columns(
    estimate: dict[str, float], truth: dict[str, float], lat_step: float, lon_step: float
) -> dict[str, float]:
    out: dict[str, float] = {}
    log_parts: list[float] = []
    for name in PARAMETERS:
        out[f"truth_{name}"] = float(truth[name])
        out[f"error_abs_{name}"] = abs(float(estimate[name]) - float(truth[name]))
        if name in POSITIVE_PARAMETERS:
            log_error = float(np.log(float(estimate[name]) / float(truth[name])))
            out[f"error_logratio_{name}"] = log_error
            log_parts.append(log_error)
    dlat_cells = (float(estimate["advec_lat"]) - float(truth["advec_lat"])) / abs(lat_step)
    dlon_cells = (float(estimate["advec_lon"]) - float(truth["advec_lon"])) / abs(lon_step)
    out["advection_error"] = float(
        np.hypot(estimate["advec_lat"] - truth["advec_lat"], estimate["advec_lon"] - truth["advec_lon"])
    )
    out["advection_error_grid_cells"] = float(np.hypot(dlat_cells, dlon_cells))
    out["positive_parameter_log_error"] = float(np.linalg.norm(log_parts))
    out["combined_parameter_error"] = float(np.linalg.norm(log_parts + [dlat_cells, dlon_cells]))
    return out


def conditional_eigen_curve(
    model: StrategyClusterVecchiaFit,
    params: torch.Tensor,
    beta: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not model.is_precomputed:
        raise RuntimeError("Model must be precomputed before conditional eigen diagnostic")
    eigen_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    x_chunks: list[torch.Tensor] = []
    log_det_half = params.new_tensor(0.0)
    n_blocks = 0
    chunk_size = max(1, int(args.diag_chunk_size))
    with torch.no_grad():
        for batch in model._cluster_batches:
            target_slice = slice(batch.max_cond_points, batch.max_cond_points + batch.target_size)
            for start in range(0, int(batch.X.shape[0]), chunk_size):
                end = min(start + chunk_size, int(batch.X.shape[0]))
                cov = model.matern_cov_batched(params, batch.X[start:end])
                chol = torch.linalg.cholesky(cov)
                z_x = torch.linalg.solve_triangular(chol, batch.Locs[start:end], upper=False)
                z_y = torch.linalg.solve_triangular(chol, batch.Y[start:end], upper=False)
                u_x = z_x[:, target_slice, :]
                u_y = z_y[:, target_slice, 0]
                chol_tt = chol[:, target_slice, target_slice]
                cond_y = torch.bmm(chol_tt, u_y.unsqueeze(-1))
                cond_x = torch.bmm(chol_tt, u_x)
                cond_cov = torch.bmm(chol_tt, chol_tt.transpose(1, 2))
                evals, evecs = torch.linalg.eigh(0.5 * (cond_cov + cond_cov.transpose(1, 2)))
                evals = evals.clamp_min(EPS).flip(dims=[1])
                evecs = evecs.flip(dims=[2])
                sqrt_evals = torch.sqrt(evals)
                y_scores = torch.bmm(evecs.transpose(1, 2), cond_y).squeeze(-1) / sqrt_evals
                x_scores = torch.bmm(evecs.transpose(1, 2), cond_x) / sqrt_evals.unsqueeze(-1)
                log_det_half += torch.log(torch.diagonal(chol_tt, dim1=1, dim2=2)).sum()
                n_blocks += end - start
                eigen_chunks.append(evals.reshape(-1))
                y_chunks.append(y_scores.reshape(-1))
                x_chunks.append(x_scores.reshape(-1, x_scores.shape[-1]))
    eigenvalues = torch.cat(eigen_chunks).to(dtype=DTYPE)
    y_scores = torch.cat(y_chunks).to(dtype=DTYPE)
    x_scores = torch.cat(x_chunks).to(dtype=DTYPE)
    finite = torch.isfinite(eigenvalues) & torch.isfinite(y_scores) & torch.isfinite(x_scores).all(dim=1)
    eigenvalues, y_scores, x_scores = eigenvalues[finite], y_scores[finite], x_scores[finite]
    beta_vec = beta.detach().reshape(-1).to(device=x_scores.device, dtype=DTYPE)
    residual = y_scores - x_scores @ beta_vec
    xtx = x_scores.T @ x_scores
    leverage = torch.einsum(
        "ij,jk,ik->i",
        x_scores,
        torch.linalg.pinv(xtx + torch.eye(xtx.shape[0], device=xtx.device, dtype=DTYPE) * 1e-6),
        x_scores,
    ).clamp(0.0, 1.0 - 1e-10)
    expected_increment = 1.0 - leverage
    y2 = residual.square()
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    y2 = y2[order]
    expected_increment = expected_increment[order]
    csum = torch.cumsum(y2, dim=0)
    expected = torch.cumsum(expected_increment, dim=0)
    residual_df = float(expected[-1].detach().cpu().item())
    width = float(args.brown_bridge_q) * math.sqrt(2.0 * max(residual_df, 1.0))
    bridge = float(torch.max(torch.abs(csum - expected)).detach().cpu().item()) / math.sqrt(
        2.0 * max(residual_df, 1.0)
    )
    curve = pd.DataFrame(
        {
            "frac_index": (expected / max(residual_df, EPS)).detach().cpu().numpy(),
            "conditional_eigenvalue": eigenvalues.detach().cpu().numpy(),
            "scaled_cumsum": (csum / max(residual_df, EPS)).detach().cpu().numpy(),
            "scaled_expected": (expected / max(residual_df, EPS)).detach().cpu().numpy(),
            "scaled_band_lower": (expected / max(residual_df, EPS) - width / max(residual_df, EPS)).detach().cpu().numpy(),
            "scaled_band_upper": (expected / max(residual_df, EPS) + width / max(residual_df, EPS)).detach().cpu().numpy(),
        }
    )
    conditional_loss_total = float((log_det_half + 0.5 * y2.sum()).detach().cpu().item())
    summary = {
        "n_conditional_blocks": int(n_blocks),
        "n_conditional_scores": int(len(curve)),
        "residual_df": residual_df,
        "mean_y2": float(y2.sum().detach().cpu().item() / max(residual_df, EPS)),
        "max_abs_bridge_scaled": bridge,
        "conditional_loss_total": conditional_loss_total,
        "conditional_loss_per_score": conditional_loss_total / max(len(curve), 1),
        "min_conditional_eigenvalue": float(eigenvalues[-1].detach().cpu().item()),
        "max_conditional_eigenvalue": float(eigenvalues[0].detach().cpu().item()),
    }
    return curve, summary


def resample_curve(curve: pd.DataFrame, n_grid: int) -> pd.DataFrame:
    grid = np.linspace(1.0 / int(n_grid), 1.0, int(n_grid))
    x = curve["frac_index"].to_numpy(dtype=np.float64)
    out = pd.DataFrame(
        {
            "frac_index": grid,
            "scaled_cumsum": np.interp(grid, x, curve["scaled_cumsum"]),
            "scaled_expected": grid,
            "conditional_eigenvalue": np.interp(grid, x, curve["conditional_eigenvalue"]),
        }
    )
    return out


def plot_eigen_comparison(
    curves: dict[str, pd.DataFrame], summaries: dict[str, dict[str, Any]], title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    y_max = 1.05
    for geometry in EIGEN_GEOMETRIES:
        curve = curves[geometry]
        union_nll = summaries[geometry].get("union_graph_nll", np.nan)
        nll_label = (
            f"NLL native/union={summaries[geometry]['final_native_nll']:.5f}/"
            f"{union_nll:.5f}"
            if np.isfinite(union_nll)
            else f"NLL={summaries[geometry]['final_native_nll']:.5f}"
        )
        label = (
            f"{geometry}: {nll_label}, "
            f"D={summaries[geometry]['max_abs_bridge_scaled']:.3f}"
        )
        ax.plot(
            curve["frac_index"],
            curve["scaled_cumsum"],
            color=GEOMETRY_COLORS[geometry],
            lw=1.8,
            label=label,
        )
        y_max = max(y_max, float(np.nanmax(curve["scaled_cumsum"])) * 1.04)
    grid = np.linspace(0.0, 1.0, 200)
    ax.plot(grid, grid, color="0.4", lw=1.1, label="expected")
    residual_df = float(np.mean([summaries[g]["residual_df"] for g in EIGEN_GEOMETRIES]))
    band = BROWN_BRIDGE_Q95 * math.sqrt(2.0 / max(residual_df, 1.0))
    ax.plot(grid, grid - band, color="0.65", ls="--", lw=0.9)
    ax.plot(grid, grid + band, color="0.65", ls="--", lw=0.9)
    ax.set(xlim=(0, 1), ylim=(0, y_max))
    ax.set_xlabel("projected expected-df fraction, decreasing conditional eigenvalue")
    ax.set_ylabel("cumulative squared conditional score / residual df")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fit_one_geometry(
    geometry: str,
    asset: DayAsset,
    seed: dict[str, Any],
    init: dict[str, float],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[float], pd.DataFrame | None, dict[str, Any] | None]:
    model = build_geometry_model(geometry, asset, seed, device, args)
    t0 = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - t0
    condition_summary = conditioning_block_summary(model, len(asset.keys))
    raw_init = physical_to_raw({**init, "advec_lat": seed["seed_lat"], "advec_lon": seed["seed_lon"]})
    params = [Parameter(torch.tensor(value, dtype=DTYPE, device=device)) for value in raw_init]
    optimizer = model.set_optimizer(
        params,
        lr=float(args.lbfgs_lr),
        max_iter=int(args.lbfgs_eval),
        max_eval=int(args.lbfgs_eval),
        history_size=int(args.lbfgs_history),
    )
    t1 = time.perf_counter()
    if args.suppress_fit_prints:
        with contextlib.redirect_stdout(io.StringIO()):
            returned, step_index = model.fit_vecc_lbfgs(
                params, optimizer, max_steps=int(args.lbfgs_steps), grad_tol=float(args.grad_tol)
            )
    else:
        returned, step_index = model.fit_vecc_lbfgs(
            params, optimizer, max_steps=int(args.lbfgs_steps), grad_tol=float(args.grad_tol)
        )
    fit_s = time.perf_counter() - t1
    raw_final = [float(param.detach().item()) for param in params]
    params_tensor = torch.as_tensor(raw_final, dtype=DTYPE, device=device)
    with torch.no_grad():
        final_nll = float(model.vecchia_batched_likelihood(params_tensor).detach().cpu().item())
        beta = model.get_gls_beta(params_tensor).detach()
    estimate = raw_to_physical(raw_final)
    gradients = [abs(float(param.grad.detach().item())) for param in params if param.grad is not None]
    row: dict[str, Any] = {
        "dataset_id": asset.dataset_id,
        "data_kind": asset.data_kind,
        "year": asset.year,
        "month": asset.month,
        "day": asset.day,
        "date": asset.date,
        "geometry": geometry,
        "status": "ok",
        "error": "",
        "smooth": float(args.smooth),
        "block_shape": "4x4",
        "lag_pattern": "6/4/3" if geometry != "union" else "6/(4U4U4)/(3U3U3)",
        "initializer": seed["method"],
        "init_advec_lat": float(seed["seed_lat"]),
        "init_advec_lon": float(seed["seed_lon"]),
        "final_native_nll": final_nll,
        "fit_returned_nll": float(returned[-1]),
        "outer_steps": int(step_index) + 1,
        "max_abs_gradient": max(gradients) if gradients else np.nan,
        "precompute_s": precompute_s,
        "fit_s": fit_s,
        "total_fit_s": precompute_s + fit_s,
        **{f"est_{name}": float(estimate[name]) for name in PARAMETERS},
        **model.cluster_summary(),
        **condition_summary,
    }
    if asset.truth is not None:
        row.update(
            parameter_error_columns(
                estimate, asset.truth, float(seed["lat_step"]), float(seed["lon_step"])
            )
        )
    eigen_curve = None
    eigen_summary = None
    if geometry in EIGEN_GEOMETRIES:
        t_diag = time.perf_counter()
        eigen_curve, eigen_summary = conditional_eigen_curve(model, params_tensor, beta, args)
        eigen_summary = {
            **eigen_summary,
            "dataset_id": asset.dataset_id,
            "data_kind": asset.data_kind,
            "date": asset.date,
            "geometry": geometry,
            "final_native_nll": final_nll,
            "diag_s": time.perf_counter() - t_diag,
        }
    del model, params, optimizer, params_tensor, beta
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, raw_final, eigen_curve, eigen_summary


def cross_evaluate(
    asset: DayAsset,
    seed: dict[str, Any],
    fitted_raw: dict[str, list[float]],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    source_vectors = dict(fitted_raw)
    if asset.truth is not None:
        source_vectors["truth"] = physical_to_raw(asset.truth)
    rows: list[dict[str, Any]] = []
    for eval_geometry in GEOMETRIES:
        model = build_geometry_model(eval_geometry, asset, seed, device, args)
        t0 = time.perf_counter()
        model.precompute_conditioning_sets()
        precompute_s = time.perf_counter() - t0
        for source_fit, raw in source_vectors.items():
            params = torch.as_tensor(raw, dtype=DTYPE, device=device)
            with torch.no_grad():
                nll = float(model.vecchia_batched_likelihood(params).detach().cpu().item())
            rows.append(
                {
                    "dataset_id": asset.dataset_id,
                    "data_kind": asset.data_kind,
                    "date": asset.date,
                    "evaluation_geometry": eval_geometry,
                    "source_fit": source_fit,
                    "nll_per_target": nll,
                    "evaluation_precompute_s": precompute_s,
                }
            )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def resolve_device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if bool(args.require_cuda) and device.type != "cuda":
        raise RuntimeError("--require-cuda was requested")
    return device


def run_task(args: argparse.Namespace) -> None:
    selection, selected = load_selection(args.selection_file)
    task_index = int(args.task_index)
    if task_index < 0 or task_index >= len(selected):
        raise IndexError(f"task-index {task_index} outside 0..{len(selected)-1}")
    spec = selected[task_index]
    task_dir = Path(args.output_root) / f"task_{task_index:02d}_{spec['dataset_id']}"
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(task_dir / "selection_manifest.json", selection)
    write_json(task_dir / "dataset_spec.json", spec)
    device = resolve_device(args)
    print(f"Loading {spec['dataset_id']} ({spec['data_kind']}, {spec['date']})", flush=True)
    asset = load_real_asset(spec, args) if spec["data_kind"] == "real" else load_synthetic_asset(spec, args)
    write_json(
        task_dir / "asset_manifest.json",
        {
            **spec,
            "source_path": asset.source_path,
            "keys": asset.keys,
            "n_valid": asset.n_valid,
            "n_total": asset.n_total,
            "valid_rate": asset.n_valid / asset.n_total,
            "center_value": asset.center_value,
            "truth": asset.truth,
        },
    )
    seed = m3_q3_seed(asset, args)
    pd.DataFrame([{**spec, **seed}]).to_csv(task_dir / "initializer.csv", index=False)
    init = dict(asset.truth) if asset.truth is not None else dict(DEFAULT_REAL_INIT)
    fit_rows: list[dict[str, Any]] = []
    fitted_raw: dict[str, list[float]] = {}
    curves: dict[str, pd.DataFrame] = {}
    eigen_summaries: dict[str, dict[str, Any]] = {}
    for geometry in GEOMETRIES:
        print(f"Fitting {geometry} on {asset.dataset_id}", flush=True)
        row, raw, curve, eigen_summary = fit_one_geometry(
            geometry, asset, seed, init, device, args
        )
        fit_rows.append(row)
        fitted_raw[geometry] = raw
        pd.DataFrame(fit_rows).to_csv(task_dir / "fits.csv", index=False)
        write_json(task_dir / "fitted_raw_parameters.json", fitted_raw)
        if curve is not None and eigen_summary is not None:
            sampled = resample_curve(curve, int(args.resample_grid)).assign(
                dataset_id=asset.dataset_id,
                data_kind=asset.data_kind,
                date=asset.date,
                geometry=geometry,
            )
            curves[geometry] = sampled
            eigen_summaries[geometry] = eigen_summary
            sampled.to_csv(task_dir / f"conditional_eigen_curve_{geometry}.csv", index=False)
            pd.DataFrame(eigen_summaries.values()).to_csv(task_dir / "eigen_summary.csv", index=False)
    cross_rows = cross_evaluate(asset, seed, fitted_raw, device, args)
    pd.DataFrame(cross_rows).to_csv(task_dir / "cross_likelihoods.csv", index=False)
    union_graph_nll = {
        str(row["source_fit"]): float(row["nll_per_target"])
        for row in cross_rows
        if row["evaluation_geometry"] == "union" and row["source_fit"] in GEOMETRIES
    }
    union_fit_nll = union_graph_nll["union"]
    for row in fit_rows:
        geometry = str(row["geometry"])
        row["union_graph_nll_at_fit"] = union_graph_nll[geometry]
        row["union_graph_nll_gap_from_union_fit"] = (
            union_graph_nll[geometry] - union_fit_nll
        )
    pd.DataFrame(fit_rows).to_csv(task_dir / "fits.csv", index=False)
    for geometry in EIGEN_GEOMETRIES:
        eigen_summaries[geometry]["union_graph_nll"] = union_graph_nll[geometry]
        eigen_summaries[geometry]["union_graph_nll_gap_from_union_fit"] = (
            union_graph_nll[geometry] - union_fit_nll
        )
    pd.DataFrame(eigen_summaries.values()).to_csv(task_dir / "eigen_summary.csv", index=False)
    plot_eigen_comparison(
        curves,
        {geometry: {**eigen_summaries[geometry], **next(row for row in fit_rows if row["geometry"] == geometry)} for geometry in EIGEN_GEOMETRIES},
        f"{asset.dataset_id} ({asset.date}): four lag643 geometries",
        task_dir / "conditional_eigen_four_geometries.png",
    )
    write_json(
        task_dir / "run_config.json",
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()),
            "args": vars(args),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "dataset": spec,
            "geometry_note": (
                "adapted uses calibrated signed 2-D corridors: t-1 [0.5v,1.5v], "
                "t-2 [0,2v]; shifted uses calibrated v/2v centers; fixed uses the "
                "target center; all past offsets follow the -v covariance convention; "
                "union is the exact deduplicated set union of all three"
            ),
        },
    )
    (task_dir / "COMPLETE").write_text(datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")
    print(f"Completed {asset.dataset_id}: {task_dir}", flush=True)


def read_task_tables(output_root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(output_root.glob(f"task_*/{filename}"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def aggregate_eigen_curves(
    output_root: Path,
    fits: pd.DataFrame,
    cross: pd.DataFrame,
) -> pd.DataFrame:
    paths = sorted(output_root.glob("task_*/conditional_eigen_curve_*.csv"))
    if not paths:
        return pd.DataFrame()
    curves = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    curves.to_csv(output_root / "all_conditional_eigen_curves.csv", index=False)
    summary = (
        curves.groupby(["data_kind", "geometry", "frac_index"], as_index=False)
        .agg(
            scaled_cumsum_mean=("scaled_cumsum", "mean"),
            scaled_cumsum_sd=("scaled_cumsum", "std"),
            conditional_eigenvalue_mean=("conditional_eigenvalue", "mean"),
            n_datasets=("dataset_id", "nunique"),
        )
    )
    summary.to_csv(output_root / "mean_conditional_eigen_curves.csv", index=False)
    nll_means: dict[tuple[str, str], float] = {}
    if not fits.empty and "final_native_nll" in fits:
        nll_means = (
            fits[fits["status"] == "ok"]
            .groupby(["data_kind", "geometry"])["final_native_nll"]
            .mean()
            .to_dict()
        )
    union_nll_means: dict[tuple[str, str], float] = {}
    if not cross.empty:
        union_nll_means = (
            cross[
                (cross["evaluation_geometry"] == "union")
                & (cross["source_fit"].isin(GEOMETRIES))
            ]
            .groupby(["data_kind", "source_fit"])["nll_per_target"]
            .mean()
            .to_dict()
        )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharex=True)
    for ax, data_kind in zip(axes, ("real", "synthetic")):
        for geometry in EIGEN_GEOMETRIES:
            group = summary[(summary["data_kind"] == data_kind) & (summary["geometry"] == geometry)]
            if group.empty:
                continue
            x = group["frac_index"].to_numpy()
            mean = group["scaled_cumsum_mean"].to_numpy()
            sd = group["scaled_cumsum_sd"].fillna(0.0).to_numpy()
            mean_nll = nll_means.get((data_kind, geometry), np.nan)
            mean_union_nll = union_nll_means.get((data_kind, geometry), np.nan)
            label = (
                f"{geometry} (mean NLL native/union="
                f"{mean_nll:.5f}/{mean_union_nll:.5f})"
                if np.isfinite(mean_nll) and np.isfinite(mean_union_nll)
                else geometry
            )
            ax.plot(
                x,
                mean,
                color=GEOMETRY_COLORS[geometry],
                lw=2.0,
                label=label,
            )
            ax.fill_between(
                x,
                mean - sd,
                mean + sd,
                color=GEOMETRY_COLORS[geometry],
                alpha=0.12,
            )
        grid = np.linspace(0.0, 1.0, 200)
        ax.plot(grid, grid, color="0.4", lw=1.0)
        ax.set_title(f"{data_kind}: mean of selected days")
        ax.set_xlabel("projected expected-df fraction")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("mean cumulative squared score / residual df")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        output_root / "conditional_eigen_mean_four_geometries.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    return curves


def plot_union_reference(cross: pd.DataFrame, output_root: Path) -> None:
    union = cross[cross["evaluation_geometry"] == "union"].copy()
    pivot = union.pivot_table(
        index=["dataset_id", "data_kind", "date"], columns="source_fit", values="nll_per_target"
    ).reset_index()
    for geometry in GEOMETRIES:
        if geometry in pivot and "union" in pivot:
            pivot[f"{geometry}_minus_union_fit_nll"] = pivot[geometry] - pivot["union"]
    pivot.to_csv(output_root / "union_reference_likelihood_gaps.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    for ax, data_kind in zip(axes, ("real", "synthetic")):
        group = pivot[pivot["data_kind"] == data_kind].sort_values("date")
        x = np.arange(len(group))
        for offset, geometry in zip((-0.18, 0.0, 0.18), BASE_GEOMETRIES):
            col = f"{geometry}_minus_union_fit_nll"
            ax.scatter(
                x + offset,
                group[col],
                color=GEOMETRY_COLORS[geometry],
                s=42,
                label=geometry,
            )
        ax.axhline(0.0, color="0.4", lw=1.0)
        ax.set_xticks(x, group["date"], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{data_kind}: union-graph NLL gap")
        ax.set_ylabel("NLL(source fit on union graph) - NLL(union fit)")
        ax.grid(axis="y", alpha=0.22)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_root / "union_reference_likelihood_gaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_synthetic_errors(fits: pd.DataFrame, output_root: Path) -> None:
    synthetic = fits[(fits["data_kind"] == "synthetic") & (fits["status"] == "ok")].copy()
    if synthetic.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax, metric, title in (
        (axes[0], "combined_parameter_error", "combined parameter error"),
        (axes[1], "advection_error_grid_cells", "advection error (grid cells)"),
    ):
        dates = sorted(synthetic["date"].unique())
        x = np.arange(len(dates))
        for offset, geometry in zip((-0.27, -0.09, 0.09, 0.27), GEOMETRIES):
            group = synthetic[synthetic["geometry"] == geometry].set_index("date").reindex(dates)
            ax.scatter(
                x + offset,
                group[metric],
                color=GEOMETRY_COLORS[geometry],
                s=38,
                label=geometry,
            )
        ax.set_xticks(x, dates, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_root / "synthetic_parameter_errors.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    selection, selected = load_selection(args.selection_file)
    completed = sorted(output_root.glob("task_*/COMPLETE"))
    if len(completed) != len(selected) and not args.allow_partial_aggregate:
        raise RuntimeError(f"Expected {len(selected)} COMPLETE markers, found {len(completed)}")
    write_json(output_root / "selection_manifest.json", selection)
    fits = read_task_tables(output_root, "fits.csv")
    initializers = read_task_tables(output_root, "initializer.csv")
    cross = read_task_tables(output_root, "cross_likelihoods.csv")
    eigen = read_task_tables(output_root, "eigen_summary.csv")
    for name, frame in (
        ("all_fits.csv", fits),
        ("all_initializers.csv", initializers),
        ("all_cross_likelihoods.csv", cross),
        ("all_eigen_summaries.csv", eigen),
    ):
        frame.to_csv(output_root / name, index=False)
    if not fits.empty:
        summary_metrics = [
            "final_native_nll",
            "fit_s",
            "total_fit_s",
            "advection_error_grid_cells",
            "positive_parameter_log_error",
            "combined_parameter_error",
        ]
        available = [metric for metric in summary_metrics if metric in fits.columns]
        grouped = fits.groupby(["data_kind", "geometry"])[available].agg(["mean", "median", "std", "count"])
        grouped.to_csv(output_root / "fit_summary_by_kind_geometry.csv")
        plot_synthetic_errors(fits, output_root)
    if not cross.empty:
        plot_union_reference(cross, output_root)
    aggregate_eigen_curves(output_root, fits, cross)
    dates = pd.DataFrame(selected)[["dataset_id", "data_kind", "date", "year", "month", "day"]]
    dates.to_csv(output_root / "selected_dates.csv", index=False)
    date_headers = list(dates.columns)
    date_table = [
        "| " + " | ".join(date_headers) + " |",
        "| " + " | ".join("---" for _ in date_headers) + " |",
    ]
    for values in dates.itertuples(index=False, name=None):
        date_table.append("| " + " | ".join(str(value) for value in values) + " |")
    report = [
        "# Four-geometry lag643 Vecchia comparison",
        "",
        f"Aggregated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Selected data sets",
        "",
        *date_table,
        "",
        "## Geometry",
        "",
        "- adapted: M3+Q3-calibrated signed 2-D corridors; t-1 [0.5v,1.5v], t-2 [0,2v] in the past -v direction.",
        "- shifted: identical 6/4/3 budget; nearest blocks around calibrated v/2v centers in the past -v direction.",
        "- fixed: identical 6/4/3 budget; past neighborhoods centered at the target.",
        "- union: exact deduplicated union of the three candidates, up to 6/12/9 blocks.",
        "- all four optimizers start from the identical M3+Q3 advection and identical nuisance values.",
        "",
        "Native NLLs from different graphs are recorded, but the primary likelihood comparison is",
        "the cross-evaluation on the common union graph in union_reference_likelihood_gaps.csv.",
    ]
    (output_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Aggregated results: {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run-task", "aggregate"], default="run-task")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=HERE / "vecchia_adapted_vs_fixed_selection_090126.json",
    )
    parser.add_argument("--real-data-root", type=Path, default=Path("/home/jl2815/tco/data"))
    parser.add_argument(
        "--synthetic-data-root",
        type=Path,
        default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_090126"),
    )
    parser.add_argument("--hours-per-day", type=int, default=8)
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5, choices=[0.5])
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=32)
    parser.add_argument("--diag-chunk-size", type=int, default=64)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)
    parser.add_argument("--brown-bridge-q", type=float, default=BROWN_BRIDGE_Q95)
    parser.add_argument("--resample-grid", type=int, default=500)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--allow-partial-aggregate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "aggregate":
        aggregate(args)
    else:
        try:
            run_task(args)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
