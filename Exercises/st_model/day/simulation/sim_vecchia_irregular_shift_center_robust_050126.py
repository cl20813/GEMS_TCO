"""
sim_vecchia_irregular_shift_center_robust_050126.py

Amarel simulation study for shifted-center lag conditioning on real-data-like
irregular GEMS observation patterns.

Main question:
  Does a lagged conditioning set need to reuse the current-time nearest-neighbor
  list, or should each lag choose fresh nearest neighbors around the predicted
  upstream/transported location?

The standard comparison point is the best local-only reduced model:

  Irr_Cand_A20_B18_C15
      t: 20 current neighbors
      t-1: same-location anchor + 18 local neighbors around current location
      t-2: same-location anchor + 15 local neighbors around current location

Shift-center candidates keep the same-location anchor at each lag, but choose
the remaining lagged neighbors by a fresh nearest-neighbor search around
longitude-shifted centers:

  t-1 center = current lon + predicted_lag1_offset
  t-2 center = current lon + 2 * predicted_lag1_offset

For negative longitudinal advection, the upstream location at previous times is
east of the target, so the predicted offsets are positive.  The script sweeps a
range of predicted offsets to test robustness when the advection magnitude is
unknown or misspecified.

The data pipeline follows the existing real-data-like high-resolution
simulation pipeline:
  high-resolution FFT field -> real GEMS source locations -> irregular
  N_grid-row maps with NaNs for unobserved rows.  Conditioning order is computed
  on the regular grid template, then applied to the irregular source-location
  rows, matching the real-data Vecchia pipeline.

Example:
  conda activate faiss_env
  python sim_vecchia_irregular_shift_center_robust_050126.py --num-iters 1
"""

import gc
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.fft
import typer
from sklearn.neighbors import BallTree

AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
_src = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, _src)

from GEMS_TCO import configuration as config
from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO.data_loader import load_data_dynamic_processed


is_amarel = os.path.exists(config.amarel_data_load_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
T_STEPS = 8

P_LABELS = [
    "sigmasq", "range_lat", "range_lon", "range_time",
    "advec_lat", "advec_lon", "nugget",
]
P_COLS = [
    "sigmasq_est", "range_lat_est", "range_lon_est", "range_t_est",
    "advec_lat_est", "advec_lon_est", "nugget_est",
]
SPATIAL_KEYS = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def build_high_res_grid(lat_range, lon_range, lat_factor=100, lon_factor=10):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lat_max, lat_min = max(lat_range), min(lat_range)
    lats = torch.arange(lat_min - 0.1, lat_max + 0.1, dlat, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon, device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    """Generate one FFT circulant-embedding field on the high-res grid."""
    cpu = torch.device("cpu")
    f32 = torch.float32
    nx, ny, nt = len(lats_hr), len(lons_hr), t_steps
    px, py, pt = 2 * nx, 2 * ny, 2 * nt

    lx = torch.arange(px, device=cpu, dtype=f32) * dlat
    lx[px // 2:] -= px * dlat
    ly = torch.arange(py, device=cpu, dtype=f32) * dlon
    ly[py // 2:] -= py * dlon
    lt = torch.arange(pt, device=cpu, dtype=f32)
    lt[pt // 2:] -= pt

    params_cpu = params.cpu().float()
    lx_g, ly_g, lt_g = torch.meshgrid(lx, ly, lt, indexing="ij")
    cov = get_covariance_on_grid(lx_g, ly_g, lt_g, params_cpu)
    spec = torch.fft.fftn(cov)
    spec.real = torch.clamp(spec.real, min=0)
    noise = torch.fft.fftn(torch.randn(px, py, pt, device=cpu, dtype=f32))
    field = torch.fft.ifftn(torch.sqrt(spec.real) * noise).real[:nx, :ny, :nt]
    return field.to(dtype=DTYPE, device=DEVICE)


def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    """Obs-to-cell 1:1 assignment, matching step3 behavior."""
    n_grid = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(n_grid, -1, dtype=np.int64)

    dist_to_cell, cell_for_obs = grid_tree.query(np.radians(src_np_valid), k=1)
    dist_to_cell = dist_to_cell.flatten()
    cell_for_obs = cell_for_obs.flatten()

    assignment = np.full(n_grid, -1, dtype=np.int64)
    best_dist = np.full(n_grid, np.inf)
    for obs_i, (cell_j, dist) in enumerate(zip(cell_for_obs, dist_to_cell)):
        if dist < best_dist[cell_j]:
            assignment[cell_j] = obs_i
            best_dist[cell_j] = dist

    filled = assignment >= 0
    if filled.any():
        win_obs = assignment[filled]
        lat_diff = np.abs(src_np_valid[win_obs, 0] - grid_coords_np[filled, 0])
        lon_diff = np.abs(src_np_valid[win_obs, 1] - grid_coords_np[filled, 1])
        too_far = (lat_diff > DELTA_LAT_BASE / 2) | (lon_diff > DELTA_LON_BASE / 2)
        assignment[np.where(filled)[0][too_far]] = -1

    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing="ij")
    hr_coords_np = torch.stack([hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric="haversine")

    grid_coords_np = grid_coords.cpu().numpy()
    n_grid = len(grid_coords_np)
    grid_tree = BallTree(np.radians(grid_coords_np), metric="haversine")

    step3_assignment_per_t = []
    hr_idx_per_t = []
    src_locs_per_t = []

    for key in sorted_keys:
        df = ref_day_map.get(key)
        if df is None or len(df) == 0:
            step3_assignment_per_t.append(np.full(n_grid, -1, dtype=np.int64))
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
            src_locs_per_t.append(torch.zeros((0, 2), dtype=DTYPE, device=DEVICE))
            continue

        src_np = df[["Source_Latitude", "Source_Longitude"]].values
        valid_mask = ~np.isnan(src_np).any(axis=1)
        src_np_valid = src_np[valid_mask]

        assignment = apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree)
        step3_assignment_per_t.append(assignment)

        if len(src_np_valid) > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np_valid), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))

        src_locs_per_t.append(torch.tensor(src_np_valid, device=DEVICE, dtype=DTYPE))

    return step3_assignment_per_t, hr_idx_per_t, src_locs_per_t


def compute_grid_ordering(grid_coords, mm_cond_number):
    coords_np = grid_coords.cpu().numpy()
    ord_mm = _orderings.maxmin_cpp(coords_np)
    nns = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


def assemble_irregular_map(
    field,
    step3_assignment_per_t,
    hr_idx_per_t,
    src_locs_per_t,
    sorted_keys,
    grid_coords,
    true_params,
    t_offset=21.0,
):
    """Build irregular N_grid-row maps with NaN rows for unobserved cells."""
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    n_grid = grid_coords.shape[0]
    field_flat = field.reshape(-1, T_STEPS)

    irr_map = {}
    for t_idx, key in enumerate(sorted_keys):
        t_val = float(t_offset + t_idx)
        assign = step3_assignment_per_t[t_idx]
        hr_idx = hr_idx_per_t[t_idx]
        src_locs = src_locs_per_t[t_idx]
        n_valid = hr_idx.shape[0]

        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0

        rows = torch.full((n_grid, 11), float("nan"), device=DEVICE, dtype=DTYPE)
        rows[:, 3] = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(n_grid, -1)

        if n_valid > 0:
            gp_vals = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(n_valid, device=DEVICE, dtype=DTYPE) * nugget_std
            assign_t = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            filled = assign_t >= 0
            win_obs = assign_t[filled]
            rows[filled, 0] = src_locs[win_obs, 0]
            rows[filled, 1] = src_locs[win_obs, 1]
            rows[filled, 2] = sim_vals[win_obs]

        irr_map[key] = rows.detach()

    return irr_map


def true_to_log_params(true_dict):
    phi2 = 1.0 / true_dict["range_lon"]
    phi1 = true_dict["sigmasq"] * phi2
    phi3 = (true_dict["range_lon"] / true_dict["range_lat"]) ** 2
    phi4 = (true_dict["range_lon"] / true_dict["range_time"]) ** 2
    return [
        np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
        true_dict["advec_lat"], true_dict["advec_lon"], np.log(true_dict["nugget"]),
    ]


def backmap_params(out_params):
    p = out_params
    if isinstance(p[0], torch.Tensor):
        p = [x.item() if x.numel() == 1 else x[0].item() for x in p[:7]]
    else:
        p = [float(x) for x in p[:7]]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {
        "sigmasq": np.exp(p[0]) / phi2,
        "range_lat": rlon / phi3 ** 0.5,
        "range_lon": rlon,
        "range_time": rlon / phi4 ** 0.5,
        "advec_lat": p[4],
        "advec_lon": p[5],
        "nugget": np.exp(p[6]),
    }


def rmsre_for_keys(est, true_dict, keys, zero_thresh=0.01):
    vals = []
    for key in keys:
        tv = true_dict[key]
        if abs(tv) < zero_thresh:
            continue
        vals.append(((est[key] - tv) / abs(tv)) ** 2)
    return float(np.sqrt(np.mean(vals))) if vals else float("nan")


def calculate_metrics(out_params, true_dict):
    est = backmap_params(out_params)
    metrics = {
        "overall_rmsre": rmsre_for_keys(est, true_dict, P_LABELS),
        "spatial_rmsre": rmsre_for_keys(est, true_dict, SPATIAL_KEYS),
        "range_time_re": abs(est["range_time"] - true_dict["range_time"]) / abs(true_dict["range_time"]),
        "advec_rmsre": rmsre_for_keys(est, true_dict, ADVECTION_KEYS),
        "nugget_re": abs(est["nugget"] - true_dict["nugget"]) / abs(true_dict["nugget"]),
    }
    return metrics, est


def make_random_init(rng, true_log, init_noise):
    noisy = list(true_log)
    for idx in [0, 1, 2, 3, 6]:
        noisy[idx] = true_log[idx] + rng.uniform(-init_noise, init_noise)
    for idx in [4, 5]:
        scale = max(abs(true_log[idx]), 0.05)
        noisy[idx] = true_log[idx] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


def total_conditioning(limit_a, limit_b, limit_c):
    return int(limit_a + (1 + limit_b) + (1 + limit_c))


class fit_vecchia_lbfgs_advec_mixed(kernels_vecchia.fit_vecchia_lbfgs):
    """Vecchia model that mixes local lag neighbors with fixed upstream points.

    `limit_B` and `limit_C` are local-neighbor counts.  Additional upstream
    points are supplied by `lag1_advec_cell_offsets` and
    `lag2_advec_cell_offsets`, measured in longitude grid cells.  If a shifted
    point falls outside the grid, is missing, or duplicates an existing point,
    the vacant slot is filled from the next local lag neighbor.
    """

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        nns_map: Dict[str, Any],
        mm_cond_number: int,
        nheads: int,
        limit_A: int = 20,
        limit_B: int = 14,
        limit_C: int = 8,
        daily_stride: int = 2,
        spatial_coords: Optional[np.ndarray] = None,
        lon_resolution: float = DELTA_LON_BASE,
        lag1_advec_cell_offsets: Sequence[int] = (2, 3),
        lag2_advec_cell_offsets: Sequence[int] = (4, 5),
    ):
        super().__init__(
            smooth,
            input_map,
            nns_map,
            mm_cond_number,
            nheads,
            limit_A=limit_A,
            limit_B=limit_B,
            limit_C=limit_C,
            daily_stride=daily_stride,
        )
        self.spatial_coords = spatial_coords
        self.lon_resolution = float(lon_resolution)
        self.lag1_advec_cell_offsets = tuple(int(v) for v in lag1_advec_cell_offsets)
        self.lag2_advec_cell_offsets = tuple(int(v) for v in lag2_advec_cell_offsets)

    def _spatial_coords_np(self, n_points: int) -> np.ndarray:
        if self.spatial_coords is not None:
            coords_np = np.asarray(self.spatial_coords[:n_points], dtype=np.float64)
        else:
            all_data = [
                torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                for d in self.input_map.values()
            ]
            coords_np = all_data[0][:n_points, :2].cpu().numpy().astype(np.float64)

        coords_np = coords_np.copy()
        nan_mask = np.isnan(coords_np).any(axis=1)
        coords_np[nan_mask] = np.array([0.0, 1000.0])
        return coords_np

    def _build_offset_lookup(self, n_points: int, offsets: Sequence[int]) -> np.ndarray:
        """Return nearest local indices for eastward longitude cell offsets."""
        from sklearn.neighbors import BallTree

        coords_np = self._spatial_coords_np(n_points)
        if not offsets:
            return np.empty((n_points, 0), dtype=np.int64)

        tree = BallTree(np.radians(coords_np), metric="haversine")
        lats = coords_np[:, 0]
        lons = coords_np[:, 1]
        lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
        lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
        base_ids = np.arange(n_points, dtype=np.int64)

        lookups = []
        for offset in offsets:
            target_lat = lats
            target_lon = lons + int(offset) * self.lon_resolution
            inside = (
                (target_lat >= lat_min)
                & (target_lat <= lat_max)
                & (target_lon >= lon_min)
                & (target_lon <= lon_max)
            )
            q = np.column_stack([np.radians(target_lat), np.radians(target_lon)])
            _, idx = tree.query(q, k=1)
            lookup = idx.flatten().astype(np.int64)
            lookup[~inside] = base_ids[~inside]
            lookups.append(lookup)
        return np.column_stack(lookups)

    @staticmethod
    def _valid_local_ids(values: Iterable[int], upper: int) -> List[int]:
        return [int(v) for v in values if int(v) < upper]

    def precompute_conditioning_sets(self):
        limit_A, limit_B, limit_C = self.limit_A, self.limit_B, self.limit_C
        daily_stride = self.daily_stride
        n_adv_B = len(self.lag1_advec_cell_offsets)
        n_adv_C = len(self.lag2_advec_cell_offsets)

        max_dim_A = limit_A
        max_dim_AB = limit_A + (limit_B + 1) + n_adv_B
        max_dim_ABC = max_dim_AB + (limit_C + 1) + n_adv_C

        n_stored = next((len(m) for m in self.nns_map if len(m) > 0), 0)
        print(
            "Pre-computing AdvecMixed Vecchia "
            f"[A={max_dim_A}, AB={max_dim_AB}, ABC={max_dim_ABC}, "
            f"B_local={limit_B}, C_local={limit_C}, "
            f"lag1_cells={self.lag1_advec_cell_offsets}, "
            f"lag2_cells={self.lag2_advec_cell_offsets}, stored={n_stored}]...",
            end=" ",
        )

        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        Real_Data = torch.cat(all_data_list, dim=0).to(self.device, dtype=torch.float32)
        n_real, num_cols = Real_Data.shape

        is_nan_real = torch.isnan(Real_Data[:, 2])
        valid_lats = Real_Data[~is_nan_real, 0]
        self.lat_mean_val = (
            valid_lats.mean().item()
            if valid_lats.numel() > 0
            else Real_Data[:, 0].mean().item()
        )
        print(f"[Mean Lat: {self.lat_mean_val:.4f}]", end=" ")
        is_nan_mask_np = is_nan_real.cpu().numpy()

        n_dummies = max_dim_ABC
        dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float32)
        for k in range(n_dummies):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        Full_Data = torch.cat([Real_Data, dummy_block], dim=0)
        dummy_start = n_real
        is_nan_mask_np = np.append(is_nan_mask_np, np.zeros(n_dummies, dtype=bool))

        key_list = list(self.input_map.keys())
        day_lengths = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        n_time_steps = len(key_list)
        use_set_C = daily_stride < n_time_steps

        n_pts_per_day = day_lengths[0]
        lag1_lookup = self._build_offset_lookup(n_pts_per_day, self.lag1_advec_cell_offsets)
        lag2_lookup = self._build_offset_lookup(n_pts_per_day, self.lag2_advec_cell_offsets)

        heads_indices = []
        batch_list_A = []
        batch_list_AB = []
        batch_list_ABC = []

        def add_valid_neighbors(indices_to_check, current_indices, cap):
            count = 0
            for idx in indices_to_check:
                if count >= cap:
                    break
                idx = int(idx)
                if idx not in current_indices and not is_nan_mask_np[idx]:
                    current_indices.append(idx)
                    count += 1

        for time_idx, key in enumerate(key_list):
            day_len = day_lengths[time_idx]
            offset = cumulative_len[time_idx]

            for local_idx in range(min(day_len, self.nheads)):
                idx = offset + local_idx
                if not is_nan_mask_np[idx]:
                    heads_indices.append(idx)
            if self.nheads >= day_len:
                continue

            for local_idx in range(self.nheads, day_len):
                target_idx = offset + local_idx
                if is_nan_mask_np[target_idx]:
                    continue

                current_indices = []

                add_valid_neighbors(
                    (offset + self.nns_map[local_idx]).tolist(),
                    current_indices,
                    cap=limit_A,
                )

                has_B = time_idx > 0
                has_C = time_idx >= daily_stride
                nbs = self.nns_map[local_idx]

                if has_B:
                    prev_off = cumulative_len[time_idx - 1]
                    prev_len = day_lengths[time_idx - 1]

                    if local_idx < prev_len:
                        add_valid_neighbors([prev_off + local_idx], current_indices, cap=1)

                    lag1_ids = []
                    if local_idx < lag1_lookup.shape[0]:
                        lag1_ids = self._valid_local_ids(lag1_lookup[local_idx], prev_len)
                    lag1_exclude = {v for v in lag1_ids if v != local_idx}
                    local_candidates = [
                        prev_off + int(v)
                        for v in nbs
                        if int(v) < prev_len and int(v) not in lag1_exclude
                    ]
                    add_valid_neighbors(local_candidates, current_indices, cap=limit_B)

                    for adv_id in lag1_ids:
                        before = len(current_indices)
                        if adv_id != local_idx:
                            add_valid_neighbors([prev_off + adv_id], current_indices, cap=1)
                        if len(current_indices) == before:
                            add_valid_neighbors(local_candidates, current_indices, cap=1)

                if has_C:
                    pd_idx = time_idx - daily_stride
                    pd_off = cumulative_len[pd_idx]
                    pd_len = day_lengths[pd_idx]

                    if local_idx < pd_len:
                        add_valid_neighbors([pd_off + local_idx], current_indices, cap=1)

                    lag2_ids = []
                    if local_idx < lag2_lookup.shape[0]:
                        lag2_ids = self._valid_local_ids(lag2_lookup[local_idx], pd_len)
                    lag2_exclude = {v for v in lag2_ids if v != local_idx}
                    local_candidates = [
                        pd_off + int(v)
                        for v in nbs
                        if int(v) < pd_len and int(v) not in lag2_exclude
                    ]
                    add_valid_neighbors(local_candidates, current_indices, cap=limit_C)

                    for adv_id in lag2_ids:
                        before = len(current_indices)
                        if adv_id != local_idx:
                            add_valid_neighbors([pd_off + adv_id], current_indices, cap=1)
                        if len(current_indices) == before:
                            add_valid_neighbors(local_candidates, current_indices, cap=1)

                if has_C:
                    max_d, target_list = max_dim_ABC, batch_list_ABC
                elif has_B:
                    max_d, target_list = max_dim_AB, batch_list_AB
                else:
                    max_d, target_list = max_dim_A, batch_list_A

                n_valid = len(current_indices)
                if n_valid < max_d:
                    row = [dummy_start + k for k in range(max_d - n_valid)] + current_indices
                else:
                    row = current_indices[-max_d:]
                target_list.append(row)

        heads_tensor = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
        self.Heads_data = (
            Full_Data[heads_tensor].contiguous().to(torch.float64)
            if len(heads_indices) > 0
            else torch.empty((0, num_cols), device=self.device, dtype=torch.float64)
        )

        def build_tensors(idx_list, max_d):
            if not idx_list:
                return None, None, None, None, None
            T = torch.tensor(idx_list, device=self.device, dtype=torch.long)
            G = Full_Data[T]
            X = G[..., [0, 1, 3]].contiguous().to(torch.float64)
            Y = G[..., 2].unsqueeze(-1).contiguous().to(torch.float64)
            ones = torch.ones_like(G[..., 0]).unsqueeze(-1)
            lat = (G[..., 0] - self.lat_mean_val).unsqueeze(-1)
            dums = G[..., 4:11]
            Locs = torch.cat([ones, lat, dums], dim=-1).contiguous().to(torch.float64)
            is_dummy = (T >= dummy_start).unsqueeze(-1)
            Locs = Locs.masked_fill(is_dummy, 0.0)
            Y = Y.masked_fill(is_dummy, 0.0)
            return X, Y, Locs, T, is_dummy

        self.X_A, self.Y_A, self.Locs_A, self._T_A, self._is_dummy_A = build_tensors(
            batch_list_A, max_dim_A
        )
        self.X_AB, self.Y_AB, self.Locs_AB, self._T_AB, self._is_dummy_AB = build_tensors(
            batch_list_AB, max_dim_AB
        )
        self.X_ABC, self.Y_ABC, self.Locs_ABC, self._T_ABC, self._is_dummy_ABC = build_tensors(
            batch_list_ABC, max_dim_ABC
        )

        self._heads_tensor_stored = heads_tensor if len(heads_indices) > 0 else None
        self._dummy_start_stored = dummy_start
        self._n_real_stored = n_real
        self._n_dummies_stored = n_dummies
        self.n_tails = len(batch_list_A) + len(batch_list_AB) + len(batch_list_ABC)

        print(
            f"[Set C: {use_set_C}] Done. "
            f"(Heads: {len(heads_indices)}, "
            f"Tails A/AB/ABC: {len(batch_list_A)}/{len(batch_list_AB)}/{len(batch_list_ABC)})"
        )
        self.is_precomputed = True
        return self


class fit_vecchia_lbfgs_shift_center(kernels_vecchia.fit_vecchia_lbfgs):
    """Vecchia model using fresh lagged NN searches around shifted centers.

    Each lag keeps the same-location temporal anchor, then uses a shifted
    upstream center to choose a new local-neighbor list:

      Set B: same loc at t-1 + shifted center + limit_B NN around shifted center
      Set C: same loc at t-daily_stride + shifted center + limit_C NN around shifted center

    If the shifted center leaves the observed longitude range, it falls back to
    the same local index.  That keeps boundary behavior conservative and avoids
    silently snapping to an artificial edge point.
    """

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        nns_map: Dict[str, Any],
        mm_cond_number: int,
        nheads: int,
        limit_A: int = 20,
        limit_B: int = 16,
        limit_C: int = 10,
        daily_stride: int = 2,
        spatial_coords: Optional[np.ndarray] = None,
        lag1_lon_offset: float = 0.16,
    ):
        super().__init__(
            smooth,
            input_map,
            nns_map,
            mm_cond_number,
            nheads,
            limit_A=limit_A,
            limit_B=limit_B,
            limit_C=limit_C,
            daily_stride=daily_stride,
        )
        self.spatial_coords = spatial_coords
        self.lag1_lon_offset = float(abs(lag1_lon_offset))

    def _spatial_coords_np(self, n_points: int) -> np.ndarray:
        if self.spatial_coords is not None:
            coords_np = np.asarray(self.spatial_coords[:n_points], dtype=np.float64)
        else:
            all_data = [
                torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                for d in self.input_map.values()
            ]
            coords_np = all_data[0][:n_points, :2].cpu().numpy().astype(np.float64)

        coords_np = coords_np.copy()
        nan_mask = np.isnan(coords_np).any(axis=1)
        coords_np[nan_mask] = np.array([0.0, 1000.0])
        return coords_np

    def _build_shift_lookup(self, n_points: int, multiplier: float) -> np.ndarray:
        coords_np = self._spatial_coords_np(n_points)
        tree = BallTree(np.radians(coords_np), metric="haversine")

        lats = coords_np[:, 0]
        lons = coords_np[:, 1]
        valid = ~np.isnan(coords_np).any(axis=1)
        lon_min = float(np.nanmin(lons[valid]))
        lon_max = float(np.nanmax(lons[valid]))
        base_ids = np.arange(n_points, dtype=np.int64)

        target_lons = lons + multiplier * self.lag1_lon_offset
        outside = (~valid) | (target_lons < lon_min) | (target_lons > lon_max)
        query = np.column_stack([np.radians(lats), np.radians(target_lons)])
        _, idx = tree.query(query, k=1)
        lookup = idx.flatten().astype(np.int64)
        lookup[outside] = base_ids[outside]
        return lookup

    def precompute_conditioning_sets(self):
        limit_A, limit_B, limit_C = self.limit_A, self.limit_B, self.limit_C
        daily_stride = self.daily_stride

        max_dim_A = limit_A
        max_dim_AB = limit_A + (limit_B + 1) + 1
        max_dim_ABC = max_dim_AB + (limit_C + 1) + 1

        n_stored = next((len(m) for m in self.nns_map if len(m) > 0), 0)
        print(
            "Pre-computing ShiftCenter Vecchia "
            f"[A={max_dim_A}, AB={max_dim_AB}, ABC={max_dim_ABC}, "
            f"lag1_offset={self.lag1_lon_offset:.4f}, "
            f"lag2_offset={2.0 * self.lag1_lon_offset:.4f}, stored={n_stored}]...",
            end=" ",
        )

        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        Real_Data = torch.cat(all_data_list, dim=0).to(self.device, dtype=torch.float32)
        n_real, num_cols = Real_Data.shape

        is_nan_real = torch.isnan(Real_Data[:, 2])
        valid_lats = Real_Data[~is_nan_real, 0]
        self.lat_mean_val = (
            valid_lats.mean().item()
            if valid_lats.numel() > 0
            else Real_Data[:, 0].mean().item()
        )
        print(f"[Mean Lat: {self.lat_mean_val:.4f}]", end=" ")
        is_nan_mask_np = is_nan_real.cpu().numpy()

        n_dummies = max_dim_ABC
        dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float32)
        for k in range(n_dummies):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        Full_Data = torch.cat([Real_Data, dummy_block], dim=0)
        dummy_start = n_real
        is_nan_mask_np = np.append(is_nan_mask_np, np.zeros(n_dummies, dtype=bool))

        key_list = list(self.input_map.keys())
        day_lengths = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        n_time_steps = len(key_list)
        use_set_C = daily_stride < n_time_steps

        n_pts_per_day = day_lengths[0]
        lag1_center = self._build_shift_lookup(n_pts_per_day, multiplier=1.0)
        lag2_center = self._build_shift_lookup(n_pts_per_day, multiplier=2.0)

        heads_indices = []
        batch_list_A = []
        batch_list_AB = []
        batch_list_ABC = []

        def add_valid_neighbors(indices_to_check, current_indices, cap):
            count = 0
            for idx in indices_to_check:
                if count >= cap:
                    break
                idx = int(idx)
                if idx not in current_indices and not is_nan_mask_np[idx]:
                    current_indices.append(idx)
                    count += 1

        for time_idx, key in enumerate(key_list):
            day_len = day_lengths[time_idx]
            offset = cumulative_len[time_idx]

            for local_idx in range(min(day_len, self.nheads)):
                idx = offset + local_idx
                if not is_nan_mask_np[idx]:
                    heads_indices.append(idx)
            if self.nheads >= day_len:
                continue

            for local_idx in range(self.nheads, day_len):
                target_idx = offset + local_idx
                if is_nan_mask_np[target_idx]:
                    continue

                current_indices = []
                add_valid_neighbors(
                    (offset + self.nns_map[local_idx]).tolist(),
                    current_indices,
                    cap=limit_A,
                )

                has_B = time_idx > 0
                has_C = time_idx >= daily_stride

                if has_B:
                    prev_off = cumulative_len[time_idx - 1]
                    prev_len = day_lengths[time_idx - 1]

                    if local_idx < prev_len:
                        add_valid_neighbors([prev_off + local_idx], current_indices, cap=1)

                    center_B = int(lag1_center[local_idx]) if local_idx < len(lag1_center) else local_idx
                    if center_B >= prev_len:
                        center_B = local_idx
                    nbs_B = (
                        self.nns_map[center_B]
                        if center_B < len(self.nns_map)
                        else np.array([], dtype=np.int64)
                    )
                    b_candidates = [
                        prev_off + int(v)
                        for v in nbs_B
                        if int(v) < prev_len and int(v) not in {local_idx, center_B}
                    ]
                    add_valid_neighbors(b_candidates, current_indices, cap=limit_B)

                    before_center = len(current_indices)
                    if center_B != local_idx:
                        add_valid_neighbors([prev_off + center_B], current_indices, cap=1)
                    if len(current_indices) == before_center:
                        add_valid_neighbors(b_candidates, current_indices, cap=1)

                if has_C:
                    pd_idx = time_idx - daily_stride
                    pd_off = cumulative_len[pd_idx]
                    pd_len = day_lengths[pd_idx]

                    if local_idx < pd_len:
                        add_valid_neighbors([pd_off + local_idx], current_indices, cap=1)

                    center_C = int(lag2_center[local_idx]) if local_idx < len(lag2_center) else local_idx
                    if center_C >= pd_len:
                        center_C = local_idx
                    nbs_C = (
                        self.nns_map[center_C]
                        if center_C < len(self.nns_map)
                        else np.array([], dtype=np.int64)
                    )
                    c_candidates = [
                        pd_off + int(v)
                        for v in nbs_C
                        if int(v) < pd_len and int(v) not in {local_idx, center_C}
                    ]
                    add_valid_neighbors(c_candidates, current_indices, cap=limit_C)

                    before_center = len(current_indices)
                    if center_C != local_idx:
                        add_valid_neighbors([pd_off + center_C], current_indices, cap=1)
                    if len(current_indices) == before_center:
                        add_valid_neighbors(c_candidates, current_indices, cap=1)

                if has_C:
                    max_d, target_list = max_dim_ABC, batch_list_ABC
                elif has_B:
                    max_d, target_list = max_dim_AB, batch_list_AB
                else:
                    max_d, target_list = max_dim_A, batch_list_A

                n_valid = len(current_indices)
                if n_valid < max_d:
                    row = [dummy_start + k for k in range(max_d - n_valid)] + current_indices
                else:
                    row = current_indices[-max_d:]
                target_list.append(row)

        heads_tensor = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
        self.Heads_data = (
            Full_Data[heads_tensor].contiguous().to(torch.float64)
            if len(heads_indices) > 0
            else torch.empty((0, num_cols), device=self.device, dtype=torch.float64)
        )

        def build_tensors(idx_list, max_d):
            if not idx_list:
                return None, None, None, None, None
            T = torch.tensor(idx_list, device=self.device, dtype=torch.long)
            G = Full_Data[T]
            X = G[..., [0, 1, 3]].contiguous().to(torch.float64)
            Y = G[..., 2].unsqueeze(-1).contiguous().to(torch.float64)
            ones = torch.ones_like(G[..., 0]).unsqueeze(-1)
            lat = (G[..., 0] - self.lat_mean_val).unsqueeze(-1)
            dums = G[..., 4:11]
            Locs = torch.cat([ones, lat, dums], dim=-1).contiguous().to(torch.float64)
            is_dummy = (T >= dummy_start).unsqueeze(-1)
            Locs = Locs.masked_fill(is_dummy, 0.0)
            Y = Y.masked_fill(is_dummy, 0.0)
            return X, Y, Locs, T, is_dummy

        self.X_A, self.Y_A, self.Locs_A, self._T_A, self._is_dummy_A = build_tensors(
            batch_list_A, max_dim_A
        )
        self.X_AB, self.Y_AB, self.Locs_AB, self._T_AB, self._is_dummy_AB = build_tensors(
            batch_list_AB, max_dim_AB
        )
        self.X_ABC, self.Y_ABC, self.Locs_ABC, self._T_ABC, self._is_dummy_ABC = build_tensors(
            batch_list_ABC, max_dim_ABC
        )

        self._heads_tensor_stored = heads_tensor if len(heads_indices) > 0 else None
        self._dummy_start_stored = dummy_start
        self._n_real_stored = n_real
        self._n_dummies_stored = n_dummies
        self.n_tails = len(batch_list_A) + len(batch_list_AB) + len(batch_list_ABC)

        print(
            f"[Set C: {use_set_C}] Done. "
            f"(Heads: {len(heads_indices)}, "
            f"Tails A/AB/ABC: {len(batch_list_A)}/{len(batch_list_AB)}/{len(batch_list_ABC)})"
        )
        self.is_precomputed = True
        return self


def parse_shift_offsets(offsets: str) -> List[float]:
    return [float(x.strip()) for x in offsets.split(",") if x.strip()]


def parse_shift_budgets(budgets: str) -> List[Tuple[int, int]]:
    out = []
    for item in budgets.split(","):
        item = item.strip()
        if not item:
            continue
        b_raw, c_raw = item.split(":")
        out.append((int(b_raw), int(c_raw)))
    return out


def make_shift_center_specs(
    base_limit_a: int,
    base_limit_b: int,
    base_limit_c: int,
    ratio_limit_a: int,
    ratio_limit_b: int,
    ratio_limit_c: int,
    shift_offsets: Sequence[float],
    shift_budgets: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "model": f"Irr_Full_A{base_limit_a:02d}_B{base_limit_b:02d}_C{base_limit_c:02d}",
            "allocation": "baseline full lag budget",
            "kernel": "std",
            "limit_A": base_limit_a,
            "limit_B": base_limit_b,
            "limit_C": base_limit_c,
        },
        {
            "model": "Irr_Cand_A20_B18_C15",
            "allocation": "best standard reduced baseline: t-1 0.9x, t-2 0.75x",
            "kernel": "std",
            "limit_A": 20,
            "limit_B": 18,
            "limit_C": 15,
        },
        {
            "model": f"Irr_Ratio_A{ratio_limit_a:02d}_B{ratio_limit_b:02d}_C{ratio_limit_c:02d}",
            "allocation": "aggressive standard local-only control",
            "kernel": "std",
            "limit_A": ratio_limit_a,
            "limit_B": ratio_limit_b,
            "limit_C": ratio_limit_c,
        },
    ]

    for b, c in shift_budgets:
        for offset in shift_offsets:
            off_label = f"{offset:.3f}".replace(".", "p")
            specs.append({
                "model": f"Irr_ShiftNN_A20_B{b:02d}_C{c:02d}_off{off_label}",
                "allocation": (
                    f"fresh lag NN around predicted upstream centers; "
                    f"lag1={offset:.3f}, lag2={2.0 * offset:.3f}"
                ),
                "kernel": "shift_center",
                "limit_A": 20,
                "limit_B": int(b),
                "limit_C": int(c),
                "shift_lag1_lon_offset": float(offset),
                "shift_lag2_lon_offset": float(2.0 * offset),
            })
    return specs


def fit_irregular_model(
    spec: Dict,
    irr_map_ord: Dict[str, torch.Tensor],
    nns_grid,
    ordered_grid_coords_np,
    initial_vals,
    smooth,
    mm_cond_number,
    nheads,
    daily_stride,
    lbfgs_lr,
    lbfgs_eval,
    lbfgs_hist,
    lbfgs_steps,
):
    params = [
        torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
        for val in initial_vals
    ]
    kernel = spec.get("kernel", "std")
    if kernel == "std":
        model = kernels_vecchia.fit_vecchia_lbfgs(
            smooth=smooth,
            input_map=irr_map_ord,
            nns_map=nns_grid,
            mm_cond_number=mm_cond_number,
            nheads=nheads,
            limit_A=spec["limit_A"],
            limit_B=spec["limit_B"],
            limit_C=spec["limit_C"],
            daily_stride=daily_stride,
        )
    elif kernel == "advec_mixed":
        model = fit_vecchia_lbfgs_advec_mixed(
            smooth=smooth,
            input_map=irr_map_ord,
            nns_map=nns_grid,
            mm_cond_number=mm_cond_number,
            nheads=nheads,
            limit_A=spec["limit_A"],
            limit_B=spec["limit_B"],
            limit_C=spec["limit_C"],
            daily_stride=daily_stride,
            spatial_coords=ordered_grid_coords_np,
            lon_resolution=spec.get("lon_resolution", DELTA_LON_BASE),
            lag1_advec_cell_offsets=spec.get("lag1_advec_cell_offsets", (2, 3)),
            lag2_advec_cell_offsets=spec.get("lag2_advec_cell_offsets", (4, 5)),
        )
    elif kernel == "shift_center":
        model = fit_vecchia_lbfgs_shift_center(
            smooth=smooth,
            input_map=irr_map_ord,
            nns_map=nns_grid,
            mm_cond_number=mm_cond_number,
            nheads=nheads,
            limit_A=spec["limit_A"],
            limit_B=spec["limit_B"],
            limit_C=spec["limit_C"],
            daily_stride=daily_stride,
            spatial_coords=ordered_grid_coords_np,
            lag1_lon_offset=spec.get("shift_lag1_lon_offset", 0.16),
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    model.precompute_conditioning_sets()
    optimizer = model.set_optimizer(
        params, lr=lbfgs_lr, max_iter=lbfgs_eval, history_size=lbfgs_hist
    )
    t0 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, optimizer, max_steps=lbfgs_steps, grad_tol=1e-5)
    elapsed = time.time() - t0
    loss = float(out[-1])
    return out, loss, int(fit_iter) + 1, elapsed


def build_model_summary(df):
    metric_cols = [
        "loss", "overall_rmsre", "spatial_rmsre", "range_time_re",
        "advec_rmsre", "nugget_re", "total_s", "fit_iter",
    ]
    rows = []
    ok = df[df["error"].fillna("") == ""].copy()
    group_cols = [
        "model", "allocation", "kernel", "limit_A", "limit_B", "limit_C",
        "lag1_advec_offsets", "lag2_advec_offsets", "shift_lag1_lon_offset",
        "shift_lag2_lon_offset", "total_conditioning",
    ]
    for keys, sub in ok.groupby(group_cols):
        row = {
            "model": keys[0],
            "allocation": keys[1],
            "kernel": keys[2],
            "limit_A": keys[3],
            "limit_B": keys[4],
            "limit_C": keys[5],
            "lag1_advec_offsets": keys[6],
            "lag2_advec_offsets": keys[7],
            "shift_lag1_lon_offset": keys[8],
            "shift_lag2_lon_offset": keys[9],
            "total_conditioning": keys[10],
            "n": len(sub),
        }
        for col in metric_cols:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_median"] = float("nan")
                row[f"{col}_p10"] = float("nan")
                row[f"{col}_p90"] = float("nan")
                row[f"{col}_p90_p10"] = float("nan")
                continue
            p10, p90 = np.percentile(vals, [10, 90])
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_p10"] = float(p10)
            row[f"{col}_p90"] = float(p90)
            row[f"{col}_p90_p10"] = float(p90 - p10)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("overall_rmsre_mean")


def build_param_summary(df, true_dict):
    rows = []
    ok = df[df["error"].fillna("") == ""].copy()
    for model, sub in ok.groupby("model"):
        for par, col in zip(P_LABELS, P_COLS):
            tv = true_dict[par]
            denom = abs(tv) if abs(tv) >= 0.01 else 1.0
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            re = np.abs((vals - tv) / denom)
            p10, p90 = np.percentile(re, [10, 90])
            rows.append({
                "model": model,
                "parameter": par,
                "true": tv,
                "rmsre": float(np.sqrt(np.mean(re ** 2))),
                "mean_re": float(np.mean(re)),
                "median_re": float(np.median(re)),
                "p10_re": float(p10),
                "p90_re": float(p90),
                "p90_p10_re": float(p90 - p10),
                "estimate_mean": float(sub[col].mean()),
                "estimate_sd": float(sub[col].std()),
            })
    return pd.DataFrame(rows)


@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100, help="Maxmin neighbor list size"),
    nheads: int = typer.Option(0, help="Vecchia head points per time step"),
    base_limit_a: int = typer.Option(20, help="Full model Set A neighbors"),
    base_limit_b: int = typer.Option(20, help="Full model Set B local neighbors"),
    base_limit_c: int = typer.Option(20, help="Full model Set C local neighbors"),
    ratio_limit_a: int = typer.Option(20, help="Reduced model Set A neighbors"),
    ratio_limit_b: int = typer.Option(16, help="Reduced model Set B local neighbors"),
    ratio_limit_c: int = typer.Option(10, help="Reduced model Set C local neighbors"),
    daily_stride: int = typer.Option(2, help="Set C stride; 2 means t-2"),
    num_iters: int = typer.Option(300, help="Simulation iterations"),
    years: str = typer.Option("2022,2023,2024,2025", help="Years for observation patterns"),
    month: int = typer.Option(7, help="Reference month"),
    lat_range: str = typer.Option("-3,2", help="lat_min,lat_max"),
    lon_range: str = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor: int = typer.Option(100, help="High-res lat multiplier"),
    lon_factor: int = typer.Option(10, help="High-res lon multiplier"),
    true_advec_lat: float = typer.Option(0.25, help="True latitudinal advection used in the simulator"),
    true_advec_lon: float = typer.Option(-0.16, help="True longitudinal advection used in the simulator"),
    shift_offsets: str = typer.Option(
        "0.10,0.126,0.16,0.20,0.25",
        help="Comma-separated predicted lag-1 upstream longitude offsets for shift-center models",
    ),
    shift_budgets: str = typer.Option(
        "16:10,14:8,18:15",
        help="Comma-separated B:C lag budgets for shift-center models; A is fixed at 20",
    ),
    init_noise: float = typer.Option(0.7, help="Uniform log-space init noise"),
    lbfgs_steps: int = typer.Option(5, help="Outer LBFGS steps"),
    lbfgs_eval: int = typer.Option(20, help="LBFGS max_iter per outer step"),
    lbfgs_hist: int = typer.Option(10, help="LBFGS history size"),
    seed: int = typer.Option(42, help="Random seed"),
    out_prefix: str = typer.Option("sim_vecchia_irregular_shift_center_robust", help="Output prefix"),
) -> None:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    lat_r = [float(x) for x in lat_range.split(",")]
    lon_r = [float(x) for x in lon_range.split(",")]
    years_list = [y.strip() for y in years.split(",")]

    print(f"Device : {DEVICE}")
    print(f"Region : lat {lat_r}, lon {lon_r}")
    print(f"Years  : {years_list} month={month}")
    print(f"High-res : lat x{lat_factor}, lon x{lon_factor}")
    print(f"Iterations : {num_iters}")
    print(f"True advec_lat : {true_advec_lat}")
    print(f"True advec_lon : {true_advec_lon}")
    print(f"Shift offsets  : {shift_offsets}")
    print(f"Shift budgets  : {shift_budgets}")

    output_path = Path(config.amarel_estimates_day_path if is_amarel else config.mac_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%m%d%y")
    raw_csv = output_path / f"{out_prefix}_{date_tag}_raw.csv"
    model_summary_csv = output_path / f"{out_prefix}_{date_tag}_model_summary.csv"
    param_summary_csv = output_path / f"{out_prefix}_{date_tag}_param_summary.csv"

    # Same active real-data-like simulation scenario used in prior Vecchia tests.
    true_dict = {
        "sigmasq": 10.0,
        "range_lat": 0.5,
        "range_lon": 0.6,
        "range_time": 2.5,
        "advec_lat": true_advec_lat,
        "advec_lon": true_advec_lon,
        "nugget": 1.2,
    }
    true_log = true_to_log_params(true_dict)
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    shift_offset_values = parse_shift_offsets(shift_offsets)
    shift_budget_values = parse_shift_budgets(shift_budgets)
    model_specs = make_shift_center_specs(
        base_limit_a=base_limit_a,
        base_limit_b=base_limit_b,
        base_limit_c=base_limit_c,
        ratio_limit_a=ratio_limit_a,
        ratio_limit_b=ratio_limit_b,
        ratio_limit_c=ratio_limit_c,
        shift_offsets=shift_offset_values,
        shift_budgets=shift_budget_values,
    )

    for spec in model_specs:
        n_adv_b = len(spec.get("lag1_advec_cell_offsets", ()))
        n_adv_c = len(spec.get("lag2_advec_cell_offsets", ()))
        spec["lag1_advec_offsets"] = ",".join(map(str, spec.get("lag1_advec_cell_offsets", ())))
        spec["lag2_advec_offsets"] = ",".join(map(str, spec.get("lag2_advec_cell_offsets", ())))
        spec["shift_lag1_lon_offset"] = float(spec.get("shift_lag1_lon_offset", 0.0))
        spec["shift_lag2_lon_offset"] = float(spec.get("shift_lag2_lon_offset", 0.0))
        if "total_conditioning" not in spec:
            extra_shift_slots = 2 if spec.get("kernel") == "shift_center" else 0
            spec["total_conditioning"] = (
                total_conditioning(spec["limit_A"], spec["limit_B"], spec["limit_C"])
                + n_adv_b
                + n_adv_c
                + extra_shift_slots
            )
        spec["lag1_ratio_actual"] = spec["limit_B"] / max(spec["limit_A"], 1)
        spec["lag2_ratio_actual"] = spec["limit_C"] / max(spec["limit_A"], 1)

    print("\nModel specs")
    for spec in model_specs:
        print(
            f"  {spec['model']}: kernel={spec['kernel']} "
            f"A={spec['limit_A']} B={spec['limit_B']} C={spec['limit_C']} "
            f"advB=({spec['lag1_advec_offsets']}) advC=({spec['lag2_advec_offsets']}) "
            f"shift=({spec['shift_lag1_lon_offset']:.3f},{spec['shift_lag2_lon_offset']:.3f}) "
            f"total={spec['total_conditioning']} "
            f"(B/A={spec['lag1_ratio_actual']:.2f}, C/A={spec['lag2_ratio_actual']:.2f})"
        )

    print("\n[Setup 1/5] Loading GEMS observation patterns...")
    data_path = config.amarel_data_load_path if is_amarel else config.mac_data_load_path
    data_loader = load_data_dynamic_processed(data_path)
    year_dfmaps = {}
    year_tco_maps = {}
    for yr in years_list:
        df_map_yr, _, _, _ = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1],
            mm_cond_number=mm_cond_number,
            years_=[yr],
            months_=[month],
            lat_range=lat_r,
            lon_range=lon_r,
            is_whittle=False,
        )
        year_dfmaps[yr] = df_map_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots loaded")

        yr2 = str(yr)[2:]
        tco_path = Path(data_path) / f"pickle_{yr}" / f"tco_grid_{yr2}_{month:02d}.pkl"
        if tco_path.exists():
            with open(tco_path, "rb") as f:
                year_tco_maps[yr] = pickle.load(f)
            print(f"    tco_grid: {len(year_tco_maps[yr])} slots loaded")
        else:
            year_tco_maps[yr] = {}
            print(f"    [WARN] Missing tco_grid: {tco_path}")

    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT_BASE, device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0], lon_r[1] + 0.0001, DELTA_LON_BASE, device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing="ij")
    grid_coords = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    n_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat x {len(lons_grid)} lon = {n_grid} cells")

    print("[Setup 3/5] Building high-res grid and precomputing mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat x {len(lons_hr)} lon = {len(lats_hr) * len(lons_hr):,} cells")

    dummy_keys = [f"t{i}" for i in range(T_STEPS)]
    all_day_mappings = []
    for yr in years_list:
        df_map_yr = year_dfmaps[yr]
        all_sorted = sorted(df_map_yr.keys())
        n_days_yr = len(all_sorted) // T_STEPS
        print(f"  {yr}: precomputing {n_days_yr} day-patterns...", flush=True)
        for d_idx in range(n_days_yr):
            day_keys = all_sorted[d_idx * T_STEPS : (d_idx + 1) * T_STEPS]
            if len(day_keys) < T_STEPS:
                continue
            ref_day = {
                k: year_tco_maps[yr].get(k.split("_", 2)[-1], pd.DataFrame())
                for k in day_keys
            }
            s3, hr_idx, src = precompute_mapping_indices(ref_day, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))

    if not all_day_mappings:
        raise RuntimeError("No usable day-patterns were found.")
    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    print("[Setup 4/5] Computing shared grid-based maxmin ordering...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, mm_cond_number)
    ordered_grid_coords_np = grid_coords[ord_grid].detach().cpu().numpy()
    print(f"  Ordering complete: N_grid={n_grid}, mm_cond_number={mm_cond_number}")

    print("[Setup 5/5] Verifying one assembled irregular map...")
    yr0, d0, s3_0, hr0, src0 = all_day_mappings[0]
    field0 = generate_field_values(lats_hr, lons_hr, T_STEPS, true_params, dlat_hr, dlon_hr)
    irr0 = assemble_irregular_map(field0, s3_0, hr0, src0, dummy_keys, grid_coords, true_params)
    del field0
    first = list(irr0.values())[0]
    n_valid = int((~torch.isnan(first[:, 2])).sum().item())
    print(f"  Sample pattern {yr0} day {d0}: {n_valid}/{n_grid} valid rows at t0")
    del irr0
    torch.cuda.empty_cache()
    gc.collect()

    lbfgs_lr = 1.0
    records = []

    for it in range(num_iters):
        print(f"\n{'=' * 72}")
        print(f"Iteration {it + 1}/{num_iters}")
        print(f"{'=' * 72}")

        yr_it, d_it, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t = all_day_mappings[
            rng.integers(len(all_day_mappings))
        ]
        initial_vals = make_random_init(rng, true_log, init_noise)
        init_orig = backmap_params(initial_vals)
        print(f"Obs pattern: {yr_it} day {d_it}")
        print(
            f"Init: sigmasq={init_orig['sigmasq']:.3f}, "
            f"range_lon={init_orig['range_lon']:.3f}, nugget={init_orig['nugget']:.3f}"
        )

        try:
            field = generate_field_values(lats_hr, lons_hr, T_STEPS, true_params, dlat_hr, dlon_hr)
            irr_map = assemble_irregular_map(
                field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                dummy_keys, grid_coords, true_params,
            )
            del field
            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}
            del irr_map
        except Exception as exc:
            print(f"[SKIP] Iteration assembly failed: {type(exc).__name__}: {exc}")
            continue

        for spec in model_specs:
            print(f"\n--- {spec['model']} ---")
            pre_t = time.time()
            error_msg = ""
            try:
                out, loss, fit_iter, fit_s = fit_irregular_model(
                    spec, irr_map_ord, nns_grid, ordered_grid_coords_np, initial_vals, v, mm_cond_number,
                    nheads, daily_stride, lbfgs_lr, lbfgs_eval, lbfgs_hist, lbfgs_steps,
                )
                total_s = time.time() - pre_t
                metrics, est = calculate_metrics(out, true_dict)
                print(
                    f"loss={loss:.4f} overall={metrics['overall_rmsre']:.4f} "
                    f"spatial={metrics['spatial_rmsre']:.4f} time={total_s:.1f}s"
                )
            except Exception as exc:
                total_s = time.time() - pre_t
                loss = fit_iter = fit_s = float("nan")
                metrics = {
                    "overall_rmsre": float("nan"),
                    "spatial_rmsre": float("nan"),
                    "range_time_re": float("nan"),
                    "advec_rmsre": float("nan"),
                    "nugget_re": float("nan"),
                }
                est = {k: float("nan") for k in P_LABELS}
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"FAILED: {error_msg}")

            records.append({
                "iter": it + 1,
                "obs_year": yr_it,
                "obs_day": d_it,
                "model": spec["model"],
                "allocation": spec["allocation"],
                "kernel": spec["kernel"],
                "limit_A": spec["limit_A"],
                "limit_B": spec["limit_B"],
                "limit_C": spec["limit_C"],
                "lag1_advec_offsets": spec["lag1_advec_offsets"],
                "lag2_advec_offsets": spec["lag2_advec_offsets"],
                "shift_lag1_lon_offset": spec["shift_lag1_lon_offset"],
                "shift_lag2_lon_offset": spec["shift_lag2_lon_offset"],
                "true_advec_lat": true_dict["advec_lat"],
                "true_advec_lon": true_dict["advec_lon"],
                "lag1_ratio_actual": spec["lag1_ratio_actual"],
                "lag2_ratio_actual": spec["lag2_ratio_actual"],
                "total_conditioning": spec["total_conditioning"],
                "loss": round(loss, 6) if np.isfinite(loss) else np.nan,
                "overall_rmsre": round(metrics["overall_rmsre"], 6),
                "spatial_rmsre": round(metrics["spatial_rmsre"], 6),
                "range_time_re": round(metrics["range_time_re"], 6),
                "advec_rmsre": round(metrics["advec_rmsre"], 6),
                "nugget_re": round(metrics["nugget_re"], 6),
                "fit_iter": fit_iter,
                "fit_s": round(fit_s, 3) if np.isfinite(fit_s) else np.nan,
                "total_s": round(total_s, 3),
                "sigmasq_est": round(est["sigmasq"], 6),
                "range_lat_est": round(est["range_lat"], 6),
                "range_lon_est": round(est["range_lon"], 6),
                "range_t_est": round(est["range_time"], 6),
                "advec_lat_est": round(est["advec_lat"], 6),
                "advec_lon_est": round(est["advec_lon"], 6),
                "nugget_est": round(est["nugget"], 6),
                "init_sigmasq": round(init_orig["sigmasq"], 6),
                "init_range_lon": round(init_orig["range_lon"], 6),
                "init_nugget": round(init_orig["nugget"], 6),
                "error": error_msg,
            })

            torch.cuda.empty_cache()
            gc.collect()

        df_now = pd.DataFrame(records)
        df_now.to_csv(raw_csv, index=False)
        ok_now = df_now[df_now["error"].fillna("") == ""]
        if not ok_now.empty:
            model_summary = build_model_summary(df_now)
            param_summary = build_param_summary(df_now, true_dict)
            model_summary.to_csv(model_summary_csv, index=False)
            param_summary.to_csv(param_summary_csv, index=False)
            if not model_summary.empty:
                print("\nRunning model summary")
                cols = [
                    "model", "kernel", "shift_lag1_lon_offset", "n", "loss_mean", "overall_rmsre_mean",
                    "overall_rmsre_p90_p10", "spatial_rmsre_mean",
                    "spatial_rmsre_p90_p10", "advec_rmsre_mean",
                    "advec_rmsre_p90_p10", "total_s_mean", "total_s_p90_p10",
                ]
                print(model_summary[cols].to_string(index=False))

            if not param_summary.empty:
                print("\nRunning parameter summary")
                param_cols_show = [
                    "model", "parameter", "rmsre", "mean_re", "median_re",
                    "p10_re", "p90_re", "p90_p10_re",
                ]
                print(param_summary[param_cols_show].to_string(index=False))

        del irr_map_ord
        torch.cuda.empty_cache()
        gc.collect()

    df_final = pd.DataFrame(records)
    df_final.to_csv(raw_csv, index=False)
    model_summary = build_model_summary(df_final)
    param_summary = build_param_summary(df_final, true_dict)
    model_summary.to_csv(model_summary_csv, index=False)
    param_summary.to_csv(param_summary_csv, index=False)

    print(f"\nSaved raw results: {raw_csv}")
    print(f"Saved model summary: {model_summary_csv}")
    print(f"Saved parameter summary: {param_summary_csv}")
    if not model_summary.empty:
        print("\nFinal model summary")
        print(model_summary.to_string(index=False))


if __name__ == "__main__":
    app()
