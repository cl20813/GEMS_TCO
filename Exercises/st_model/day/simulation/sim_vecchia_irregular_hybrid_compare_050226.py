"""
sim_vecchia_irregular_hybrid_compare_050226.py

Amarel simulation study comparing local-only baseline, hybrid fresh-mix,
and pure fresh-shift conditioning strategies on real-data-like irregular
GEMS observation patterns.

Four model families:
  1. Irr_Cand_A20_B18_C15                   (std,          total=55) — baseline
  2. Hybrid_NearLocal_L16F02_C12F02_Op*      (hybrid_fresh, total=54) — local16+fresh2 / local12+fresh2
  3. Hybrid_Lean_L08F04_C4F03_Op*            (hybrid_fresh, total=41) — local8+fresh4 / local4+fresh3
  4. FreshShift_A20_B18_C12_Op*              (hybrid_fresh, total=52) — all-fresh shifted NN (lag1_local=0)

Each hybrid/shift model is tested at two predicted lag-1 offsets (e.g. 0.063 and 0.126).
Godambe information matrix (SE and relative SE) is computed every iteration.

SLURM array varies true_advec_lon: -0.10, -0.16, -0.25.

Example:
  conda activate faiss_env
  python sim_vecchia_irregular_hybrid_compare_050226.py --num-iters 1 --true-advec-lon -0.16
"""

import gc
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

P_LABELS = ["sigmasq", "range_lat", "range_lon", "range_time", "advec_lat", "advec_lon", "nugget"]
P_COLS = ["sigmasq_est", "range_lat_est", "range_lon_est", "range_t_est",
          "advec_lat_est", "advec_lon_est", "nugget_est"]
SPATIAL_KEYS   = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]

# Godambe constants
HESSIAN_EPS    = 1e-4
SCORE_EPS      = 1e-5
H_RIDGE_SCALE  = 1e-6
GODAMBE_J_METHOD        = "block"
GODAMBE_BLOCK_LAT_WIDTH  = 0.50
GODAMBE_BLOCK_LON_WIDTH  = 0.50
GODAMBE_BLOCK_TIME_WIDTH = 2.0

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def assemble_irregular_map(field, step3_assignment_per_t, hr_idx_per_t,
                            src_locs_per_t, sorted_keys, grid_coords, true_params, t_offset=21.0):
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
    return [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
            true_dict["advec_lat"], true_dict["advec_lon"], np.log(true_dict["nugget"])]


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
        "sigmasq":    np.exp(p[0]) / phi2,
        "range_lat":  rlon / phi3 ** 0.5,
        "range_lon":  rlon,
        "range_time": rlon / phi4 ** 0.5,
        "advec_lat":  p[4],
        "advec_lon":  p[5],
        "nugget":     np.exp(p[6]),
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
        "advec_rmsre":   rmsre_for_keys(est, true_dict, ADVECTION_KEYS),
        "nugget_re":     abs(est["nugget"] - true_dict["nugget"]) / abs(true_dict["nugget"]),
    }
    for par in P_LABELS:
        tv = true_dict[par]
        denom = abs(tv) if abs(tv) >= 0.01 else 1.0
        metrics[f"{par}_re"] = abs(est[par] - tv) / denom
    return metrics, est


def make_random_init(rng, true_log, init_noise):
    noisy = list(true_log)
    for idx in [0, 1, 2, 3, 6]:
        noisy[idx] = true_log[idx] + rng.uniform(-init_noise, init_noise)
    for idx in [4, 5]:
        scale = max(abs(true_log[idx]), 0.05)
        noisy[idx] = true_log[idx] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


def transform_log_phi_to_physical(p):
    phi1, phi2, phi3, phi4 = (torch.exp(p[i]) for i in range(4))
    rlon = 1.0 / phi2
    return torch.stack([
        phi1 / phi2,
        rlon / torch.sqrt(phi3),
        rlon,
        rlon / torch.sqrt(phi4),
        p[4],
        p[5],
        torch.exp(p[6]),
    ])


def relative_se_summary(se_by_key, denom_dict, keys, zero_thresh=0.01):
    vals = []
    for key in keys:
        denom = abs(denom_dict[key])
        if denom >= zero_thresh:
            vals.append((se_by_key[key] / denom) ** 2)
        else:
            vals.append(se_by_key[key] ** 2)
    return float(np.sqrt(np.mean(vals)))


# ---------------------------------------------------------------------------
# Godambe helpers
# ---------------------------------------------------------------------------

def finite_diff_hessian(nll_fn, p, eps=HESSIAN_EPS):
    n = p.shape[0]
    H = torch.zeros(n, n, device=p.device, dtype=p.dtype)
    for i in range(n):
        p_p = p.detach().clone()
        p_m = p.detach().clone()
        p_p[i] += eps
        p_m[i] -= eps
        p_p.requires_grad_(True)
        p_m.requires_grad_(True)
        g_p = torch.autograd.grad(nll_fn(p_p), p_p)[0].detach()
        g_m = torch.autograd.grad(nll_fn(p_m), p_m)[0].detach()
        H[i] = (g_p - g_m) / (2.0 * eps)
    return (H + H.T) / 2.0


def vecchia_per_unit_target_coords(model):
    chunks = []
    if model.Heads_data is not None and model.Heads_data.shape[0] > 0:
        chunks.append(model.Heads_data[:, [0, 1, 3]].to(dtype=DTYPE))
    for X_b in [model.X_A, model.X_AB, model.X_ABC]:
        if X_b is not None and X_b.shape[0] > 0:
            chunks.append(X_b[:, -1, :].to(dtype=DTYPE))
    if not chunks:
        return torch.empty((0, 3), device=DEVICE, dtype=DTYPE)
    return torch.cat(chunks, dim=0)


def make_block_ids(target_coords):
    lat = target_coords[:, 0]
    lon = target_coords[:, 1]
    tim = target_coords[:, 2]
    lat_id = torch.floor((lat - lat.min()) / GODAMBE_BLOCK_LAT_WIDTH).to(torch.long)
    lon_id = torch.floor((lon - lon.min()) / GODAMBE_BLOCK_LON_WIDTH).to(torch.long)
    if GODAMBE_BLOCK_TIME_WIDTH is None or GODAMBE_BLOCK_TIME_WIDTH <= 0:
        time_id = torch.zeros_like(lat_id)
    else:
        time_id = torch.floor((tim - tim.min()) / GODAMBE_BLOCK_TIME_WIDTH).to(torch.long)
    n_lon   = int(lon_id.max().item()) + 1 if lon_id.numel() else 1
    n_time  = int(time_id.max().item()) + 1 if time_id.numel() else 1
    raw_id  = (lat_id * n_lon + lon_id) * n_time + time_id
    _, block_id = torch.unique(raw_id, sorted=True, return_inverse=True)
    return block_id


def score_cov_per_unit_centered(score_mat):
    n_units = score_mat.shape[1]
    score_mean = score_mat.mean(dim=1)
    score_centered = score_mat - score_mean.unsqueeze(1)
    if n_units > 1:
        return score_centered @ score_centered.T / (n_units * (n_units - 1))
    return score_mat @ score_mat.T / max(n_units ** 2, 1)


def score_cov_block_cluster(score_mat, target_coords):
    n_units = score_mat.shape[1]
    scores = score_mat.T.contiguous()
    block_id = make_block_ids(target_coords)
    n_blocks = int(block_id.max().item()) + 1 if block_id.numel() else 0
    block_scores = torch.zeros((n_blocks, scores.shape[1]), device=DEVICE, dtype=DTYPE)
    block_scores.index_add_(0, block_id, scores)
    if n_blocks > 1:
        centered = block_scores - block_scores.mean(dim=0, keepdim=True)
        J = centered.T @ centered * (n_blocks / (n_blocks - 1)) / (n_units ** 2)
    else:
        J = block_scores.T @ block_scores / max(n_units ** 2, 1)
    return J, n_blocks


def compute_vecchia_godambe(model, raw_params, true_dict):
    p_hat = torch.tensor(raw_params[:7], device=DEVICE, dtype=DTYPE, requires_grad=True)

    def nll(p):
        return model.vecchia_batched_likelihood(p)

    H = finite_diff_hessian(nll, p_hat)
    eig = torch.linalg.eigvalsh(H).detach()
    h_abs_min = torch.clamp(torch.min(torch.abs(eig)), min=1e-12)
    h_cond = float((torch.max(torch.abs(eig)) / h_abs_min).detach().cpu())
    beta_hat = model.get_gls_beta(p_hat).detach()

    def per_unit_losses(p):
        return model.vecchia_per_unit_nll_terms(p, beta_hat)

    cols = []
    for k in range(p_hat.shape[0]):
        pp = p_hat.detach().clone()
        pm = p_hat.detach().clone()
        pp[k] += SCORE_EPS
        pm[k] -= SCORE_EPS
        with torch.no_grad():
            cols.append((per_unit_losses(pp) - per_unit_losses(pm)) / (2.0 * SCORE_EPS))
    score_mat = torch.stack(cols)
    n_units = score_mat.shape[1]
    target_coords = vecchia_per_unit_target_coords(model)
    if target_coords.shape[0] != n_units:
        raise RuntimeError(
            f"target/score mismatch: targets={target_coords.shape[0]}, scores={n_units}"
        )

    score_mean = score_mat.mean(dim=1)
    p_grad = p_hat.detach().clone().requires_grad_(True)
    profile_grad = torch.autograd.grad(nll(p_grad), p_grad)[0].detach()
    score_grad_diff = profile_grad - score_mean

    J_uncentered = score_mat @ score_mat.T / (n_units ** 2)
    J_centered = score_cov_per_unit_centered(score_mat)
    J_block, n_blocks = score_cov_block_cluster(score_mat, target_coords)
    if GODAMBE_J_METHOD == "block":
        J_main = J_block
    elif GODAMBE_J_METHOD == "per_unit_centered":
        J_main = J_centered
    elif GODAMBE_J_METHOD == "per_unit_uncentered":
        J_main = J_uncentered
    else:
        raise ValueError(f"Unknown GODAMBE_J_METHOD={GODAMBE_J_METHOD!r}")

    eye = torch.eye(H.shape[0], device=DEVICE, dtype=DTYPE)
    h_scale = torch.clamp(torch.mean(torch.abs(torch.diag(H))), min=1.0)
    H_inv = torch.linalg.pinv(H + eye * h_scale * H_RIDGE_SCALE)
    Jac = torch.autograd.functional.jacobian(transform_log_phi_to_physical, p_hat)

    def summarize_J(J):
        G_raw = H_inv @ J @ H_inv
        G_phys = Jac @ G_raw @ Jac.T
        se = torch.sqrt(torch.clamp(torch.diag(G_phys), min=0.0)).detach().cpu().numpy()
        se_by_key = dict(zip(P_LABELS, [float(x) for x in se]))
        return se_by_key, {
            "spatial": relative_se_summary(se_by_key, true_dict, SPATIAL_KEYS),
            "overall": relative_se_summary(se_by_key, true_dict, P_LABELS),
            "advec":   relative_se_summary(se_by_key, true_dict, ADVECTION_KEYS),
            "nugget":  se_by_key["nugget"] / abs(true_dict["nugget"]),
        }

    se_main,       rel_main       = summarize_J(J_main)
    se_block,      rel_block      = summarize_J(J_block)
    se_centered,   rel_centered   = summarize_J(J_centered)
    se_uncentered, rel_uncentered = summarize_J(J_uncentered)
    return {
        "gim_j_method":                       GODAMBE_J_METHOD,
        "gim_n_units":                         int(n_units),
        "gim_n_blocks":                        int(n_blocks),
        "gim_h_cond_abs":                      h_cond,
        "gim_score_mean_max_abs":              float(torch.max(torch.abs(score_mean)).detach().cpu()),
        "gim_profile_grad_max_abs":            float(torch.max(torch.abs(profile_grad)).detach().cpu()),
        "gim_score_profile_diff_max_abs":      float(torch.max(torch.abs(score_grad_diff)).detach().cpu()),
        "gim_spatial_rel_se":                  rel_main["spatial"],
        "gim_overall_rel_se":                  rel_main["overall"],
        "gim_advec_rel_se":                    rel_main["advec"],
        "gim_nugget_rel_se":                   rel_main["nugget"],
        "gim_spatial_rel_se_block":            rel_block["spatial"],
        "gim_spatial_rel_se_perunit_centered": rel_centered["spatial"],
        "gim_spatial_rel_se_uncentered":       rel_uncentered["spatial"],
        **{f"gim_se_{k}": v for k, v in se_main.items()},
    }


# ---------------------------------------------------------------------------
# Hybrid fresh class (local NN + fresh shifted-center NN at each lag)
# ---------------------------------------------------------------------------

class fit_vecchia_lbfgs_fresh_hybrid(kernels_vecchia.fit_vecchia_lbfgs):
    """Vecchia kernel mixing local lag neighbors with fresh shifted-center lag neighbors.

    At each lag:
      same-location anchor + local NN (nns_map, first limit_B/limit_C entries)
                           + fresh NN around shifted upstream center

    lag*_fresh_count includes the shifted center as first candidate, then
    max-min NN around that center. Duplicates are skipped; next NN backfills.
    """

    def __init__(self, smooth, input_map, nns_map, mm_cond_number, nheads,
                 limit_A=20, limit_B=16, limit_C=12, daily_stride=2,
                 spatial_coords=None, lag1_lon_offset=0.063,
                 lag1_fresh_count=2, lag2_fresh_count=2):
        super().__init__(smooth, input_map, nns_map, mm_cond_number, nheads,
                         limit_A=limit_A, limit_B=limit_B, limit_C=limit_C,
                         daily_stride=daily_stride)
        self.spatial_coords = spatial_coords
        self.lag1_lon_offset  = float(abs(lag1_lon_offset))
        self.lag1_fresh_count = int(lag1_fresh_count)
        self.lag2_fresh_count = int(lag2_fresh_count)

    def _spatial_coords_np(self, n_points):
        if self.spatial_coords is not None:
            coords_np = np.asarray(self.spatial_coords[:n_points], dtype=np.float64)
        else:
            all_data = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                        for d in self.input_map.values()]
            coords_np = all_data[0][:n_points, :2].cpu().numpy().astype(np.float64)
        coords_np = coords_np.copy()
        nan_mask = np.isnan(coords_np).any(axis=1)
        coords_np[nan_mask] = np.array([0.0, 1000.0])
        return coords_np

    def _build_shift_lookup(self, n_points, multiplier):
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
        limit_A   = int(self.limit_A)
        lag1_local = int(self.limit_B)
        lag2_local = int(self.limit_C)
        lag1_fresh = int(self.lag1_fresh_count)
        lag2_fresh = int(self.lag2_fresh_count)
        daily_stride = int(self.daily_stride)

        max_dim_A   = limit_A
        max_dim_AB  = limit_A + 1 + lag1_local + lag1_fresh
        max_dim_ABC = max_dim_AB + 1 + lag2_local + lag2_fresh

        n_stored = next((len(m) for m in self.nns_map if len(m) > 0), 0)
        print(
            "Pre-computing FreshHybrid Vecchia "
            f"[A={max_dim_A}, AB={max_dim_AB}, ABC={max_dim_ABC}, "
            f"B=local{lag1_local}+fresh{lag1_fresh}, "
            f"C=local{lag2_local}+fresh{lag2_fresh}, "
            f"lag1_offset={self.lag1_lon_offset:.4f}, stored={n_stored}]...",
            end=" ",
        )

        all_data_list = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                         for d in self.input_map.values()]
        Real_Data = torch.cat(all_data_list, dim=0).to(self.device, dtype=torch.float32)
        n_real, num_cols = Real_Data.shape

        is_nan_real = torch.isnan(Real_Data[:, 2])
        valid_lats = Real_Data[~is_nan_real, 0]
        self.lat_mean_val = (valid_lats.mean().item() if valid_lats.numel() > 0
                             else Real_Data[:, 0].mean().item())
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

        key_list      = list(self.input_map.keys())
        day_lengths   = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        n_time_steps  = len(key_list)
        use_set_C     = daily_stride < n_time_steps

        n_pts_per_day = day_lengths[0]
        lag1_center   = self._build_shift_lookup(n_pts_per_day, multiplier=1.0)
        lag2_center   = self._build_shift_lookup(n_pts_per_day, multiplier=2.0)

        heads_indices = []
        batch_list_A  = []
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
            offset  = cumulative_len[time_idx]

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
                nbs_current = (self.nns_map[local_idx] if local_idx < len(self.nns_map)
                               else np.array([], dtype=np.int64))
                add_valid_neighbors((offset + nbs_current).tolist(), current_indices, cap=limit_A)

                has_B = time_idx > 0
                has_C = time_idx >= daily_stride

                if has_B:
                    prev_off  = cumulative_len[time_idx - 1]
                    prev_len  = day_lengths[time_idx - 1]

                    if local_idx < prev_len:
                        add_valid_neighbors([prev_off + local_idx], current_indices, cap=1)

                    local_candidates = [
                        prev_off + int(v)
                        for v in nbs_current
                        if int(v) < prev_len and int(v) != local_idx
                    ]
                    add_valid_neighbors(local_candidates, current_indices, cap=lag1_local)

                    center_B = int(lag1_center[local_idx]) if local_idx < len(lag1_center) else local_idx
                    if center_B >= prev_len:
                        center_B = local_idx
                    nbs_B = (self.nns_map[center_B] if center_B < len(self.nns_map)
                             else np.array([], dtype=np.int64))
                    fresh_candidates_B = [prev_off + center_B] + [
                        prev_off + int(v)
                        for v in nbs_B
                        if int(v) < prev_len and int(v) != local_idx
                    ]
                    add_valid_neighbors(fresh_candidates_B, current_indices, cap=lag1_fresh)

                if has_C:
                    pd_idx = time_idx - daily_stride
                    pd_off  = cumulative_len[pd_idx]
                    pd_len  = day_lengths[pd_idx]

                    if local_idx < pd_len:
                        add_valid_neighbors([pd_off + local_idx], current_indices, cap=1)

                    local_candidates = [
                        pd_off + int(v)
                        for v in nbs_current
                        if int(v) < pd_len and int(v) != local_idx
                    ]
                    add_valid_neighbors(local_candidates, current_indices, cap=lag2_local)

                    center_C = int(lag2_center[local_idx]) if local_idx < len(lag2_center) else local_idx
                    if center_C >= pd_len:
                        center_C = local_idx
                    nbs_C = (self.nns_map[center_C] if center_C < len(self.nns_map)
                             else np.array([], dtype=np.int64))
                    fresh_candidates_C = [pd_off + center_C] + [
                        pd_off + int(v)
                        for v in nbs_C
                        if int(v) < pd_len and int(v) != local_idx
                    ]
                    add_valid_neighbors(fresh_candidates_C, current_indices, cap=lag2_fresh)

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
            X    = G[..., [0, 1, 3]].contiguous().to(torch.float64)
            Y    = G[..., 2].unsqueeze(-1).contiguous().to(torch.float64)
            ones = torch.ones_like(G[..., 0]).unsqueeze(-1)
            lat  = (G[..., 0] - self.lat_mean_val).unsqueeze(-1)
            dums = G[..., 4:11]
            Locs = torch.cat([ones, lat, dums], dim=-1).contiguous().to(torch.float64)
            is_dummy = (T >= dummy_start).unsqueeze(-1)
            Locs = Locs.masked_fill(is_dummy, 0.0)
            Y    = Y.masked_fill(is_dummy, 0.0)
            return X, Y, Locs, T, is_dummy

        self.X_A,   self.Y_A,   self.Locs_A,   self._T_A,   self._is_dummy_A   = build_tensors(batch_list_A,   max_dim_A)
        self.X_AB,  self.Y_AB,  self.Locs_AB,  self._T_AB,  self._is_dummy_AB  = build_tensors(batch_list_AB,  max_dim_AB)
        self.X_ABC, self.Y_ABC, self.Locs_ABC, self._T_ABC, self._is_dummy_ABC = build_tensors(batch_list_ABC, max_dim_ABC)

        self._heads_tensor_stored = heads_tensor if len(heads_indices) > 0 else None
        self._dummy_start_stored  = dummy_start
        self._n_real_stored       = n_real
        self._n_dummies_stored    = n_dummies
        self.n_tails = len(batch_list_A) + len(batch_list_AB) + len(batch_list_ABC)

        print(
            f"[Set C: {use_set_C}] Done. "
            f"(Heads: {len(heads_indices)}, "
            f"Tails A/AB/ABC: {len(batch_list_A)}/{len(batch_list_AB)}/{len(batch_list_ABC)})"
        )
        self.is_precomputed = True
        return self


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

def make_model_specs() -> List[Dict[str, Any]]:
    """Return the fixed list of 7 model configurations for this experiment."""
    return [
        # ---- baseline -------------------------------------------------------
        {
            "model":             "Irr_Cand_A20_B18_C15",
            "group":             "local_baseline",
            "kernel":            "std",
            "limit_A":           20, "limit_B": 18, "limit_C": 15,
            "lag1_local_count":  18, "lag1_fresh_count": 0,
            "lag2_local_count":  15, "lag2_fresh_count": 0,
            "pred_lag1_lon_offset": 0.0,
            "total_conditioning":   55,   # 20 + (1+18) + (1+15)
        },
        # ---- hybrid NearLocal (54 budget) -----------------------------------
        {
            "model":             "Hybrid_NearLocal_L16F02_C12F02_Op0p063",
            "group":             "hybrid_nearlocal",
            "kernel":            "hybrid_fresh",
            "limit_A":           20, "limit_B": 16, "limit_C": 12,
            "lag1_local_count":  16, "lag1_fresh_count": 2,
            "lag2_local_count":  12, "lag2_fresh_count": 2,
            "pred_lag1_lon_offset": 0.063,
            "total_conditioning":   54,   # 20 + (1+16+2) + (1+12+2)
        },
        {
            "model":             "Hybrid_NearLocal_L16F02_C12F02_Op0p126",
            "group":             "hybrid_nearlocal",
            "kernel":            "hybrid_fresh",
            "limit_A":           20, "limit_B": 16, "limit_C": 12,
            "lag1_local_count":  16, "lag1_fresh_count": 2,
            "lag2_local_count":  12, "lag2_fresh_count": 2,
            "pred_lag1_lon_offset": 0.126,
            "total_conditioning":   54,
        },
        # ---- hybrid Lean (41 budget) ----------------------------------------
        {
            "model":             "Hybrid_Lean_L08F04_C4F03_Op0p060",
            "group":             "hybrid_lean",
            "kernel":            "hybrid_fresh",
            "limit_A":           20, "limit_B": 8, "limit_C": 4,
            "lag1_local_count":  8,  "lag1_fresh_count": 4,
            "lag2_local_count":  4,  "lag2_fresh_count": 3,
            "pred_lag1_lon_offset": 0.060,
            "total_conditioning":   41,   # 20 + (1+8+4) + (1+4+3)
        },
        {
            "model":             "Hybrid_Lean_L08F04_C4F03_Op0p126",
            "group":             "hybrid_lean",
            "kernel":            "hybrid_fresh",
            "limit_A":           20, "limit_B": 8, "limit_C": 4,
            "lag1_local_count":  8,  "lag1_fresh_count": 4,
            "lag2_local_count":  4,  "lag2_fresh_count": 3,
            "pred_lag1_lon_offset": 0.126,
            "total_conditioning":   41,
        },
        # ---- pure fresh shift-center (~52 budget) ---------------------------
        {
            "model":             "FreshShift_A20_B18_C12_Op0p063",
            "group":             "fresh_shift",
            "kernel":            "hybrid_fresh",
            "limit_A":           20, "limit_B": 0, "limit_C": 0,
            "lag1_local_count":  0,  "lag1_fresh_count": 18,
            "lag2_local_count":  0,  "lag2_fresh_count": 12,
            "pred_lag1_lon_offset": 0.063,
            "total_conditioning":   52,   # 20 + (1+0+18) + (1+0+12)
        },
        {
            "model":             "FreshShift_A20_B18_C12_Op0p126",
            "group":             "fresh_shift",
            "kernel":            "hybrid_fresh",
            "limit_A":           20, "limit_B": 0, "limit_C": 0,
            "lag1_local_count":  0,  "lag1_fresh_count": 18,
            "lag2_local_count":  0,  "lag2_fresh_count": 12,
            "pred_lag1_lon_offset": 0.126,
            "total_conditioning":   52,   # 20 + (1+0+18) + (1+0+12)
        },
    ]


# ---------------------------------------------------------------------------
# Fitting + Godambe
# ---------------------------------------------------------------------------

def fit_irregular_model(spec, irr_map_ord, nns_grid, ordered_grid_coords_np,
                         initial_vals, smooth, mm_cond_number, nheads, daily_stride,
                         lbfgs_lr, lbfgs_eval, lbfgs_hist, lbfgs_steps,
                         compute_godambe=True, true_dict=None):
    params = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
              for val in initial_vals]
    kernel = spec["kernel"]

    t_pre = time.time()
    if kernel == "std":
        model = kernels_vecchia.fit_vecchia_lbfgs(
            smooth=smooth, input_map=irr_map_ord, nns_map=nns_grid,
            mm_cond_number=mm_cond_number, nheads=nheads,
            limit_A=spec["limit_A"], limit_B=spec["limit_B"], limit_C=spec["limit_C"],
            daily_stride=daily_stride,
        )
    elif kernel == "hybrid_fresh":
        model = fit_vecchia_lbfgs_fresh_hybrid(
            smooth=smooth, input_map=irr_map_ord, nns_map=nns_grid,
            mm_cond_number=mm_cond_number, nheads=nheads,
            limit_A=spec["limit_A"],
            limit_B=spec["lag1_local_count"],
            limit_C=spec["lag2_local_count"],
            daily_stride=daily_stride,
            spatial_coords=ordered_grid_coords_np,
            lag1_lon_offset=spec["pred_lag1_lon_offset"],
            lag1_fresh_count=spec["lag1_fresh_count"],
            lag2_fresh_count=spec["lag2_fresh_count"],
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    model.precompute_conditioning_sets()
    precompute_s = time.time() - t_pre

    optimizer = model.set_optimizer(params, lr=lbfgs_lr, max_iter=lbfgs_eval,
                                     history_size=lbfgs_hist)
    t_fit = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, optimizer, max_steps=lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t_fit

    loss = float(out[-1])
    metrics, est = calculate_metrics(out, true_dict)

    godambe = {}
    gim_s = 0.0
    if compute_godambe and true_dict is not None:
        t_gim = time.time()
        try:
            godambe = compute_vecchia_godambe(model, [float(x) for x in out[:7]], true_dict)
        except Exception as exc:
            godambe = {"gim_error": f"{type(exc).__name__}: {exc}"}
        gim_s = time.time() - t_gim

    return out, loss, int(fit_iter) + 1, precompute_s, fit_s, gim_s, metrics, est, godambe


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "loss", "overall_rmsre", "spatial_rmsre",
    "sigmasq_re", "range_lat_re", "range_lon_re",
    "range_time_re", "advec_lat_re", "advec_lon_re", "nugget_re",
    "advec_rmsre", "total_s", "fit_iter",
]
GIM_METRIC_COLS = [
    "gim_se_advec_lon", "gim_relse_advec_lon",
    "gim_se_range_lon", "gim_se_range_time",
    "gim_spatial_rel_se", "gim_overall_rel_se", "gim_advec_rel_se",
    "gim_h_cond_abs",
    "gim_spatial_rel_se_block", "gim_spatial_rel_se_perunit_centered",
    "gim_spatial_rel_se_uncentered",
]


def build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["error"].fillna("") == ""].copy()
    group_cols = ["model", "group", "kernel", "pred_lag1_lon_offset",
                  "lag1_local_count", "lag1_fresh_count",
                  "lag2_local_count", "lag2_fresh_count", "total_conditioning",
                  "true_advec_lon"]
    rows = []
    for keys, sub in ok.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n"] = len(sub)
        for col in METRIC_COLS + GIM_METRIC_COLS:
            vals = sub[col].dropna().values if col in sub.columns else np.array([])
            if len(vals) == 0:
                for sfx in ("_mean", "_median", "_p10", "_p90", "_p90_p10"):
                    row[f"{col}{sfx}"] = float("nan")
                continue
            p10, p90 = np.percentile(vals, [10, 90])
            row[f"{col}_mean"]    = float(np.mean(vals))
            row[f"{col}_median"]  = float(np.median(vals))
            row[f"{col}_p10"]     = float(p10)
            row[f"{col}_p90"]     = float(p90)
            row[f"{col}_p90_p10"] = float(p90 - p10)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("overall_rmsre_mean") if not out.empty else out


def build_param_summary(df: pd.DataFrame, true_dict: Dict) -> pd.DataFrame:
    rows = []
    ok = df[df["error"].fillna("") == ""].copy()
    for model, sub in ok.groupby("model"):
        for par, col in zip(P_LABELS, P_COLS):
            tv    = true_dict[par]
            denom = abs(tv) if abs(tv) >= 0.01 else 1.0
            vals  = sub[col].dropna().values
            if len(vals) == 0:
                continue
            re = np.abs((vals - tv) / denom)
            p10, p90 = np.percentile(re, [10, 90])
            rows.append({
                "model":         model,
                "parameter":     par,
                "true":          tv,
                "rmsre":         float(np.sqrt(np.mean(re ** 2))),
                "mean_re":       float(np.mean(re)),
                "median_re":     float(np.median(re)),
                "p10_re":        float(p10),
                "p90_re":        float(p90),
                "p90_p10_re":    float(p90 - p10),
                "estimate_mean": float(sub[col].mean()),
                "estimate_sd":   float(sub[col].std()),
            })
    return pd.DataFrame(rows)


def build_gim_summary(df: pd.DataFrame, true_dict: Dict) -> pd.DataFrame:
    """Parameter-level Godambe SE summary: mean/median/p90-p10 per model."""
    gim_se_cols = {par: f"gim_se_{par}" for par in P_LABELS}
    rows = []
    ok = df[df["error"].fillna("") == ""].copy()
    for model, sub in ok.groupby("model"):
        for par, col in gim_se_cols.items():
            if col not in sub.columns:
                continue
            tv    = true_dict[par]
            denom = abs(tv) if abs(tv) >= 0.01 else 1.0
            vals  = sub[col].dropna().values
            if len(vals) == 0:
                continue
            relse = vals / denom
            p10, p90 = np.percentile(relse, [10, 90])
            rows.append({
                "model":          model,
                "parameter":      par,
                "true":           tv,
                "gim_se_mean":    float(np.mean(vals)),
                "gim_se_median":  float(np.median(vals)),
                "gim_relse_mean": float(np.mean(relse)),
                "gim_relse_median": float(np.median(relse)),
                "gim_relse_p10":  float(p10),
                "gim_relse_p90":  float(p90),
                "gim_relse_p90_p10": float(p90 - p10),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def cli(
    v:               float = typer.Option(0.5,  help="Matern smoothness"),
    mm_cond_number:  int   = typer.Option(100,  help="Maxmin neighbor list size"),
    nheads:          int   = typer.Option(0,    help="Vecchia head points per time step"),
    daily_stride:    int   = typer.Option(2,    help="Set C stride; 2 means t-2"),
    num_iters:       int   = typer.Option(300,  help="Simulation iterations"),
    years:           str   = typer.Option("2022,2023,2024,2025", help="Years for observation patterns"),
    month:           int   = typer.Option(7,    help="Reference month"),
    lat_range:       str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:       str   = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor:      int   = typer.Option(100,  help="High-res lat multiplier"),
    lon_factor:      int   = typer.Option(10,   help="High-res lon multiplier"),
    true_advec_lat:  float = typer.Option(0.08,  help="True latitudinal advection"),
    true_advec_lon:  str   = typer.Option("-0.10,-0.16,-0.25", help="Comma-separated true lon advection values (cycled per iteration)"),
    init_noise:      float = typer.Option(0.7,  help="Uniform log-space init noise"),
    lbfgs_steps:     int   = typer.Option(5,    help="Outer LBFGS steps"),
    lbfgs_eval:      int   = typer.Option(20,   help="LBFGS max_iter per outer step"),
    lbfgs_hist:      int   = typer.Option(10,   help="LBFGS history size"),
    compute_godambe: bool  = typer.Option(True, help="Compute Godambe SE each iteration"),
    seed:            int   = typer.Option(42,   help="Random seed"),
    out_prefix:      str   = typer.Option("sim_vecchia_irregular_hybrid_compare",
                                           help="Output file prefix"),
) -> None:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    lat_r     = [float(x) for x in lat_range.split(",")]
    lon_r     = [float(x) for x in lon_range.split(",")]
    years_list = [y.strip() for y in years.split(",")]
    advec_lon_list = [float(x.strip()) for x in true_advec_lon.split(",")]

    print(f"Device     : {DEVICE}")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"Years      : {years_list}  month={month}")
    print(f"Iterations : {num_iters}")
    print(f"True advec_lat : {true_advec_lat}")
    print(f"True advec_lon list (cycled) : {advec_lon_list}")
    print(f"Godambe    : {compute_godambe}")

    output_path = Path(config.amarel_estimates_day_path if is_amarel
                       else config.mac_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag          = datetime.now().strftime("%m%d%y")
    raw_csv           = output_path / f"{out_prefix}_{date_tag}_raw.csv"
    model_summary_csv = output_path / f"{out_prefix}_{date_tag}_model_summary.csv"
    param_summary_csv = output_path / f"{out_prefix}_{date_tag}_param_summary.csv"
    gim_summary_csv   = output_path / f"{out_prefix}_{date_tag}_gim_summary.csv"

    # default true_dict for setup verification (uses first advec_lon value)
    true_dict_default = {
        "sigmasq": 10.0, "range_lat": 0.5, "range_lon": 0.6,
        "range_time": 2.5, "advec_lat": true_advec_lat,
        "advec_lon": advec_lon_list[0], "nugget": 1.2,
    }
    true_params_default = torch.tensor(true_to_log_params(true_dict_default), device=DEVICE, dtype=DTYPE)

    model_specs = make_model_specs()
    print("\nModel specs")
    for s in model_specs:
        print(
            f"  {s['model']}: kernel={s['kernel']}  "
            f"A={s['limit_A']}  B_local={s['lag1_local_count']}  "
            f"B_fresh={s['lag1_fresh_count']}  "
            f"C_local={s['lag2_local_count']}  C_fresh={s['lag2_fresh_count']}  "
            f"offset={s['pred_lag1_lon_offset']:.3f}  total={s['total_conditioning']}"
        )

    print("\n[Setup 1/5] Loading GEMS observation patterns...")
    data_path = config.amarel_data_load_path if is_amarel else config.mac_data_load_path
    data_loader  = load_data_dynamic_processed(data_path)
    year_dfmaps  = {}
    year_tco_maps = {}
    for yr in years_list:
        df_map_yr, _, _, _ = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1], mm_cond_number=mm_cond_number,
            years_=[yr], months_=[month], lat_range=lat_r, lon_range=lon_r, is_whittle=False,
        )
        year_dfmaps[yr] = df_map_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots loaded")
        yr2 = str(yr)[2:]
        tco_path = Path(data_path) / f"pickle_{yr}" / f"tco_grid_{yr2}_{month:02d}.pkl"
        if tco_path.exists():
            with open(tco_path, "rb") as f:
                year_tco_maps[yr] = pickle.load(f)
            print(f"    tco_grid: {len(year_tco_maps[yr])} slots")
        else:
            year_tco_maps[yr] = {}
            print(f"    [WARN] Missing tco_grid: {tco_path}")

    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 1e-4, DELTA_LAT_BASE, device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]   + 1e-4, DELTA_LON_BASE, device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing="ij")
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    n_grid       = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat x {len(lons_grid)} lon = {n_grid} cells")

    print("[Setup 3/5] Building high-res grid and precomputing mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat x {len(lons_hr)} lon = {len(lats_hr)*len(lons_hr):,}")

    dummy_keys = [f"t{i}" for i in range(T_STEPS)]
    all_day_mappings = []
    for yr in years_list:
        df_map_yr  = year_dfmaps[yr]
        all_sorted = sorted(df_map_yr.keys())
        n_days_yr  = len(all_sorted) // T_STEPS
        print(f"  {yr}: precomputing {n_days_yr} day-patterns...", flush=True)
        for d_idx in range(n_days_yr):
            day_keys = all_sorted[d_idx * T_STEPS: (d_idx + 1) * T_STEPS]
            if len(day_keys) < T_STEPS:
                continue
            ref_day = {k: year_tco_maps[yr].get(k.split("_", 2)[-1], pd.DataFrame())
                       for k in day_keys}
            s3, hr_idx, src = precompute_mapping_indices(ref_day, lats_hr, lons_hr,
                                                          grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))

    if not all_day_mappings:
        raise RuntimeError("No usable day-patterns found.")
    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    print("[Setup 4/5] Computing shared grid-based maxmin ordering...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, mm_cond_number)
    ordered_grid_coords_np = grid_coords[ord_grid].detach().cpu().numpy()
    print(f"  Ordering complete: N_grid={n_grid}")

    print("[Setup 5/5] Verifying one assembled irregular map...")
    yr0, d0, s3_0, hr0, src0 = all_day_mappings[0]
    field0 = generate_field_values(lats_hr, lons_hr, T_STEPS, true_params_default, dlat_hr, dlon_hr)
    irr0   = assemble_irregular_map(field0, s3_0, hr0, src0, dummy_keys, grid_coords, true_params_default)
    del field0
    n_valid0 = int((~torch.isnan(list(irr0.values())[0][:, 2])).sum().item())
    print(f"  Sample {yr0} day {d0}: {n_valid0}/{n_grid} valid rows at t0")
    del irr0
    torch.cuda.empty_cache()
    gc.collect()

    lbfgs_lr = 1.0
    records  = []

    for it in range(num_iters):
        print(f"\n{'=' * 72}")
        print(f"Iteration {it + 1}/{num_iters}")
        print(f"{'=' * 72}")

        true_advec_lon_it = advec_lon_list[it % len(advec_lon_list)]
        true_dict = {
            "sigmasq": 10.0, "range_lat": 0.5, "range_lon": 0.6,
            "range_time": 2.5, "advec_lat": true_advec_lat,
            "advec_lon": true_advec_lon_it, "nugget": 1.2,
        }
        true_log    = true_to_log_params(true_dict)
        true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)
        print(f"True advec_lon this iter: {true_advec_lon_it}")

        yr_it, d_it, s3_it, hr_it, src_it = all_day_mappings[rng.integers(len(all_day_mappings))]
        initial_vals = make_random_init(rng, true_log, init_noise)
        init_orig    = backmap_params(initial_vals)
        print(f"Obs pattern: {yr_it} day {d_it}")
        print(
            f"Init: sigmasq={init_orig['sigmasq']:.3f}, "
            f"range_lon={init_orig['range_lon']:.3f}, "
            f"nugget={init_orig['nugget']:.3f}"
        )

        try:
            field   = generate_field_values(lats_hr, lons_hr, T_STEPS, true_params, dlat_hr, dlon_hr)
            irr_map = assemble_irregular_map(field, s3_it, hr_it, src_it, dummy_keys,
                                              grid_coords, true_params)
            del field
            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}
            del irr_map
        except Exception as exc:
            print(f"[SKIP] Assembly failed: {type(exc).__name__}: {exc}")
            continue

        for spec in model_specs:
            print(f"\n--- {spec['model']} ---")
            pre_t     = time.time()
            error_msg = ""
            try:
                out, loss, fit_iter, pre_s, fit_s, gim_s, metrics, est, godambe = fit_irregular_model(
                    spec, irr_map_ord, nns_grid, ordered_grid_coords_np, initial_vals,
                    v, mm_cond_number, nheads, daily_stride,
                    lbfgs_lr, lbfgs_eval, lbfgs_hist, lbfgs_steps,
                    compute_godambe=compute_godambe, true_dict=true_dict,
                )
                total_s = time.time() - pre_t
                adv_lon_est = est["advec_lon"]
                gim_adv_relse = godambe.get("gim_se_advec_lon", float("nan"))
                if np.isfinite(gim_adv_relse) and abs(true_dict["advec_lon"]) >= 0.01:
                    gim_adv_relse /= abs(true_dict["advec_lon"])
                print(
                    f"loss={loss:.4f}  overall={metrics['overall_rmsre']:.4f}  "
                    f"advec_lon_est={adv_lon_est:.4f}  "
                    f"gim_relse_advec_lon={gim_adv_relse:.4f}  "
                    f"time={total_s:.1f}s"
                )
            except Exception as exc:
                total_s  = time.time() - pre_t
                loss = pre_s = fit_s = gim_s = fit_iter = float("nan")
                metrics = {k: float("nan") for k in
                           ["overall_rmsre", "spatial_rmsre", "range_time_re", "advec_rmsre", "nugget_re",
                            "sigmasq_re", "range_lat_re", "range_lon_re", "advec_lat_re", "advec_lon_re"]}
                est      = {k: float("nan") for k in P_LABELS}
                godambe  = {}
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"FAILED: {error_msg}")

            true_av_abs = abs(true_dict["advec_lon"]) if abs(true_dict["advec_lon"]) >= 0.01 else 1.0
            gim_se_av   = godambe.get("gim_se_advec_lon", float("nan"))
            records.append({
                "iter":         it + 1,
                "obs_year":     yr_it,
                "obs_day":      d_it,
                "model":        spec["model"],
                "group":        spec["group"],
                "kernel":       spec["kernel"],
                "limit_A":      spec["limit_A"],
                "limit_B":      spec["limit_B"],
                "limit_C":      spec["limit_C"],
                "lag1_local_count":    spec["lag1_local_count"],
                "lag1_fresh_count":    spec["lag1_fresh_count"],
                "lag2_local_count":    spec["lag2_local_count"],
                "lag2_fresh_count":    spec["lag2_fresh_count"],
                "pred_lag1_lon_offset": spec["pred_lag1_lon_offset"],
                "total_conditioning":  spec["total_conditioning"],
                "true_advec_lat":  true_dict["advec_lat"],
                "true_advec_lon":  true_dict["advec_lon"],
                "loss":          round(loss, 6) if np.isfinite(loss) else np.nan,
                "overall_rmsre": round(metrics["overall_rmsre"], 6),
                "spatial_rmsre": round(metrics["spatial_rmsre"], 6),
                "sigmasq_re":    round(metrics["sigmasq_re"],    6),
                "range_lat_re":  round(metrics["range_lat_re"],  6),
                "range_lon_re":  round(metrics["range_lon_re"],  6),
                "range_time_re": round(metrics["range_time_re"], 6),
                "advec_lat_re":  round(metrics["advec_lat_re"],  6),
                "advec_lon_re":  round(metrics["advec_lon_re"],  6),
                "nugget_re":     round(metrics["nugget_re"],     6),
                "advec_rmsre":   round(metrics["advec_rmsre"],   6),
                "fit_iter":      fit_iter,
                "precompute_s":  round(pre_s, 3) if np.isfinite(pre_s) else np.nan,
                "fit_s":         round(fit_s, 3) if np.isfinite(fit_s) else np.nan,
                "gim_s":         round(gim_s, 3) if np.isfinite(gim_s) else np.nan,
                "total_s":       round(total_s, 3),
                "sigmasq_est":   round(est.get("sigmasq",    float("nan")), 6),
                "range_lat_est": round(est.get("range_lat",  float("nan")), 6),
                "range_lon_est": round(est.get("range_lon",  float("nan")), 6),
                "range_t_est":   round(est.get("range_time", float("nan")), 6),
                "advec_lat_est": round(est.get("advec_lat",  float("nan")), 6),
                "advec_lon_est": round(est.get("advec_lon",  float("nan")), 6),
                "nugget_est":    round(est.get("nugget",     float("nan")), 6),
                "init_sigmasq":  round(init_orig["sigmasq"],   6),
                "init_range_lon": round(init_orig["range_lon"], 6),
                "init_nugget":   round(init_orig["nugget"],    6),
                # Godambe columns
                "gim_j_method":                       godambe.get("gim_j_method", ""),
                "gim_n_units":                         godambe.get("gim_n_units",  float("nan")),
                "gim_n_blocks":                        godambe.get("gim_n_blocks", float("nan")),
                "gim_h_cond_abs":                      godambe.get("gim_h_cond_abs", float("nan")),
                "gim_score_mean_max_abs":              godambe.get("gim_score_mean_max_abs", float("nan")),
                "gim_profile_grad_max_abs":            godambe.get("gim_profile_grad_max_abs", float("nan")),
                "gim_score_profile_diff_max_abs":      godambe.get("gim_score_profile_diff_max_abs", float("nan")),
                "gim_spatial_rel_se":                  godambe.get("gim_spatial_rel_se", float("nan")),
                "gim_overall_rel_se":                  godambe.get("gim_overall_rel_se", float("nan")),
                "gim_advec_rel_se":                    godambe.get("gim_advec_rel_se",   float("nan")),
                "gim_nugget_rel_se":                   godambe.get("gim_nugget_rel_se",  float("nan")),
                "gim_spatial_rel_se_block":            godambe.get("gim_spatial_rel_se_block",            float("nan")),
                "gim_spatial_rel_se_perunit_centered": godambe.get("gim_spatial_rel_se_perunit_centered", float("nan")),
                "gim_spatial_rel_se_uncentered":       godambe.get("gim_spatial_rel_se_uncentered",       float("nan")),
                "gim_se_sigmasq":      godambe.get("gim_se_sigmasq",   float("nan")),
                "gim_se_range_lat":    godambe.get("gim_se_range_lat", float("nan")),
                "gim_se_range_lon":    godambe.get("gim_se_range_lon", float("nan")),
                "gim_se_range_time":   godambe.get("gim_se_range_time",  float("nan")),
                "gim_se_advec_lat":    godambe.get("gim_se_advec_lat",   float("nan")),
                "gim_se_advec_lon":    gim_se_av,
                "gim_se_nugget":       godambe.get("gim_se_nugget", float("nan")),
                "gim_relse_advec_lon": (gim_se_av / true_av_abs
                                        if np.isfinite(gim_se_av) else float("nan")),
                "error": error_msg,
            })

            torch.cuda.empty_cache()
            gc.collect()

        # ---- per-iteration running summary ----------------------------------
        df_now = pd.DataFrame(records)
        df_now.to_csv(raw_csv, index=False)
        ok_now = df_now[df_now["error"].fillna("") == ""]
        if not ok_now.empty:
            msumm  = build_model_summary(df_now)
            psumm  = build_param_summary(df_now, true_dict)
            gimsumm = build_gim_summary(df_now, true_dict)
            msumm.to_csv(model_summary_csv,   index=False)
            psumm.to_csv(param_summary_csv,   index=False)
            gimsumm.to_csv(gim_summary_csv,   index=False)

            if not msumm.empty:
                print("\nRunning model summary")
                show_cols = [
                    "model", "true_advec_lon", "pred_lag1_lon_offset", "total_conditioning", "n",
                    "loss_mean", "loss_p90_p10",
                    "overall_rmsre_mean", "overall_rmsre_p90_p10",
                    "sigmasq_re_mean", "sigmasq_re_p90_p10",
                    "range_lat_re_mean", "range_lat_re_p90_p10",
                    "range_lon_re_mean", "range_lon_re_p90_p10",
                    "range_time_re_mean", "range_time_re_p90_p10",
                    "advec_lat_re_mean", "advec_lat_re_p90_p10",
                    "advec_lon_re_mean", "advec_lon_re_p90_p10",
                    "nugget_re_mean", "nugget_re_p90_p10",
                    "total_s_mean",
                ]
                show_cols = [c for c in show_cols if c in msumm.columns]
                print(msumm[show_cols].to_string(index=False))

            if not psumm.empty:
                print("\nRunning parameter summary")
                pcols = ["model", "parameter", "rmsre", "mean_re", "median_re",
                         "p10_re", "p90_re", "p90_p10_re"]
                print(psumm[pcols].to_string(index=False))

            if not gimsumm.empty and compute_godambe:
                print("\nRunning GIM summary (all parameters)")
                gcols = ["model", "parameter", "gim_relse_mean", "gim_relse_median",
                         "gim_relse_p10", "gim_relse_p90", "gim_relse_p90_p10"]
                gcols = [c for c in gcols if c in gimsumm.columns]
                print(gimsumm[gcols].to_string(index=False))

        del irr_map_ord
        torch.cuda.empty_cache()
        gc.collect()

    # ---- final save ---------------------------------------------------------
    df_final = pd.DataFrame(records)
    df_final.to_csv(raw_csv, index=False)
    msumm   = build_model_summary(df_final)
    psumm   = build_param_summary(df_final, true_dict)
    gimsumm = build_gim_summary(df_final, true_dict)
    msumm.to_csv(model_summary_csv,  index=False)
    psumm.to_csv(param_summary_csv,  index=False)
    gimsumm.to_csv(gim_summary_csv,  index=False)

    print(f"\nSaved raw:           {raw_csv}")
    print(f"Saved model summary: {model_summary_csv}")
    print(f"Saved param summary: {param_summary_csv}")
    print(f"Saved GIM summary:   {gim_summary_csv}")

    if not msumm.empty:
        print("\nFinal model summary")
        print(msumm.to_string(index=False))


if __name__ == "__main__":
    app()
