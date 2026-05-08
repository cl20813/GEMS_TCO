"""
sim_vecchia_highres_colver3_vs_hybrid_050726.py

Amarel high-resolution complete-grid simulation comparing equal-sized
conditioning strategies:
  1. Column V3 rigid reverse-L/downward-right, nominal m=42
  2. Hybrid Lean batched Vecchia, nominal m=41

The script prints and saves running summaries after each iteration:
  - model summary: loss/time/RMSRE means and p90-p10 spreads
  - parameter summary: parameter-wise relative error RMSRE and p90-p10 spreads
"""

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.fft


AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
SRC = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, SRC)

from GEMS_TCO import orderings as _orderings
from GEMS_TCO.kernels_vecchia_hybrid import HybridVecchiaFit
from GEMS_TCO.kernel_vecchia_col_ver3 import ReverseLColumnVecchiaFitV3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
ROUND_DECIMALS = 4

DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
T_STEPS = 8
SMOOTH = 0.5

P_LABELS = ["sigmasq", "range_lat", "range_lon", "range_time", "advec_lat", "advec_lon", "nugget"]
SPATIAL_KEYS = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]


TRUE_DICT = {
    "sigmasq": 10.0,
    "range_lat": 0.30,
    "range_lon": 0.40,
    "range_time": 2.0,
    "advec_lat": 0.08,
    "advec_lon": -0.16,
    "nugget": 2.5,
}

COLUMN_V3_SPEC = {
    "model": "ColumnV3_Up3_Right3_Down14_Lag2",
    "head_right_cols": 3,
    "above_count": 3,
    "right_col_count": 3,
    "per_lag_conditioning_count": 14,
    "lag_count": 2,
    "include_lag_self": False,
}

HYBRID_SPEC = {
    "model": "Hybrid_Lean_L08F04_C4F03_Op0p063",
    "limit_A": 20,
    "lag1_local_count": 8,
    "lag1_fresh_count": 4,
    "lag2_local_count": 4,
    "lag2_fresh_count": 3,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def true_to_log_params(d):
    phi2 = 1.0 / d["range_lon"]
    phi1 = d["sigmasq"] * phi2
    phi3 = (d["range_lon"] / d["range_lat"]) ** 2
    phi4 = (d["range_lon"] / d["range_time"]) ** 2
    return [
        np.log(phi1),
        np.log(phi2),
        np.log(phi3),
        np.log(phi4),
        d["advec_lat"],
        d["advec_lon"],
        np.log(d["nugget"]),
    ]


def backmap_params(out_params):
    p = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in out_params[:7]]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {
        "sigmasq": np.exp(p[0]) / phi2,
        "range_lat": rlon / phi3**0.5,
        "range_lon": rlon,
        "range_time": rlon / phi4**0.5,
        "advec_lat": p[4],
        "advec_lon": p[5],
        "nugget": np.exp(p[6]),
    }


def make_random_init(rng, true_log, init_noise):
    noisy = list(true_log)
    for i in [0, 1, 2, 3, 6]:
        noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
    for i in [4, 5]:
        noisy[i] = true_log[i] + rng.uniform(-0.03, 0.03)
    return noisy


def rmsre_for_keys(est, truth, keys, zero_thresh=0.01):
    vals = []
    for k in keys:
        tv = truth[k]
        if abs(tv) >= zero_thresh:
            vals.append(((est[k] - tv) / abs(tv)) ** 2)
        else:
            vals.append(abs(est[k] - tv) ** 2)
    return float(np.sqrt(np.mean(vals)))


def calculate_metrics(out_params, truth):
    est = backmap_params(out_params)
    row = {
        "overall_rmsre": rmsre_for_keys(est, truth, P_LABELS),
        "spatial_rmsre": rmsre_for_keys(est, truth, SPATIAL_KEYS),
        "advec_rmsre": rmsre_for_keys(est, truth, ADVECTION_KEYS),
    }
    for p in P_LABELS:
        denom = abs(truth[p]) if abs(truth[p]) >= 0.01 else 1.0
        row[f"{p}_re"] = abs(est[p] - truth[p]) / denom
    row["est"] = est
    return row


def template_diagnostics(model):
    groups = getattr(model, "Grouped_Batches", [])
    if not groups:
        return {
            "n_templates": 0,
            "largest_template_n": 0,
            "median_template_n": 0.0,
            "mean_template_n": 0.0,
            "mean_m_by_template": 0.0,
            "median_m_by_template": 0.0,
            "max_m_by_template": 0,
        }
    group_sizes = np.asarray([int(g["target_idx"].shape[0]) for g in groups], dtype=np.int64)
    m_sizes = np.asarray([int(g["offsets"].shape[0]) for g in groups], dtype=np.int64)
    return {
        "n_templates": int(len(groups)),
        "largest_template_n": int(group_sizes.max()),
        "median_template_n": float(np.median(group_sizes)),
        "mean_template_n": float(group_sizes.mean()),
        "mean_m_by_template": float(m_sizes.mean()),
        "median_m_by_template": float(np.median(m_sizes)),
        "max_m_by_template": int(m_sizes.max()),
    }


def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def build_target_grid(lat_range, lon_range, lat_factor, lon_factor):
    delta_lat = DELTA_LAT_BASE * lat_factor
    delta_lon = DELTA_LON_BASE * lon_factor
    lat0, lat1 = float(min(lat_range)), float(max(lat_range))
    lon0, lon1 = float(min(lon_range)), float(max(lon_range))
    n_lat = int(np.floor((lat1 - lat0) / delta_lat + 1e-9)) + 1
    n_lon = int(np.floor((lon1 - lon0) / delta_lon + 1e-9)) + 1
    lats = lat0 + torch.arange(n_lat, device=DEVICE, dtype=DTYPE) * delta_lat
    lons = lon0 + torch.arange(n_lon, device=DEVICE, dtype=DTYPE) * delta_lon
    lats = torch.round(lats * 10000) / 10000
    lons = torch.round(lons * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats, lons, indexing="ij")
    grid_coords = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    return lats, lons, grid_coords, delta_lat, delta_lon


def generate_field_values(n_lat, n_lon, t_steps, params, delta_lat, delta_lon):
    cpu = torch.device("cpu")
    f32 = torch.float32
    px, py, pt = 2 * n_lat, 2 * n_lon, 2 * t_steps
    lx = torch.arange(px, device=cpu, dtype=f32) * delta_lat
    lx[px // 2 :] -= px * delta_lat
    ly = torch.arange(py, device=cpu, dtype=f32) * delta_lon
    ly[py // 2 :] -= py * delta_lon
    lt = torch.arange(pt, device=cpu, dtype=f32)
    lt[pt // 2 :] -= pt
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing="ij")
    C = get_covariance_on_grid(Lx, Ly, Lt, params.cpu().float())
    S = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(px, py, pt, device=cpu, dtype=f32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:n_lat, :n_lon, :t_steps]
    return field.to(device=DEVICE, dtype=DTYPE)


def assemble_reg_map(field, grid_coords, true_params, t_offset=21.0):
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    n_grid = grid_coords.shape[0]
    field_flat = field.reshape(n_grid, field.shape[-1])
    reg_map = {}
    for t_idx in range(field.shape[-1]):
        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0
        rows = torch.zeros((n_grid, 11), device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords
        rows[:, 2] = field_flat[:, t_idx] + torch.randn(n_grid, device=DEVICE, dtype=DTYPE) * nugget_std
        rows[:, 3] = float(t_offset + t_idx)
        rows[:, 4:] = dummy.unsqueeze(0).expand(n_grid, -1)
        reg_map[f"t{t_idx}"] = rows.detach()
    return reg_map


def assert_complete_regular_grid(reg_map):
    total = 0
    missing = 0
    for key, rows in reg_map.items():
        miss = torch.isnan(rows[:, :4]).any(dim=1)
        n_miss = int(miss.sum().item())
        missing += n_miss
        total += int(rows.shape[0])
        if n_miss:
            raise RuntimeError(f"{key} has {n_miss} missing rows; this comparison requires complete regular grids.")
    print(f"Complete regular-grid check passed: {total:,} rows, missing={missing}", flush=True)


def compute_grid_ordering(grid_coords, mm_cond_number):
    coords_np = grid_coords.detach().cpu().numpy()
    ord_grid = _orderings.maxmin_cpp(coords_np)
    nns_grid = _orderings.find_nns_l2(locs=coords_np[ord_grid], max_nn=mm_cond_number)
    return ord_grid, nns_grid


def hybrid_tail_count(model):
    total = 0
    for x in (getattr(model, "X_A", None), getattr(model, "X_AB", None), getattr(model, "X_ABC", None)):
        if x is not None:
            total += int(x.shape[0])
    return total


def finish_row(model_name, kernel, out, n_iter, pre_s, fit_s, metrics, extra):
    row = {
        "model": model_name,
        "kernel": kernel,
        "loss": float(out[-1]),
        "fit_iter": int(n_iter),
        "precompute_s": float(pre_s),
        "fit_s": float(fit_s),
        "total_s": float(pre_s + fit_s),
        **{k: v for k, v in metrics.items() if k != "est"},
        **{f"est_{k}": v for k, v in metrics["est"].items()},
        **extra,
    }
    return row


def fit_column_v3(reg_map_ord, ordered_grid_coords_np, initial_vals, truth, args):
    params = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True) for v in initial_vals]
    model = ReverseLColumnVecchiaFitV3(
        smooth=SMOOTH,
        input_map=reg_map_ord,
        mm_cond_number=args.mm_cond_number,
        grid_coords=ordered_grid_coords_np,
        head_right_cols=COLUMN_V3_SPEC["head_right_cols"],
        above_count=COLUMN_V3_SPEC["above_count"],
        right_col_count=COLUMN_V3_SPEC["right_col_count"],
        per_lag_conditioning_count=COLUMN_V3_SPEC["per_lag_conditioning_count"],
        lag_count=COLUMN_V3_SPEC["lag_count"],
        include_lag_self=COLUMN_V3_SPEC["include_lag_self"],
        target_chunk_size=args.column_chunk_size,
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    diag = template_diagnostics(model)
    try:
        diag["n_spatial_templates"] = int(model.expected_spatial_template_count_upper_bound())
    except Exception:
        diag["n_spatial_templates"] = np.nan
    optimizer = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=args.lbfgs_hist)
    t1 = time.time()
    out, n_iter = model.fit_vecc_lbfgs(params, optimizer, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t1
    metrics = calculate_metrics(out, truth)
    return finish_row(
        COLUMN_V3_SPEC["model"],
        "column_reverse_l_v3",
        out,
        n_iter,
        pre_s,
        fit_s,
        metrics,
        {
            "total_conditioning_nominal": COLUMN_V3_SPEC["per_lag_conditioning_count"]
            * (COLUMN_V3_SPEC["lag_count"] + 1),
            "n_heads": int(model.Heads_data.shape[0]),
            "n_tails": int(model.n_tails),
            **diag,
        },
    )


def fit_hybrid_lean(reg_map_ord, nns_grid, ordered_grid_coords_np, initial_vals, truth, args):
    params = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True) for v in initial_vals]
    model = HybridVecchiaFit(
        smooth=SMOOTH,
        input_map=reg_map_ord,
        nns_map=nns_grid,
        mm_cond_number=args.mm_cond_number,
        nheads=0,
        limit_A=HYBRID_SPEC["limit_A"],
        limit_B_local=HYBRID_SPEC["lag1_local_count"],
        limit_C_local=HYBRID_SPEC["lag2_local_count"],
        daily_stride=args.daily_stride,
        spatial_coords=ordered_grid_coords_np,
        lag1_lon_offset=DELTA_LON_BASE * args.lon_factor,
        lag1_fresh_count=HYBRID_SPEC["lag1_fresh_count"],
        lag2_fresh_count=HYBRID_SPEC["lag2_fresh_count"],
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    optimizer = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=args.lbfgs_hist)
    t1 = time.time()
    out, n_iter = model.fit_vecc_lbfgs(params, optimizer, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t1
    metrics = calculate_metrics(out, truth)
    return finish_row(
        HYBRID_SPEC["model"],
        "hybrid_lean",
        out,
        n_iter,
        pre_s,
        fit_s,
        metrics,
        {
            "total_conditioning_nominal": 20 + (1 + 8 + 4) + (1 + 4 + 3),
            "n_heads": int(model.Heads_data.shape[0]),
            "n_tails": hybrid_tail_count(model),
            "n_templates": np.nan,
        },
    )


def p90_p10(x):
    return x.quantile(0.9) - x.quantile(0.1)


def round_scalar(x, digits=ROUND_DECIMALS):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return x
        return round(float(x), digits)
    return x


def round_mapping(d, digits=ROUND_DECIMALS):
    return {k: round_scalar(v, digits) for k, v in d.items()}


def round_df(df, digits=ROUND_DECIMALS):
    out = df.copy()
    float_cols = out.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].round(digits)
    return out


def save_csv_rounded(df, path):
    round_df(df).to_csv(path, index=False, float_format=f"%.{ROUND_DECIMALS}f")


def make_model_summary(df):
    rows = []
    for model, g in df.groupby("model", sort=False):
        row = {
            "model": model,
            "kernel": g["kernel"].iloc[-1],
            "n": int(g.shape[0]),
            "total_conditioning": float(g["total_conditioning_nominal"].median()),
            "n_templates_mean": float(g["n_templates"].mean()) if "n_templates" in g else np.nan,
            "loss_mean": float(g["loss"].mean()),
            "loss_p90_p10": float(p90_p10(g["loss"])),
            "overall_rmsre_mean": float(g["overall_rmsre"].mean()),
            "overall_rmsre_p90_p10": float(p90_p10(g["overall_rmsre"])),
            "spatial_rmsre_mean": float(g["spatial_rmsre"].mean()),
            "advec_rmsre_mean": float(g["advec_rmsre"].mean()),
            "total_s_mean": float(g["total_s"].mean()),
            "fit_s_mean": float(g["fit_s"].mean()),
            "precompute_s_mean": float(g["precompute_s"].mean()),
            "fits_per_hour_mean": float(3600.0 / g["total_s"].mean()) if g["total_s"].mean() > 0 else np.nan,
            "fit_only_per_hour_mean": float(3600.0 / g["fit_s"].mean()) if g["fit_s"].mean() > 0 else np.nan,
            "total_s_p90_p10": float(p90_p10(g["total_s"])),
            "fit_s_p90_p10": float(p90_p10(g["fit_s"])),
        }
        for p in P_LABELS:
            row[f"{p}_re_mean"] = float(g[f"{p}_re"].mean())
            row[f"{p}_re_p90_p10"] = float(p90_p10(g[f"{p}_re"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["overall_rmsre_mean", "loss_mean"]).reset_index(drop=True)


def make_param_summary(df):
    rows = []
    for model, g in df.groupby("model", sort=False):
        for p in P_LABELS:
            re = g[f"{p}_re"]
            rows.append(
                {
                    "model": model,
                    "parameter": p,
                    "rmsre": float(np.sqrt(np.mean(np.square(re)))),
                    "mean_re": float(re.mean()),
                    "median_re": float(re.median()),
                    "p10_re": float(re.quantile(0.1)),
                    "p90_re": float(re.quantile(0.9)),
                    "p90_p10_re": float(p90_p10(re)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "parameter"]).reset_index(drop=True)


def parse_range(s):
    a, b = [float(x.strip()) for x in s.split(",")]
    return (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lat-range", type=parse_range, default="-3,2")
    parser.add_argument("--lon-range", type=parse_range, default="121,131")
    parser.add_argument("--lat-factor", type=int, default=1)
    parser.add_argument("--lon-factor", type=int, default=1)
    parser.add_argument("--mm-cond-number", type=int, default=100)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--init-noise", type=float, default=0.25)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-hist", type=int, default=10)
    parser.add_argument("--column-chunk-size", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jl2815/tco/exercise_output/estimates/day"))
    parser.add_argument("--out-prefix", type=str, default="sim_vecchia_highres_colver3_vs_hybrid_050726")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / f"{args.out_prefix}_raw.csv"
    model_csv = args.out_dir / f"{args.out_prefix}_model_summary.csv"
    param_csv = args.out_dir / f"{args.out_prefix}_param_summary.csv"

    print("SRC:", SRC, flush=True)
    print("DEVICE:", DEVICE, flush=True)
    print("torch:", torch.__version__, flush=True)
    print("args:", vars(args), flush=True)
    print("truth:", round_mapping(TRUE_DICT), flush=True)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    lats_grid, lons_grid, grid_coords, delta_lat, delta_lon = build_target_grid(
        args.lat_range, args.lon_range, args.lat_factor, args.lon_factor
    )
    n_lat, n_lon = len(lats_grid), len(lons_grid)
    print(f"Grid: {n_lat} x {n_lon} x {T_STEPS} = {n_lat * n_lon * T_STEPS:,} rows", flush=True)
    print(f"delta_lat={delta_lat}, delta_lon={delta_lon}", flush=True)

    ord_grid, nns_grid = compute_grid_ordering(grid_coords, args.mm_cond_number)
    ordered_grid_coords_np = grid_coords[ord_grid].detach().cpu().numpy()
    print("Maxmin ordering done", flush=True)

    true_log = true_to_log_params(TRUE_DICT)
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)
    rows = []

    fitters = [
        ("column_v3", fit_column_v3),
        ("hybrid_lean", fit_hybrid_lean),
    ]

    for iter_idx in range(args.num_iters):
        iter_seed = args.seed + iter_idx
        set_seed(iter_seed)
        initial_vals = make_random_init(rng, true_log, args.init_noise)
        print("\n" + "=" * 100, flush=True)
        print(f"Iteration {iter_idx + 1}/{args.num_iters} | seed={iter_seed}", flush=True)
        print("Initial vals:", [round(x, 4) for x in initial_vals], flush=True)

        field = generate_field_values(n_lat, n_lon, T_STEPS, true_params, delta_lat, delta_lon)
        reg_map = assemble_reg_map(field, grid_coords, true_params)
        assert_complete_regular_grid(reg_map)
        del field
        reg_map_ord = {k: v[ord_grid].contiguous() for k, v in reg_map.items()}
        del reg_map

        for fit_name, fitter in fitters:
            print("\n" + "-" * 80, flush=True)
            print(f"Fitting {fit_name}", flush=True)
            try:
                if fit_name == "hybrid_lean":
                    row = fitter(reg_map_ord, nns_grid, ordered_grid_coords_np, initial_vals, TRUE_DICT, args)
                else:
                    row = fitter(reg_map_ord, ordered_grid_coords_np, initial_vals, TRUE_DICT, args)
                row.update({"iter": iter_idx, "seed": iter_seed})
                row.update({f"true_{k}": v for k, v in TRUE_DICT.items()})
                rows.append(row)
                print("RESULT:", round_mapping(row), flush=True)
            except Exception as exc:
                err_row = {
                    "iter": iter_idx,
                    "seed": iter_seed,
                    "model": fit_name,
                    "kernel": fit_name,
                    "error": repr(exc),
                }
                rows.append(err_row)
                print("ERROR:", err_row, flush=True)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        df = pd.DataFrame(rows)
        save_csv_rounded(df, raw_csv)
        ok = df[df["loss"].notna()].copy() if "loss" in df else pd.DataFrame()
        if not ok.empty:
            model_summary = make_model_summary(ok)
            param_summary = make_param_summary(ok)
            save_csv_rounded(model_summary, model_csv)
            save_csv_rounded(param_summary, param_csv)
            print("\nRunning model summary", flush=True)
            print(round_df(model_summary).to_string(index=False), flush=True)
            print("\nRunning parameter summary", flush=True)
            print(round_df(param_summary).to_string(index=False), flush=True)
            print("Saved:", raw_csv, model_csv, param_csv, flush=True)

        del reg_map_ord
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
