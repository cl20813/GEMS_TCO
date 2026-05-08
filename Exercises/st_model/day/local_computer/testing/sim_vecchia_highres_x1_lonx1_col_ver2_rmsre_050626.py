import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.fft


LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
AMAREL_SRC = "/home/jl2815/tco"
SRC = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, SRC)

from GEMS_TCO.kernel_vecchia_col_ver2 import ReverseLColumnVecchiaFitV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

RES_LAT_FACTOR = 1
RES_LON_FACTOR = 1
DELTA_LAT = 0.044 * RES_LAT_FACTOR
DELTA_LON = 0.063 * RES_LON_FACTOR
LAT_RANGE = (-3.0, 2.0)
LON_RANGE = (121.0, 131.0)
T_STEPS = 8
SEED = 123

SMOOTH = 0.5
MM_COND_NUMBER = 100
LBFGS_STEPS = 5
LBFGS_EVAL = 20
LBFGS_HIST = 10
INIT_NOISE = 0.25

TRUE_DICT = {
    "sigmasq": 10.0,
    "range_lat": 0.30,
    "range_lon": 0.40,
    "range_time": 2.0,
    "advec_lat": 0.08,
    "advec_lon": -0.16,
    "nugget": 2.5,
}

COLUMN_V2_SPEC = {
    "model": "ColumnV2_Up3_Right3_Down50_Lag2",
    "head_right_cols": 3,
    "above_count": 3,
    "right_col_count": 3,
    "per_lag_conditioning_count": 50,
    "lag_count": 2,
    "include_lag_self": False,
}

OUT_CSV = Path(
    "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/testing/log/"
    "sim_vecchia_highres_x1_lonx1_col_ver2_rmsre_050626.csv"
)

P_LABELS = ["sigmasq", "range_lat", "range_lon", "range_time", "advec_lat", "advec_lon", "nugget"]
SPATIAL_KEYS = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]


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
        row[f"{p}_re"] = abs(est[p] - truth[p]) / abs(truth[p])
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


def build_target_grid(lat_range, lon_range):
    lat0, lat1 = float(min(lat_range)), float(max(lat_range))
    lon0, lon1 = float(min(lon_range)), float(max(lon_range))
    n_lat = int(np.floor((lat1 - lat0) / DELTA_LAT + 1e-9)) + 1
    n_lon = int(np.floor((lon1 - lon0) / DELTA_LON + 1e-9)) + 1
    lats = lat0 + torch.arange(n_lat, device=DEVICE, dtype=DTYPE) * DELTA_LAT
    lons = lon0 + torch.arange(n_lon, device=DEVICE, dtype=DTYPE) * DELTA_LON
    lats = torch.round(lats * 10000) / 10000
    lons = torch.round(lons * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats, lons, indexing="ij")
    grid_coords = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    return lats, lons, grid_coords


def generate_field_values(n_lat, n_lon, t_steps, params):
    cpu = torch.device("cpu")
    f32 = torch.float32
    px, py, pt = 2 * n_lat, 2 * n_lon, 2 * t_steps
    lx = torch.arange(px, device=cpu, dtype=f32) * DELTA_LAT
    lx[px // 2 :] -= px * DELTA_LAT
    ly = torch.arange(py, device=cpu, dtype=f32) * DELTA_LON
    ly[py // 2 :] -= py * DELTA_LON
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


def build_model(reg_map, grid_coords_np):
    return ReverseLColumnVecchiaFitV2(
        smooth=SMOOTH,
        input_map=reg_map,
        mm_cond_number=MM_COND_NUMBER,
        grid_coords=grid_coords_np,
        head_right_cols=COLUMN_V2_SPEC["head_right_cols"],
        above_count=COLUMN_V2_SPEC["above_count"],
        right_col_count=COLUMN_V2_SPEC["right_col_count"],
        per_lag_conditioning_count=COLUMN_V2_SPEC["per_lag_conditioning_count"],
        lag_count=COLUMN_V2_SPEC["lag_count"],
        include_lag_self=COLUMN_V2_SPEC["include_lag_self"],
        target_chunk_size=1024 if DEVICE.type == "cpu" else 4096,
    )


def fit_column_v2(reg_map, grid_coords_np, initial_vals, truth, precompute_only=False):
    params = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True) for v in initial_vals]
    model = build_model(reg_map, grid_coords_np)

    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0

    diag = template_diagnostics(model)
    try:
        diag["n_spatial_templates"] = int(model.expected_spatial_template_count_upper_bound())
    except Exception as exc:
        print("spatial template diagnostic failed:", repr(exc))
        diag["n_spatial_templates"] = np.nan
    print("COLUMN V2 DIAGNOSTICS:", diag)

    if precompute_only:
        return {
            "model": COLUMN_V2_SPEC["model"],
            "kernel": "column_reverse_l_v2",
            "total_conditioning_nominal": COLUMN_V2_SPEC["per_lag_conditioning_count"]
            * (COLUMN_V2_SPEC["lag_count"] + 1),
            "n_heads": int(model.Heads_data.shape[0]),
            "n_tails": int(model.n_tails),
            "precompute_s": pre_s,
            "fit_s": np.nan,
            "total_s": pre_s,
            **diag,
        }

    optimizer = model.set_optimizer(
        params,
        lr=1.0,
        max_iter=LBFGS_EVAL,
        max_eval=LBFGS_EVAL,
        history_size=LBFGS_HIST,
    )
    t1 = time.time()
    out, n_iter = model.fit_vecc_lbfgs(params, optimizer, max_steps=LBFGS_STEPS, grad_tol=1e-5)
    fit_s = time.time() - t1

    metrics = calculate_metrics(out, truth)
    row = {
        "model": COLUMN_V2_SPEC["model"],
        "kernel": "column_reverse_l_v2",
        "total_conditioning_nominal": COLUMN_V2_SPEC["per_lag_conditioning_count"]
        * (COLUMN_V2_SPEC["lag_count"] + 1),
        "n_heads": int(model.Heads_data.shape[0]),
        "n_tails": int(model.n_tails),
        "loss": float(out[-1]),
        "fit_iter": int(n_iter),
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        **diag,
        **{k: v for k, v in metrics.items() if k != "est"},
        **{f"est_{k}": v for k, v in metrics["est"].items()},
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="High-res simulation for Column V2 only.")
    parser.add_argument("--precompute-only", action="store_true", help="Only build templates and write diagnostics.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    print("SRC:", SRC)
    print("DEVICE:", DEVICE)
    print("torch:", torch.__version__)
    print("resolution:", DELTA_LAT, DELTA_LON)
    print("true:", TRUE_DICT)
    print("column_v2:", COLUMN_V2_SPEC)
    print("out:", args.out_csv)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    lats_grid, lons_grid, grid_coords = build_target_grid(LAT_RANGE, LON_RANGE)
    n_lat, n_lon = len(lats_grid), len(lons_grid)
    print(f"Grid: {n_lat} x {n_lon} x {T_STEPS} = {n_lat * n_lon * T_STEPS:,} rows")

    true_log = true_to_log_params(TRUE_DICT)
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)
    initial_vals = make_random_init(rng, true_log, INIT_NOISE)
    print("Initial vals:", [round(x, 4) for x in initial_vals])

    field = generate_field_values(n_lat, n_lon, T_STEPS, true_params)
    reg_map = assemble_reg_map(field, grid_coords, true_params)
    del field
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    row = fit_column_v2(
        reg_map,
        grid_coords.detach().cpu().numpy(),
        initial_vals,
        TRUE_DICT,
        precompute_only=args.precompute_only,
    )
    row.update({f"true_{k}": v for k, v in TRUE_DICT.items()})

    df = pd.DataFrame([row])
    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print(df.to_string(index=False))

    if not args.precompute_only:
        param_rows = []
        for p in P_LABELS:
            est = row[f"est_{p}"]
            true = row[f"true_{p}"]
            param_rows.append(
                {
                    "parameter": p,
                    "estimate": est,
                    "truth": true,
                    "relative_error": abs(est - true) / abs(true),
                }
            )
        print(pd.DataFrame(param_rows).to_string(index=False))


if __name__ == "__main__":
    main()
