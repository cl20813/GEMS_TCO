"""
sim_vecchia_advec_comparison_042826.py

Direct regular-grid simulation comparing:
  1. Vecc_Std   : existing kernels_vecchia.fit_vecchia_lbfgs with one extra
                  standard temporal neighbor per lag
  2. Vecc_Advec : kernels_vecchia_advec.fit_vecchia_lbfgs_advec

No DW baseline and no high-resolution observation remapping.  The simulated
field is generated directly on the target 0.044 x 0.063 regular grid.

Advection-aware conditioning:
  t      : existing spatial neighbors
  t-1    : same loc + nearest point at lon + 0.063*2
           + local neighbors around that upstream point
  t-2    : same loc + nearest point at lon + 0.063*2*2
           + local neighbors around that upstream point

Fairness control:
  Vecc_Std uses same-location-centered temporal neighbors with limit_B + 1
  and limit_C + 1.  Vecc_Advec uses the same lag-wise budget, but replaces
  those extra temporal neighbors with upstream-centered neighborhoods.

Example:
  conda activate faiss_env
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation
  python sim_vecchia_advec_comparison_042826.py --num-iters 1
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.fft
import typer

AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
_src = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, _src)

from GEMS_TCO import configuration as config
from GEMS_TCO import kernels_vecchia
from GEMS_TCO import kernels_vecchia_advec
from GEMS_TCO import orderings as _orderings

is_amarel = os.path.exists(config.amarel_data_load_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
T_STEPS = 8

MODELS = ["Vecc_Std", "Vecc_Advec"]
P_LABELS = ["sigmasq", "range_lat", "range_lon", "range_time",
            "advec_lat", "advec_lon", "nugget"]
P_COLS = ["sigmasq_est", "range_lat_est", "range_lon_est", "range_t_est",
          "advec_lat_est", "advec_lon_est", "nugget_est"]

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def build_target_grid(lat_range, lon_range):
    lats = torch.arange(min(lat_range), max(lat_range) + 0.0001,
                        DELTA_LAT_BASE, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0], lon_range[1] + 0.0001,
                        DELTA_LON_BASE, device=DEVICE, dtype=DTYPE)
    lats = torch.round(lats * 10000) / 10000
    lons = torch.round(lons * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats, lons, indexing="ij")
    grid_coords = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    return lats, lons, grid_coords


def generate_field_values(n_lat, n_lon, t_steps, params):
    """Circulant embedding field generated directly at target grid resolution."""
    cpu = torch.device("cpu")
    f32 = torch.float32
    px, py, pt = 2 * n_lat, 2 * n_lon, 2 * t_steps

    lx = torch.arange(px, device=cpu, dtype=f32) * DELTA_LAT_BASE
    lx[px // 2:] -= px * DELTA_LAT_BASE
    ly = torch.arange(py, device=cpu, dtype=f32) * DELTA_LON_BASE
    ly[py // 2:] -= py * DELTA_LON_BASE
    lt = torch.arange(pt, device=cpu, dtype=f32)
    lt[pt // 2:] -= pt

    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing="ij")
    C = get_covariance_on_grid(Lx, Ly, Lt, params_cpu)
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
        key = f"t{t_idx}"
        t_val = float(t_offset + t_idx)
        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0

        rows = torch.zeros((n_grid, 11), device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords
        rows[:, 2] = field_flat[:, t_idx] + torch.randn(n_grid, device=DEVICE, dtype=DTYPE) * nugget_std
        rows[:, 3] = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(n_grid, -1)
        reg_map[key] = rows.detach()
    return reg_map


def compute_grid_ordering(grid_coords, mm_cond_number):
    coords_np = grid_coords.detach().cpu().numpy()
    ord_mm = _orderings.maxmin_cpp(coords_np)
    nns = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


def true_to_log_params(true_dict):
    phi2 = 1.0 / true_dict["range_lon"]
    phi1 = true_dict["sigmasq"] * phi2
    phi3 = (true_dict["range_lon"] / true_dict["range_lat"]) ** 2
    phi4 = (true_dict["range_lon"] / true_dict["range_time"]) ** 2
    return [
        np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
        true_dict["advec_lat"], true_dict["advec_lon"], np.log(true_dict["nugget"])
    ]


def backmap_params(out_params):
    p = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in out_params[:7]]
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


def calculate_rmsre(out_params, true_dict, zero_thresh=0.01):
    est = backmap_params(out_params)
    e = np.array([est[k] for k in ["sigmasq", "range_lat", "range_lon",
                                   "range_time", "advec_lat", "advec_lon", "nugget"]])
    t = np.array([true_dict[k] for k in ["sigmasq", "range_lat", "range_lon",
                                         "range_time", "advec_lat", "advec_lon", "nugget"]])
    mask = np.abs(t) >= zero_thresh
    rmsre = float(np.sqrt(np.mean(((e[mask] - t[mask]) / np.abs(t[mask])) ** 2)))
    mae_zero = float(np.mean(np.abs(e[~mask] - t[~mask]))) if (~mask).any() else float("nan")
    return rmsre, mae_zero, est


def make_random_init(rng, true_log, init_noise):
    noisy = list(true_log)
    for i in [0, 1, 2, 3, 6]:
        noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
    for i in [4, 5]:
        scale = max(abs(true_log[i]), 0.05)
        noisy[i] = true_log[i] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


def fit_one_model(model_name, reg_map_ord, nns_grid, ordered_grid_coords_np,
                  initial_vals, smooth, mm_cond_number, nheads, limit_a,
                  limit_b, limit_c, daily_stride, advec_lon_offset,
                  lbfgs_lr, lbfgs_eval, lbfgs_hist, lbfgs_steps):
    params = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
              for val in initial_vals]
    if model_name == "Vecc_Std":
        model = kernels_vecchia.fit_vecchia_lbfgs(
            smooth=smooth, input_map=reg_map_ord, nns_map=nns_grid,
            mm_cond_number=mm_cond_number, nheads=nheads,
            limit_A=limit_a, limit_B=limit_b + 1, limit_C=limit_c + 1,
            daily_stride=daily_stride)
    elif model_name == "Vecc_Advec":
        model = kernels_vecchia_advec.fit_vecchia_lbfgs_advec(
            smooth=smooth, input_map=reg_map_ord, nns_map=nns_grid,
            mm_cond_number=mm_cond_number, nheads=nheads,
            limit_A=limit_a, limit_B=limit_b, limit_C=limit_c,
            daily_stride=daily_stride,
            spatial_coords=ordered_grid_coords_np,
            advec_lon_offset=advec_lon_offset)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.precompute_conditioning_sets()
    optimizer = model.set_optimizer(params, lr=lbfgs_lr, max_iter=lbfgs_eval,
                                    history_size=lbfgs_hist)
    t0 = time.time()
    out, n_iter = model.fit_vecc_lbfgs(params, optimizer,
                                       max_steps=lbfgs_steps, grad_tol=1e-5)
    elapsed = time.time() - t0
    loss = float(out[-1])
    return out, loss, int(n_iter), elapsed


@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100, help="Vecchia neighbors used by maxmin nns"),
    nheads: int = typer.Option(0, help="Vecchia head points per time step"),
    limit_a: int = typer.Option(20, help="Set A neighbors"),
    limit_b: int = typer.Option(20, help="Set B neighbors"),
    limit_c: int = typer.Option(20, help="Set C neighbors"),
    daily_stride: int = typer.Option(2, help="Temporal lag for Set C; use 2 for t-2"),
    num_iters: int = typer.Option(300, help="Simulation iterations"),
    lat_range: str = typer.Option("-3,2", help="lat_min,lat_max"),
    lon_range: str = typer.Option("121,131", help="lon_min,lon_max"),
    advec_lon_offset: float = typer.Option(
        DELTA_LON_BASE * 2, help="t-1 upstream longitude offset; default 0.063*2"),
    init_noise: float = typer.Option(0.7, help="Uniform noise half-width in log space"),
    lbfgs_steps: int = typer.Option(5, help="Outer LBFGS steps per model fit"),
    lbfgs_eval: int = typer.Option(20, help="LBFGS max_iter per outer step"),
    lbfgs_hist: int = typer.Option(10, help="LBFGS history size"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    lat_r = [float(x) for x in lat_range.split(",")]
    lon_r = [float(x) for x in lon_range.split(",")]


    # stronger signal / cleaner field sensitivity
    #true_dict = {
    #    "sigmasq": 13.0,
    #    "range_lat": 0.25,
    #    "range_lon": 0.30,
    #    "range_time": 2.5,
    #    "advec_lat": 0.08,
    #    "advec_lon": 0.126,
    #   "nugget": 1.2,
    #}

    true_dict = {
        "sigmasq": 10.0,
        "range_lat": 0.3,
        "range_lon": 0.40,
        "range_time": 2,
        "advec_lat": 0.08,
        "advec_lon": -0.126,
        "nugget": 2.5,
    }


    true_log = true_to_log_params(true_dict)
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)
    true_vals = [true_dict["sigmasq"], true_dict["range_lat"], true_dict["range_lon"],
                 true_dict["range_time"], true_dict["advec_lat"],
                 true_dict["advec_lon"], true_dict["nugget"]]

    print(f"Device       : {DEVICE}")
    print(f"Models       : {MODELS}")
    print(f"Region       : lat {lat_r}, lon {lon_r}")
    print(f"Resolution   : dlat={DELTA_LAT_BASE}, dlon={DELTA_LON_BASE}")
    print(f"Advec offset : t-1 lon + {advec_lon_offset:.4f}, "
          f"t-{daily_stride} lon + {2 * advec_lon_offset:.4f}")
    print(f"True advec   : lat={true_dict['advec_lat']}, lon={true_dict['advec_lon']}")
    print(f"Iterations   : {num_iters}, LBFGS outer steps={lbfgs_steps}")

    output_path = Path(config.amarel_estimates_day_path if is_amarel
                       else config.mac_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%m%d%y")
    csv_raw = f"sim_vecchia_advec_comparison_{date_tag}.csv"
    csv_summary = f"sim_vecchia_advec_summary_{date_tag}.csv"

    print("\n[Setup 1/2] Building target grid...")
    lats_grid, lons_grid, grid_coords = build_target_grid(lat_r, lon_r)
    n_lat, n_lon = len(lats_grid), len(lons_grid)
    n_grid = grid_coords.shape[0]
    print(f"  Grid: {n_lat} lat x {n_lon} lon x {T_STEPS} time = {n_grid * T_STEPS:,} rows")

    print("[Setup 2/2] Computing shared maxmin ordering...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, mm_cond_number)
    ordered_grid_coords_np = grid_coords[ord_grid].detach().cpu().numpy()
    print(f"  Ordering done: N_grid={n_grid}, mm_cond_number={mm_cond_number}")

    records = []
    skipped = 0

    for it in range(num_iters):
        print(f"\n{'=' * 60}")
        print(f"  Iteration {it + 1}/{num_iters}  (skipped: {skipped})")
        print(f"{'=' * 60}")

        initial_vals = make_random_init(rng, true_log, init_noise)
        init_orig = backmap_params(initial_vals)
        print(f"  Init sigmasq={init_orig['sigmasq']:.3f} "
              f"range_lon={init_orig['range_lon']:.3f} "
              f"advec_lon={init_orig['advec_lon']:.3f} "
              f"nugget={init_orig['nugget']:.3f}")

        try:
            field = generate_field_values(n_lat, n_lon, T_STEPS, true_params)
            reg_map = assemble_reg_map(field, grid_coords, true_params)
            del field
            reg_map_ord = {k: v[ord_grid] for k, v in reg_map.items()}

            iter_rows = []
            for model_name in MODELS:
                print(f"--- {model_name} ---")
                out, loss, n_fit_iter, elapsed = fit_one_model(
                    model_name=model_name,
                    reg_map_ord=reg_map_ord,
                    nns_grid=nns_grid,
                    ordered_grid_coords_np=ordered_grid_coords_np,
                    initial_vals=initial_vals,
                    smooth=v,
                    mm_cond_number=mm_cond_number,
                    nheads=nheads,
                    limit_a=limit_a,
                    limit_b=limit_b,
                    limit_c=limit_c,
                    daily_stride=daily_stride,
                    advec_lon_offset=advec_lon_offset,
                    lbfgs_lr=1.0,
                    lbfgs_eval=lbfgs_eval,
                    lbfgs_hist=lbfgs_hist,
                    lbfgs_steps=lbfgs_steps,
                )
                rmsre, mae_zero, est = calculate_rmsre(out, true_dict)
                print(f"  loss={loss:.6f}  RMSRE={rmsre:.4f}  "
                      f"zero-true-MAE={mae_zero:.4f}  ({elapsed:.1f}s)")

                iter_rows.append({
                    "iter": it + 1,
                    "model": model_name,
                    "rmsre": round(rmsre, 6),
                    "mae_zero_true": round(mae_zero, 6) if not np.isnan(mae_zero) else float("nan"),
                    "loss": round(loss, 6),
                    "fit_iter": n_fit_iter,
                    "time_s": round(elapsed, 2),
                    "sigmasq_est": round(est["sigmasq"], 6),
                    "range_lat_est": round(est["range_lat"], 6),
                    "range_lon_est": round(est["range_lon"], 6),
                    "range_t_est": round(est["range_time"], 6),
                    "advec_lat_est": round(est["advec_lat"], 6),
                    "advec_lon_est": round(est["advec_lon"], 6),
                    "nugget_est": round(est["nugget"], 6),
                    "init_sigmasq": round(init_orig["sigmasq"], 4),
                    "init_range_lon": round(init_orig["range_lon"], 4),
                    "init_advec_lon": round(init_orig["advec_lon"], 4),
                })
            records.extend(iter_rows)

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] Iteration {it + 1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)

        n_done = len([r for r in records if r["model"] == MODELS[0]])
        print(f"\n  -- Running summary ({n_done} completed / {it + 1} attempted) --")
        print_summary(records, true_vals)

    print(f"\n{'=' * 60}")
    print(f"  DONE: {len([r for r in records if r['model'] == MODELS[0]])} completed, {skipped} skipped")
    print(f"{'=' * 60}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_raw, index=False)
    summary_rows = print_summary(records, true_vals, final=True)
    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\nSaved:\n  {output_path / csv_raw}\n  {output_path / csv_summary}")


def print_summary(records, true_vals, final=False):
    df = pd.DataFrame(records)
    cw = 10
    title = "FINAL SUMMARY" if final else "Running parameter summary"
    print(f"\n  {title}")
    summary_rows = []

    for m in MODELS:
        sub = df[df["model"] == m]
        print(f"\n  [{m}]")
        print(f"  {'param':<12} {'true':>{cw}}  {'mean':>{cw}}  {'median':>{cw}}  "
              f"{'bias':>{cw}}  {'RMSRE':>{cw}}  {'RMSRE_med':>{cw}}  {'P90-P10':>{cw}}")
        print(f"  {'-' * 94}")

        for lbl, col, tv in zip(P_LABELS, P_COLS, true_vals):
            vals = sub[col].dropna().values
            mean_ = float(np.mean(vals))
            med_ = float(np.median(vals))
            bias_ = mean_ - tv
            p10_ = float(np.percentile(vals, 10))
            p90_ = float(np.percentile(vals, 90))
            p9p1_ = p90_ - p10_
            sr = {
                "model": m,
                "parameter": lbl,
                "true": tv,
                "mean": round(mean_, 6),
                "median": round(med_, 6),
                "bias": round(bias_, 6),
                "sd": round(float(np.std(vals)), 6),
                "P10": round(p10_, 6),
                "P90": round(p90_, 6),
                "P90_P10": round(p9p1_, 6),
            }
            if abs(tv) >= 0.01:
                rmsre_ = float(np.sqrt(np.mean(((vals - tv) / abs(tv)) ** 2)))
                rmsre_m = float(np.median(np.abs((vals - tv) / abs(tv))))
                sr["RMSRE"] = round(rmsre_, 6)
                sr["RMSRE_median"] = round(rmsre_m, 6)
                print(f"  {lbl:<12} {tv:>{cw}.4f}  {mean_:>{cw}.4f}  {med_:>{cw}.4f}  "
                      f"{bias_:>{cw}.4f}  {rmsre_:>{cw}.4f}  {rmsre_m:>{cw}.4f}  {p9p1_:>{cw}.4f}")
            else:
                mae_ = float(np.mean(np.abs(vals - tv)))
                mae_m = float(np.median(np.abs(vals - tv)))
                sr["MAE"] = round(mae_, 6)
                sr["MAE_median"] = round(mae_m, 6)
                print(f"  {lbl:<12} {tv:>{cw}.4f}  {mean_:>{cw}.4f}  {med_:>{cw}.4f}  "
                      f"{bias_:>{cw}.4f}  {'MAE':>{cw}}  {mae_m:>{cw}.4f}  {p9p1_:>{cw}.4f}")
            summary_rows.append(sr)

        rv = sub["rmsre"].dropna().values
        lv = sub["loss"].dropna().values
        tvs = sub["time_s"].dropna().values
        print(f"  {'-' * 94}")
        print(f"  {'Overall':<12} {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
              f"{np.mean(rv):>{cw}.4f}  {np.median(rv):>{cw}.4f}  "
              f"{np.percentile(rv, 90) - np.percentile(rv, 10):>{cw}.4f}")
        print(f"  {'loss':<12} {'':>{cw}}  {np.mean(lv):>{cw}.4f}  {np.median(lv):>{cw}.4f}  "
              f"{'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
              f"{np.percentile(lv, 90) - np.percentile(lv, 10):>{cw}.4f}")
        print(f"  {'time_s':<12} {'':>{cw}}  {np.mean(tvs):>{cw}.4f}  {np.median(tvs):>{cw}.4f}  "
              f"{'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
              f"{np.percentile(tvs, 90) - np.percentile(tvs, 10):>{cw}.4f}")

        summary_rows.append({
            "model": m,
            "parameter": "Overall",
            "true": float("nan"),
            "mean_rmsre": round(float(np.mean(rv)), 6),
            "median_rmsre": round(float(np.median(rv)), 6),
            "rmsre_P90_P10": round(float(np.percentile(rv, 90) - np.percentile(rv, 10)), 6),
            "mean_loss": round(float(np.mean(lv)), 6),
            "median_loss": round(float(np.median(lv)), 6),
            "loss_P90_P10": round(float(np.percentile(lv, 90) - np.percentile(lv, 10)), 6),
            "mean_time_s": round(float(np.mean(tvs)), 6),
            "median_time_s": round(float(np.median(tvs)), 6),
            "time_s_P90_P10": round(float(np.percentile(tvs, 90) - np.percentile(tvs, 10)), 6),
        })

    return summary_rows


if __name__ == "__main__":
    app()
