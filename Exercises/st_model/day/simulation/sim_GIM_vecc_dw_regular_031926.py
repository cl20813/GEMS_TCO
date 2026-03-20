import sys
import gc
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import typer
from pathlib import Path
from typing import List

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

# --- GLOBAL SETTINGS ---
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE    = torch.float64
NUM_SIMS = 100
DELTA_LAT = 0.044
DELTA_LON = 0.063


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Covariance & FFT simulation ──────────────────────────────────────────────

def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lags_x - params[4] * lags_t
    u_lon = lags_y - params[5] * lags_t
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lags_t.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def generate_regular_data(params_tensor, grid_config):
    lats, lons, t_def = grid_config['lats'], grid_config['lons'], grid_config['t_def']
    Nx, Ny, Nt = len(lats), len(lons), t_def
    dlat = float(lats[1] - lats[0]) if Nx > 1 else DELTA_LAT
    dlon = float(lons[1] - lons[0]) if Ny > 1 else DELTA_LON
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt

    lx = torch.arange(Px, device=DEVICE, dtype=DTYPE) * dlat; lx[Px//2:] -= Px*dlat
    ly = torch.arange(Py, device=DEVICE, dtype=DTYPE) * dlon; ly[Py//2:] -= Py*dlon
    lt = torch.arange(Pt, device=DEVICE, dtype=DTYPE);        lt[Pt//2:] -= Pt

    L_x, L_y, L_t = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_model_covariance_on_grid(L_x, L_y, L_t, params_tensor)
    S = torch.fft.fftn(C); S.real = torch.clamp(S.real, min=0)
    field = torch.fft.ifftn(torch.sqrt(S.real) * torch.fft.fftn(
        torch.randn(Px, Py, Pt, device=DEVICE, dtype=DTYPE))).real[:Nx, :Ny, :Nt]

    nugget_std = torch.sqrt(torch.exp(params_tensor[6]))
    grid_lat, grid_lon = torch.meshgrid(lats, lons, indexing='ij')
    flat_lats, flat_lons = grid_lat.flatten(), grid_lon.flatten()

    input_map, agg_list = {}, []
    for t in range(t_def):
        obs = field[:, :, t].flatten() + torch.randn_like(flat_lats) * nugget_std + grid_config['mean']
        row = torch.stack([flat_lats, flat_lons, obs,
                           torch.full_like(flat_lats, 21.0 + t)], dim=1).detach()
        input_map[f't_{t:02d}'] = row
        agg_list.append(row)
    return input_map, torch.cat(agg_list, dim=0)


# ── Utilities ────────────────────────────────────────────────────────────────

def transform_raw_to_log_phi(raw: list) -> list:
    sigmasq, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget = raw
    phi2 = 1.0 / max(range_lon, 1e-4)
    phi1 = sigmasq * phi2
    phi3 = (range_lon / max(range_lat, 1e-4)) ** 2
    phi4 = (range_lon / max(range_time, 1e-4)) ** 2
    return [math.log(phi1), math.log(phi2), math.log(phi3), math.log(phi4),
            advec_lat, advec_lon, math.log(max(nugget, 1e-8))]


def transform_log_phi_to_physical(p: torch.Tensor) -> torch.Tensor:
    phi1, phi2, phi3, phi4 = (torch.exp(p[i]) for i in range(4))
    rlon = 1.0 / phi2
    return torch.stack([phi1/phi2, rlon/torch.sqrt(phi3), rlon,
                        rlon/torch.sqrt(phi4), p[4], p[5], torch.exp(p[6])])


# ── Main CLI ──────────────────────────────────────────────────────────────────

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    start_day: int = typer.Option(..., help="Start day (1-based)"),
    end_day:   int = typer.Option(..., help="End day (1-based, inclusive)"),
    v: float = typer.Option(0.5, help="Matern smoothness"),
    mm_cond_number: int = typer.Option(30, help="Vecchia neighbors"),
    nheads: int = typer.Option(1000, help="Head points"),
    limit_a: int = typer.Option(16, help="Set A neighbors"),
    limit_b: int = typer.Option(16, help="Set B neighbors"),
    limit_c: int = typer.Option(16, help="Set C neighbors"),
    daily_stride: int = typer.Option(2, help="Set C stride"),
    years: List[str] = typer.Option(['2022,2024,2025'], help="Comma-separated years"),
    month: int = typer.Option(7, help="Month"),
    num_sims: int = typer.Option(100, help="GIM bootstrap samples"),
) -> None:

    set_seed(2025)
    years_list = [y.strip() for y in years[0].split(',')]
    print(f"Device: {DEVICE}")
    print(f"Years: {years_list}, Month: {month}, Days: {start_day}-{end_day}")

    output_path = Path(config.amarel_estimates_day_path) / "GIM"
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load fitted estimates ─────────────────────────────────────────────────
    est_path = Path(config.amarel_estimates_day_path)
    # Expects real_dw_*.json and real_vecc_*.json already produced
    dw_files   = sorted(est_path.glob("real_dw_summary_LBFGS_*.json"))
    vecc_files = sorted(est_path.glob("real_vecc_*.json"))
    if not dw_files or not vecc_files:
        print("[Error] Could not find fitted estimate files. Run fit scripts first.")
        raise SystemExit(1)

    import json
    dw_records   = json.loads(dw_files[-1].read_text())
    vecc_records = json.loads(vecc_files[-1].read_text())

    # Index by day string for lookup
    dw_by_day   = {r['day']: r['params'][:7] for r in dw_records}
    vecc_by_day = {r['day']: r['params'][:7] for r in vecc_records}

    data_load_instance = load_data_dynamic_processed(config.amarel_data_load_path)

    p_names = ["SigmaSq", "RangeLat", "RangeLon", "RangeTime",
               "AdvecLat", "AdvecLon", "Nugget"]

    for year in years_list:
        print(f'\n{"="*60}')
        print(f'=== Year {year} ===')
        print(f'{"="*60}')

        out_file = output_path / f"GIM_{year}_{month:02d}_days{start_day}_to_{end_day}.csv"

        df_map, ord_mm, nns_map, monthly_mean = data_load_instance.load_maxmin_ordered_data_bymonthyear(
            mm_cond_number=mm_cond_number,
            years_=[year], months_=[month],
            lat_range=[-3, 2], lon_range=[121, 131],
            is_whittle=False
        )

        results = []
        dwl = debiased_whittle.debiased_whittle_likelihood()

        for day in range(start_day - 1, end_day):
            day_str = f"{year}-{month:02d}-{day+1}"
            if day_str not in dw_by_day or day_str not in vecc_by_day:
                print(f"[Skip] No estimates found for {day_str}")
                continue

            t0 = time.time()
            gc.collect(); torch.cuda.empty_cache()
            print(f"\n>>> {day_str}")

            hour_indices = [day * 8, (day + 1) * 8]

            # ── DW GIM ───────────────────────────────────────────────────────
            print("  [1/2] DW GIM...")
            dw_map, dw_agg = data_load_instance.load_working_data(
                df_map, monthly_mean, hour_indices,
                ord_mm=None, dtype=DTYPE, keep_ori=False
            )
            real_agg_dw = dw_agg.to(DEVICE)

            grid_cfg = {
                'lats': torch.unique(real_agg_dw[:, 0]),
                'lons': torch.unique(real_agg_dw[:, 1]),
                't_def': int(torch.unique(real_agg_dw[:, 3]).shape[0]),
                'mean': float(monthly_mean)
            }

            dw_log_phi = torch.tensor(
                transform_raw_to_log_phi(dw_by_day[day_str]),
                device=DEVICE, dtype=DTYPE, requires_grad=True
            )

            db = debiased_whittle.debiased_whittle_preprocess(
                [real_agg_dw], [dw_map], day_idx=0,
                params_list=dw_by_day[day_str],
                lat_range=[-3, 2], lon_range=[121.0, 131.0]
            )
            cur_df = db.generate_spatially_filtered_days(-3, 2, 121, 131).to(DEVICE)
            unique_times = torch.unique(cur_df[:, 3])
            time_slices  = [cur_df[cur_df[:, 3] == t] for t in unique_times]

            J_vec, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered(
                time_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE)
            I_obs   = dwl.calculate_sample_periodogram_vectorized(J_vec)
            t_auto  = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

            def nll_dw(p):
                loss = dwl.whittle_likelihood_loss_tapered(
                    p, I_obs, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
                return loss[0] if isinstance(loss, tuple) else loss

            H_dw     = torch.autograd.functional.hessian(nll_dw, dw_log_phi)
            H_inv_dw = torch.linalg.inv(H_dw + torch.eye(7, device=DEVICE) * 1e-5)

            grads_dw = []
            for _ in range(num_sims):
                with torch.no_grad():
                    _, b_agg = generate_regular_data(dw_log_phi, grid_cfg)
                b_slices = [b_agg[b_agg[:, 3] == t] for t in unique_times]
                J_b, bn1, bn2, _, bt = dwl.generate_Jvector_tapered(
                    b_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE)
                I_b = dwl.calculate_sample_periodogram_vectorized(J_b)
                ta  = t_auto if (bn1==n1 and bn2==n2) else \
                      dwl.calculate_taper_autocorrelation_fft(bt, bn1, bn2, DEVICE)
                if dw_log_phi.grad is not None: dw_log_phi.grad.zero_()
                loss = dwl.whittle_likelihood_loss_tapered(
                    dw_log_phi, I_b, bn1, bn2, p_time, ta, DELTA_LAT, DELTA_LON)
                (loss[0] if isinstance(loss, tuple) else loss).backward()
                grads_dw.append(dw_log_phi.grad.detach().clone())
                del b_agg, b_slices, J_b, I_b, loss

            J_dw    = torch.stack(grads_dw).T @ torch.stack(grads_dw) / num_sims
            GIM_dw  = H_inv_dw @ J_dw @ H_inv_dw
            Jac_dw  = torch.autograd.functional.jacobian(transform_log_phi_to_physical, dw_log_phi)
            SE_dw   = torch.sqrt(torch.diag(Jac_dw @ GIM_dw @ Jac_dw.T)).detach().cpu().numpy()
            Pt_dw   = transform_log_phi_to_physical(dw_log_phi).detach().cpu().numpy()

            del real_agg_dw, dw_map, H_dw, J_dw, GIM_dw, grads_dw
            gc.collect(); torch.cuda.empty_cache()

            # ── Vecchia GIM ───────────────────────────────────────────────────
            print("  [2/2] Vecchia GIM...")
            vecc_map, _ = data_load_instance.load_working_data(
                df_map, monthly_mean, hour_indices,
                ord_mm=ord_mm, dtype=DTYPE, keep_ori=True
            )
            real_map_vecc = {k: v.to(DEVICE) for k, v in vecc_map.items()}

            vc_log_phi = torch.tensor(
                transform_raw_to_log_phi(vecc_by_day[day_str]),
                device=DEVICE, dtype=DTYPE, requires_grad=True
            )

            model_vc = kernels_vecchia.fit_vecchia_lbfgs(
                smooth=v, input_map=real_map_vecc,
                nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nheads,
                limit_A=limit_a, limit_B=limit_b, limit_C=limit_c, daily_stride=daily_stride
            )
            model_vc.precompute_conditioning_sets()

            def nll_vc(p): return model_vc.vecchia_batched_likelihood(p)

            H_vc     = torch.autograd.functional.hessian(nll_vc, vc_log_phi)
            H_inv_vc = torch.linalg.inv(H_vc + torch.eye(7, device=DEVICE) * 1e-5)

            grads_vc = []
            for _ in range(num_sims):
                with torch.no_grad():
                    b_map, _ = generate_regular_data(vc_log_phi, grid_cfg)
                model_vc.input_map = {k: b_map[k][ord_mm] for k in b_map}
                model_vc.precompute_conditioning_sets()
                if vc_log_phi.grad is not None: vc_log_phi.grad.zero_()
                model_vc.vecchia_batched_likelihood(vc_log_phi).backward()
                grads_vc.append(vc_log_phi.grad.detach().clone())
                del b_map

            J_vc    = torch.stack(grads_vc).T @ torch.stack(grads_vc) / num_sims
            GIM_vc  = H_inv_vc @ J_vc @ H_inv_vc
            Jac_vc  = torch.autograd.functional.jacobian(transform_log_phi_to_physical, vc_log_phi)
            SE_vc   = torch.sqrt(torch.diag(Jac_vc @ GIM_vc @ Jac_vc.T)).detach().cpu().numpy()
            Pt_vc   = transform_log_phi_to_physical(vc_log_phi).detach().cpu().numpy()

            # ── Save ─────────────────────────────────────────────────────────
            row = {"Year": year, "Month": month, "Day": day + 1}
            for k, name in enumerate(p_names):
                row[f"DW_Est_{name}"] = round(float(Pt_dw[k]), 4)
                row[f"DW_SE_{name}"]  = round(float(SE_dw[k]), 4)
                row[f"VC_Est_{name}"] = round(float(Pt_vc[k]), 4)
                row[f"VC_SE_{name}"]  = round(float(SE_vc[k]), 4)
            results.append(row)
            pd.DataFrame(results).to_csv(out_file, index=False)

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s → {out_file.name}")
            print(f"  {'Param':<10} | {'DW Est':>8} | {'DW SE':>8} || {'VC Est':>8} | {'VC SE':>8}")
            for k, name in enumerate(p_names):
                print(f"  {name:<10} | {Pt_dw[k]:>8.4f} | {SE_dw[k]:>8.4f} || "
                      f"{Pt_vc[k]:>8.4f} | {SE_vc[k]:>8.4f}")

            del real_map_vecc, vecc_map, model_vc, H_vc, J_vc, GIM_vc, grads_vc
            gc.collect(); torch.cuda.empty_cache()

    print("\n[Done] All GIM results saved.")


if __name__ == "__main__":
    app()
