"""
sim_GIM_vecc_irr_dw_031926.py  —  Amarel (cluster) version

Computes GIM sandwich standard errors for DW and Vecchia-Irr models using
**observed information** for J (no parametric bootstrap):
  - DW  J: per-frequency score outer products  (n1*n2 - 1 terms)
  - VC  J: per-unit conditional score outer products  (N_heads + N_tails terms)

Loops over all days in July for years 2022–2025 (or a subset), producing
a single combined CSV with running and final summaries.

Run via sbatch; see slurm_gim_vecc_irr_dw_031926.md for the job script.

Output
------
[Mean GIM SE]: asymptotic standard errors from the Godambe sandwich estimator —
the primary output. DW SE vs VC SE with DW/VC ratio per parameter.

[Mean estimate ± SD]: descriptive summary of daily parameter estimates.
"""
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
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.float64
DELTA_LAT = 0.044
DELTA_LON = 0.063


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Utilities ─────────────────────────────────────────────────────────────────

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


# ── Finite-difference Hessian ─────────────────────────────────────────────────

def finite_diff_hessian(nll_fn, p, eps=1e-4):
    n = p.shape[0]
    H = torch.zeros(n, n, device=p.device, dtype=p.dtype)
    for i in range(n):
        p_p = p.detach().clone(); p_p[i] += eps; p_p.requires_grad_(True)
        p_m = p.detach().clone(); p_m[i] -= eps; p_m.requires_grad_(True)
        g_p = torch.autograd.grad(nll_fn(p_p), p_p)[0].detach()
        g_m = torch.autograd.grad(nll_fn(p_m), p_m)[0].detach()
        H[i] = (g_p - g_m) / (2.0 * eps)
        torch.cuda.empty_cache()
    return (H + H.T) / 2.0


# ── Summary helpers ───────────────────────────────────────────────────────────

P_NAMES = ["SigmaSq", "RangeLat", "RangeLon", "RangeTime", "AdvecLat", "AdvecLon", "Nugget"]
_EST_COLS_DW  = [f"DW_Est_{p}"  for p in P_NAMES]
_SE_COLS_DW   = [f"DW_SE_{p}"   for p in P_NAMES]
_EST_COLS_VC  = [f"VC_Est_{p}"  for p in P_NAMES]
_SE_COLS_VC   = [f"VC_SE_{p}"   for p in P_NAMES]


def print_running_summary(records: list, n_total: int):
    """Print running summary after each completed day."""
    n_done = len(records)
    print(f"\n  ── Running summary ({n_done} days done / {n_total} total) ──")
    cw = 10

    # ── Mean GIM SE ──────────────────────────────────────────────────────────
    print(f"  {'param':<11} {'DW SE':>{cw}} {'VC SE':>{cw}} {'DW/VC':>{cw}}")
    print(f"  {'-'*45}")
    for p, sc_dw, sc_vc in zip(P_NAMES, _SE_COLS_DW, _SE_COLS_VC):
        dw_se = np.mean([r[sc_dw] for r in records])
        vc_se = np.mean([r[sc_vc] for r in records])
        print(f"  {p:<11} {dw_se:>{cw}.4f} {vc_se:>{cw}.4f} {dw_se/(vc_se+1e-12):>{cw}.4f}")

    # ── Timing ───────────────────────────────────────────────────────────────
    print(f"\n  mean elapsed: {np.mean([r['elapsed'] for r in records]):.1f}s per day")


def print_final_summary(records: list):
    """Print final summary after all days are processed."""
    n_done = len(records)
    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — GIM  ({n_done} days, July 2022–2025)")
    print(f"{'='*75}")
    cw = 10

    # ── Mean GIM SE ──────────────────────────────────────────────────────────
    print(f"\n  [Mean GIM SE per param]")
    print(f"  {'Parameter':<14} {'DW SE':>{cw}} {'VC SE':>{cw}} {'DW/VC':>{cw}}")
    print(f"  {'-'*48}")
    for p, sc_dw, sc_vc in zip(P_NAMES, _SE_COLS_DW, _SE_COLS_VC):
        dw_se = np.mean([r[sc_dw] for r in records])
        vc_se = np.mean([r[sc_vc] for r in records])
        print(f"  {p:<14} {dw_se:>{cw}.4f} {vc_se:>{cw}.4f} {dw_se/(vc_se+1e-12):>{cw}.4f}")

    # ── Timing ───────────────────────────────────────────────────────────────
    print(f"\n  Total elapsed: {sum(r['elapsed'] for r in records)/3600:.2f}h  "
          f"(mean {np.mean([r['elapsed'] for r in records]):.1f}s/day)")


def build_summary_csv(records: list) -> list:
    """Return rows for the summary CSV (mean GIM SE per param, DW vs VC)."""
    rows = []
    for p, sc_dw, sc_vc in zip(P_NAMES, _SE_COLS_DW, _SE_COLS_VC):
        dw_mean_se = float(np.mean([r[sc_dw] for r in records]))
        vc_mean_se = float(np.mean([r[sc_vc] for r in records]))
        rows.append({
            'parameter':      p,
            'dw_mean_se':     round(dw_mean_se, 6),
            'vc_mean_se':     round(vc_mean_se, 6),
            'dw_vc_se_ratio': round(dw_mean_se / (vc_mean_se + 1e-12), 6),
        })
    return rows


# ── Main CLI ───────────────────────────────────────────────────────────────────

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    years:          str   = typer.Option("2022,2023,2024,2025", help="Comma-separated years"),
    days:           str   = typer.Option("1,28",   help="Day range 'start,end' (1-based, inclusive)"),
    month:          int   = typer.Option(7,         help="Month"),
    v:              float = typer.Option(0.5,       help="Matern smoothness"),
    mm_cond_number: int   = typer.Option(100,       help="Vecchia neighbors"),
    nheads:         int   = typer.Option(0,          help="Head points"),
    limit_a:        int   = typer.Option(20,        help="Set A neighbors"),
    limit_b:        int   = typer.Option(20,        help="Set B neighbors"),
    limit_c:        int   = typer.Option(20,        help="Set C neighbors"),
    daily_stride:   int   = typer.Option(2,         help="Set C stride"),
) -> None:

    set_seed(2025)
    years_list = [y.strip() for y in years.split(',')]
    day_start, day_end = [int(x) for x in days.split(',')]
    n_total = len(years_list) * (day_end - day_start + 1)

    print(f"Device : {DEVICE}")
    print(f"Years  : {years_list}  |  Days {day_start}–{day_end}  |  ~{n_total} day-jobs")
    print(f"Observed J — no bootstrap")

    output_path = Path(config.amarel_estimates_day_path) / "GIM"
    output_path.mkdir(parents=True, exist_ok=True)
    out_raw     = output_path / "GIM_all_july_22_23_24_25_obsJ.csv"
    out_summary = output_path / "GIM_summary_july_22_23_24_25_obsJ.csv"

    # ── Load fitted estimates (all years, both methods) ───────────────────────
    est_path = Path(config.amarel_estimates_day_path) / "july_22_23_24_25"
    dw_csv   = est_path / "real_dw_july_22_23_24_25.csv"
    vecc_csv = est_path / "real_vecc_july_22_23_24_25_h0_mm100.csv"
    if not dw_csv.exists() or not vecc_csv.exists():
        print(f"[Error] Missing estimate CSVs in {est_path}")
        raise SystemExit(1)

    _param_cols = ['sigma', 'range_lat', 'range_lon', 'range_time',
                   'advec_lat', 'advec_lon', 'nugget']
    def _norm_day(d: str) -> str:
        """Normalize 'YYYY-M-D' or 'YYYY-MM-DD' → 'YYYY-MM-DD' (zero-padded)."""
        parts = str(d).split('-')
        return f'{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}'

    dw_df   = pd.read_csv(dw_csv)
    vecc_df = pd.read_csv(vecc_csv)
    dw_by_day   = {_norm_day(row['day']): [row[c] for c in _param_cols] for _, row in dw_df.iterrows()}
    vecc_by_day = {_norm_day(row['day']): [row[c] for c in _param_cols] for _, row in vecc_df.iterrows()}
    print(f"Loaded estimate CSVs — {len(dw_by_day)} DW days, {len(vecc_by_day)} VC days available")

    _eps = 1e-5
    dwl  = debiased_whittle.debiased_whittle_likelihood()
    data_load_instance = load_data_dynamic_processed(config.amarel_data_load_path)

    records = []
    n_skip  = 0

    # ── Main loop: year → day ─────────────────────────────────────────────────
    for year_str in years_list:
        print(f"\n{'#'*65}")
        print(f"  Loading spatial data for {year_str}-{month:02d}...")
        print(f"{'#'*65}")

        df_map_yr, ord_mm, nns_map, monthly_mean_yr = \
            data_load_instance.load_maxmin_ordered_data_bymonthyear(
                mm_cond_number=mm_cond_number,
                years_=[year_str], months_=[month],
                lat_range=[-3, 2], lon_range=[121, 131],
                is_whittle=False,
            )

        for day in range(day_start, day_end + 1):
            day_str  = f"{year_str}-{month:02d}-{day:02d}"
            day_idx  = day - 1
            n_done   = len(records)

            print(f"\n{'='*65}")
            print(f"  {day_str}  ({n_done+1}/{n_total}  skipped:{n_skip})")
            print(f"{'='*65}")

            if day_str not in dw_by_day or day_str not in vecc_by_day:
                n_skip += 1
                print(f"  [SKIP] No estimates found for {day_str}")
                continue

            print(f"  DW   params: {[round(x,4) for x in dw_by_day[day_str]]}")
            print(f"  Vecc params: {[round(x,4) for x in vecc_by_day[day_str]]}")

            t0         = time.time()
            hour_indices = [day_idx * 8, (day_idx + 1) * 8]

            try:
                # ── Load spatial data for this day ────────────────────────────
                dw_map, dw_agg = data_load_instance.load_working_data(
                    df_map_yr, monthly_mean_yr, hour_indices,
                    ord_mm=None, dtype=DTYPE, keep_ori=False,
                )
                real_agg_dw  = dw_agg.to(DEVICE)
                vecc_map, _  = data_load_instance.load_working_data(
                    df_map_yr, monthly_mean_yr, hour_indices,
                    ord_mm=ord_mm, dtype=DTYPE, keep_ori=True,
                )
                real_map_vecc = {k: v.to(DEVICE) for k, v in vecc_map.items()}

                # ── [1/2] DW GIM ──────────────────────────────────────────────
                print(f"  [1/2] DW GIM...")
                dw_log_phi = torch.tensor(
                    transform_raw_to_log_phi(dw_by_day[day_str]),
                    device=DEVICE, dtype=DTYPE, requires_grad=True,
                )

                db = debiased_whittle.debiased_whittle_preprocess(
                    [real_agg_dw], [dw_map], day_idx=0,
                    params_list=dw_by_day[day_str],
                    lat_range=[-3, 2], lon_range=[121.0, 131.0],
                )
                cur_df       = db.generate_spatially_filtered_days(-3, 2, 121, 131).to(DEVICE)
                unique_times = torch.unique(cur_df[:, 3])
                time_slices  = [cur_df[cur_df[:, 3] == t] for t in unique_times]

                J_vec, n1, n2, p_time, taper_grid, obs_masks = dwl.generate_Jvector_tapered_mv(
                    time_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE)
                I_obs  = dwl.calculate_sample_periodogram_vectorized(J_vec)
                t_auto = dwl.calculate_taper_autocorrelation_multivariate(taper_grid, obs_masks, n1, n2, DEVICE)
                del obs_masks

                def nll_dw(p):
                    loss = dwl.whittle_likelihood_loss_tapered(
                        p, I_obs, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
                    return loss[0] if isinstance(loss, tuple) else loss

                H_dw     = finite_diff_hessian(nll_dw, dw_log_phi)
                H_inv_dw = torch.linalg.inv(H_dw + torch.eye(7, device=DEVICE) * 1e-5)
                torch.cuda.empty_cache()

                # DW observed J (per-frequency FD JVP)
                I_obs_c = I_obs.to(dtype=torch.complex128)

                def per_freq_losses_dw(p):
                    I_exp = dwl.expected_periodogram_fft_tapered(
                        p, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
                    eye_m = torch.eye(p_time, dtype=torch.complex128, device=DEVICE)
                    diag_load = max(
                        torch.abs(I_exp.diagonal(dim1=-2, dim2=-1)).mean().item() * 1e-8, 1e-9)
                    I_exp_s = I_exp + eye_m * diag_load
                    _, logdet = torch.linalg.slogdet(I_exp_s)
                    solved    = torch.linalg.solve(I_exp_s, I_obs_c)
                    trace     = torch.einsum('...ii->...', solved).real
                    return (logdet + trace).reshape(-1)[1:]

                _cols_dw = []
                for _k in range(dw_log_phi.shape[0]):
                    _pp = dw_log_phi.detach().clone(); _pp[_k] += _eps
                    _pm = dw_log_phi.detach().clone(); _pm[_k] -= _eps
                    with torch.no_grad():
                        _cols_dw.append(
                            (per_freq_losses_dw(_pp) - per_freq_losses_dw(_pm)) / (2.0 * _eps))
                J_mat_dw = torch.stack(_cols_dw)
                n_freq   = J_mat_dw.shape[1]
                J_dw     = J_mat_dw @ J_mat_dw.T / n_freq ** 2
                torch.cuda.empty_cache()

                GIM_dw = H_inv_dw @ J_dw @ H_inv_dw
                Jac_dw = torch.autograd.functional.jacobian(
                    transform_log_phi_to_physical, dw_log_phi)
                SE_dw  = torch.sqrt(torch.diag(Jac_dw @ GIM_dw @ Jac_dw.T)).detach().cpu().numpy()
                Pt_dw  = transform_log_phi_to_physical(dw_log_phi).detach().cpu().numpy()

                del real_agg_dw, dw_map, H_dw, J_dw, GIM_dw, J_mat_dw
                gc.collect(); torch.cuda.empty_cache()

                # ── [2/2] Vecchia GIM ─────────────────────────────────────────
                print(f"  [2/2] Vecchia GIM...")
                vc_log_phi = torch.tensor(
                    transform_raw_to_log_phi(vecc_by_day[day_str]),
                    device=DEVICE, dtype=DTYPE, requires_grad=True,
                )

                model_vc = kernels_vecchia.fit_vecchia_lbfgs(
                    smooth=v, input_map=real_map_vecc,
                    nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nheads,
                    limit_A=limit_a, limit_B=limit_b, limit_C=limit_c,
                    daily_stride=daily_stride,
                )
                model_vc.precompute_conditioning_sets()

                def nll_vc(p): return model_vc.vecchia_batched_likelihood(p)

                H_vc     = finite_diff_hessian(nll_vc, vc_log_phi)
                H_inv_vc = torch.linalg.inv(H_vc + torch.eye(7, device=DEVICE) * 1e-5)
                torch.cuda.empty_cache()

                beta_hat = model_vc.get_gls_beta(vc_log_phi).detach()

                def per_unit_losses_vc(p):
                    return model_vc.vecchia_per_unit_nll_terms(p, beta_hat)

                _cols_vc = []
                for _k in range(vc_log_phi.shape[0]):
                    _pp = vc_log_phi.detach().clone(); _pp[_k] += _eps
                    _pm = vc_log_phi.detach().clone(); _pm[_k] -= _eps
                    with torch.no_grad():
                        _cols_vc.append(
                            (per_unit_losses_vc(_pp) - per_unit_losses_vc(_pm)) / (2.0 * _eps))
                J_mat_vc = torch.stack(_cols_vc)
                N_units  = J_mat_vc.shape[1]
                J_vc     = J_mat_vc @ J_mat_vc.T / N_units ** 2
                torch.cuda.empty_cache()

                GIM_vc = H_inv_vc @ J_vc @ H_inv_vc
                Jac_vc = torch.autograd.functional.jacobian(
                    transform_log_phi_to_physical, vc_log_phi)
                SE_vc  = torch.sqrt(torch.diag(Jac_vc @ GIM_vc @ Jac_vc.T)).detach().cpu().numpy()
                Pt_vc  = transform_log_phi_to_physical(vc_log_phi).detach().cpu().numpy()

                del real_map_vecc, vecc_map, model_vc, H_vc, J_vc, GIM_vc, J_mat_vc
                gc.collect(); torch.cuda.empty_cache()

                # ── Record ────────────────────────────────────────────────────
                elapsed = time.time() - t0
                row = {
                    'year': year_str, 'month': month, 'day': day, 'day_str': day_str,
                    'J_method': 'observed', 'elapsed': round(elapsed, 1),
                }
                for k, name in enumerate(P_NAMES):
                    row[f'DW_Est_{name}'] = round(float(Pt_dw[k]), 6)
                    row[f'DW_SE_{name}']  = round(float(SE_dw[k]), 6)
                    row[f'VC_Est_{name}'] = round(float(Pt_vc[k]), 6)
                    row[f'VC_SE_{name}']  = round(float(SE_vc[k]), 6)
                records.append(row)

                # Print per-day table
                print(f"\n  {'Param':<10} | {'DW Est':>8} | {'DW SE':>8} || {'VC Est':>8} | {'VC SE':>8}")
                print(f"  {'-'*60}")
                for k, name in enumerate(P_NAMES):
                    print(f"  {name:<10} | {Pt_dw[k]:>8.4f} | {SE_dw[k]:>8.4f} || "
                          f"{Pt_vc[k]:>8.4f} | {SE_vc[k]:>8.4f}")
                print(f"  Elapsed: {elapsed:.1f}s")

                # Incremental CSV save
                pd.DataFrame(records).to_csv(out_raw, index=False)

                # Running summary
                print_running_summary(records, n_total)

            except Exception as e:
                import traceback
                n_skip += 1
                print(f"  [ERROR] {day_str}: {type(e).__name__}: {e}  (total skipped: {n_skip})")
                traceback.print_exc()
                gc.collect(); torch.cuda.empty_cache()
                continue

    # ── Final summary ─────────────────────────────────────────────────────────
    if records:
        print_final_summary(records)
        pd.DataFrame(build_summary_csv(records)).to_csv(out_summary, index=False)
        print(f"\n  Saved: {out_raw.name}")
        print(f"  Saved: {out_summary.name}")
    else:
        print("[Warning] No days completed successfully.")

    print("\n[Done]")


if __name__ == "__main__":
    app()
