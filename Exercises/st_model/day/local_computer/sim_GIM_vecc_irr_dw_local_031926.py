"""
sim_GIM_vecc_irr_dw_local_031926.py  —  Local (mac) version

Computes GIM sandwich standard errors for DW and Vecchia-Irr models using
**observed information** for J (no parametric bootstrap):
  - DW  J: per-frequency score outer products  (n1*n2 - 1 terms)
  - VC  J: per-unit conditional score outer products  (N_heads + N_tails terms)

Differences from Amarel version:
  - sys.path  → mac src path
  - data path → config.mac_data_load_path
  - est CSVs  → GEMS_TCO-1/outputs/day/july_22_24_25/
  - output    → GEMS_TCO-1/outputs/day/GIM/

conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
python sim_GIM_vecc_irr_dw_local_031926.py --sample-year 2022 --sample-day 22
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

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle_2110 as debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

# --- GLOBAL SETTINGS ---
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.float64
DELTA_LAT = 0.044
DELTA_LON = 0.063

LOCAL_EST_PATH   = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25")
LOCAL_OUTPUT_DIR = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/GIM")


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
    return (H + H.T) / 2.0


# ── Main CLI ──────────────────────────────────────────────────────────────────

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    sample_year: str = typer.Option('2024', help="Year of representative day"),
    sample_day:  int = typer.Option(1,      help="Day of month (1-based)"),
    month:       int = typer.Option(7,      help="Month"),
    v:           float = typer.Option(0.5,  help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100, help="Vecchia neighbors"),
    nheads:      int = typer.Option(0,      help="Head points"),
    limit_a:     int = typer.Option(20,      help="Set A neighbors"),
    limit_b:     int = typer.Option(20,      help="Set B neighbors"),
    limit_c:     int = typer.Option(20,      help="Set C neighbors"),
    daily_stride: int = typer.Option(2,     help="Set C stride"),
) -> None:

    set_seed(2025)
    day_str = f"{sample_year}-{month:02d}-{sample_day}"
    day_idx = sample_day - 1
    print(f"Device: {DEVICE}")
    print(f"Day: {day_str}  (observed J — no bootstrap)")

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = LOCAL_OUTPUT_DIR / f"GIM_{day_str}_obsJ_local.csv"

    # ── Load fitted estimates ─────────────────────────────────────────────────
    dw_csv   = LOCAL_EST_PATH / "real_dw_july_22_23_24_25.csv"
    vecc_csv = LOCAL_EST_PATH / "real_vecc_july_22_23_24_25_h0_mm100.csv"
    if not dw_csv.exists() or not vecc_csv.exists():
        print(f"[Error] CSV not found in {LOCAL_EST_PATH}")
        raise SystemExit(1)

    _param_cols = ['sigma', 'range_lat', 'range_lon', 'range_time',
                   'advec_lat', 'advec_lon', 'nugget']
    dw_df   = pd.read_csv(dw_csv)
    vecc_df = pd.read_csv(vecc_csv)
    dw_by_day   = {row['day']: [row[c] for c in _param_cols] for _, row in dw_df.iterrows()}
    vecc_by_day = {row['day']: [row[c] for c in _param_cols] for _, row in vecc_df.iterrows()}

    if day_str not in dw_by_day or day_str not in vecc_by_day:
        print(f"[Error] No estimates for {day_str}. Available: {sorted(dw_by_day.keys())[:5]} ...")
        raise SystemExit(1)

    print(f"DW   params: {[round(x,4) for x in dw_by_day[day_str]]}")
    print(f"Vecc params: {[round(x,4) for x in vecc_by_day[day_str]]}")

    # ── Load spatial data ─────────────────────────────────────────────────────
    data_load_instance = load_data_dynamic_processed(config.mac_data_load_path)
    df_map, ord_mm, nns_map, monthly_mean = data_load_instance.load_maxmin_ordered_data_bymonthyear(
        mm_cond_number=mm_cond_number,
        years_=[sample_year], months_=[month],
        lat_range=[-3, 2], lon_range=[121, 131],
        is_whittle=False
    )
    hour_indices = [day_idx * 8, (day_idx + 1) * 8]

    p_names = ["SigmaSq", "RangeLat", "RangeLon", "RangeTime", "AdvecLat", "AdvecLon", "Nugget"]
    dwl = debiased_whittle.debiased_whittle_likelihood()
    t0  = time.time()

    # ── Load DW data ──────────────────────────────────────────────────────────
    dw_map, dw_agg = data_load_instance.load_working_data(
        df_map, monthly_mean, hour_indices,
        ord_mm=None, dtype=DTYPE, keep_ori=False
    )
    real_agg_dw = dw_agg.to(DEVICE)

    # ── Load Vecchia data ─────────────────────────────────────────────────────
    vecc_map, _ = data_load_instance.load_working_data(
        df_map, monthly_mean, hour_indices,
        ord_mm=ord_mm, dtype=DTYPE, keep_ori=True
    )
    real_map_vecc = {k: v.to(DEVICE) for k, v in vecc_map.items()}

    # ── DW GIM ────────────────────────────────────────────────────────────────
    print(f"\n[1/2] DW GIM  ({day_str})...")
    dw_log_phi = torch.tensor(
        transform_raw_to_log_phi(dw_by_day[day_str]),
        device=DEVICE, dtype=DTYPE, requires_grad=True
    )

    db = debiased_whittle.debiased_whittle_preprocess(
        [real_agg_dw], [dw_map], day_idx=0,
        params_list=dw_by_day[day_str],
        lat_range=[-3, 2], lon_range=[121.0, 131.0]
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

    H_dw       = finite_diff_hessian(nll_dw, dw_log_phi)
    eigvals_dw = torch.linalg.eigvalsh(H_dw)
    print(f"  H_dw eigenvalues: {eigvals_dw.detach().cpu().numpy().round(4)}")
    print(f"  H_dw condition number: {(eigvals_dw.max()/eigvals_dw.abs().min()).item():.2e}")
    H_inv_dw = torch.linalg.inv(H_dw + torch.eye(7, device=DEVICE) * 1e-5)

    # ── DW J: observed information (per-frequency score outer products) ───────
    # J = (1/n_freq²) Σ_ω ∇ℓ_ω · ∇ℓ_ω^T   (excluding DC at ω=(0,0))
    print("  Computing DW observed J via per-frequency Jacobian...")
    I_obs_c = I_obs.to(dtype=torch.complex128)

    def per_freq_losses_dw(p):
        """Per-frequency NLL terms (DC excluded), shape (n1*n2 - 1,)."""
        I_exp = dwl.expected_periodogram_fft_tapered(p, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
        eye_m = torch.eye(p_time, dtype=torch.complex128, device=DEVICE)
        diag_load = max(torch.abs(I_exp.diagonal(dim1=-2, dim2=-1)).mean().item() * 1e-8, 1e-9)
        I_exp_s = I_exp + eye_m * diag_load
        _, logdet = torch.linalg.slogdet(I_exp_s)                # (n1, n2) real
        solved    = torch.linalg.solve(I_exp_s, I_obs_c)         # (n1, n2, p, p) complex
        trace     = torch.einsum('...ii->...', solved).real       # (n1, n2) real
        return (logdet + trace).reshape(-1)[1:]                   # exclude DC at index 0

    # Column-wise FD JVP: 7 params → 14 forward passes, O(n_freq) memory each
    _eps = 1e-5
    _cols_dw = []
    for _k in range(dw_log_phi.shape[0]):
        _pp = dw_log_phi.detach().clone(); _pp[_k] += _eps
        _pm = dw_log_phi.detach().clone(); _pm[_k] -= _eps
        with torch.no_grad():
            _cols_dw.append((per_freq_losses_dw(_pp) - per_freq_losses_dw(_pm)) / (2.0 * _eps))
    J_mat_dw = torch.stack(_cols_dw)          # (7, n_freq)
    n_freq   = J_mat_dw.shape[1]
    J_dw     = J_mat_dw @ J_mat_dw.T / n_freq ** 2

    print(f"  DW n_freq={n_freq}  |  J_dw diag: {torch.diag(J_dw).detach().cpu().numpy().round(4)}")
    print(f"  H_dw diag:          {torch.diag(H_dw).detach().cpu().numpy().round(4)}")

    GIM_dw = H_inv_dw @ J_dw @ H_inv_dw
    Jac_dw = torch.autograd.functional.jacobian(transform_log_phi_to_physical, dw_log_phi)
    SE_dw  = torch.sqrt(torch.diag(Jac_dw @ GIM_dw @ Jac_dw.T)).detach().cpu().numpy()
    Pt_dw  = transform_log_phi_to_physical(dw_log_phi).detach().cpu().numpy()

    del real_agg_dw, dw_map, H_dw, J_dw, GIM_dw
    gc.collect()

    # ── Vecchia-Irr GIM ───────────────────────────────────────────────────────
    print(f"\n[2/2] Vecchia-Irr GIM  ({day_str})...")
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

    H_vc       = finite_diff_hessian(nll_vc, vc_log_phi)
    eigvals_vc = torch.linalg.eigvalsh(H_vc)
    print(f"  H_vc eigenvalues: {eigvals_vc.detach().cpu().numpy().round(4)}")
    print(f"  H_vc condition number: {(eigvals_vc.max()/eigvals_vc.abs().min()).item():.2e}")
    H_inv_vc = torch.linalg.inv(H_vc + torch.eye(7, device=DEVICE) * 1e-5)

    model_vc.input_map = real_map_vecc

    # ── Vecchia J: observed information (per-unit score outer products) ───────
    # Fix beta at MLE estimate, then differentiate per-unit NLL terms w.r.t. params.
    # J = (1/N²) Σ_i ∇LL_i · ∇LL_i^T   where LL_i = log(σ_i) + 0.5 * r_i²
    print("  Computing Vecchia observed J via per-unit Jacobian...")
    beta_hat = model_vc.get_gls_beta(vc_log_phi).detach()

    def per_unit_losses_vc(p):
        """Per-unit NLL terms (heads + tails), shape (total_N,)."""
        return model_vc.vecchia_per_unit_nll_terms(p, beta_hat)

    # Column-wise FD JVP: 7 params → 14 forward passes, O(N_units) memory each
    _cols_vc = []
    for _k in range(vc_log_phi.shape[0]):
        _pp = vc_log_phi.detach().clone(); _pp[_k] += _eps
        _pm = vc_log_phi.detach().clone(); _pm[_k] -= _eps
        with torch.no_grad():
            _cols_vc.append((per_unit_losses_vc(_pp) - per_unit_losses_vc(_pm)) / (2.0 * _eps))
    J_mat_vc = torch.stack(_cols_vc)          # (7, N_units)
    N_units  = J_mat_vc.shape[1]
    J_vc     = J_mat_vc @ J_mat_vc.T / N_units ** 2

    print(f"  Vecchia N_units={N_units}  |  J_vc diag: {torch.diag(J_vc).detach().cpu().numpy().round(6)}")
    print(f"  H_vc diag:                  {torch.diag(H_vc).detach().cpu().numpy().round(6)}")

    GIM_vc = H_inv_vc @ J_vc @ H_inv_vc
    Jac_vc = torch.autograd.functional.jacobian(transform_log_phi_to_physical, vc_log_phi)
    SE_vc  = torch.sqrt(torch.diag(Jac_vc @ GIM_vc @ Jac_vc.T)).detach().cpu().numpy()
    Pt_vc  = transform_log_phi_to_physical(vc_log_phi).detach().cpu().numpy()

    # ── Save & print ──────────────────────────────────────────────────────────
    row = {"day": day_str, "J_method": "observed"}
    for k, name in enumerate(p_names):
        row[f"DW_Est_{name}"]     = round(float(Pt_dw[k]), 4)
        row[f"DW_SE_{name}"]      = round(float(SE_dw[k]), 4)
        row[f"VC_Irr_Est_{name}"] = round(float(Pt_vc[k]), 4)
        row[f"VC_Irr_SE_{name}"]  = round(float(SE_vc[k]), 4)
    pd.DataFrame([row]).to_csv(out_file, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  GIM results — {day_str}  ({elapsed:.1f}s total)")
    print(f"{'='*65}")
    print(f"  {'Param':<10} | {'DW Est':>8} | {'DW SE':>8} || {'VC Est':>8} | {'VC SE':>8}")
    print(f"  {'-'*63}")
    for k, name in enumerate(p_names):
        print(f"  {name:<10} | {Pt_dw[k]:>8.4f} | {SE_dw[k]:>8.4f} || "
              f"{Pt_vc[k]:>8.4f} | {SE_vc[k]:>8.4f}")
    print(f"\n  Saved: {out_file}")

    del real_map_vecc, vecc_map, model_vc, H_vc, J_vc, GIM_vc
    gc.collect()
    print("\n[Done]")


if __name__ == "__main__":
    app()
