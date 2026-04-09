"""
sim_dw_sign_040826.py

Simulation study: 2-way sign ambiguity diagnostic for 4 reduced DW models.

Each model is a simplified version of DW_raw with one advec direction and
optionally isotropic spatial range.  After L-BFGS converges, the likelihood
is evaluated at the converged point (pp) AND the sign-flipped point (mm).

  L(pp) ≈ L(mm)   →  structural sign ambiguity
  L(pp) << L(mm)  →  correct optimization, no ambiguity

Models
------
  1  DW_iso_lat   : isotropic range, advec_lat free   (5 params)
  2  DW_iso_lon   : isotropic range, advec_lon free   (5 params)
  3  DW_aniso_lat : anisotropic range, advec_lat free (6 params)
  4  DW_aniso_lon : anisotropic range, advec_lon free (6 params)

Usage
-----
  python sim_dw_sign_040826.py --model 1 --num-iters 200 --sign-start 30
  python sim_dw_sign_040826.py --model 3 --num-iters 200
"""

import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.fft
import pandas as pd
import typer
from pathlib import Path

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")

from GEMS_TCO import debiased_whittle_raw    as dw_raw_module
from GEMS_TCO import debiased_whittle_reduced as dw_red_module
from GEMS_TCO import configuration as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT = 0.044
DELTA_LON = 0.063

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ─────────────────────────────────────────────────────────────────────────────
# True parameters for each model
# ─────────────────────────────────────────────────────────────────────────────
#
# iso range = geometric mean of aniso range_lat × range_lon
_range_iso = np.sqrt(0.154 * 0.195)   # ≈ 0.1734

TRUE_PARAMS = {
    # iso models: single range; advec = only active direction
    1: {'sigmasq': 13.059, 'range':     _range_iso, 'range_time': 1.0,
        'advec': 0.0218,  'nugget': 0.247, 'advec_name': 'advec_lat',
        'range_lat': _range_iso, 'range_lon': _range_iso},
    2: {'sigmasq': 13.059, 'range':     _range_iso, 'range_time': 1.0,
        'advec': -0.1689, 'nugget': 0.247, 'advec_name': 'advec_lon',
        'range_lat': _range_iso, 'range_lon': _range_iso},
    # aniso models: separate lat/lon ranges
    3: {'sigmasq': 13.059, 'range_lat': 0.154, 'range_lon': 0.195, 'range_time': 1.0,
        'advec': 0.0218,  'nugget': 0.247, 'advec_name': 'advec_lat'},
    4: {'sigmasq': 13.059, 'range_lat': 0.154, 'range_lon': 0.195, 'range_time': 1.0,
        'advec': -0.1689, 'nugget': 0.247, 'advec_name': 'advec_lon'},
}

MODEL_CLASS = {
    1: dw_red_module.DW_iso_lat,
    2: dw_red_module.DW_iso_lon,
    3: dw_red_module.DW_aniso_lat,
    4: dw_red_module.DW_aniso_lon,
}

MODEL_NAMES = {
    1: 'DW_iso_lat',
    2: 'DW_iso_lon',
    3: 'DW_aniso_lat',
    4: 'DW_aniso_lon',
}


# ─────────────────────────────────────────────────────────────────────────────
# Param construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def true_to_7log(model_id, td):
    """Build full 7-param log tensor for data generation."""
    if model_id in (1, 2):
        r    = td['range']
        phi2 = 1.0 / r
        phi1 = td['sigmasq'] * phi2
        phi3 = 1.0                         # isotropic → phi3 = 1, log_phi3 = 0
        phi4 = (r / td['range_time']) ** 2
    else:
        phi2 = 1.0 / td['range_lon']
        phi1 = td['sigmasq'] * phi2
        phi3 = (td['range_lon'] / td['range_lat']) ** 2
        phi4 = (td['range_lon'] / td['range_time']) ** 2

    advec_lat = td['advec'] if td['advec_name'] == 'advec_lat' else 0.0
    advec_lon = td['advec'] if td['advec_name'] == 'advec_lon' else 0.0
    return [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
            advec_lat, advec_lon, np.log(td['nugget'])]


def true_to_reduced_log(model_id, td):
    """Build reduced param log vector (N_PARAMS elements) from true params."""
    if model_id in (1, 2):
        r    = td['range']
        phi2 = 1.0 / r
        phi1 = td['sigmasq'] * phi2
        phi4 = (r / td['range_time']) ** 2
        return [np.log(phi1), np.log(phi2), np.log(phi4), td['advec'], np.log(td['nugget'])]
    else:
        phi2 = 1.0 / td['range_lon']
        phi1 = td['sigmasq'] * phi2
        phi3 = (td['range_lon'] / td['range_lat']) ** 2
        phi4 = (td['range_lon'] / td['range_time']) ** 2
        return [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                td['advec'], np.log(td['nugget'])]


def make_init_reduced(model_id, td, rng, noise=0.7):
    """Perturb true reduced log params for random init."""
    base = true_to_reduced_log(model_id, td)
    noisy = list(base)
    if model_id in (1, 2):
        # indices: 0,1,2=log scale  3=advec  4=log scale
        for i in [0, 1, 2, 4]:
            noisy[i] = base[i] + rng.uniform(-noise, noise)
        scale = max(abs(base[3]), 0.05)
        noisy[3] = base[3] + rng.uniform(-2 * scale, 2 * scale)
    else:
        # indices: 0,1,2,3=log scale  4=advec  5=log scale
        for i in [0, 1, 2, 3, 5]:
            noisy[i] = base[i] + rng.uniform(-noise, noise)
        scale = max(abs(base[4]), 0.05)
        noisy[4] = base[4] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


# ─────────────────────────────────────────────────────────────────────────────
# Data generation  (identical to sim_dw_bimodal: FFT-based field on full grid)
# ─────────────────────────────────────────────────────────────────────────────

def get_covariance_on_grid(lx, ly, lt, params7):
    """7-param anisotropic Matérn-1 covariance on a grid of lags."""
    params7 = torch.clamp(params7, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params7[i]) for i in range(4))
    u_lat = lx - params7[4] * lt
    u_lon = ly - params7[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def generate_field_on_grid(lats_grid, lons_grid, t_steps, true_params7, dlat, dlon):
    """Generate a random spatio-temporal field via spectral simulation."""
    CPU = torch.device("cpu"); F32 = torch.float32
    Nx, Ny, Nt = len(lats_grid), len(lons_grid), t_steps
    Px, Py, Pt = 2 * Nx, 2 * Ny, 2 * Nt

    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt

    p7 = true_params7.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid(Lx, Ly, Lt, p7)
    S = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


def assemble_grid_dataset(field, grid_coords, true_params7, t_offset=21.0):
    """Assemble per-time-slice tensors with optional nugget noise."""
    nugget_std = torch.sqrt(torch.exp(true_params7[6]))
    N_lat, N_lon, T = field.shape
    N_grid = N_lat * N_lon
    grid_map, grid_list = {}, []
    field_flat = field.reshape(N_grid, T)

    for t_idx in range(T):
        t_val = float(t_offset + t_idx)
        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0
        rows = torch.zeros(N_grid, 11, device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords
        rows[:, 2]  = (field_flat[:, t_idx] +
                       torch.randn(N_grid, device=DEVICE, dtype=DTYPE) * nugget_std)
        rows[:, 3]  = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid, -1)
        grid_map[f't{t_idx}'] = rows.detach()
        grid_list.append(rows.detach())

    return grid_map, torch.cat(grid_list, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2-Way sign check
# ─────────────────────────────────────────────────────────────────────────────

def sign_check_2way(params_list, model_cls, I_samp, n1, n2, p_time,
                    t_auto, device, true_advec):
    """
    Evaluate Whittle likelihood at converged point (pp) and sign-flipped point (mm).

    advec_idx is read from model_cls.ADVEC_IDX.

    Returns dict with L_pp, L_mm, dL (=L(pp)-L(mm)), verdict, advec_conv.
    """
    advec_idx = model_cls.ADVEC_IDX

    with torch.no_grad():
        p_base = torch.cat([p.detach().clone() for p in params_list])
        advec_conv = p_base[advec_idx].item()

        # pp: converged as-is
        L_pp_t = model_cls.whittle_likelihood_loss_tapered(
            p_base.to(device), I_samp, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
        L_pp = L_pp_t.item() if not (torch.isnan(L_pp_t) or torch.isinf(L_pp_t)) else float('nan')

        # mm: flip advec sign
        p_mm = p_base.clone()
        p_mm[advec_idx] = -p_base[advec_idx]
        L_mm_t = model_cls.whittle_likelihood_loss_tapered(
            p_mm.to(device), I_samp, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
        L_mm = L_mm_t.item() if not (torch.isnan(L_mm_t) or torch.isinf(L_mm_t)) else float('nan')

    dL = (L_pp - L_mm) if not (np.isnan(L_pp) or np.isnan(L_mm)) else float('nan')

    print(f"\n  ── 2-Way Sign Check ({model_cls.ADVEC_NAME}) ──────────────────────────────")
    print(f"  Converged {model_cls.ADVEC_NAME}: {advec_conv:+.4f}   True: {true_advec:+.4f}")
    print(f"  L(pp = conv):  {L_pp:.6f}")
    print(f"  L(mm = flip):  {L_mm:.6f}")
    print(f"  ΔL = L(pp)-L(mm): {dL:+.4f}")

    if np.isnan(dL):
        verdict = 'NaN'
    elif abs(dL) < 0.5:
        verdict = 'SYM'    # structural symmetry
    elif dL < -0.5:
        verdict = 'CORRECT' # pp wins (converged point is better)
    else:
        verdict = 'WRONG'  # mm wins (flip is better → wrong sign converged)

    verdict_msg = {
        'SYM':     'STRUCTURAL SYMMETRY  (|ΔL| < 0.5 — +/- signs indistinguishable)',
        'CORRECT': f'CORRECT MODE  (converged wins, gap={-dL:.3f})',
        'WRONG':   f'OPT. FAILURE / WRONG SIGN  (flip wins, gap={dL:.3f})',
        'NaN':     'NaN — could not evaluate',
    }[verdict]

    print(f"  Verdict: {verdict_msg}")
    print(f"  ─────────────────────────────────────────────────────────────────\n")

    return {
        'L_pp':       L_pp,
        'L_mm':       L_mm,
        'dL':         dL,
        'advec_conv': advec_conv,
        'verdict':    verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Running / final summaries
# ─────────────────────────────────────────────────────────────────────────────

def print_running_summary(records, sc_records, model_id, true_d):
    """Print estimate statistics and sign-check summary."""
    mc   = MODEL_CLASS[model_id]
    n    = len(records)
    if n == 0:
        return
    true_advec = true_d['advec']
    advec_name = true_d['advec_name']

    print(f"\n  ── Running summary ({n} iters) — {MODEL_NAMES[model_id]} ──")

    # Estimate stats
    advec_vals = np.array([r['advec_est'] for r in records])
    pos = advec_vals[advec_vals > 0]
    neg = advec_vals[advec_vals <= 0]
    print(f"  {advec_name} true={true_advec:+.4f}  "
          f"mean={advec_vals.mean():.4f}  med={np.median(advec_vals):.4f}")
    print(f"  Sign split:  >0 n={len(pos):3d} mean={pos.mean():.4f}" if len(pos) else
          f"  Sign split:  >0 n=  0")
    print(f"               ≤0 n={len(neg):3d} mean={neg.mean():.4f}" if len(neg) else
          f"               ≤0 n=  0")

    if sc_records:
        n_sc  = len(sc_records)
        n_sym = sum(1 for r in sc_records if r['verdict'] == 'SYM')
        n_cor = sum(1 for r in sc_records if r['verdict'] == 'CORRECT')
        n_wrg = sum(1 for r in sc_records if r['verdict'] == 'WRONG')
        dLs   = [r['dL'] for r in sc_records if not np.isnan(r.get('dL', float('nan')))]
        print(f"\n  [2-Way Sign Check   n={n_sc}]")
        print(f"  SYM    (|ΔL|<0.5)   : {n_sym:3d}  ({100*n_sym/n_sc:.1f}%)")
        print(f"  CORRECT (conv wins) : {n_cor:3d}  ({100*n_cor/n_sc:.1f}%)")
        print(f"  WRONG   (flip wins) : {n_wrg:3d}  ({100*n_wrg/n_sc:.1f}%)")
        if dLs:
            print(f"  ΔL = L(pp)-L(mm): mean={np.mean(dLs):+.4f}  "
                  f"med={np.median(dLs):+.4f}  "
                  f"P5={np.percentile(dLs,5):.3f}  P95={np.percentile(dLs,95):.3f}")

    rmsre_last5 = [f"{r['rmsre']:.4f}" for r in records[-5:]]
    print(f"  [RMSRE last 5] {rmsre_last5}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(records, sc_records, true_d, plot_dir, model_id, n_iters):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Patch

    mc         = MODEL_CLASS[model_id]
    model_name = MODEL_NAMES[model_id]
    advec_name = true_d['advec_name']
    true_advec = true_d['advec']

    df = pd.DataFrame(records)

    COL_A = '#1976D2'   # advec > 0
    COL_B = '#E53935'   # advec ≤ 0

    # ── 1. KDE + scatter: advec estimate ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name} — {advec_name} Estimate  ({n_iters} iterations)',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    vals = df['advec_est'].dropna().values
    if len(vals) >= 3:
        try:
            kde = gaussian_kde(vals)
            xs  = np.linspace(vals.min() - 0.5*abs(vals.max()-vals.min()),
                              vals.max() + 0.5*abs(vals.max()-vals.min()), 300)
            ax.plot(xs, kde(xs), color=COL_A, lw=2.5)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=COL_A)
        except Exception:
            ax.hist(vals, bins=30, color=COL_A, alpha=0.5)
    ax.axvline(true_advec, color='black', ls='--', lw=1.5, label=f'true={true_advec:.4f}')
    ax.axvline(0, color='gray', ls=':', lw=0.8)
    ax.set_title(f'KDE — {advec_name}', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    colors = np.where(df['advec_est'].values > 0, COL_A, COL_B)
    ax = axes[1]
    ax.scatter(df['iter'].values, df['advec_est'].values, c=colors, s=20, alpha=0.6)
    ax.axhline(true_advec, color='black', ls='--', lw=1.5, label=f'true={true_advec:.4f}')
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    if len(vals) >= 20:
        rm = pd.Series(df['advec_est'].values).rolling(20, center=True, min_periods=5).median().values
        ax.plot(df['iter'].values, rm, color='darkorange', lw=1.5, label='roll. median')
    ax.set_xlabel('Iteration'); ax.set_ylabel(advec_name)
    ax.set_title(f'Scatter — iter vs {advec_name}', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    legend_el = [Patch(fc=COL_A, label=f'{advec_name} > 0'),
                 Patch(fc=COL_B, label=f'{advec_name} ≤ 0')]
    fig.legend(handles=legend_el, loc='lower right', fontsize=9, bbox_to_anchor=(0.99, 0.01))
    plt.tight_layout()
    pname = f'{model_name}_advec.png'
    plt.savefig(plot_dir / pname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pname}")

    # ── 2. Sign check: ΔL timeline ──────────────────────────────────────────
    if not sc_records:
        return
    sc_df = pd.DataFrame(sc_records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{model_name} — 2-Way Sign Check: ΔL = L(pp)−L(mm)',
                 fontsize=13, fontweight='bold')

    verdict_colors = {'SYM': '#FF6F00', 'CORRECT': '#1976D2',
                      'WRONG': '#E53935', 'NaN': 'gray'}
    vc = [verdict_colors.get(v, 'gray') for v in sc_df['verdict'].values]

    ax0 = axes[0]
    ax0.scatter(sc_df['iter'].values, sc_df['dL'].values, c=vc, s=25, alpha=0.8)
    ax0.axhline(0,    color='black', ls='-',  lw=0.8)
    ax0.axhline(0.5,  color='green', ls='--', lw=1.0, alpha=0.6, label='±0.5 threshold')
    ax0.axhline(-0.5, color='green', ls='--', lw=1.0, alpha=0.6)
    ax0.set_xlabel('Iteration'); ax0.set_ylabel('ΔL = L(pp)−L(mm)')
    ax0.set_title('ΔL over iterations (colored by verdict)', fontsize=10)
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.2)

    ax1 = axes[1]
    ax1.scatter(sc_df['iter'].values, sc_df['advec_conv'].values, c=vc, s=25, alpha=0.8)
    ax1.axhline(true_advec, color='black', ls='--', lw=1.5, label=f'true={true_advec:.4f}')
    ax1.axhline(0, color='gray', ls=':', lw=0.8)
    ax1.set_xlabel('Iteration'); ax1.set_ylabel(f'{advec_name} (converged)')
    ax1.set_title(f'Converged {advec_name} — colored by verdict', fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.2)

    legend_el = [Patch(fc=verdict_colors[v], label=v)
                 for v in ('SYM', 'CORRECT', 'WRONG')]
    fig.legend(handles=legend_el, loc='lower center', fontsize=9,
               ncol=3, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    pname = f'{model_name}_signcheck.png'
    plt.savefig(plot_dir / pname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pname}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    model:      int   = typer.Option(1,   help="Model ID: 1=iso_lat, 2=iso_lon, 3=aniso_lat, 4=aniso_lon"),
    num_iters:  int   = typer.Option(200, help="Simulation iterations"),
    sign_start: int   = typer.Option(30,  help="Start 2-way sign check after this many iters"),
    lat_range:  str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:  str   = typer.Option("121,131", help="lon_min,lon_max"),
    init_noise: float = typer.Option(0.7,  help="Uniform noise half-width in log space"),
    seed:       int   = typer.Option(42,   help="Random seed"),
) -> None:

    if model not in MODEL_CLASS:
        print(f"Error: --model must be 1,2,3,4.  Got {model}.")
        raise typer.Exit(1)

    mc         = MODEL_CLASS[model]
    model_name = MODEL_NAMES[model]
    true_d     = TRUE_PARAMS[model]
    true_advec = true_d['advec']

    rng   = np.random.default_rng(seed)
    lat_r = [float(x) for x in lat_range.split(',')]
    lon_r = [float(x) for x in lon_range.split(',')]

    print(f"Device      : {DEVICE}")
    print(f"Model       : {model_name}  (model ID={model})")
    print(f"n_params    : {mc.N_PARAMS}")
    print(f"Region      : lat {lat_r}, lon {lon_r}")
    print(f"True advec  : {mc.ADVEC_NAME} = {true_advec:+.4f}")
    print(f"Sign check  : starts at iter {sign_start}")
    print(f"Init noise  : ±{init_noise} log-space")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_dir = output_path / 'plots' / f'sign_{model_name}'
    plot_dir.mkdir(parents=True, exist_ok=True)

    date_tag   = datetime.now().strftime("%m%d%y")
    csv_est    = f"sim_dw_sign_m{model}_{date_tag}.csv"
    csv_sc     = f"sim_dw_sign_m{model}_check_{date_tag}.csv"

    # ── True 7-param log tensor (for data generation) ───────────────────────
    true_7log    = true_to_7log(model, true_d)
    true_params7 = torch.tensor(true_7log, device=DEVICE, dtype=DTYPE)

    # ── Build target grid ────────────────────────────────────────────────────
    print("\n[Setup] Building target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001, DELTA_LON,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat × {len(lons_grid)} lon = {N_grid} cells")

    # ── Shared DW infrastructure ──────────────────────────────────────────────
    # Taper, periodogram, expected-periodogram are inherited from debiased_whittle_raw
    dwl_raw = dw_raw_module.debiased_whittle_likelihood()

    LBFGS_STEPS = 5
    LC, NC, VC, TC = 0, 1, 2, 3

    records    = []
    sc_records = []
    skipped    = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  [{model_name}] Iteration {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*60}")

        init_vals = make_init_reduced(model, true_d, rng, noise=init_noise)

        try:
            # ── Generate data ────────────────────────────────────────────────
            field    = generate_field_on_grid(lats_grid, lons_grid, 8, true_params7,
                                              DELTA_LAT, DELTA_LON)
            grid_map, grid_agg = assemble_grid_dataset(field, grid_coords, true_params7)
            del field

            # ── Preprocess: demean (raw passthrough, no spatial filter) ──────
            db = dw_raw_module.debiased_whittle_preprocess(
                [grid_agg], [grid_map], day_idx=0,
                params_list=true_7log,
                lat_range=lat_r, lon_range=lon_r
            )
            cur_df     = db.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
            unique_t   = torch.unique(cur_df[:, TC])
            time_slices = [cur_df[cur_df[:, TC] == t] for t in unique_t]

            # ── DFT + taper + autocorrelation ────────────────────────────────
            J_vec, n1, n2, p_time, taper, obs_masks = dwl_raw.generate_Jvector_tapered_mv(
                time_slices, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE)
            I_samp = dwl_raw.calculate_sample_periodogram_vectorized(J_vec)
            t_auto = dwl_raw.calculate_taper_autocorrelation_multivariate(
                taper, obs_masks, n1, n2, DEVICE)
            del obs_masks

            # ── Optimize ─────────────────────────────────────────────────────
            p_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                    for val in init_vals]

            opt = torch.optim.LBFGS(
                p_dw, lr=1.0, max_iter=20, max_eval=100,
                history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5)

            t0 = time.time()
            _, _, steps = mc.run_lbfgs_tapered(
                params_list=p_dw, optimizer=opt, I_sample=I_samp,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
                max_steps=LBFGS_STEPS, device=DEVICE)
            t_elapsed = time.time() - t0

            est = mc.backmap(p_dw)
            advec_est = est['advec']

            # ── RMSRE (vs true natural-scale params) ─────────────────────────
            if model in (1, 2):
                true_arr = np.array([true_d['sigmasq'], true_d['range'], true_d['range_time'],
                                     true_d['advec'], true_d['nugget']])
                est_arr  = np.array([est['sigmasq'], est['range'], est['range_time'],
                                     est['advec'], est['nugget']])
            else:
                true_arr = np.array([true_d['sigmasq'], true_d['range_lat'], true_d['range_lon'],
                                     true_d['range_time'], true_d['advec'], true_d['nugget']])
                est_arr  = np.array([est['sigmasq'], est['range_lat'], est['range_lon'],
                                     est['range_time'], est['advec'], est['nugget']])
            rmsre = float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2)))

            print(f"  RMSRE={rmsre:.4f}  ({t_elapsed:.1f}s)  "
                  f"grid={n1}×{n2}  p={p_time}  "
                  f"{mc.ADVEC_NAME}_est={advec_est:+.4f}  true={true_advec:+.4f}")

            # ── 2-Way sign check ─────────────────────────────────────────────
            if it + 1 >= sign_start:
                sc = sign_check_2way(p_dw, mc, I_samp, n1, n2, p_time,
                                     t_auto, DEVICE, true_advec)
                sc['iter'] = it + 1
                sc_records.append(sc)
                pd.DataFrame(sc_records).to_csv(output_path / csv_sc, index=False)

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ───────────────────────────────────────────────────────────
        row = {
            'iter':      it + 1,
            'rmsre':     round(rmsre,     6),
            'time_s':    round(t_elapsed, 2),
            'advec_est': round(advec_est, 6),
            'advec_true': true_advec,
        }
        if model in (1, 2):
            row.update({'sigmasq_est': round(est['sigmasq'], 6),
                        'range_est':   round(est['range'],   6),
                        'range_t_est': round(est['range_time'], 6),
                        'nugget_est':  round(est['nugget'],  6)})
        else:
            row.update({'sigmasq_est':   round(est['sigmasq'],    6),
                        'range_lat_est': round(est['range_lat'],  6),
                        'range_lon_est': round(est['range_lon'],  6),
                        'range_t_est':   round(est['range_time'], 6),
                        'nugget_est':    round(est['nugget'],     6)})
        records.append(row)

        pd.DataFrame(records).to_csv(output_path / csv_est, index=False)
        print_running_summary(records, sc_records, model, true_d)

    # ─────────────────────────────────────────────────────────────────────────
    # Final
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done: {num_iters} iters, {skipped} skipped  [{model_name}]")
    print(f"{'='*60}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_est, index=False)

    if sc_records:
        sc_df = pd.DataFrame(sc_records)
        sc_df.to_csv(output_path / csv_sc, index=False)
        n_sc  = len(sc_records)
        n_sym = sum(1 for r in sc_records if r['verdict'] == 'SYM')
        n_cor = sum(1 for r in sc_records if r['verdict'] == 'CORRECT')
        n_wrg = sum(1 for r in sc_records if r['verdict'] == 'WRONG')
        dLs   = [r['dL'] for r in sc_records if not np.isnan(r.get('dL', float('nan')))]
        print(f"\n  === Final 2-Way Sign Check Summary [{model_name}] ===")
        print(f"  n_check = {n_sc}")
        print(f"  SYM     (|ΔL|<0.5): {n_sym:3d}  ({100*n_sym/n_sc:.1f}%)")
        print(f"  CORRECT             : {n_cor:3d}  ({100*n_cor/n_sc:.1f}%)")
        print(f"  WRONG               : {n_wrg:3d}  ({100*n_wrg/n_sc:.1f}%)")
        if dLs:
            print(f"  mean ΔL = {np.mean(dLs):+.4f}   med ΔL = {np.median(dLs):+.4f}")
        print(f"  Key:  SYM → spectral likelihood cannot distinguish +/- {mc.ADVEC_NAME}")

    try:
        make_plots(records, sc_records, true_d, plot_dir, model, num_iters)
    except Exception as e:
        print(f"  [Plot error]: {e}")

    print(f"\n  CSV  → {output_path / csv_est}")
    print(f"  SC   → {output_path / csv_sc}")
    print(f"  Plots→ {plot_dir}/")


if __name__ == "__main__":
    app()
