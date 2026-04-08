"""
sim_dw_bimodal_040826.py

Simulation study: DW_raw bimodality diagnostic on a complete regular grid.

Design
------
Same data generation as sim_vdw_grid (direct FFT on target grid, no step3,
no NaN).  Only DW_raw is fitted — Vecchia removed for speed.

Two diagnostic additions
------------------------
1. Scatter plot  : iteration index vs estimate for every parameter.
   Shows clustering at modes visually.

2. Bimodal likelihood check  (activated after --bimodal-start iterations):
   When DW converges to point A, also evaluate the DW likelihood at
   point B = same params but advec_lat and advec_lon sign-flipped.

   L(A) ≈ L(B)  →  structural sign ambiguity  (two equivalent spectral modes)
   L(A) << L(B) →  optimization failure        (A is the correct global mode)

   This directly tests whether the bimodality in advec estimates is a real
   feature of the DW likelihood surface or an optimization artifact.

Usage:
  python sim_dw_bimodal_040826.py --num-iters 1
  python sim_dw_bimodal_040826.py --num-iters 300 --bimodal-start 50
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

from GEMS_TCO import debiased_whittle_raw as dw_raw_module
from GEMS_TCO import configuration as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

MODEL   = 'DW_raw'
P_COLS  = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
           'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
P_LABELS = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
            'advec_lat', 'advec_lon', 'nugget']

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ── Covariance kernel ──────────────────────────────────────────────────────────

def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


# ── FFT field on target grid ───────────────────────────────────────────────────

def generate_field_on_grid(lats_grid, lons_grid, t_steps, params, dlat, dlon):
    CPU = torch.device("cpu")
    F32 = torch.float32
    Nx, Ny, Nt = len(lats_grid), len(lons_grid), t_steps
    Px, Py, Pt = 2 * Nx, 2 * Ny, 2 * Nt

    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt

    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid(Lx, Ly, Lt, params_cpu)
    S = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Assemble complete-grid dataset ────────────────────────────────────────────

def assemble_grid_dataset(field, grid_coords, true_params, t_offset=21.0):
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_lat, N_lon, T = field.shape
    N_grid = N_lat * N_lon
    grid_map, grid_list = {}, []
    field_flat = field.reshape(N_grid, T)

    for t_idx in range(T):
        key   = f't{t_idx}'
        t_val = float(t_offset + t_idx)
        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0
        rows = torch.zeros(N_grid, 11, device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords
        rows[:, 2]  = field_flat[:, t_idx] + torch.randn(N_grid, device=DEVICE, dtype=DTYPE) * nugget_std
        rows[:, 3]  = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid, -1)
        grid_map[key] = rows.detach()
        grid_list.append(rows.detach())

    return grid_map, torch.cat(grid_list, dim=0)


# ── Metrics ────────────────────────────────────────────────────────────────────

def backmap_params(out_params):
    p = out_params
    if isinstance(p[0], torch.Tensor):
        p = [x.item() if x.numel() == 1 else x[0].item() for x in p]
    p = [float(x) for x in p]
    phi2 = np.exp(p[1]); phi3 = np.exp(p[2]); phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {
        'sigmasq':    np.exp(p[0]) / phi2,
        'range_lat':  rlon / phi3 ** 0.5,
        'range_lon':  rlon,
        'range_time': rlon / phi4 ** 0.5,
        'advec_lat':  p[4],
        'advec_lon':  p[5],
        'nugget':     np.exp(p[6]),
    }


def calculate_rmsre(out_params, true_dict):
    est     = backmap_params(out_params)
    est_arr = np.array([est['sigmasq'], est['range_lat'], est['range_lon'],
                        est['range_time'], est['advec_lat'], est['advec_lon'], est['nugget']])
    tru_arr = np.array([true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                        true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                        true_dict['nugget']])
    return float(np.sqrt(np.mean(((est_arr - tru_arr) / np.abs(tru_arr)) ** 2))), est


# ── Bimodal likelihood check ───────────────────────────────────────────────────

def params_list_to_tensor(params_list):
    """Convert list of 1-element tensors → flat tensor (detached)."""
    return torch.cat([p.detach().clone() for p in params_list])


def eval_dw_loss(params_tensor, I_samp, n1, n2, p_time, t_auto, device, dwl):
    """Evaluate DW raw loss (no grad) at a given flat parameter tensor."""
    pt = params_tensor.to(device).detach()
    with torch.no_grad():
        loss = dwl.whittle_likelihood_loss_tapered(
            pt, I_samp, n1, n2, p_time, t_auto, 0.044, 0.063)
    return loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else float('nan')


# Sign combinations: (s_lat, s_lon) applied to the converged advec values
# pp = original, mp = flip lat, pm = flip lon, mm = flip both
SIGN_COMBOS = [
    ('pp', +1, +1),
    ('mp', -1, +1),
    ('pm', +1, -1),
    ('mm', -1, -1),
]


def bimodal_check(params_list, I_samp, n1, n2, p_time, t_auto, device, dwl, true_dict):
    """
    Evaluate DW likelihood at all 4 sign combinations of (advec_lat, advec_lon).

    SIGN_COMBOS:
      pp (+lat, +lon)  — converged as-is
      mp (-lat, +lon)  — flip lat only
      pm (+lat, -lon)  — flip lon only
      mm (-lat, -lon)  — flip both

    Winner = combo with MINIMUM loss (best fit).

    Interpretation:
      All 4 L nearly equal        → strong structural symmetry (periodogram invariance)
      pp wins, others much worse  → optimization found global minimum, no issue
      pp loses to another combo   → optimization failure (converged to wrong mode)
      two combos tie              → partial sign ambiguity (one advec direction only)
    """
    p_base = params_list_to_tensor(params_list)
    est_base = backmap_params([p_base[i].unsqueeze(0) for i in range(7)])
    alat_base = est_base['advec_lat']
    alon_base = est_base['advec_lon']

    results = {}
    for tag, s_lat, s_lon in SIGN_COMBOS:
        p_c = p_base.clone()
        p_c[4] = s_lat * p_base[4]   # pp: keep as-is; others: negate lat/lon/both
        p_c[5] = s_lon * p_base[5]
        L = eval_dw_loss(p_c, I_samp, n1, n2, p_time, t_auto, device, dwl)
        alat_c = s_lat * alat_base
        alon_c = s_lon * alon_base
        results[tag] = {'L': L, 'advec_lat': alat_c, 'advec_lon': alon_c}

    # Identify winner (minimum L)
    valid = {k: v for k, v in results.items() if not np.isnan(v['L'])}
    winner = min(valid, key=lambda k: valid[k]['L']) if valid else 'nan'
    L_min  = valid[winner]['L'] if winner != 'nan' else float('nan')

    # ΔL relative to pp (converged point) for each combo
    L_pp = results['pp']['L']

    print(f"\n  ── 4-Way Sign Check (advec_lat × advec_lon) ──────────────────")
    print(f"  Converged: advec_lat={alat_base:+.4f}, advec_lon={alon_base:+.4f}")
    print(f"  True     : advec_lat={true_dict['advec_lat']:+.4f},"
          f" advec_lon={true_dict['advec_lon']:+.4f}")
    print(f"  {'Combo':<6} {'s_lat':>6} {'s_lon':>6}  {'advec_lat':>10}  "
          f"{'advec_lon':>10}  {'L':>12}  {'ΔL vs pp':>10}")
    print(f"  {'-'*70}")
    for tag, s_lat, s_lon in SIGN_COMBOS:
        r   = results[tag]
        dL  = (r['L'] - L_pp) if not np.isnan(r['L']) and not np.isnan(L_pp) else float('nan')
        mrk = ' ◄ WINNER' if tag == winner else ''
        print(f"  {tag:<6} {s_lat:>+6}  {s_lon:>+6}  {r['advec_lat']:>+10.4f}  "
              f"{r['advec_lon']:>+10.4f}  {r['L']:>12.6f}  {dL:>+10.4f}{mrk}")

    # Verdict
    if winner == 'nan':
        verdict = "NaN — could not evaluate"
    elif winner == 'pp':
        max_other = max(results[k]['L'] for k in ('mp', 'pm', 'mm')
                        if not np.isnan(results[k]['L']))
        gap = max_other - L_pp
        if gap < 0.5:
            verdict = "STRUCTURAL SYMMETRY  (pp wins but |ΔL| < 0.5 vs others)"
        else:
            verdict = f"CORRECT MODE         (pp is global min, gap={gap:.3f})"
    else:
        gap = results[winner]['L'] - L_pp   # negative means winner < pp
        verdict = (f"OPT. FAILURE / WRONG SIGN  "
                   f"(winner={winner}, ΔL(winner-pp)={gap:+.3f})")

    print(f"\n  Winner: {winner}  →  {verdict}")
    print(f"  ──────────────────────────────────────────────────────────────")

    out = {
        'advec_lat_conv': alat_base,
        'advec_lon_conv': alon_base,
        'winner': winner,
    }
    for tag, _, _ in SIGN_COMBOS:
        out[f'L_{tag}']        = results[tag]['L']
        out[f'advec_lat_{tag}'] = results[tag]['advec_lat']
        out[f'advec_lon_{tag}'] = results[tag]['advec_lon']
        dL = (results[tag]['L'] - L_pp) if not np.isnan(results[tag]['L']) and not np.isnan(L_pp) else float('nan')
        out[f'dL_{tag}_vs_pp'] = dL
    return out


# ── Running summary ────────────────────────────────────────────────────────────

def _bm_winner_table(bm_records):
    """Count and ΔL stats for each sign combo across bm_records."""
    if not bm_records:
        return
    n = len(bm_records)
    print(f"\n  [4-Way Sign Check — winner distribution  (n={n})]")
    print(f"  {'Combo':<6}  {'s_lat':>6} {'s_lon':>6}  "
          f"{'n_wins':>8}  {'%':>6}  {'mean ΔL vs pp':>15}  {'med ΔL':>10}")
    print(f"  {'-'*70}")
    for tag, s_lat, s_lon in SIGN_COMBOS:
        n_win = sum(1 for r in bm_records if r.get('winner') == tag)
        dLs   = [r[f'dL_{tag}_vs_pp'] for r in bm_records
                 if not np.isnan(r.get(f'dL_{tag}_vs_pp', float('nan')))]
        mean_dL = np.mean(dLs)  if dLs else float('nan')
        med_dL  = np.median(dLs) if dLs else float('nan')
        print(f"  {tag:<6}  {s_lat:>+6}  {s_lon:>+6}  "
              f"{n_win:>8}  {100*n_win/n:>5.1f}%  "
              f"{mean_dL:>15.4f}  {med_dL:>10.4f}")

    # Cross-tab: advec_lat sign × advec_lon sign of converged estimate
    print(f"\n  [Converged sign cross-tab  (advec_lat × advec_lon)]")
    print(f"  {'':16}  advec_lon < 0    advec_lon >= 0")
    for sign_lbl, cond_lat in [('advec_lat > 0 ', lambda r: r['advec_lat_conv'] > 0),
                                ('advec_lat <= 0', lambda r: r['advec_lat_conv'] <= 0)]:
        n_neg = sum(1 for r in bm_records if cond_lat(r) and r['advec_lon_conv'] < 0)
        n_pos = sum(1 for r in bm_records if cond_lat(r) and r['advec_lon_conv'] >= 0)
        print(f"  {sign_lbl}    {n_neg:>8} ({100*n_neg/n:4.1f}%)   "
              f"{n_pos:>8} ({100*n_pos/n:4.1f}%)")


def print_running_summary(records, true_dict, it, bm_records=None):
    tv_list = [true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]
    cw = 14
    n_done = len(records)

    print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
    hdr = f"  {'param':<13} {'true':>{cw}}  {'mean':>{cw}}  {'median':>{cw}}"
    print(hdr); print(f"  {'-'*60}")

    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        vals = np.array([r[col] for r in records])
        if len(vals) == 0:
            continue
        print(f"  {lbl:<13} {tv:{cw}.4f}  {vals.mean():{cw}.4f}  {np.median(vals):{cw}.4f}")

    print(f"\n  [5-number: Min | Q1 | Q2 | Q3 | Max]")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        vals = np.array([r[col] for r in records])
        if len(vals) < 2:
            continue
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        print(f"  {lbl:<13} {vals.min():.4f} | {q1:.4f} | {q2:.4f} | {q3:.4f} | {vals.max():.4f}")

    for metric_lbl, fn in [
        ('RMSRE',   lambda v, tv: float(np.sqrt(np.mean(((v - tv) / abs(tv)) ** 2)))),
        ('MdARE',   lambda v, tv: float(np.median(np.abs((v - tv) / abs(tv))))),
        ('P90-P10', lambda v, _tv: float(np.percentile(v, 90) - np.percentile(v, 10))),
    ]:
        print(f"\n  [{metric_lbl}]")
        for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
            vals = np.array([r[col] for r in records])
            if len(vals) >= 2:
                print(f"  {lbl:<13} {fn(vals, tv):.4f}")

    print(f"\n  [advec_lat split: >0 vs <=0]")
    alat = np.array([r['advec_lat_est'] for r in records])
    pos = alat[alat > 0]; neg = alat[alat <= 0]
    print(f"  advec_lat > 0  : n={len(pos):3d}  mean={pos.mean():.4f}" if len(pos) else "  advec_lat > 0  : n=  0")
    print(f"  advec_lat <= 0 : n={len(neg):3d}  mean={neg.mean():.4f}" if len(neg) else "  advec_lat <= 0 : n=  0")

    print(f"\n  [advec_lon split: <0 vs >=0]")
    alon = np.array([r['advec_lon_est'] for r in records])
    neg2 = alon[alon < 0]; pos2 = alon[alon >= 0]
    print(f"  advec_lon < 0  : n={len(neg2):3d}  mean={neg2.mean():.4f}" if len(neg2) else "  advec_lon < 0  : n=  0")
    print(f"  advec_lon >= 0 : n={len(pos2):3d}  mean={pos2.mean():.4f}" if len(pos2) else "  advec_lon >= 0 : n=  0")

    if bm_records:
        _bm_winner_table(bm_records)

    print(f"\n  [Overall RMSRE — last 5]")
    rmsres = [r['rmsre'] for r in records][-5:]
    print(f"    {[f'{v:.4f}' for v in rmsres]}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def make_plots(df, true_dict, plot_dir, n_iters):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    tv_list = [true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]

    COL_A = '#1976D2'   # advec_lat > 0 (near true)
    COL_B = '#E53935'   # advec_lat <= 0 (wrong sign)

    # ── 1. KDE distribution plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()
    fig.suptitle(f'DW_raw — Parameter Estimate Distributions  ({n_iters} iterations)',
                 fontsize=14, fontweight='bold')

    for idx, (lbl, col, tv) in enumerate(zip(P_LABELS, P_COLS, tv_list)):
        ax = axes[idx]
        vals = df[col].dropna().values
        if len(vals) < 3:
            ax.set_title(lbl); continue
        try:
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min() - 0.1*(vals.max()-vals.min()),
                             vals.max() + 0.1*(vals.max()-vals.min()), 300)
            ax.plot(xs, kde(xs), color=COL_A, lw=2.5)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=COL_A)
        except Exception:
            ax.hist(vals, bins=30, color=COL_A, alpha=0.5)
        ax.axvline(tv, color='black', ls='--', lw=1.5, label=f'true={tv:.4f}')
        ax.set_title(f'{lbl}  (true={tv:.4f})', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    if len(P_LABELS) < len(axes):
        axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'dw_bimodal_kde.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dw_bimodal_kde.png")

    # ── 2. Scatter plots: iteration vs estimate ───────────────────────────────
    # Color by advec_lat sign: blue = >0 (near true), red = <=0 (wrong sign)
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()
    fig.suptitle(f'DW_raw — Scatter: Iteration vs Estimate  (colored by advec_lat sign)',
                 fontsize=13, fontweight='bold')

    colors = np.where(df['advec_lat_est'].values > 0, COL_A, COL_B)

    for idx, (lbl, col, tv) in enumerate(zip(P_LABELS, P_COLS, tv_list)):
        ax = axes[idx]
        iters = df['iter'].values
        vals  = df[col].values
        ax.scatter(iters, vals, c=colors, s=20, alpha=0.6, linewidths=0)
        ax.axhline(tv, color='black', ls='--', lw=1.5, label=f'true={tv:.4f}')
        # Running median (window=20)
        if len(vals) >= 20:
            rm = pd.Series(vals).rolling(20, center=True, min_periods=5).median().values
            ax.plot(iters, rm, color='darkorange', lw=1.5, alpha=0.8, label='roll. median')
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_title(f'{lbl}  (true={tv:.4f})', fontsize=10)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.grid(True, alpha=0.2)

    # Legend patch
    from matplotlib.patches import Patch
    legend_elements = [Patch(fc=COL_A, label='advec_lat > 0'),
                       Patch(fc=COL_B, label='advec_lat ≤ 0')]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
               bbox_to_anchor=(0.98, 0.01))

    if len(P_LABELS) < len(axes):
        axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'dw_bimodal_scatter.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dw_bimodal_scatter.png")

    # ── 3. advec 2D scatter: advec_lat vs advec_lon ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    alat = df['advec_lat_est'].values
    alon = df['advec_lon_est'].values
    ax.scatter(alat, alon, c=colors, s=25, alpha=0.7, linewidths=0)
    ax.scatter([true_dict['advec_lat']], [true_dict['advec_lon']],
               marker='*', s=250, color='gold', edgecolors='black', lw=1.2,
               zorder=10, label='True')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('advec_lat estimate', fontsize=11)
    ax.set_ylabel('advec_lon estimate', fontsize=11)
    ax.set_title(f'DW_raw — advec_lat vs advec_lon  ({n_iters} iters)', fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=COL_A, label='advec_lat > 0'),
                        Patch(fc=COL_B, label='advec_lat ≤ 0'),
                        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='gold',
                                   markersize=12, label='True')],
              fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'dw_bimodal_advec2d.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dw_bimodal_advec2d.png")

    # ── 4. Bimodal check: ΔL = L(A) - L(B) over iterations ──────────────────
    # (bimodal ΔL plot called separately via make_bimodal_plot)


def make_bimodal_plot(bm_records, plot_dir):
    if not bm_records:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    iters    = [r['iter'] for r in bm_records]
    alat_A   = [r['advec_lat_conv'] for r in bm_records]
    alon_A   = [r['advec_lon_conv'] for r in bm_records]
    winners  = [r['winner'] for r in bm_records]

    # Color map for 4 combos
    COMBO_COLORS = {'pp': '#1976D2', 'mp': '#E53935', 'pm': '#43A047', 'mm': '#FF6F00'}
    win_colors = [COMBO_COLORS.get(w, 'gray') for w in winners]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle('4-Way Sign Check: L(pp) vs L(mp) vs L(pm) vs L(mm)',
                 fontsize=13, fontweight='bold')

    # Panel (0,0)-(0,3): ΔL per combo over iterations
    combo_labels = {'pp': '(+lat,+lon) pp', 'mp': '(−lat,+lon) mp',
                    'pm': '(+lat,−lon) pm', 'mm': '(−lat,−lon) mm'}
    for idx, (tag, _, _) in enumerate(SIGN_COMBOS):
        ax = axes[idx // 2][idx % 2]
        dLs = [r.get(f'dL_{tag}_vs_pp', float('nan')) for r in bm_records]
        pt_colors = [COMBO_COLORS[tag]] * len(iters)
        ax.scatter(iters, dLs, c=pt_colors, s=20, alpha=0.7)
        ax.axhline(0,    color='gray',  ls='-',  lw=0.8)
        ax.axhline(0.5,  color='green', ls='--', lw=1.0, alpha=0.6)
        ax.axhline(-0.5, color='green', ls='--', lw=1.0, alpha=0.6)
        ax.set_title(f'ΔL for {combo_labels[tag]}', fontsize=10)
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel('ΔL vs pp', fontsize=9)
        ax.grid(True, alpha=0.2)

    # Panel (2,0): advec_lat_conv colored by winner
    ax5 = axes[2][0]
    ax5.scatter(iters, alat_A, c=win_colors, s=20, alpha=0.8)
    ax5.axhline(0, color='gray', ls='--', lw=0.8)
    ax5.set_xlabel('Iteration', fontsize=9)
    ax5.set_ylabel('advec_lat (converged)', fontsize=9)
    ax5.set_title('Converged advec_lat — colored by winner combo', fontsize=10)
    ax5.grid(True, alpha=0.2)

    # Panel (2,1): advec scatter lat vs lon colored by winner
    ax6 = axes[2][1]
    ax6.scatter(alat_A, alon_A, c=win_colors, s=20, alpha=0.8)
    ax6.axvline(0, color='gray', ls='--', lw=0.8)
    ax6.axhline(0, color='gray', ls='--', lw=0.8)
    ax6.set_xlabel('advec_lat (converged)', fontsize=9)
    ax6.set_ylabel('advec_lon (converged)', fontsize=9)
    ax6.set_title('advec 2D — colored by winner combo', fontsize=10)
    ax6.grid(True, alpha=0.2)

    legend_el = [Patch(fc=COMBO_COLORS[t], label=combo_labels[t])
                 for t, _, _ in SIGN_COMBOS]
    fig.legend(handles=legend_el, loc='lower center', fontsize=9,
               ncol=4, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(plot_dir / 'dw_bimodal_4way.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dw_bimodal_4way.png")


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    num_iters: int = typer.Option(10, help="Simulation iterations"),
    bimodal_start: int = typer.Option(50, help="Start bimodal check after this many iters"),
    lat_range: str = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range: str = typer.Option("121,131", help="lon_min,lon_max"),
    init_noise: float = typer.Option(0.7, help="Uniform noise half-width in log space"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:

    rng = np.random.default_rng(seed)
    lat_r = [float(x) for x in lat_range.split(',')]
    lon_r = [float(x) for x in lon_range.split(',')]

    print(f"Device        : {DEVICE}")
    print(f"Region        : lat {lat_r}, lon {lon_r}")
    print(f"Model         : DW_raw (identity filter, per-slice demean, DC excluded)")
    print(f"Data          : direct grid FFT — no step3, no NaN")
    print(f"Bimodal check : starts at iter {bimodal_start}")
    print(f"Init noise    : ±{init_noise} log-space")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_dw_bimodal_{date_tag}.csv"
    csv_bm      = f"sim_dw_bimodal_check_{date_tag}.csv"

    # ── True parameters ──────────────────────────────────────────────────────
    true_dict = {
        'sigmasq':    13.059,
        'range_lat':  0.154,
        'range_lon':  0.195,
        'range_time': 1.0,
        'advec_lat':  0.0218,
        'advec_lon':  -0.1689,
        'nugget':     0.247,
    }
    phi2 = 1.0 / true_dict['range_lon']
    phi1 = true_dict['sigmasq'] * phi2
    phi3 = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4 = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    def make_init(rng):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            scale = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng.uniform(-2 * scale, 2 * scale)
        return noisy

    # ── Build target grid ────────────────────────────────────────────────────
    print("\n[Setup 1/2] Building target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    dlat, dlon = DELTA_LAT_BASE, DELTA_LON_BASE
    print(f"  Grid: {len(lats_grid)} lat × {len(lons_grid)} lon = {N_grid} cells  "
          f"(δlat={dlat:.4f}°, δlon={dlon:.4f}°)")

    # ── DW setup ─────────────────────────────────────────────────────────────
    print("[Setup 2/2] Verifying dataset structure...")
    dwl = dw_raw_module.debiased_whittle_likelihood()
    _f0 = generate_field_on_grid(lats_grid, lons_grid, 8, true_params, dlat, dlon)
    _gm0, _ga0 = assemble_grid_dataset(_f0, grid_coords, true_params)
    _first = list(_gm0.values())[0]
    print(f"  Sample: {(~torch.isnan(_first[:,2])).sum().item()}/{N_grid} valid (should be {N_grid})")
    del _f0, _gm0, _ga0

    LBFGS_EVAL = 20
    LBFGS_HIST = 10
    DWL_STEPS  = 5
    LC, NC, VC, TC = 0, 1, 2, 3

    records    = []
    bm_records = []
    skipped    = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iteration {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*60}")

        initial_vals = make_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Init: sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  "
              f"nugget={init_orig['nugget']:.3f}")

        try:
            field = generate_field_on_grid(lats_grid, lons_grid, 8, true_params, dlat, dlon)
            grid_map, grid_agg = assemble_grid_dataset(field, grid_coords, true_params)
            del field

            p_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                    for val in initial_vals]

            db = dw_raw_module.debiased_whittle_preprocess(
                [grid_agg], [grid_map], day_idx=0,
                params_list=[true_dict['sigmasq'], true_dict['range_lat'],
                             true_dict['range_lon'], true_dict['range_time'],
                             true_dict['advec_lat'], true_dict['advec_lon'],
                             true_dict['nugget']],
                lat_range=lat_r, lon_range=lon_r
            )
            cur_df = db.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
            unique_t    = torch.unique(cur_df[:, TC])
            time_slices = [cur_df[cur_df[:, TC] == t] for t in unique_t]

            J_vec, n1, n2, p_time, taper, obs_masks = dwl.generate_Jvector_tapered_mv(
                time_slices, dwl.cgn_hamming, LC, NC, VC, DEVICE)
            I_samp = dwl.calculate_sample_periodogram_vectorized(J_vec)
            t_auto = dwl.calculate_taper_autocorrelation_multivariate(
                taper, obs_masks, n1, n2, DEVICE)
            del obs_masks

            opt_dw = torch.optim.LBFGS(
                p_dw, lr=1.0, max_iter=20, max_eval=LBFGS_EVAL,
                history_size=LBFGS_HIST, line_search_fn="strong_wolfe",
                tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, _, _ = dwl.run_lbfgs_tapered(
                params_list=p_dw, optimizer=opt_dw, I_sample=I_samp,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
                max_steps=DWL_STEPS, device=DEVICE)
            t_dw = time.time() - t0

            rmsre_dw, est_dw = calculate_rmsre([p.item() for p in p_dw], true_dict)
            print(f"  RMSRE = {rmsre_dw:.4f}  ({t_dw:.1f}s)  grid: {n1}×{n2}, p={p_time}")

            # ── Bimodal likelihood check ───────────────────────────────────
            if it + 1 >= bimodal_start:
                bm = bimodal_check(p_dw, I_samp, n1, n2, p_time, t_auto,
                                   DEVICE, dwl, true_dict)
                bm['iter'] = it + 1
                bm_records.append(bm)
                pd.DataFrame(bm_records).to_csv(output_path / csv_bm, index=False)

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        records.append({
            'iter':          it + 1,
            'rmsre':         round(rmsre_dw,           6),
            'time_s':        round(t_dw,               2),
            'sigmasq_est':   round(est_dw['sigmasq'],   6),
            'range_lat_est': round(est_dw['range_lat'], 6),
            'range_lon_est': round(est_dw['range_lon'], 6),
            'range_t_est':   round(est_dw['range_time'],6),
            'advec_lat_est': round(est_dw['advec_lat'], 6),
            'advec_lon_est': round(est_dw['advec_lon'], 6),
            'nugget_est':    round(est_dw['nugget'],    6),
            'init_sigmasq':  round(init_orig['sigmasq'],  4),
            'init_rlon':     round(init_orig['range_lon'],4),
        })

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)
        print_running_summary(records, true_dict, it, bm_records=bm_records)

    # ── Final ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done: {num_iters} iters, {skipped} skipped")
    print(f"{'='*60}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_raw, index=False)

    # ── Bimodal summary ───────────────────────────────────────────────────────
    if bm_records:
        bm_df = pd.DataFrame(bm_records)
        n = len(bm_df)
        print(f"\n{'='*65}")
        print(f"  FINAL 4-WAY SIGN CHECK SUMMARY  ({n} checks, iter ≥ {bimodal_start})")
        print(f"{'='*65}")
        _bm_winner_table(bm_records)

        # ΔL distribution per combo
        print(f"\n  [ΔL distribution per combo  (relative to pp = converged)]")
        print(f"  {'Combo':<6}  {'mean':>10}  {'median':>10}  {'std':>8}  "
              f"{'|ΔL|<0.5':>10}  {'|ΔL|<1.0':>10}")
        print(f"  {'-'*65}")
        for tag, _, _ in SIGN_COMBOS:
            col = f'dL_{tag}_vs_pp'
            dLs = bm_df[col].dropna().values
            if len(dLs) == 0:
                continue
            n05 = (np.abs(dLs) < 0.5).sum()
            n10 = (np.abs(dLs) < 1.0).sum()
            print(f"  {tag:<6}  {np.mean(dLs):>10.4f}  {np.median(dLs):>10.4f}  "
                  f"{np.std(dLs):>8.4f}  {n05:>5}/{n:>3}={100*n05/n:4.1f}%  "
                  f"{n10:>5}/{n:>3}={100*n10/n:4.1f}%")

        # Verdict counts
        winners = bm_df['winner'].value_counts()
        n_pp = winners.get('pp', 0)
        n_other = n - n_pp
        print(f"\n  pp (converged) wins: {n_pp}/{n} ({100*n_pp/n:.1f}%)")
        print(f"  Other combo wins  : {n_other}/{n} ({100*n_other/n:.1f}%)  ← opt. failure")

        # All-close check: are all 4 L values within 0.5 of each other?
        n_all_sym = 0
        for r in bm_records:
            Ls = [r[f'L_{tag}'] for tag, _, _ in SIGN_COMBOS
                  if not np.isnan(r.get(f'L_{tag}', float('nan')))]
            if len(Ls) == 4 and (max(Ls) - min(Ls)) < 0.5:
                n_all_sym += 1
        print(f"  All 4 L within 0.5: {n_all_sym}/{n} ({100*n_all_sym/n:.1f}%)"
              f"  ← structural 4-way symmetry")
    else:
        bm_df = pd.DataFrame()

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        plot_dir = output_path / "plots" / "dw_bimodal"
        plot_dir.mkdir(parents=True, exist_ok=True)
        make_plots(df_final, true_dict, plot_dir, num_iters)
        if bm_records:
            make_bimodal_plot(bm_records, plot_dir)
        print(f"\n  Plots saved to: {plot_dir}")
    except Exception as e:
        import traceback
        print(f"  [Plot error] {e}")
        traceback.print_exc()

    print(f"\n  Saved: {csv_raw}")
    if bm_records:
        print(f"  Saved: {csv_bm}")


if __name__ == "__main__":
    app()
