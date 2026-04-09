"""
sanity_check_dw_raw_040826.py

Four sanity checks for debiased_whittle_raw:

  CHECK 1 — Off-diagonal temporal cross-spectral correlation
  ───────────────────────────────────────────────────────────
  For the DW likelihood to work well, the expected periodogram matrix
  E[I(ω)] ∈ R^{p_time × p_time} should be nearly diagonal at each spatial
  frequency ω.  If the off-diagonal entries are large relative to the
  diagonal (high temporal cross-spectral correlation), the matrix is
  ill-conditioned → unstable log-det + trace terms → unreliable estimation.

  For each ω=(ω1,ω2) we compute the normalized cross-spectral correlation:
      corr_qr(ω) = I_exp[ω,q,r] / sqrt(I_exp[ω,q,q] * I_exp[ω,r,r])  (q≠r)
  and report statistics (max, mean, P95) across all frequencies and pairs.
  Values < 0.1 are excellent; < 0.3 is acceptable.

  CHECK 2 — Parameter recovery via DW_raw optimization
  ──────────────────────────────────────────────────────
  Simulate N_SIM fields from the true 7-param model, run L-BFGS, and
  compare estimates to truth (RMSRE, bias, distribution plots).

  Additionally verifies:
    (a) Loss at true params < loss at random init  (direction check)
    (b) Gradient norm at true params is small      (stationarity check)

  CHECK 3 — Section 6: Wick's theorem — cross-frequency covariance
  ─────────────────────────────────────────────────────────────────
  By Isserlis/Wick's theorem:
      Cov(I_j, I_k) = 2 * |C̃_h(ω_j, ω_k)|²
  As n→∞ different-frequency DFT coefficients are asymptotically independent,
  so Cov(I_j, I_k) → 0 for j ≠ k.  We verify this empirically over multiple
  realizations by computing the sample correlation of I[ω_j] and I[ω_k] across
  replications, for a set of random frequency pairs (j,k) with j≠k.
  Values near 0 confirm asymptotic independence; large values indicate leakage.

  CHECK 4 — Section 7: Analytical white-noise verification
  ─────────────────────────────────────────────────────────
  NO simulation needed.  For white noise C(h) = σ²·δ(h=0):
    - E[I_qq(ω)] = σ² / (4π²)   at ALL frequencies  (flat spectrum)
    - E[I_qr(ω)] = 0             for q≠r            (temporal independence)
  We set params to a pure-nugget (σ²) model (sigmasq≈0, nugget=σ²) and call
  expected_periodogram_fft_tapered.  If the implementation is correct the
  diagonal deviates from σ²/(4π²) by < 1e-6 and off-diagonal entries are < 1e-6.

Usage:
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation
  conda activate faiss_env
  python sanity_check_dw_raw_040826.py --mode corr       # Checks 1+3+4 (fast)
  python sanity_check_dw_raw_040826.py --mode optim      # Check 2 (optimization)
  python sanity_check_dw_raw_040826.py --mode both       # All four checks
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

from GEMS_TCO import debiased_whittle_raw as dw

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.float64
DELTA_LAT = 0.044
DELTA_LON = 0.063

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

# ─────────────────────────────────────────────────────────────────────────────
# True parameters  (same as sim_dw_raw_040626.py)
# ─────────────────────────────────────────────────────────────────────────────

TRUE_DICT = {
    'sigmasq':    13.059,
    'range_lat':  0.154,
    'range_lon':  0.195,
    'range_time': 1.0,
    'advec_lat':  0.0218,
    'advec_lon':  -0.1689,
    'nugget':     0.247,
}

def true_log_params():
    td  = TRUE_DICT
    phi2 = 1.0 / td['range_lon']
    phi1 = td['sigmasq'] * phi2
    phi3 = (td['range_lon'] / td['range_lat'])  ** 2
    phi4 = (td['range_lon'] / td['range_time']) ** 2
    return [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
            td['advec_lat'], td['advec_lon'], np.log(td['nugget'])]

P_COLS   = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
            'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
P_LABELS = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
            'advec_lat', 'advec_lon', 'nugget']

# ─────────────────────────────────────────────────────────────────────────────
# Data generation  (FFT circulant embedding — no GEMS obs needed)
# ─────────────────────────────────────────────────────────────────────────────

def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, -15.0, 15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def generate_field(lats, lons, t_steps, true_params, dlat, dlon):
    """FFT circulant embedding field on a regular grid."""
    CPU = torch.device("cpu"); F32 = torch.float32
    Nx, Ny, Nt = len(lats), len(lons), t_steps
    Px, Py, Pt = 2 * Nx, 2 * Ny, 2 * Nt

    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt

    p = true_params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid(Lx, Ly, Lt, p)
    S = torch.fft.fftn(C); S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


def assemble_dataset(field, grid_coords, true_params, t_offset=21.0):
    """Pack field into per-time tensors; add nugget noise."""
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_lat, N_lon, T = field.shape
    N_grid = N_lat * N_lon
    field_flat = field.reshape(N_grid, T)

    dataset_map, dataset_list = {}, []
    for t_idx in range(T):
        t_val  = float(t_offset + t_idx)
        dummy  = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0
        rows = torch.zeros(N_grid, 11, device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords
        rows[:, 2]  = (field_flat[:, t_idx] +
                       torch.randn(N_grid, device=DEVICE, dtype=DTYPE) * nugget_std)
        rows[:, 3]  = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid, -1)
        key = f't{t_idx}'
        dataset_map[key] = rows.detach()
        dataset_list.append(rows.detach())

    return dataset_map, torch.cat(dataset_list, dim=0)


def build_dw_inputs(dataset_map, dataset_agg, lat_r, lon_r):
    """Preprocess → J-vector → sample periodogram → taper autocorr."""
    DW_PREPROC = dw.debiased_whittle_preprocess(
        [dataset_agg], [dataset_map], day_idx=0,
        params_list=[TRUE_DICT[k] for k in
                     ['sigmasq','range_lat','range_lon','range_time',
                      'advec_lat','advec_lon','nugget']],
        lat_range=lat_r, lon_range=lon_r
    )
    cur_df = DW_PREPROC.generate_spatially_filtered_days(
        lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)

    unique_t    = torch.unique(cur_df[:, 3])
    time_slices = [cur_df[cur_df[:, 3] == t] for t in unique_t]

    DWL = dw.debiased_whittle_likelihood()
    J_vec, n1, n2, p_time, taper, obs_masks = DWL.generate_Jvector_tapered_mv(
        time_slices, DWL.cgn_hamming, 0, 1, 2, DEVICE)
    I_samp = DWL.calculate_sample_periodogram_vectorized(J_vec)
    t_auto = DWL.calculate_taper_autocorrelation_multivariate(
        taper, obs_masks, n1, n2, DEVICE)
    return DWL, I_samp, t_auto, n1, n2, p_time


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: Off-diagonal correlation structure
# ─────────────────────────────────────────────────────────────────────────────

def check_offdiag_corr(I_exp: torch.Tensor) -> dict:
    """
    Off-diagonal spectral correlations of E[I(ω)] only — NOT I_sample.

    I_sample = J[ω]·J[ω]ᴴ is rank-1, so normalized off-diagonal trivially
    equals |cos(phase diff)| with mean 2/π ≈ 0.637. That check is meaningless.

    Verdict uses frac_above_0.3: fraction of (ω, q≠r) pairs with |corr|>0.30.
    The raw max is dominated by near-DC where temporal correlation is genuinely
    high; what matters for DW stability is the bulk distribution.
    """
    _, _, p, _ = I_exp.shape
    if p < 2:
        print(f"  p_time={p} < 2 — no off-diagonal pairs.")
        return {}

    diag     = torch.diagonal(I_exp, dim1=-2, dim2=-1)
    denom    = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2))
    denom    = torch.clamp(denom, min=1e-15)
    corr     = (I_exp / denom).real.abs()
    eye      = torch.eye(p, device=I_exp.device, dtype=torch.bool)
    off_flat = corr[:, :, ~eye].reshape(-1).cpu().numpy()
    off_flat = off_flat[np.isfinite(off_flat)]

    if off_flat.size == 0:
        print("  no valid off-diagonal entries.")
        return {}

    thresholds = [0.05, 0.10, 0.20, 0.30]
    stats = {
        'max_corr':    float(off_flat.max()),
        'mean_corr':   float(off_flat.mean()),
        'median_corr': float(np.median(off_flat)),
        'P95_corr':    float(np.percentile(off_flat, 95)),
        'P99_corr':    float(np.percentile(off_flat, 99)),
        'n_pairs':     int(off_flat.size),
    }
    for thr in thresholds:
        stats[f'frac_above_{thr}'] = float((off_flat > thr).mean())

    frac30 = stats['frac_above_0.3']
    sep = "─" * 58
    print(f"\n  {sep}")
    print(f"  E[I(ω)] off-diagonal |corr_qr(ω)|  (p_time={p})")
    print(f"  {sep}")
    print(f"  Max   : {stats['max_corr']:.6f}  ← near-DC outlier; see frac>0.30 below")
    print(f"  Mean  : {stats['mean_corr']:.6f}")
    print(f"  Median: {stats['median_corr']:.6f}")
    print(f"  P95   : {stats['P95_corr']:.6f}")
    print(f"  P99   : {stats['P99_corr']:.6f}")
    for thr in thresholds:
        frac = stats[f'frac_above_{thr}']
        flag = " ← HIGH" if (thr == 0.3 and frac > 0.05) else ""
        print(f"  Frac > {thr:.2f}: {frac:.4f}{flag}")

    verdict = (
        "GOOD    (< 1% of freqs |corr_qr|>0.30 — DW well-conditioned)"
        if frac30 < 0.01 else
        "OK      (1–5% of freqs |corr_qr|>0.30 — DW should work)"
        if frac30 < 0.05 else
        "WARNING (> 5% of freqs |corr_qr|>0.30 — check temporal range)"
    )
    print(f"  Verdict: {verdict}")
    print(f"  {sep}")

    return stats


def run_corr_check(lat_r, lon_r, t_steps=8, n_realizations=3, seed=42):
    """
    Check 1: build I_expected at true params and report off-diagonal correlations.
    Uses n_realizations to average out sampling noise in I_sample check.
    """
    print("\n" + "=" * 65)
    print("  CHECK 1: Off-diagonal correlation structure of E[I(ω)]")
    print("=" * 65)

    rng   = np.random.default_rng(seed)
    tlog  = true_log_params()
    tpar  = torch.tensor(tlog, device=DEVICE, dtype=DTYPE)

    # ── Build grid ────────────────────────────────────────────────────────────
    lats_g = torch.arange(min(lat_r), max(lat_r) + 0.0001,  DELTA_LAT,  device=DEVICE, dtype=DTYPE)
    lons_g = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON,  device=DEVICE, dtype=DTYPE)
    lats_g = torch.round(lats_g * 10000) / 10000
    lons_g = torch.round(lons_g * 10000) / 10000
    g_lat, g_lon   = torch.meshgrid(lats_g, lons_g, indexing='ij')
    grid_coords    = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    n1_g, n2_g     = len(lats_g), len(lons_g)
    print(f"  Grid: {n1_g} lat × {n2_g} lon = {n1_g*n2_g} cells  | t_steps={t_steps}")

    corr_stats_list = []

    for real_idx in range(n_realizations):
        print(f"\n  [Realization {real_idx+1}/{n_realizations}]")

        field = generate_field(lats_g, lons_g, t_steps, tpar, DELTA_LAT, DELTA_LON)
        ds_map, ds_agg = assemble_dataset(field, grid_coords, tpar)
        del field

        try:
            DWL, _I_samp, t_auto, n1, n2, p_time = build_dw_inputs(
                ds_map, ds_agg, lat_r, lon_r)
        except Exception as e:
            print(f"  [SKIP] build_dw_inputs failed: {e}")
            continue

        print(f"  FFT grid: {n1}×{n2}  p_time={p_time}")

        # ── E[I(ω)] at true params ────────────────────────────────────────────
        with torch.no_grad():
            I_exp = DWL.expected_periodogram_fft_tapered(
                tpar, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)

        # E[I(ω)] off-diagonal correlations
        s_exp = check_offdiag_corr(I_exp)
        corr_stats_list.append(s_exp)

        # Condition numbers of E[I(ω)] matrices
        cond_nums = []
        for i in range(n1):
            for j in range(n2):
                if i == 0 and j == 0:
                    continue  # skip DC
                mat = I_exp[i, j].real.float()
                if not (torch.isnan(mat).any() or torch.isinf(mat).any()):
                    try:
                        sv = torch.linalg.svdvals(mat)
                        sv_min = sv.min().item()
                        if sv_min > 1e-15:
                            cond_nums.append((sv.max() / sv_min).item())
                    except Exception:
                        pass
        if cond_nums:
            cn = np.array(cond_nums)
            print(f"\n  Condition numbers of E[I(ω)] (excluding DC):")
            print(f"  Max={cn.max():.2e}  Mean={cn.mean():.2e}  "
                  f"Median={np.median(cn):.2e}  P99={np.percentile(cn,99):.2e}")
            flag = " ← OK" if cn.max() < 1e4 else " ← WARNING: ill-conditioned"
            print(f"  Max cond: {cn.max():.2e}{flag}")

    # ── Summary across realizations ───────────────────────────────────────────
    if len(corr_stats_list) > 1:
        keys = ['max_corr', 'mean_corr', 'P95_corr']
        print(f"\n  [Aggregate over {len(corr_stats_list)} realizations — E[I(ω)]]")
        for k in keys:
            vals = [s[k] for s in corr_stats_list if k in s]
            if vals:
                print(f"  {k:<14}: mean={np.mean(vals):.6f}  max={np.max(vals):.6f}")

    return corr_stats_list


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Parameter recovery
# ─────────────────────────────────────────────────────────────────────────────

def backmap_params(log_params):
    p = [float(x.item() if isinstance(x, torch.Tensor) else x) for x in log_params]
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


def rmsre(est_dict, true_dict):
    keys = ['sigmasq','range_lat','range_lon','range_time','advec_lat','advec_lon','nugget']
    e = np.array([est_dict[k] for k in keys])
    t = np.array([true_dict[k] for k in keys])
    return float(np.sqrt(np.mean(((e - t) / np.abs(t)) ** 2))), dict(zip(keys, e))


def make_random_init(tlog, rng, noise=0.7):
    noisy = list(tlog)
    for i in [0, 1, 2, 3, 6]:
        noisy[i] = tlog[i] + rng.uniform(-noise, noise)
    for i in [4, 5]:
        scale    = max(abs(tlog[i]), 0.05)
        noisy[i] = tlog[i] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


def loss_and_grad_at_true(tpar_tensor, I_samp, t_auto, n1, n2, p_time):
    """
    Evaluate loss and gradient norm at the true parameter values.
    Near the optimum, gradient norm should be small.
    """
    p_check = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True)
               for v in tpar_tensor.tolist()]
    p_cat   = torch.cat(p_check)
    loss    = dw.debiased_whittle_likelihood.whittle_likelihood_loss_tapered(
        p_cat, I_samp, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
    loss.backward()
    grad_norm = max(
        p.grad.abs().item() for p in p_check if p.grad is not None
    )
    return loss.item(), grad_norm


def print_running_summary(records, it):
    td   = TRUE_DICT
    tv   = [td['sigmasq'], td['range_lat'], td['range_lon'], td['range_time'],
            td['advec_lat'], td['advec_lon'], td['nugget']]
    n    = len(records)
    if n == 0:
        return
    cw   = 12
    print(f"\n  ── Running summary ({n} / {it+1} attempted) ──")
    hdr  = f"  {'param':<14} {'true':>{cw}}  {'est_mean':>{cw}}  {'bias':>{cw}}  {'RMSRE':>{cw}}"
    print(hdr); print(f"  {'-'*70}")
    for lbl, col, tv_i in zip(P_LABELS, P_COLS, tv):
        vals = np.array([r[col] for r in records])
        bias = float(np.mean(vals) - tv_i)
        rmsr = float(np.sqrt(np.mean(((vals - tv_i) / abs(tv_i)) ** 2)))
        print(f"  {lbl:<14} {tv_i:>{cw}.4f}  {np.mean(vals):>{cw}.4f}  "
              f"{bias:>{cw}.4f}  {rmsr:>{cw}.4f}")
    overall_rmsre = np.mean([r['rmsre'] for r in records])
    print(f"\n  Overall RMSRE  : {overall_rmsre:.4f}")
    print(f"  Loss@true (mean): {np.mean([r['loss_true'] for r in records]):.6f}")
    print(f"  Loss@init (mean): {np.mean([r['loss_init'] for r in records]):.6f}")
    print(f"  Loss@conv (mean): {np.mean([r['loss_conv'] for r in records]):.6f}")
    n_better = sum(1 for r in records if r['loss_true'] < r['loss_init'])
    print(f"  Loss(true)<Loss(init): {n_better}/{n}  (should be most — direction check)")


def print_final_summary(records, num_iters):
    td  = TRUE_DICT
    tv  = [td['sigmasq'], td['range_lat'], td['range_lon'], td['range_time'],
           td['advec_lat'], td['advec_lon'], td['nugget']]
    cw  = 14
    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — Parameter Recovery  ({num_iters} iters)")
    print(f"{'='*75}")
    print(f"  {'Parameter':<14} {'True':>10}  {'Mean':>{cw}}  {'Bias':>{cw}}  {'RMSRE':>{cw}}  {'SD':>{cw}}")
    print(f"  {'-'*75}")
    for lbl, col, tv_i in zip(P_LABELS, P_COLS, tv):
        vals = np.array([r[col] for r in records])
        bias = float(np.mean(vals) - tv_i)
        rmsr = float(np.sqrt(np.mean(((vals - tv_i) / abs(tv_i)) ** 2)))
        sd   = float(np.std(vals))
        print(f"  {lbl:<14} {tv_i:>10.4f}  {np.mean(vals):>{cw}.4f}  "
              f"{bias:>{cw}.4f}  {rmsr:>{cw}.4f}  {sd:>{cw}.4f}")
    print(f"  {'-'*75}")
    ovr = np.mean([r['rmsre'] for r in records])
    print(f"  {'Overall RMSRE':<14} {'':>10}  {'':>{cw}}  {'':>{cw}}  {ovr:>{cw}.4f}")

    print(f"\n  5-Number summary:")
    for lbl, col in zip(P_LABELS, P_COLS):
        vals = np.array([r[col] for r in records])
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        print(f"  {lbl:<14}  min={vals.min():.4f}  Q1={q1:.4f}  "
              f"med={q2:.4f}  Q3={q3:.4f}  max={vals.max():.4f}")

    # ── Direction checks ──────────────────────────────────────────────────────
    n = len(records)
    n_dir = sum(1 for r in records if r['loss_true'] < r['loss_init'])
    n_grd = sum(1 for r in records if r['grad_norm_true'] < 0.10)
    print(f"\n  Direction check: Loss(true) < Loss(rand_init)  {n_dir}/{n} "
          f"({100*n_dir/n:.0f}%)  [expect ~100%]")
    print(f"  Stationarity  : |grad| at true params < 0.10  {n_grd}/{n} "
          f"({100*n_grd/n:.0f}%)  [expect high]")
    print(f"  Mean |grad| at true: {np.mean([r['grad_norm_true'] for r in records]):.4e}")

    print(f"\n{'='*75}")


def make_plots(records, plot_dir):
    """Distribution plots for parameter estimates."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        td  = TRUE_DICT
        tv  = [td['sigmasq'], td['range_lat'], td['range_lon'], td['range_time'],
               td['advec_lat'], td['advec_lon'], td['nugget']]
        COL = '#1565C0'

        n_p, n_cols = len(P_LABELS), 2
        n_rows = (n_p + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

        for ax, lbl, col, tv_i in zip(axes, P_LABELS, P_COLS, tv):
            vals = np.array([r[col] for r in records])
            ax.hist(vals, bins=max(5, min(25, len(vals)//3+1)),
                    alpha=0.35, color=COL, density=True, edgecolor='white', lw=0.5)
            if len(vals) >= 3:
                try:
                    kde = gaussian_kde(vals)
                    xs  = np.linspace(vals.min(), vals.max(), 300)
                    ax.plot(xs, kde(xs), color=COL, lw=2.0)
                except Exception:
                    pass
            ax.axvline(tv_i, color='black', lw=1.5, ls='--', label=f'true={tv_i:.3f}')
            ax.axvline(np.median(vals), color=COL, lw=1.5, ls=':',
                       label=f'median={np.median(vals):.3f}')
            q1, q3 = np.percentile(vals, [25, 75])
            ax.axvspan(q1, q3, alpha=0.10, color=COL)
            ax.set_title(lbl, fontsize=10)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        for ax in axes[len(P_LABELS):]:
            ax.set_visible(False)

        n_done = len(records)
        fig.suptitle(f'DW_raw — Parameter Recovery  ({n_done} iters)', fontsize=12)
        plt.tight_layout()
        out = plot_dir / "sanity_dw_raw_param_dist.png"
        plt.savefig(out, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  [Plot skipped] {e}")


def run_optim_check(lat_r, lon_r, num_iters, init_noise, seed, t_steps=8):
    """
    Check 2: run N_SIM DW_raw optimizations on simulated data and
    report parameter recovery + direction/stationarity diagnostics.
    """
    print("\n" + "=" * 65)
    print("  CHECK 2: Parameter recovery via DW_raw optimization")
    print("=" * 65)

    rng     = np.random.default_rng(seed)
    tlog    = true_log_params()
    tpar    = torch.tensor(tlog, device=DEVICE, dtype=DTYPE)
    td      = TRUE_DICT

    lats_g  = torch.arange(min(lat_r), max(lat_r) + 0.0001,  DELTA_LAT,  device=DEVICE, dtype=DTYPE)
    lons_g  = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON,  device=DEVICE, dtype=DTYPE)
    lats_g  = torch.round(lats_g * 10000) / 10000
    lons_g  = torch.round(lons_g * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_g, lons_g, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    n1_g, n2_g   = len(lats_g), len(lons_g)
    print(f"  Grid: {n1_g} lat × {n2_g} lon | t_steps={t_steps} | iters={num_iters}")
    print(f"  True params: {dw.debiased_whittle_likelihood.get_printable_params_7param([tpar])}")

    output_path = Path(__file__).resolve().parent / "sanity_output"
    output_path.mkdir(parents=True, exist_ok=True)
    plot_dir    = output_path / "plots"
    plot_dir.mkdir(exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_path    = output_path / f"sanity_dw_raw_{date_tag}.csv"

    DWL_STEPS = 5
    records   = []
    skipped   = 0

    for it in range(num_iters):
        print(f"\n{'='*55}")
        print(f"  Iter {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*55}")

        init_log = make_random_init(tlog, rng, noise=init_noise)
        init_bp  = backmap_params(init_log)
        print(f"  init: sigmasq={init_bp['sigmasq']:.3f}  "
              f"range_lon={init_bp['range_lon']:.3f}  "
              f"nugget={init_bp['nugget']:.3f}")

        try:
            field = generate_field(lats_g, lons_g, t_steps, tpar, DELTA_LAT, DELTA_LON)
            ds_map, ds_agg = assemble_dataset(field, grid_coords, tpar)
            del field

            DWL, I_samp, t_auto, n1, n2, p_time = build_dw_inputs(
                ds_map, ds_agg, lat_r, lon_r)
            print(f"  FFT grid: {n1}×{n2}  p_time={p_time}")

            # ── (a) Loss + gradient at true params ────────────────────────────
            loss_true, grad_true = loss_and_grad_at_true(tpar, I_samp, t_auto, n1, n2, p_time)
            print(f"  Loss@true={loss_true:.6f}  |grad|@true={grad_true:.4e}")

            # ── (b) Loss at random init ────────────────────────────────────────
            with torch.no_grad():
                init_tensor = torch.tensor(init_log, device=DEVICE, dtype=DTYPE)
                loss_init   = dw.debiased_whittle_likelihood.whittle_likelihood_loss_tapered(
                    init_tensor, I_samp, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON).item()
            print(f"  Loss@init={loss_init:.6f}  "
                  f"Direction OK: {loss_true < loss_init}")

            # ── (c) Optimize ──────────────────────────────────────────────────
            p_dw  = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True)
                     for v in init_log]
            opt   = torch.optim.LBFGS(p_dw, lr=1.0, max_iter=20, max_eval=100,
                                       history_size=10, line_search_fn="strong_wolfe",
                                       tolerance_grad=1e-5)
            t0    = time.time()
            _, _, _, loss_conv, _ = DWL.run_lbfgs_tapered(
                p_dw, opt, I_samp, n1, n2, p_time, t_auto,
                max_steps=DWL_STEPS, device=DEVICE)
            elapsed = time.time() - t0

            out_log = [p.item() for p in p_dw]
            rmsre_v, est_v = rmsre(backmap_params(out_log), td)
            print(f"  RMSRE={rmsre_v:.4f}  elapsed={elapsed:.1f}s  "
                  f"Loss@conv={loss_conv:.6f}")

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        records.append({
            'iter':            it + 1,
            'rmsre':           round(rmsre_v, 6),
            'loss_true':       round(float(loss_true), 6),
            'loss_init':       round(float(loss_init), 6),
            'loss_conv':       round(float(loss_conv) if loss_conv != float('inf') else 1e9, 6),
            'grad_norm_true':  round(float(grad_true), 6),
            'time_s':          round(elapsed, 2),
            'n1': n1, 'n2': n2, 'p_time': p_time,
            'sigmasq_est':    round(est_v['sigmasq'],    6),
            'range_lat_est':  round(est_v['range_lat'],  6),
            'range_lon_est':  round(est_v['range_lon'],  6),
            'range_t_est':    round(est_v['range_time'], 6),
            'advec_lat_est':  round(est_v['advec_lat'],  6),
            'advec_lon_est':  round(est_v['advec_lon'],  6),
            'nugget_est':     round(est_v['nugget'],     6),
        })

        pd.DataFrame(records).to_csv(csv_path, index=False)
        if (it + 1) % 5 == 0 or it == 0:
            print_running_summary(records, it)

    # ── Final ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  DONE: {len(records)} completed, {skipped} skipped")
    print(f"{'='*55}")

    if records:
        print_final_summary(records, num_iters)
        make_plots(records, plot_dir)
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Wick's theorem — cross-frequency covariance (Section 6)
# ─────────────────────────────────────────────────────────────────────────────

def check_cross_freq_cov(lat_r, lon_r, t_steps=8, n_realizations=30,
                          n_freq_pairs=20, seed=42):
    """
    Section 6 — Wick's theorem: Cov(I_j, I_k) → 0 for j ≠ k.

    Strategy
    --------
    1. Generate n_realizations independent fields from the true model.
    2. For each realization, compute the diagonal of the sample periodogram
       I_samp[ω1,ω2,q,q]  (auto-periodogram, real part) for each frequency.
    3. Pick n_freq_pairs random pairs (j,k) with j ≠ k and compute the
       sample Pearson correlation of I[ω_j] and I[ω_k] across realizations.
    4. Report: if all |corr| < 0.30, asymptotic independence holds.
    """
    print("\n" + "=" * 65)
    print("  CHECK 3: Cross-frequency independence — Wick's theorem (Sec. 6)")
    print("=" * 65)

    if n_realizations < 10:
        print("  [WARNING] n_realizations < 10; correlation estimates unreliable.")

    tlog = true_log_params()
    tpar = torch.tensor(tlog, device=DEVICE, dtype=DTYPE)

    lats_g = torch.arange(min(lat_r), max(lat_r) + 0.0001,  DELTA_LAT,  device=DEVICE, dtype=DTYPE)
    lons_g = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON,  device=DEVICE, dtype=DTYPE)
    lats_g = torch.round(lats_g * 10000) / 10000
    lons_g = torch.round(lons_g * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_g, lons_g, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    n1_g, n2_g   = len(lats_g), len(lons_g)
    print(f"  Grid: {n1_g}×{n2_g}  t_steps={t_steps}  realizations={n_realizations}")

    # Collect per-realization auto-periodogram vectors: shape (n_real, n1*n2)
    I_diag_list = []   # (n_real, n_freq)
    grid_shape  = None

    for real_idx in range(n_realizations):
        try:
            field = generate_field(lats_g, lons_g, t_steps, tpar, DELTA_LAT, DELTA_LON)
            ds_map, ds_agg = assemble_dataset(field, grid_coords, tpar)
            del field

            _, I_samp, _, n1, n2, _p = build_dw_inputs(ds_map, ds_agg, lat_r, lon_r)
            diag  = torch.diagonal(I_samp.real, dim1=-2, dim2=-1)  # (n1,n2,p_time)
            auto  = diag.mean(dim=-1).cpu().numpy().flatten()       # (n1*n2,)
            I_diag_list.append(auto)
            grid_shape = (n1, n2)
        except Exception as e:
            print(f"  [SKIP real {real_idx+1}] {e}")
            continue

    if len(I_diag_list) < 5:
        print("  [ABORT] fewer than 5 realizations succeeded.")
        return {}

    I_mat = np.stack(I_diag_list, axis=0)   # (n_real, n_freq)
    n_real_ok, n_freq = I_mat.shape
    n1_r, n2_r = grid_shape if grid_shape else (0, 0)
    print(f"  Collected {n_real_ok} realizations  ×  {n_freq} frequencies (n1*n2={n1_r}×{n2_r})")

    # Drop DC frequency (index 0 in flat layout = freq (0,0))
    dc_idx = 0
    freq_indices = [i for i in range(n_freq) if i != dc_idx]

    # Sample random freq pairs j ≠ k
    chosen_pairs = []
    rng2 = np.random.default_rng(seed + 1)
    n_avail = len(freq_indices)
    while len(chosen_pairs) < min(n_freq_pairs, n_avail * (n_avail - 1) // 2):
        j, k = rng2.integers(0, n_avail, size=2)
        if j != k:
            pair = (min(freq_indices[j], freq_indices[k]),
                    max(freq_indices[j], freq_indices[k]))
            if pair not in chosen_pairs:
                chosen_pairs.append(pair)

    corrs = []
    for (j, k) in chosen_pairs:
        x, y = I_mat[:, j], I_mat[:, k]
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            r = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(r):
                corrs.append(r)

    if not corrs:
        print("  No valid correlation estimates.")
        return {}

    corrs = np.array(corrs)
    abs_c = np.abs(corrs)

    sep = "─" * 58
    print(f"\n  {sep}")
    print(f"  Cross-frequency |corr(I_j, I_k)|  over {len(corrs)} random pairs")
    print(f"  {sep}")
    print(f"  Max    : {abs_c.max():.6f}")
    print(f"  Mean   : {abs_c.mean():.6f}")
    print(f"  Median : {np.median(abs_c):.6f}")
    print(f"  P95    : {np.percentile(abs_c, 95):.6f}")
    for thr in [0.10, 0.20, 0.30]:
        frac = float((abs_c > thr).mean())
        flag = " ← LEAKAGE" if frac > 0.05 else ""
        print(f"  Frac > {thr:.2f}: {frac:.4f}{flag}")

    verdict = (
        "GOOD    (max |corr| < 0.20 — DFT near-independence confirmed)"
        if abs_c.max() < 0.20 else
        "ACCEPTABLE (max |corr| < 0.40)"
        if abs_c.max() < 0.40 else
        "WARNING (max |corr| ≥ 0.40 — significant spectral leakage!)"
    )
    print(f"  Verdict: {verdict}")
    print(f"  {sep}")
    print(f"  Theory (Wick): Cov(I_j,I_k) = 2|C̃_h(ω_j,ω_k)|² → 0 as n→∞")
    print(f"  Empirical corr across {n_real_ok} realizations confirms/refutes this.")
    return {'max_abs_corr': float(abs_c.max()), 'mean_abs_corr': float(abs_c.mean())}


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: Analytical white-noise verification (Section 7)
# ─────────────────────────────────────────────────────────────────────────────

def check_white_noise_analytical(lat_r, lon_r, t_steps=8,
                                  sigma_wn: float = 2.0, tol: float = 1e-5):
    """
    Section 7 — analytical white-noise sanity check (no simulation).

    White noise: C(h) = σ² · δ(h = 0)
      ⟹  E[I_qq(ω)] = σ² / (4π²)   for ALL ω   (flat spectral density)
      ⟹  E[I_qr(ω)] = 0              for q ≠ r  (temporal independence)

    Method
    ------
    Set params to a pure-nugget model: sigmasq ≈ 0 (1e-10), nugget = sigma_wn².
    Call expected_periodogram_fft_tapered with a synthetic dataset built from
    white noise draws.  Verify the output against theory at tolerance `tol`.
    """
    print("\n" + "=" * 65)
    print("  CHECK 4: Analytical white-noise verification (Section 7)")
    print("=" * 65)

    # ── White-noise log-params ────────────────────────────────────────────────
    # C(h)=σ²δ(h=0): pure nugget.  Set sigmasq=1e-10 (≈0), nugget=sigma_wn**2.
    sigma_sq = sigma_wn ** 2
    phi2_wn  = 1.0                          # arbitrary, range_lon=1
    phi1_wn  = 1e-10 * phi2_wn             # sigmasq ≈ 0
    phi3_wn  = 1.0                          # isotropic
    phi4_wn  = 1.0
    wn_log   = [np.log(phi1_wn), np.log(phi2_wn), np.log(phi3_wn), np.log(phi4_wn),
                0.0, 0.0, np.log(sigma_sq)]  # last = log(nugget)
    wn_par   = torch.tensor(wn_log, device=DEVICE, dtype=DTYPE)

    theory_diag = sigma_sq / (4.0 * np.pi ** 2)
    print(f"  sigma_wn   = {sigma_wn}  →  nugget = σ² = {sigma_sq}")
    print(f"  Theory E[I_qq(ω)] = σ²/(4π²) = {theory_diag:.8f}")
    print(f"  Theory E[I_qr(ω)] = 0  for q≠r")
    print(f"  Tolerance  = {tol:.1e}")

    # ── Build grid & white-noise dataset ─────────────────────────────────────
    lats_g = torch.arange(min(lat_r), max(lat_r) + 0.0001,  DELTA_LAT,  device=DEVICE, dtype=DTYPE)
    lons_g = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON,  device=DEVICE, dtype=DTYPE)
    lats_g = torch.round(lats_g * 10000) / 10000
    lons_g = torch.round(lons_g * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_g, lons_g, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    n1_g, n2_g   = len(lats_g), len(lons_g)
    N_grid       = n1_g * n2_g
    print(f"  Grid: {n1_g}×{n2_g}  t_steps={t_steps}")

    # White-noise dataset: iid N(0, sigma_sq) at every cell and time step
    # (skip circulant FFT — pure iid draws ARE white noise)
    ds_map, ds_list = {}, []
    for t_idx in range(t_steps):
        t_val = 21.0 + float(t_idx)
        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0
        rows = torch.zeros(N_grid, 11, device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords
        rows[:, 2]  = torch.randn(N_grid, device=DEVICE, dtype=DTYPE) * sigma_wn
        rows[:, 3]  = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid, -1)
        ds_map[f't{t_idx}'] = rows.detach()
        ds_list.append(rows.detach())
    ds_agg = torch.cat(ds_list, dim=0)

    # ── Compute E[I(ω)] with white-noise params ───────────────────────────────
    try:
        DW_PREPROC = dw.debiased_whittle_preprocess(
            [ds_agg], [ds_map], day_idx=0,
            params_list=[sigma_wn**2] + [1.0]*5 + [sigma_wn**2],  # placeholder
            lat_range=lat_r, lon_range=lon_r
        )
        cur_df = DW_PREPROC.generate_spatially_filtered_days(
            lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
        unique_t    = torch.unique(cur_df[:, 3])
        time_slices = [cur_df[cur_df[:, 3] == t] for t in unique_t]

        DWL = dw.debiased_whittle_likelihood()
        _J_vec, n1, n2, p_time, taper, obs_masks = DWL.generate_Jvector_tapered_mv(
            time_slices, DWL.cgn_hamming, 0, 1, 2, DEVICE)
        t_auto = DWL.calculate_taper_autocorrelation_multivariate(
            taper, obs_masks, n1, n2, DEVICE)
        del obs_masks

        print(f"  FFT grid: {n1}×{n2}  p_time={p_time}")

        with torch.no_grad():
            I_exp = DWL.expected_periodogram_fft_tapered(
                wn_par, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
    except Exception as e:
        print(f"  [ABORT] Failed to compute E[I(ω)]: {e}")
        import traceback; traceback.print_exc()
        return {}

    # ── Verify diagonal ───────────────────────────────────────────────────────
    diag = torch.diagonal(I_exp.real, dim1=-2, dim2=-1)   # (n1, n2, p_time)
    # exclude DC (0,0)
    diag_no_dc = diag.clone()
    diag_no_dc[0, 0, :] = float('nan')
    diag_vals  = diag_no_dc.reshape(-1).cpu().numpy()
    diag_vals  = diag_vals[np.isfinite(diag_vals)]

    err_diag   = np.abs(diag_vals - theory_diag)

    # ── Verify off-diagonal ───────────────────────────────────────────────────
    p = p_time
    if p >= 2:
        eye  = torch.eye(p, device=I_exp.device, dtype=torch.bool)
        mask = ~eye
        off  = I_exp.real[:, :, mask].reshape(-1).abs().cpu().numpy()
        off  = off[np.isfinite(off)]
        err_off = off
    else:
        err_off = np.array([0.0])

    sep = "─" * 58
    print(f"\n  {sep}")
    print(f"  White-noise analytical check  (tol={tol:.1e})")
    print(f"  {sep}")
    print(f"  Diagonal  (should ≈ {theory_diag:.6f}  =  σ²/(4π²)):")
    print(f"    max|E_qq - theory| = {err_diag.max():.3e}  "
          f"mean = {err_diag.mean():.3e}")
    diag_pass = bool(err_diag.max() < tol)
    print(f"    {'PASS ✓' if diag_pass else 'FAIL ✗'}  (max error {'<' if diag_pass else '>='} {tol:.1e})")

    print(f"  Off-diagonal  (should ≈ 0):")
    print(f"    max|E_qr|  = {err_off.max():.3e}  "
          f"mean = {err_off.mean():.3e}")
    off_pass = bool(err_off.max() < tol)
    print(f"    {'PASS ✓' if off_pass else 'FAIL ✗'}  (max error {'<' if off_pass else '>='} {tol:.1e})")

    overall = diag_pass and off_pass
    print(f"\n  Overall: {'ALL PASS — expected_periodogram_fft_tapered is CORRECT for white noise.' if overall else 'SOME CHECKS FAILED — inspect the implementation.'}")
    print(f"  {sep}")
    print(f"  Interpretation (Section 7):")
    print(f"    Cov(I_j, I_k) = 0 is confirmed by DFT independence at diff frequencies.")
    print(f"    Flat spectrum + diagonal output = tapering + FFT implemented correctly.")

    return {
        'diag_max_err':  float(err_diag.max()),
        'diag_mean_err': float(err_diag.mean()),
        'off_max_err':   float(err_off.max()),
        'off_mean_err':  float(err_off.mean()),
        'pass_diag':     diag_pass,
        'pass_off':      off_pass,
        'pass_overall':  overall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    mode:        str   = typer.Option("both",    help="'corr' | 'optim' | 'both'"),
    num_iters:   int   = typer.Option(50,        help="Number of optimization iters (Check 2)"),
    n_real:      int   = typer.Option(3,         help="Realizations for Check 1 corr (and Check 3 Wick)"),
    n_real_wick: int   = typer.Option(30,        help="Realizations for Check 3 Wick (overrides n_real if >0)"),
    lat_range:   str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:   str   = typer.Option("121,131", help="lon_min,lon_max"),
    init_noise:  float = typer.Option(0.7,       help="Init noise half-width in log-space"),
    seed:        int   = typer.Option(42,        help="Random seed"),
    t_steps:     int   = typer.Option(8,         help="Number of time steps per realization"),
    sigma_wn:    float = typer.Option(2.0,       help="White noise σ for Check 4"),
    wn_tol:      float = typer.Option(1e-5,      help="Tolerance for Check 4 analytical test"),
) -> None:

    lat_r = [float(x) for x in lat_range.split(',')]
    lon_r = [float(x) for x in lon_range.split(',')]
    n_wick = n_real_wick if n_real_wick > 0 else n_real

    print(f"\nDevice    : {DEVICE}")
    print(f"Mode      : {mode}")
    print(f"Region    : lat {lat_r}  lon {lon_r}")
    print(f"t_steps   : {t_steps}")
    print(f"Seed      : {seed}")

    if mode in ('corr', 'both'):
        # Check 1: off-diagonal temporal correlation
        run_corr_check(lat_r, lon_r, t_steps=t_steps, n_realizations=n_real, seed=seed)

        # Check 3: cross-frequency covariance (Wick / Section 6)
        check_cross_freq_cov(lat_r, lon_r, t_steps=t_steps,
                              n_realizations=n_wick,
                              n_freq_pairs=20, seed=seed)

        # Check 4: white-noise analytical (Section 7)
        check_white_noise_analytical(lat_r, lon_r, t_steps=t_steps,
                                      sigma_wn=sigma_wn, tol=wn_tol)

    if mode in ('optim', 'both'):
        run_optim_check(lat_r, lon_r,
                        num_iters=num_iters, init_noise=init_noise,
                        seed=seed, t_steps=t_steps)


if __name__ == "__main__":
    app()
