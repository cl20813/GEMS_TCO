"""
make_pacf_report.py  (corrected)

Key fix — within-day PACF:
  The naive approach of flattening the (n_days×8) matrix into one long series
  is WRONG because cross-day boundaries create 17h gaps that get treated as
  consecutive 1h lags, contaminating lag-7 and lag-8 PACF values.

  Correct approach:
    1. Compute pooled within-day autocorrelations ρ(1)…ρ(7) by pairing
       Y[d, h] with Y[d, h-k] for each lag k, pooled across all days.
    2. Run Levinson-Durbin recursion on ρ(1)…ρ(7) to get true PACF values.
    3. SE ≈ 1/sqrt(n_eff) where n_eff = total valid within-day pairs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.tsa.stattools import pacf as _pacf, acf as _acf
from pathlib import Path

DATA_DIR = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/seasonal_ar/")
OUT_PDF  = DATA_DIR / "seasonal_ar_pacf_report.pdf"

SLOTS_PER_DAY = 8
CONF_MULT     = 1.96

# ── Load & rebuild Y matrix ───────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "spatial_means.csv")
day_order = (df.groupby(['year','month','day_idx'])['hours_elapsed']
               .min().reset_index().sort_values('hours_elapsed'))
mat_rows = []
for _, row in day_order.iterrows():
    yr, mo, d_idx = int(row['year']), int(row['month']), int(row['day_idx'])
    sub = df[(df['year']==yr)&(df['month']==mo)&(df['day_idx']==d_idx)]
    vec = np.full(SLOTS_PER_DAY, np.nan)
    for _, r in sub.iterrows():
        vec[int(r['slot'])] = r['spatial_mean_c']
    mat_rows.append(vec)
Y = np.array(mat_rows)          # (n_days, 8)
n_days = Y.shape[0]

daily_z   = np.nanmean(Y, axis=1)
mo_grp    = (df.groupby(['year','month'])['spatial_mean_c']
               .mean().reset_index().sort_values(['year','month']))
monthly_z = mo_grp['spatial_mean_c'].values


def conf_band(n): return CONF_MULT / np.sqrt(n)

def safe_pacf(x, max_lag):
    x = np.asarray(x, dtype=float)
    valid = x[np.isfinite(x)]
    if len(valid) < max_lag + 5: return None, len(valid)
    nlags = min(max_lag, len(valid)//3 - 1)
    try:    return _pacf(valid, nlags=nlags, method='ywm'), len(valid)
    except: return None, len(valid)

def safe_acf(x, max_lag):
    x = np.asarray(x, dtype=float)
    valid = x[np.isfinite(x)]
    if len(valid) < max_lag + 5: return None, len(valid)
    try:    return _acf(valid, nlags=min(max_lag, len(valid)-2), fft=True), len(valid)
    except: return None, len(valid)

def levinson_durbin_pacf(rho):
    """
    Given within-day autocorrelations rho[1..K] (rho[0]=1 assumed),
    return PACF values alpha[1..K] via Levinson-Durbin recursion.
    rho : array of length K  (rho[k-1] = lag-k correlation)
    """
    K   = len(rho)
    phi = np.zeros((K+1, K+1))   # phi[k, j] = k-th order AR coeff j
    pacf_vals = np.zeros(K+1)    # pacf_vals[0] = 1, pacf_vals[k] = alpha_k

    pacf_vals[0] = 1.0
    phi[1, 1]    = rho[0]
    pacf_vals[1] = rho[0]

    for k in range(2, K+1):
        num = rho[k-1] - sum(phi[k-1, j] * rho[k-1-j] for j in range(1, k))
        den = 1.0       - sum(phi[k-1, j] * rho[j-1]   for j in range(1, k))
        if abs(den) < 1e-12:
            break
        phi[k, k] = num / den
        pacf_vals[k] = phi[k, k]
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]

    return pacf_vals   # length K+1, index 0..K


# ═══════════════════════════════════════════════════════════════════════
# HOURLY — correct within-day PACF
# ═══════════════════════════════════════════════════════════════════════
max_within = 7  # lags 1..7 (max possible within 8 slots)

# Step 1: pooled within-day autocorrelations ρ(k), k=1..7
rho_within = []
n_pairs_within = []
for k in range(1, max_within + 1):
    xs, ys = [], []
    for d in range(n_days):
        for h in range(k, SLOTS_PER_DAY):
            if np.isfinite(Y[d, h]) and np.isfinite(Y[d, h-k]):
                xs.append(Y[d, h-k])
                ys.append(Y[d, h])
    if len(xs) > 5:
        rho_within.append(np.corrcoef(xs, ys)[0, 1])
        n_pairs_within.append(len(xs))
    else:
        rho_within.append(0.0)
        n_pairs_within.append(0)

rho_within    = np.array(rho_within)
n_eff_within  = int(np.mean(n_pairs_within))
conf_w        = conf_band(n_eff_within)

# Step 2: Levinson-Durbin PACF from ρ(1..7)
pacf_within = levinson_durbin_pacf(rho_within)  # index 0..7

# ═══════════════════════════════════════════════════════════════════════
# HOURLY — cross-day PACF per slot + average
# ═══════════════════════════════════════════════════════════════════════
max_cross_lag = min(20, n_days - 5)
cross_pacf_per_slot = []
n_per_slot = []
for h in range(SLOTS_PER_DAY):
    col = Y[:, h]
    pc, n = safe_pacf(col, max_cross_lag)
    cross_pacf_per_slot.append(pc)
    n_per_slot.append(n)

avg_cross_pacf = np.full(max_cross_lag + 1, np.nan)
for lag_idx in range(max_cross_lag + 1):
    vals = [cross_pacf_per_slot[h][lag_idx]
            for h in range(SLOTS_PER_DAY)
            if cross_pacf_per_slot[h] is not None
            and lag_idx < len(cross_pacf_per_slot[h])]
    if vals:
        avg_cross_pacf[lag_idx] = np.nanmean(vals)
conf_cross = conf_band(int(np.nanmean(n_per_slot)))

# ═══════════════════════════════════════════════════════════════════════
# DAILY
# ═══════════════════════════════════════════════════════════════════════
max_daily_lag = min(30, len(daily_z)//3 - 1)
pc_daily, n_daily = safe_pacf(daily_z, max_daily_lag)
ac_daily, _       = safe_acf(daily_z,  max_daily_lag)
conf_daily        = conf_band(n_daily)

# ═══════════════════════════════════════════════════════════════════════
# MONTHLY
# ═══════════════════════════════════════════════════════════════════════
max_mo_lag = len(monthly_z) - 2
pc_monthly, n_mo = safe_pacf(monthly_z, max_mo_lag)
ac_monthly, _    = safe_acf(monthly_z,  max_mo_lag)
conf_mo          = conf_band(n_mo)


def bar_plot(ax, vals, conf, title, xlabels=None, xlabel='Lag',
             color='steelblue', note=None, is_pacf=True):
    if vals is None:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title); return
    x = range(len(vals))
    ax.bar(x, vals, width=0.6, color=color, alpha=0.8)
    ax.axhline( conf, ls='--', color='red', lw=0.9, label=f'±{conf:.3f} (95%)')
    ax.axhline(-conf, ls='--', color='red', lw=0.9)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel('PACF' if is_pacf else 'ACF')
    ax.set_title(title); ax.legend(fontsize=7)
    if xlabels:
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=30)
    if note:
        ax.text(0.98, 0.97, note, transform=ax.transAxes,
                fontsize=7, ha='right', va='top', color='gray')


with PdfPages(OUT_PDF) as pdf:

    # ── Page 1: Hourly within-day ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        'Hourly AR — Within-day structure  (W lags)\n'
        'Correct method: pooled ρ(k) → Levinson-Durbin PACF  '
        '(avoids 17h gap contamination)', fontsize=10)

    # Marginal (ACF-like) ρ(k)
    axes[0].bar(range(1, max_within+1), rho_within,
                width=0.6, color='steelblue', alpha=0.8)
    axes[0].axhline( conf_w, ls='--', color='red', lw=0.9,
                     label=f'±{conf_w:.3f} (95%)')
    axes[0].axhline(-conf_w, ls='--', color='red', lw=0.9)
    axes[0].axhline(0, color='black', lw=0.5)
    axes[0].set_xticks(range(1, max_within+1))
    axes[0].set_xticklabels([f'W{k}' for k in range(1, max_within+1)])
    axes[0].set_ylabel('Pearson r')
    axes[0].set_title(f'Within-day marginal correlation ρ(k)\n(pooled across all days, n≈{n_eff_within})')
    axes[0].legend(fontsize=7)

    # PACF (Levinson-Durbin)
    bar_plot(axes[1], pacf_within[1:], conf_w,
             'Within-day PACF  (Levinson-Durbin)\n← W7 significant or not?',
             xlabels=[f'W{k}' for k in range(1, max_within+1)],
             xlabel='Within-day lag',
             note=f'n_eff≈{n_eff_within} pairs')

    # Difference: ρ vs PACF
    axes[2].bar(range(1, max_within+1),
                rho_within - pacf_within[1:],
                width=0.6, color='darkorange', alpha=0.8)
    axes[2].axhline(0, color='black', lw=0.8)
    axes[2].set_xticks(range(1, max_within+1))
    axes[2].set_xticklabels([f'W{k}' for k in range(1, max_within+1)])
    axes[2].set_ylabel('ρ(k) − PACF(k)')
    axes[2].set_title('Marginal − Partial correlation\n(≈0 means independent given shorter lags)')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── Page 2: Hourly cross-day ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Hourly AR — Cross-day structure  (D lags = same slot k days ago)\n'
        'AIC said all D1–D14 useless (ΔAIC>3982) — PACF confirms?', fontsize=10)

    ax = axes[0]
    for h in range(SLOTS_PER_DAY):
        pc = cross_pacf_per_slot[h]
        if pc is not None:
            ax.plot(range(len(pc)), pc, alpha=0.55, lw=0.9, label=f'slot{h}')
    ax.axhline( conf_cross, ls='--', color='red', lw=0.9,
                label=f'±{conf_cross:.3f}')
    ax.axhline(-conf_cross, ls='--', color='red', lw=0.9)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Cross-day lag (days)')
    ax.set_ylabel('PACF')
    ax.set_title('Cross-day PACF per slot\n(D1=yesterday same slot … D20)')
    ax.legend(fontsize=6, ncol=2)

    bar_plot(axes[1], avg_cross_pacf, conf_cross,
             'Average cross-day PACF across 8 slots\n(any bar outside band → that D-lag is useful)',
             xlabel='Cross-day lag (days)',
             color='darkorange',
             note=f'avg n≈{int(np.nanmean(n_per_slot))} obs/slot')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── Page 3: Daily ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f'Daily AR — ACF & PACF   (n={n_daily})\n'
        'Best AIC: L1+L8+L28  (ΔAIC_2nd=1.26 — ambiguous) · '
        'L1+L7+L28 is ΔAIC=3.33', fontsize=10)

    axes[0].plot(daily_z, lw=0.7, color='steelblue')
    axes[0].set_xlabel('Day index')
    axes[0].set_ylabel('Centered O₃')
    axes[0].set_title('Daily mean series')

    bar_plot(axes[1], ac_daily, conf_daily,
             f'Daily ACF (lag 0–{max_daily_lag})',
             xlabel='Lag (days)', color='steelblue', is_pacf=False,
             note=f'n={n_daily}')
    for lag, lbl, col in [(1,'L1','red'),(7,'L7','green'),
                           (8,'L8','purple'),(14,'L14','brown'),(28,'L28','orange')]:
        if ac_daily is not None and lag < len(ac_daily):
            axes[1].axvline(lag, color=col, lw=0.8, ls=':', alpha=0.7)
            axes[1].text(lag, max(ac_daily)*0.85, lbl,
                         color=col, fontsize=6, ha='center')

    bar_plot(axes[2], pc_daily, conf_daily,
             f'Daily PACF (lag 0–{max_daily_lag})\n← L7 vs L8 vs L28 significant?',
             xlabel='Lag (days)', color='darkorange',
             note=f'n={n_daily}')
    for lag, lbl, col in [(1,'L1','red'),(7,'L7','green'),
                           (8,'L8','purple'),(14,'L14','brown'),(28,'L28','orange')]:
        if pc_daily is not None and lag < len(pc_daily):
            axes[2].axvline(lag, color=col, lw=0.8, ls=':', alpha=0.7)
            axes[2].text(lag, max(pc_daily)*0.85, lbl,
                         color=col, fontsize=6, ha='center')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── Page 4: Monthly ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f'Monthly AR — ACF & PACF   (n={n_mo})\n'
        'Best AIC: Baseline (ΔAIC_2nd=65.8) — no autocorrelation expected', fontsize=10)

    axes[0].plot(monthly_z, 'o-', lw=0.8, color='steelblue')
    axes[0].set_xlabel('Month index')
    axes[0].set_ylabel('Centered O₃')
    axes[0].set_title(f'Monthly mean series (n={len(monthly_z)})')

    bar_plot(axes[1], ac_monthly, conf_mo,
             'Monthly ACF', xlabel='Lag (months)',
             color='steelblue', is_pacf=False, note=f'n={n_mo}')
    if ac_monthly is not None and len(ac_monthly) > 6:
        axes[1].axvline(6, color='orange', lw=0.9, ls=':')
        axes[1].text(6, max(ac_monthly)*0.85, 'L6\n(prev yr)',
                     color='orange', fontsize=6, ha='center')

    bar_plot(axes[2], pc_monthly, conf_mo,
             'Monthly PACF\n← all bars inside band = no structure',
             xlabel='Lag (months)', color='darkorange', note=f'n={n_mo}')
    if pc_monthly is not None and len(pc_monthly) > 6:
        axes[2].axvline(6, color='orange', lw=0.9, ls=':')
        axes[2].text(6, max(pc_monthly)*0.85, 'L6\n(prev yr)',
                     color='orange', fontsize=6, ha='center')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"Saved: {OUT_PDF}")
