"""
seasonal_ar_test_local_033026.py

Seasonal AR structure analysis — spatial-mean ozone at three temporal scales.

Key issue (hourly):
  Data has 8 CONSECUTIVE hours per day (e.g. 08:00–15:00), then a 16h gap.
  → lag-1 within same day    = 1h actual gap        (valid short-range lag)
  → lag-1 across day boundary = 17h actual gap       (NOT a 1h lag — invalid!)
  → same-slot yesterday      = 24h actual gap        (cross-day lag-1d)
  Solution: reshape into (n_days × 8) matrix, keep within/cross-day lags separate.

Three scales:
  1. Hourly  — (n_days × 8) matrix
               Wk = Y[d, h-k] (within same day, k-hour lag)
               Dk = Y[d-k, h] (same slot k days ago, 24k-hour lag)
               ~160 model specs tested
  2. Daily   — nanmean of 8 slots per day
               All individual lags 1-28, all pairs, selected triples/quads
               ~250 model specs tested
  3. Monthly — mean per year-month
               Year-over-year correlation + notes (only 4 points if MONTHS=[7])

Output (outputs/day/seasonal_ar/):
  spatial_means.csv
  seasonal_ar_hourly_acf.png / _daily_acf.png / _monthly_acf.png
  seasonal_ar_aic_hourly.csv / _daily.csv / _monthly.csv
  seasonal_ar_summary.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import acf as _acf, pacf as _pacf
warnings.filterwarnings('ignore')

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

# ── Config ────────────────────────────────────────────────────────────────────
YEARS         = ['2022', '2023', '2024', '2025']
MONTHS        = [4, 5, 6, 7, 8, 9]   # April–September (6 months × 4 years = 24 monthly points)
LAT_RANGE     = [-3.0, 2.0]
LON_RANGE     = [121.0, 131.0]
MM_COND       = 10
SLOTS_PER_DAY = 8

OUT_DIR = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/seasonal_ar/")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_spatial_means():
    """
    Returns DataFrame sorted by hours_elapsed:
      year, month, day_idx, slot (0-7), hours_elapsed, spatial_mean, n_obs,
      spatial_mean_c  (centered per year-month)
    """
    data_loader = load_data_dynamic_processed(config.mac_data_load_path)
    records = []

    for yr in YEARS:
        for mo in MONTHS:
            print(f"  Loading {yr}-{mo:02d}...", flush=True)
            df_map, _, _, _ = data_loader.load_maxmin_ordered_data_bymonthyear(
                lat_lon_resolution=[1, 1], mm_cond_number=MM_COND,
                years_=[yr], months_=[mo],
                lat_range=LAT_RANGE, lon_range=LON_RANGE,
                is_whittle=False)

            sorted_keys = sorted(df_map.keys())
            n_days = len(sorted_keys) // SLOTS_PER_DAY

            for d_idx in range(n_days):
                day_keys = sorted_keys[d_idx * SLOTS_PER_DAY:(d_idx + 1) * SLOTS_PER_DAY]
                for slot, key in enumerate(day_keys):
                    df = df_map[key]
                    o3    = pd.to_numeric(df['ColumnAmountO3'], errors='coerce').dropna()
                    hours = pd.to_numeric(df['Hours_elapsed'],  errors='coerce').dropna()
                    if len(o3) == 0 or len(hours) == 0:
                        continue
                    records.append({
                        'year':          int(yr),
                        'month':         mo,
                        'day_idx':       d_idx,
                        'slot':          slot,
                        'hours_elapsed': float(hours.median()),
                        'spatial_mean':  float(o3.mean()),
                        'n_obs':         int(len(o3)),
                    })

    df_out = pd.DataFrame(records).sort_values('hours_elapsed').reset_index(drop=True)
    df_out['spatial_mean_c'] = df_out.groupby(['year', 'month'])['spatial_mean'].transform(
        lambda x: x - x.mean())
    return df_out


# ──────────────────────────────────────────────────────────────────────────────
# 2. Build time series at three scales
# ──────────────────────────────────────────────────────────────────────────────

def build_hourly_matrix(df):
    """Returns Y (n_days, 8) centered spatial means, and day_labels."""
    day_order = (df.groupby(['year', 'month', 'day_idx'])['hours_elapsed']
                   .min().reset_index().sort_values('hours_elapsed'))
    mat_rows, day_labels = [], []
    for _, row in day_order.iterrows():
        yr, mo, d_idx = int(row['year']), int(row['month']), int(row['day_idx'])
        sub = df[(df['year'] == yr) & (df['month'] == mo) & (df['day_idx'] == d_idx)]
        vec = np.full(SLOTS_PER_DAY, np.nan)
        for _, r in sub.iterrows():
            vec[int(r['slot'])] = r['spatial_mean_c']
        mat_rows.append(vec)
        day_labels.append((yr, mo, d_idx))
    return np.array(mat_rows), day_labels


def build_daily_series(Y):
    return np.nanmean(Y, axis=1)


def build_monthly_series(df):
    grp = (df.groupby(['year', 'month'])['spatial_mean_c']
             .mean().reset_index().sort_values(['year', 'month']))
    return grp['spatial_mean_c'].values, grp[['year', 'month']].values


# ──────────────────────────────────────────────────────────────────────────────
# 3. OLS-based AIC
# ──────────────────────────────────────────────────────────────────────────────

def ols_aic(y, X):
    """AIC = n·log(σ²) + 2k after OLS. NaN rows dropped. Returns NaN if underdetermined."""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y_, X_ = y[mask], X[mask]
    n, k = len(y_), X_.shape[1]
    if n <= k:          # need at least k+1 observations
        return np.nan
    coef, *_ = np.linalg.lstsq(X_, y_, rcond=None)
    sigma2 = float(np.mean((y_ - X_ @ coef) ** 2))
    if sigma2 <= 0:
        return np.nan
    return n * np.log(sigma2) + 2 * k


def _make_aic_df(specs, y):
    rows = []
    for name, cols in specs.items():
        X   = np.column_stack(cols)
        aic = ols_aic(y, X)
        rows.append({'model': name, 'k': X.shape[1], 'AIC': aic})
    df = pd.DataFrame(rows).dropna(subset=['AIC']).sort_values('AIC').reset_index(drop=True)
    df['ΔAIC'] = (df['AIC'] - df['AIC'].iloc[0]).round(2)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. AIC tables — comprehensive grid search
# ──────────────────────────────────────────────────────────────────────────────

def hourly_aic_table(Y):
    """
    ~160 model specs on the flattened (n_days × 8) series.

    Wk = Y[d, h-k]  : within-day k-hour lag  (NaN if h < k)
    Dk = Y[d-k, h]  : same slot k days ago   (NaN if d < k)

    Lag interpretation:
      W1 = 1h ago (within same 8h block)
      D1 = 24h ago (same slot yesterday, 16h gap + 8h within)
      D7 = 7 days ago same slot (weekly pattern)
     D14 = 14 days ago same slot (biweekly)
    """
    n_days, n_slots = Y.shape
    N   = n_days * n_slots
    y   = Y.flatten()
    one = np.ones(N)

    def within_lag(k):
        f = np.full(N, np.nan)
        for d in range(n_days):
            for h in range(k, n_slots):
                f[d * n_slots + h] = Y[d, h - k]
        return f

    def cross_lag(k):
        f = np.full(N, np.nan)
        for d in range(k, n_days):
            for h in range(n_slots):
                f[d * n_slots + h] = Y[d - k, h]
        return f

    # Pre-compute all lags
    W = {k: within_lag(k) for k in range(1, SLOTS_PER_DAY)}   # W1..W7
    D = {k: cross_lag(k)  for k in range(1, 15)}              # D1..D14

    specs = {'Baseline': [one]}

    # ── Cumulative within-day only: W(1..k)
    for k in range(1, SLOTS_PER_DAY):
        name = '+'.join(f'W{i}' for i in range(1, k + 1))
        specs[name] = [one] + [W[i] for i in range(1, k + 1)]

    # ── Individual cross-day only: D1..D14
    for j in range(1, 15):
        specs[f'D{j}'] = [one, D[j]]

    # ── Cumulative within (W1..Wk) + individual cross-day (Dj)
    #    k = 1,2,3,4 ; j = 1..14
    for k in range(1, 5):
        w_block = [W[i] for i in range(1, k + 1)]
        w_name  = '+'.join(f'W{i}' for i in range(1, k + 1))
        for j in range(1, 15):
            specs[f'{w_name}+D{j}'] = [one] + w_block + [D[j]]

    # ── Cumulative within (W1..Wk) + cumulative cross (D1..Dj)
    #    k = 1,2,3 ; j = 1..5
    for k in range(1, 4):
        w_block = [W[i] for i in range(1, k + 1)]
        w_name  = '+'.join(f'W{i}' for i in range(1, k + 1))
        for j in range(1, 6):
            d_block = [D[i] for i in range(1, j + 1)]
            d_name  = '+'.join(f'D{i}' for i in range(1, j + 1))
            key = f'{w_name}+{d_name}'
            if key not in specs:
                specs[key] = [one] + w_block + d_block

    # ── W3 + two non-consecutive cross-day lags: D{j1}+D{j2}
    w3_block = [W[i] for i in range(1, 4)]
    for j1 in range(1, 8):
        for j2 in [7, 14]:
            if j2 > j1:
                specs[f'W1+W2+W3+D{j1}+D{j2}'] = [one] + w3_block + [D[j1], D[j2]]

    # ── Pure non-consecutive cross-day pairs (no within)
    for j1 in range(1, 8):
        for j2 in [7, 14]:
            if j2 > j1:
                specs[f'D{j1}+D{j2}'] = [one, D[j1], D[j2]]

    return _make_aic_df(specs, y)


def daily_aic_table(z):
    """
    ~250 model specs on the daily-mean series.
    Individual lags 1-28, all relevant pairs, triples, selected quadruples.
    """
    N   = len(z)
    one = np.ones(N)

    def lag(k):
        f = np.full(N, np.nan); f[k:] = z[:-k]; return f

    L = {k: lag(k) for k in range(1, 29)}   # L1..L28

    specs = {'Baseline': [one]}

    # ── All individual lags L1..L28
    for k in range(1, 29):
        specs[f'L{k}'] = [one, L[k]]

    # ── All pairs (L1, Lk) for k=2..28
    for k in range(2, 29):
        specs[f'L1+L{k}'] = [one, L[1], L[k]]

    # ── All pairs (L2, Lk) for k=3..28
    for k in range(3, 29):
        specs[f'L2+L{k}'] = [one, L[2], L[k]]

    # ── All pairs (L7, Lk) for k≠7
    for k in range(1, 29):
        if k != 7:
            specs[f'L7+L{k}'] = [one, L[7], L[k]]

    # ── All pairs (L28, Lk) for k=1..27
    for k in range(1, 28):
        specs[f'L28+L{k}'] = [one, L[28], L[k]]

    # ── Triples (L1, L2, Lk) for k=3..28
    for k in range(3, 29):
        specs[f'L1+L2+L{k}'] = [one, L[1], L[2], L[k]]

    # ── Triples (L1, Lk, L28) for k=2..27
    for k in range(2, 28):
        specs[f'L1+L{k}+L28'] = [one, L[1], L[k], L[28]]

    # ── Triples (L1, L7, Lk) for k≠1,7
    for k in range(2, 29):
        if k != 7:
            specs[f'L1+L7+L{k}'] = [one, L[1], L[7], L[k]]

    # ── Triples (L1, L14, Lk)
    for k in [2, 3, 7, 21, 28]:
        specs[f'L1+L14+L{k}'] = [one, L[1], L[14], L[k]]

    # ── Selected quadruples
    for k in range(1, 29):
        if k not in (1, 7, 28):
            specs[f'L1+L7+L28+L{k}'] = [one, L[1], L[7], L[28], L[k]]
    specs['L1+L2+L7+L28']  = [one, L[1], L[2], L[7],  L[28]]
    specs['L1+L2+L14+L28'] = [one, L[1], L[2], L[14], L[28]]
    specs['L1+L7+L14+L28'] = [one, L[1], L[7], L[14], L[28]]

    return _make_aic_df(specs, z)


def monthly_aic_table(z, months_per_year):
    """
    Monthly-mean AR specs. Comprehensive grid search.
    With MONTHS=[4..9] (6 months/year × 4 years = 24 points):
      S = 6  →  lag-6 = same month previous year (seasonal lag)
      lag-12 = same month 2 years ago
    Tests: individual lags 1..12, all pairs, selected triples/quads.
    """
    N   = len(z)
    one = np.ones(N)
    S   = months_per_year   # seasonal period (= 6 for Apr-Sep)
    max_lag = min(N - 2, 12)   # don't go beyond 12 months

    def lag(k):
        f = np.full(N, np.nan); f[k:] = z[:-k]; return f

    L = {k: lag(k) for k in range(1, max_lag + 1)}

    specs = {'Baseline': [one]}

    # ── All individual lags 1..max_lag
    for k in range(1, max_lag + 1):
        if N > k + 1:
            lbl = f'L{k}'
            if k == S:
                lbl += f'(=same_mo_-1yr)'
            elif k == 2 * S and 2 * S <= max_lag:
                lbl += f'(=same_mo_-2yr)'
            specs[lbl] = [one, L[k]]

    # ── All pairs (L1, Lk) for k=2..max_lag
    for k in range(2, max_lag + 1):
        if N > k + 2:
            specs[f'L1+L{k}'] = [one, L[1], L[k]]

    # ── All pairs (LS, Lk) for k≠S  (seasonal + something)
    if S <= max_lag:
        for k in range(1, max_lag + 1):
            if k != S and N > max(S, k) + 2:
                specs[f'L{S}+L{k}'] = [one, L[S], L[k]]

    # ── Pairs (L2, Lk)
    for k in range(3, max_lag + 1):
        if N > k + 2:
            specs[f'L2+L{k}'] = [one, L[2], L[k]]

    # ── Triples (L1, L2, Lk)
    for k in range(3, max_lag + 1):
        if N > k + 3:
            specs[f'L1+L2+L{k}'] = [one, L[1], L[2], L[k]]

    # ── Triples (L1, LS, Lk)
    if S <= max_lag:
        for k in range(1, max_lag + 1):
            if k not in (1, S) and N > max(1, S, k) + 3:
                specs[f'L1+L{S}+L{k}'] = [one, L[1], L[S], L[k]]

    # ── Triples (L1, L2, LS)  and  (L1, L2, L2S)
    if S <= max_lag and N > S + 3:
        specs[f'L1+L2+L{S}']   = [one, L[1], L[2], L[S]]
    if 2 * S <= max_lag and N > 2 * S + 3:
        specs[f'L1+L{S}+L{2*S}'] = [one, L[1], L[S], L[2 * S]]

    # ── Selected quadruples
    if S <= max_lag and N > S + 4:
        specs[f'L1+L2+L{S}+L{S+1}'] = [one, L[1], L[2], L[S], L[S + 1]]
    if 2 * S <= max_lag and N > 2 * S + 4:
        specs[f'L1+L2+L{S}+L{2*S}']  = [one, L[1], L[2], L[S], L[2 * S]]
        specs[f'L1+L{S}+L{S+1}+L{2*S}'] = [one, L[1], L[S], L[S+1], L[2*S]]

    df_aic = _make_aic_df(specs, z)

    # Pearson correlations at all lags (robust summary)
    corr_rows = []
    for k in range(1, max_lag + 1):
        mask = np.isfinite(z[k:]) & np.isfinite(z[:-k])
        r = np.corrcoef(z[k:][mask], z[:-k][mask])[0, 1] if mask.sum() > 1 else np.nan
        label = f'lag-{k}'
        if k == S:
            label += ' (same month prev year)'
        elif k == 2 * S:
            label += ' (same month 2yr ago)'
        elif k == 1:
            label += ' (prev month)'
        corr_rows.append({'lag': k, 'label': label,
                          'pearson_r': round(r, 4) if np.isfinite(r) else np.nan,
                          'n_pairs':   int(mask.sum())})
    df_corr = pd.DataFrame(corr_rows)
    return df_aic, df_corr


# ──────────────────────────────────────────────────────────────────────────────
# 5. Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_hourly(Y, out_path):
    n_days, n_slots = Y.shape
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Hourly spatial-mean ozone — AR structure\n'
        '8 obs/day (consecutive), 16h gap between days → W=within-day, D=cross-day',
        fontsize=12)

    # Within-day slot correlation matrix
    ax = axes[0, 0]
    corr_mat = np.full((n_slots, n_slots), np.nan)
    for h1 in range(n_slots):
        for h2 in range(n_slots):
            mask = np.isfinite(Y[:, h1]) & np.isfinite(Y[:, h2])
            if mask.sum() > 5:
                corr_mat[h1, h2] = np.corrcoef(Y[mask, h1], Y[mask, h2])[0, 1]
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(n_slots)); ax.set_xticklabels([f'h{i}' for i in range(n_slots)])
    ax.set_yticks(range(n_slots)); ax.set_yticklabels([f'h{i}' for i in range(n_slots)])
    ax.set_title('Within-day slot correlation matrix')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Same-slot cross-day ACF per slot
    ax = axes[0, 1]
    max_lag_d = min(20, n_days - 2)
    conf = 1.96 / np.sqrt(n_days)
    for h in range(n_slots):
        col = Y[:, h]; col = col[np.isfinite(col)]
        if len(col) < max_lag_d + 5:
            continue
        ac = _acf(col, nlags=max_lag_d, fft=True)
        ax.plot(range(max_lag_d + 1), ac, alpha=0.7, label=f'h{h}')
    ax.axhline( conf, ls='--', color='gray', lw=0.8)
    ax.axhline(-conf, ls='--', color='gray', lw=0.8)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Lag (days)'); ax.set_ylabel('ACF')
    ax.set_title('Same-slot ACF across days (D1, D2, ...)')
    ax.legend(fontsize=7, ncol=2, loc='upper right')

    # Within-day lag-1 ACF per slot (show how strong W1 is at each slot)
    ax = axes[1, 0]
    w1_corrs = []
    for h in range(1, n_slots):
        mask = np.isfinite(Y[:, h]) & np.isfinite(Y[:, h - 1])
        if mask.sum() > 5:
            r = np.corrcoef(Y[mask, h], Y[mask, h - 1])[0, 1]
        else:
            r = np.nan
        w1_corrs.append(r)
    ax.bar(range(1, n_slots), w1_corrs, color='steelblue', alpha=0.8)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(range(1, n_slots))
    ax.set_xticklabels([f'h{h}~h{h-1}' for h in range(1, n_slots)], rotation=30, ha='right')
    ax.set_ylabel('Pearson r')
    ax.set_title('Within-day lag-1 correlation (W1)\nfor each slot pair')

    # All slot series over time
    ax = axes[1, 1]
    for h in range(n_slots):
        ax.plot(Y[:, h], lw=0.5, alpha=0.6, label=f'h{h}')
    ax.set_xlabel('Day index'); ax.set_ylabel('Centered spatial mean O₃')
    ax.set_title('All slots over time')
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_daily(z, out_path):
    valid = z[np.isfinite(z)]
    max_lag = min(30, len(valid) // 4)
    conf = 1.96 / np.sqrt(len(valid))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Daily mean ozone — AR structure', fontsize=12)

    axes[0].plot(z, lw=0.8, color='steelblue')
    axes[0].set_xlabel('Day index'); axes[0].set_ylabel('Centered O₃')
    axes[0].set_title('Daily mean series')

    ac = _acf(valid, nlags=max_lag, fft=True)
    axes[1].bar(range(len(ac)), ac, width=0.6, color='steelblue', alpha=0.8)
    axes[1].axhline( conf, ls='--', color='red', lw=0.8)
    axes[1].axhline(-conf, ls='--', color='red', lw=0.8)
    axes[1].set_xlabel('Lag (days)'); axes[1].set_ylabel('ACF')
    axes[1].set_title('ACF — daily mean')

    pc = _pacf(valid, nlags=min(max_lag, len(valid) // 3 - 1))
    axes[2].bar(range(len(pc)), pc, width=0.6, color='darkorange', alpha=0.8)
    axes[2].axhline( conf, ls='--', color='red', lw=0.8)
    axes[2].axhline(-conf, ls='--', color='red', lw=0.8)
    axes[2].set_xlabel('Lag (days)'); axes[2].set_ylabel('PACF')
    axes[2].set_title('PACF — daily mean')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_monthly(z, mo_labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Monthly mean ozone — seasonal AR structure', fontsize=12)

    x_labels = [f"{int(l[0])}-{int(l[1]):02d}" for l in mo_labels]
    axes[0].plot(z, 'o-', lw=0.8, color='steelblue')
    axes[0].set_xticks(range(len(z)))
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Centered O₃'); axes[0].set_title('Monthly mean series')

    valid = z[np.isfinite(z)]
    if len(valid) > 3:
        max_lag = len(valid) - 1
        conf = 1.96 / np.sqrt(len(valid))
        ac = _acf(valid, nlags=max_lag, fft=False)
        axes[1].bar(range(len(ac)), ac, width=0.6, color='steelblue', alpha=0.8)
        axes[1].axhline( conf, ls='--', color='red', lw=0.8)
        axes[1].axhline(-conf, ls='--', color='red', lw=0.8)
        axes[1].set_xlabel('Lag (months in data)'); axes[1].set_ylabel('ACF')
        axes[1].set_title('ACF — monthly mean')
    else:
        axes[1].text(0.5, 0.5, f'Only {len(valid)} months — too few for ACF\n'
                     '(need more months per year for seasonal AR)',
                     ha='center', va='center', transform=axes[1].transAxes)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Seasonal AR structure test  (comprehensive grid search)")
    print(f"  Years: {YEARS}  Months: {MONTHS}")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading spatial means...")
    df = load_spatial_means()
    print(f"  {len(df)} time slots loaded")
    df.to_csv(OUT_DIR / "spatial_means.csv", index=False)

    # ── Verify gap structure ──────────────────────────────────────────────────
    print("\n[2/6] Verifying gap structure...")
    hrs = df.sort_values('hours_elapsed')['hours_elapsed'].values
    gaps = np.diff(hrs)
    within = gaps[gaps < 5]
    cross  = gaps[gaps >= 5]
    print(f"  Within-day gaps : n={len(within)}  mean={within.mean():.2f}h  (expected ~1h)")
    print(f"  Cross-day gaps  : n={len(cross)}   mean={cross.mean():.2f}h  "
          f"(expected ~16-17h  ← lag-1 across boundary ≠ 1h!)")

    # ── Build series ──────────────────────────────────────────────────────────
    print("\n[3/6] Building time series...")
    Y, day_labels          = build_hourly_matrix(df)
    daily_z                = build_daily_series(Y)
    monthly_z, mo_labels   = build_monthly_series(df)
    months_per_year        = len(MONTHS)

    print(f"  Hourly matrix   : {Y.shape}  (n_days × 8 slots)")
    print(f"  Daily series    : {len(daily_z)} days")
    print(f"  Monthly series  : {len(monthly_z)} months  (S={months_per_year})")
    if months_per_year == 1:
        print(f"  Note: Only 1 month/year ({MONTHS[0]}) → monthly lag-1 = year-over-year")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[4/6] Plotting...")
    plot_hourly(Y,             OUT_DIR / "seasonal_ar_hourly_acf.png")
    plot_daily(daily_z,        OUT_DIR / "seasonal_ar_daily_acf.png")
    plot_monthly(monthly_z, mo_labels, OUT_DIR / "seasonal_ar_monthly_acf.png")

    # ── AIC grid search ───────────────────────────────────────────────────────
    print("\n[5/6] Fitting AR models (OLS + AIC)...")

    print("\n  ── Hourly AR specs (top 30) ──")
    df_h = hourly_aic_table(Y)
    print(df_h.head(30).to_string(index=False))
    print(f"  [{len(df_h)} specs tested total]")
    df_h.to_csv(OUT_DIR / "seasonal_ar_aic_hourly.csv", index=False)

    print("\n  ── Daily AR specs (top 30) ──")
    df_d = daily_aic_table(daily_z)
    print(df_d.head(30).to_string(index=False))
    print(f"  [{len(df_d)} specs tested total]")
    df_d.to_csv(OUT_DIR / "seasonal_ar_aic_daily.csv", index=False)

    print("\n  ── Monthly AR specs ──")
    df_m, df_corr = monthly_aic_table(monthly_z, months_per_year)
    if len(df_m) > 0:
        print(df_m.to_string(index=False))
    print("\n  Year-over-year correlations:")
    print(df_corr.to_string(index=False))
    df_m.to_csv(OUT_DIR / "seasonal_ar_aic_monthly.csv", index=False)
    df_corr.to_csv(OUT_DIR / "seasonal_ar_monthly_corr.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[6/6] Summary...")
    summary = pd.DataFrame([
        {'scale':      'hourly',
         'n_obs':      int(np.isfinite(Y).sum()),
         'n_specs':    len(df_h),
         'best_model': df_h.iloc[0]['model'],
         'best_AIC':   round(df_h.iloc[0]['AIC'], 2),
         'ΔAIC_2nd':   df_h.iloc[1]['ΔAIC'] if len(df_h) > 1 else np.nan},
        {'scale':      'daily',
         'n_obs':      int(np.isfinite(daily_z).sum()),
         'n_specs':    len(df_d),
         'best_model': df_d.iloc[0]['model'],
         'best_AIC':   round(df_d.iloc[0]['AIC'], 2),
         'ΔAIC_2nd':   df_d.iloc[1]['ΔAIC'] if len(df_d) > 1 else np.nan},
        {'scale':      'monthly',
         'n_obs':      int(np.isfinite(monthly_z).sum()),
         'n_specs':    len(df_m),
         'best_model': df_m.iloc[0]['model'] if len(df_m) > 0 else 'N/A (too few points)',
         'best_AIC':   round(df_m.iloc[0]['AIC'], 2) if len(df_m) > 0 else np.nan,
         'ΔAIC_2nd':   df_m.iloc[1]['ΔAIC'] if len(df_m) > 1 else np.nan},
    ])
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "seasonal_ar_summary.csv", index=False)

    print(f"\n[Done] All outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
