"""
analyze_missing_rate_by_month_041626.py

KEY INSIGHT:  frac=0.5 threshold = natural Voronoi cell boundary of the regular grid.
Any obs within the grid region is always within STEP/2 of its nearest cell center
(by geometry), so frac=0.5 rejects nothing extra.  The missing in tco_grid is
100% coverage-driven (no satellite orbit passed over that cell that day).

To see threshold-induced missing you need frac < 0.5.  This script compares
frac = 0.5, 0.333, 0.25, 0.2 across all available months.

Per year-month, per threshold frac:
  miss_total      : fraction of N_grid cells that are NaN (= 1 - observed_rate)
  miss_no_coverage: NaN fraction with NO threshold  (only cells truly uncovered)
  miss_from_thresh: difference = cells dropped specifically by this frac threshold

Outputs
-------
  missing_analysis_041626/
    missing_rate_monthly_041626.csv
    plots/
      missing_rate_timeseries.png      — per-frac line plots over time
      missing_rate_decomposition.png   — stacked bar: no-coverage vs threshold loss
      heatmap_miss_total_frac*.png     — month × year heatmap per frac
      threshold_loss_by_frac.png       — thresh loss vs frac, monthly lines

Usage
-----
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
  conda activate faiss_env
  python analyze_missing_rate_by_month_041626.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DATA = Path("/Users/joonwonlee/Documents/GEMS_DATA")
OUT_DIR   = Path(__file__).parent / "missing_analysis_041626"
OUT_DIR.mkdir(exist_ok=True)

# ── Grid parameters (matching step3_enforce_regular_grid) ─────────────────────
LAT_S, LAT_E   = -3.0, 2.0
LON_S, LON_E   = 121.0, 131.0
STEP_LAT       = 0.044
STEP_LON       = 0.063
V_DRIFT_LON    = -0.0048           # degrees/slot (orbit westward drift)
FRACS          = [0.5, 0.333, 0.25, 0.2]   # threshold fractions to compare


def make_center_points():
    """Replicate make_center_points_wo_calibration for LAT_LON_BOUNDS."""
    lat_coords = np.arange(LAT_E - STEP_LAT, LAT_S - STEP_LAT, -STEP_LAT)
    lon_coords = np.arange(LON_E - STEP_LON, LON_S - STEP_LON, -STEP_LON)
    final_lat  = lat_coords + STEP_LAT
    final_lon  = lon_coords + STEP_LON
    center_lats = np.repeat(final_lat, len(final_lon))
    center_lons = np.tile(final_lon, len(final_lat))
    return np.column_stack([center_lats, center_lons])


CENTER_POINTS = make_center_points()
N_GRID        = len(CENTER_POINTS)
print(f"Grid: N_lat={len(np.unique(CENTER_POINTS[:,0]))}  "
      f"N_lon={len(np.unique(CENTER_POINTS[:,1]))}  "
      f"N_grid={N_GRID}")


# ── Gridification ─────────────────────────────────────────────────────────────

def count_observed_by_frac(orbit_map, frac=None):
    """
    For each time key, apply nearest-neighbor binning with the given threshold
    frac (or no threshold if frac is None), and return per-slot observed fractions.

    frac=None  → no distance limit (any nearest obs counts)
    frac=0.5   → lat_thresh = STEP_LAT/2, lon_thresh = STEP_LON/2  (Voronoi boundary)
    frac=0.333 → stricter, etc.
    """
    lat_thresh = frac * STEP_LAT if frac is not None else np.inf
    lon_thresh = frac * STEP_LON if frac is not None else np.inf

    rates = []
    sorted_keys = sorted(orbit_map.keys())
    for i, key in enumerate(sorted_keys):
        df_raw = orbit_map[key]
        if df_raw is None or len(df_raw) == 0:
            rates.append(0.0)
            continue

        t_idx        = i % 8
        shifted_grid = CENTER_POINTS + np.array([0.0, t_idx * V_DRIFT_LON])

        if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
            raw_coords = df_raw[['Latitude', 'Longitude']].values
        else:
            raw_coords = df_raw.iloc[:, :2].values

        valid_rows = ~np.isnan(raw_coords).any(axis=1)
        raw_coords = raw_coords[valid_rows]
        if len(raw_coords) == 0:
            rates.append(0.0)
            continue

        grid_tree  = KDTree(shifted_grid)
        _, grid_idx = grid_tree.query(raw_coords)

        if frac is not None:
            lat_diffs = np.abs(raw_coords[:, 0] - shifted_grid[grid_idx, 0])
            lon_diffs = np.abs(raw_coords[:, 1] - shifted_grid[grid_idx, 1])
            mask      = (lat_diffs <= lat_thresh) & (lon_diffs <= lon_thresh)
            grid_idx  = grid_idx[mask]

        rates.append(len(np.unique(grid_idx)) / N_GRID)
    return rates


# ── Available year-month pairs ─────────────────────────────────────────────────

TARGETS = []
for year in [2022, 2023, 2024, 2025]:
    yr2 = str(year)[2:]
    pkl_dir = BASE_DATA / f"pickle_{year}"
    for month in range(1, 13):
        mo2 = f"{month:02d}"
        tco_path   = pkl_dir / f"tco_grid_{yr2}_{mo2}.pkl"
        orbit_path = pkl_dir / f"orbit_map{yr2}_{mo2}.pkl"
        if tco_path.exists() and orbit_path.exists():
            TARGETS.append((year, month, tco_path, orbit_path))

print(f"Found {len(TARGETS)} year-month pairs with both tco_grid and orbit_map.")


# ── Main loop ─────────────────────────────────────────────────────────────────

records = []
for year, month, tco_path, orbit_path in TARGETS:
    print(f"  {year}-{month:02d} ...", end=" ", flush=True)

    with open(orbit_path, 'rb') as f:
        orbit_map = pickle.load(f)

    # compute observed fraction for each threshold frac (and no-threshold)
    obs_nothr = count_observed_by_frac(orbit_map, frac=None)   # no limit
    frac_obs  = {frac: count_observed_by_frac(orbit_map, frac=frac)
                 for frac in FRACS}

    mean_nothr = float(np.mean(obs_nothr))   # max possible coverage
    miss_nocover = 1.0 - mean_nothr           # truly uncovered cells

    row = {
        'year':          year,
        'month':         month,
        'year_month':    f"{year}-{month:02d}",
        'n_slots':       len(obs_nothr),
        'n_days':        len(obs_nothr) // 8,
        'N_grid':        N_GRID,
        'obs_no_thresh': round(mean_nothr,   4),
        'miss_no_coverage': round(miss_nocover, 4),
    }

    parts = [f"no_cover={miss_nocover:.3f}"]
    for frac in FRACS:
        obs_f       = frac_obs[frac]
        mean_f      = float(np.mean(obs_f))
        miss_total  = 1.0 - mean_f
        miss_thresh = mean_nothr - mean_f        # loss vs no-threshold

        # daily variability
        daily_miss = [1.0 - float(np.mean(obs_f[d*8:(d+1)*8]))
                      for d in range(len(obs_f) // 8)]
        key = f"frac{frac:.3f}".replace(".", "")

        row[f'obs_{key}']         = round(mean_f,   4)
        row[f'miss_{key}']        = round(miss_total, 4)
        row[f'thresh_loss_{key}'] = round(miss_thresh, 4)
        row[f'daily_mean_{key}']  = round(float(np.mean(daily_miss)), 4) if daily_miss else float('nan')
        row[f'daily_std_{key}']   = round(float(np.std(daily_miss)),  4) if daily_miss else float('nan')
        row[f'daily_p10_{key}']   = round(float(np.percentile(daily_miss, 10)), 4) if daily_miss else float('nan')
        row[f'daily_p90_{key}']   = round(float(np.percentile(daily_miss, 90)), 4) if daily_miss else float('nan')

        parts.append(f"f{frac}→miss={miss_total:.3f}(thresh+{miss_thresh:.3f})")

    records.append(row)
    print("  ".join(parts))

df = pd.DataFrame(records)
csv_path = OUT_DIR / "missing_rate_monthly_041626.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ── Print summary table ────────────────────────────────────────────────────────

def frac_key(frac):
    return f"frac{frac:.3f}".replace(".", "")

print(f"\n{'='*90}")
print(f"  Missing-rate summary  (N_grid={N_GRID})  "
      f"miss = total NaN fraction  |  thresh_loss = loss vs no-threshold")
print(f"{'='*90}")
cw = 7

# header
hdr = f"  {'year-mo':>8}  {'no_cov':>{cw}}"
for frac in FRACS:
    hdr += f"  {'miss_'+str(frac):>{cw+2}}  {'loss_'+str(frac):>{cw+2}}"
print(hdr); print(f"  {'-'*80}")

for _, r in df.iterrows():
    row = f"  {r['year_month']:>8}  {r['miss_no_coverage']:>{cw}.3f}"
    for frac in FRACS:
        k = frac_key(frac)
        row += f"  {r[f'miss_{k}']:>{cw+2}.3f}  {r[f'thresh_loss_{k}']:>{cw+2}.3f}"
    print(row)

# per-year averages
print(f"\n  {'Year':>8}  {'no_cov':>{cw}}")
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    row = f"  {yr:>8}  {sub['miss_no_coverage'].mean():>{cw}.3f}"
    for frac in FRACS:
        k = frac_key(frac)
        row += f"  {sub[f'miss_{k}'].mean():>{cw+2}.3f}  {sub[f'thresh_loss_{k}'].mean():>{cw+2}.3f}"
    print(row)


# ── Plots ──────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    MONTH_NAMES  = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    FRAC_COLORS  = {0.5: '#1565C0', 0.333: '#2E7D32', 0.25: '#F57C00', 0.2: '#C62828'}
    x_labels     = df['year_month'].values
    x_pos        = np.arange(len(x_labels))
    years_avail  = sorted(df['year'].unique())
    months_avail = sorted(df['month'].unique())

    # ── 1. Total missing per frac — line plot ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for frac in FRACS:
        k     = frac_key(frac)
        vals  = df[f'miss_{k}'].values
        color = FRAC_COLORS.get(frac, '#888888')
        ax.plot(x_pos, vals, marker='o', ms=5, lw=2, color=color, label=f'frac={frac}')
    # also plot no-coverage baseline
    ax.plot(x_pos, df['miss_no_coverage'].values,
            marker='x', ms=5, lw=1.5, ls='--', color='#455A64',
            label='no-coverage (baseline)')
    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels(x_labels[::2], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Missing fraction', fontsize=10)
    ax.set_ylim(0, min(1.0, df[[f'miss_{frac_key(f)}' for f in FRACS]].values.max() + 0.1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title('Monthly total missing rate by threshold frac  (2022-2025)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'missing_rate_by_frac.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: missing_rate_by_frac.png")

    # ── 2. Threshold loss (vs no-threshold) per frac — line plot ─────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for frac in FRACS:
        k     = frac_key(frac)
        vals  = df[f'thresh_loss_{k}'].values
        color = FRAC_COLORS.get(frac, '#888888')
        ax.plot(x_pos, vals, marker='o', ms=5, lw=2, color=color, label=f'frac={frac}')
    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels(x_labels[::2], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Threshold-induced missing fraction', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title('Monthly threshold-induced missing  (= total − no-threshold baseline)',
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'threshold_loss_by_frac.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: threshold_loss_by_frac.png")

    # ── 3. Stacked bar decomposition for each frac (2×2 subplots) ────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, frac in enumerate(FRACS):
        k   = frac_key(frac)
        ax  = axes[i]
        nc  = df['miss_no_coverage'].values
        tl  = df[f'thresh_loss_{k}'].values
        ax.bar(x_pos, nc, color='#7B1FA2', alpha=0.8, label='No coverage', width=0.8)
        ax.bar(x_pos, tl, bottom=nc, color=FRAC_COLORS[frac], alpha=0.8,
               label=f'Threshold loss (frac={frac})', width=0.8)
        ax.set_title(f'frac = {frac}', fontsize=11)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if i >= 2:
            ax.set_xticks(x_pos[::3])
            ax.set_xticklabels(x_labels[::3], rotation=45, ha='right', fontsize=7)
    fig.suptitle('Missing decomposition: no-coverage  vs  threshold loss  (2022-2025)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_dir / 'missing_decomposition_4frac.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: missing_decomposition_4frac.png")

    # ── 4. Heatmap: month × year for each frac's total missing ───────────────
    for frac in FRACS:
        k    = frac_key(frac)
        col  = f'miss_{k}'
        mat  = np.full((len(months_avail), len(years_avail)), np.nan)
        for ii, mo in enumerate(months_avail):
            for jj, yr in enumerate(years_avail):
                r = df[(df['year'] == yr) & (df['month'] == mo)]
                if len(r):
                    mat[ii, jj] = r[col].values[0]
        vmax = float(np.nanmax(mat)) + 0.02

        fig, ax = plt.subplots(figsize=(max(5, len(years_avail)*1.5), 6))
        im = ax.imshow(mat, aspect='auto', cmap='Blues_r', vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format='%.2f')
        ax.set_xticks(range(len(years_avail)))
        ax.set_xticklabels(years_avail, fontsize=10)
        ax.set_yticks(range(len(months_avail)))
        ax.set_yticklabels([MONTH_NAMES[m-1] for m in months_avail], fontsize=9)
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                if not np.isnan(mat[ii, jj]):
                    ax.text(jj, ii, f'{mat[ii,jj]:.2f}', ha='center', va='center',
                            fontsize=8,
                            color='white' if mat[ii,jj] > 0.6*vmax else 'black')
        ax.set_title(f'Total missing fraction  (frac={frac})', fontsize=11, fontweight='bold')
        fname = f'heatmap_miss_frac{frac}.png'
        plt.tight_layout()
        plt.savefig(plot_dir / fname, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

    # ── 5. Heatmap: no-coverage baseline ─────────────────────────────────────
    mat_nc = np.full((len(months_avail), len(years_avail)), np.nan)
    for ii, mo in enumerate(months_avail):
        for jj, yr in enumerate(years_avail):
            r = df[(df['year'] == yr) & (df['month'] == mo)]
            if len(r):
                mat_nc[ii, jj] = r['miss_no_coverage'].values[0]

    fig, ax = plt.subplots(figsize=(max(5, len(years_avail)*1.5), 6))
    vmax_nc = float(np.nanmax(mat_nc)) + 0.02
    im = ax.imshow(mat_nc, aspect='auto', cmap='Purples_r', vmin=0, vmax=vmax_nc)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format='%.2f')
    ax.set_xticks(range(len(years_avail)))
    ax.set_xticklabels(years_avail, fontsize=10)
    ax.set_yticks(range(len(months_avail)))
    ax.set_yticklabels([MONTH_NAMES[m-1] for m in months_avail], fontsize=9)
    for ii in range(mat_nc.shape[0]):
        for jj in range(mat_nc.shape[1]):
            if not np.isnan(mat_nc[ii, jj]):
                ax.text(jj, ii, f'{mat_nc[ii,jj]:.2f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if mat_nc[ii,jj] > 0.6*vmax_nc else 'black')
    ax.set_title('No-coverage missing  (baseline, no threshold)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_dir / 'heatmap_no_coverage.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: heatmap_no_coverage.png")

    # ── 6. Threshold loss vs frac — boxplot per frac across all months ────────
    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = [df[f'thresh_loss_{frac_key(f)}'].values for f in FRACS]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
    for patch, frac in zip(bp['boxes'], FRACS):
        patch.set_facecolor(FRAC_COLORS.get(frac, '#888888'))
        patch.set_alpha(0.75)
    ax.set_xticklabels([f'frac={f}' for f in FRACS], fontsize=10)
    ax.set_ylabel('Threshold-induced missing fraction', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title('Distribution of threshold-induced missing across months', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'threshold_loss_boxplot.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: threshold_loss_boxplot.png")

    print(f"\n  All plots → {plot_dir}/")

except ImportError as ie:
    print(f"\n  [Plot skipped: {ie}]")
except Exception as pe:
    import traceback
    print(f"\n  [Plot error: {pe}]")
    traceback.print_exc()


if __name__ == '__main__':
    pass
