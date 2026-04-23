"""
sim_dw_advec_lat_042026.py

4-way Debiased Whittle filter comparison — pure lat advection scenario:
  advec_lat = 0.2,  advec_lon = 0.0

  Model 1 — Raw      : no spatial filter    (debiased_whittle_raw)
  Model 2 — 2-1-1-0  : filter Z=−2X+X↓+X→ (debiased_whittle_2110)
  Model 3 — Lat-1    : filter Z=X↓−X        (debiased_whittle_lat1)
  Model 4 — Lon-1    : filter Z=X→−X        (debiased_whittle_lon1)

Usage (local test):
  python sim_dw_advec_lat_042026.py --num-iters 5 --lat-factor 10 --lon-factor 4

Usage (Amarel):
  python sim_dw_advec_lat_042026.py --num-iters 300
"""

import sys
import time
import os
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.fft
import pandas as pd
import typer
from pathlib import Path
from sklearn.neighbors import BallTree

# ── Path setup ────────────────────────────────────────────────────────────────
AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC  = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
_src = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, _src)

from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle_raw  as dw_raw
from GEMS_TCO import debiased_whittle_2110 as dw_2110
from GEMS_TCO import debiased_whittle_lat1 as dw_lat1
from GEMS_TCO import debiased_whittle_lon1 as dw_lon1
from GEMS_TCO.data_loader import load_data_dynamic_processed

is_amarel = os.path.exists(config.amarel_data_load_path)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_DW = torch.device("cpu")
DTYPE     = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

MODELS       = ['raw', 'filt_2110', 'lat1', 'lon1']
MODEL_LABELS = ['Raw', '2-1-1-0', 'Lat-1', 'Lon-1']

P_LABELS = ['sigmasq', 'range_lat', 'range_lon', 'range_t',
            'advec_lat', 'advec_lon', 'nugget']

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ── Covariance kernel ─────────────────────────────────────────────────────────

def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


# ── High-resolution FFT field ─────────────────────────────────────────────────

def build_high_res_grid(lat_range, lon_range, lat_factor=100, lon_factor=20):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lats = torch.arange(min(lat_range) - 0.1, max(lat_range) + 0.1, dlat,
                        device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon,
                        device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    CPU = torch.device("cpu"); F32 = torch.float32
    Nx, Ny, Nt = len(lats_hr), len(lons_hr), t_steps
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt
    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid(Lx, Ly, Lt, params_cpu)
    S = torch.fft.fftn(C); S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Obs → grid mapping ────────────────────────────────────────────────────────

def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N_grid = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N_grid, -1, dtype=np.int64)
    dist, cell = grid_tree.query(np.radians(src_np_valid), k=1)
    dist = dist.flatten(); cell = cell.flatten()
    assignment = np.full(N_grid, -1, dtype=np.int64)
    best = np.full(N_grid, np.inf)
    for obs_i, (c, d) in enumerate(zip(cell, dist)):
        if d < best[c]:
            assignment[c] = obs_i; best[c] = d
    filled = assignment >= 0
    if filled.any():
        win_obs  = assignment[filled]
        lat_diff = np.abs(src_np_valid[win_obs, 0] - grid_coords_np[filled, 0])
        lon_diff = np.abs(src_np_valid[win_obs, 1] - grid_coords_np[filled, 1])
        too_far  = (lat_diff > DELTA_LAT_BASE / 2) | (lon_diff > DELTA_LON_BASE / 2)
        assignment[np.where(filled)[0][too_far]] = -1
    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    """obs pool = tco_grid Source_Latitude/Source_Longitude (1 obs per cell)."""
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_np   = torch.stack([hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_np), metric='haversine')
    gc_np   = grid_coords.cpu().numpy()
    N_grid  = len(gc_np)
    gc_tree = BallTree(np.radians(gc_np),  metric='haversine')
    s3, hr_idx, src = [], [], []
    for key in sorted_keys:
        df = ref_day_map.get(key)
        if df is None or len(df) == 0:
            s3.append(np.full(N_grid, -1, dtype=np.int64))
            hr_idx.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
            src.append(torch.zeros((0, 2), dtype=DTYPE, device=DEVICE))
            continue
        snp   = df[['Source_Latitude', 'Source_Longitude']].values
        vm    = ~np.isnan(snp).any(axis=1)
        snp_v = snp[vm]
        s3.append(apply_step3_1to1(snp_v, gc_np, gc_tree))
        if len(snp_v) > 0:
            _, hri = hr_tree.query(np.radians(snp_v), k=1)
            hr_idx.append(torch.tensor(hri.flatten(), device=DEVICE))
        else:
            hr_idx.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
        src.append(torch.tensor(snp_v, device=DEVICE, dtype=DTYPE))
    return s3, hr_idx, src


def assemble_reg_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params, t_offset=21.0):
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)
    reg_map, reg_list = {}, []
    for t_idx, key in enumerate(sorted_keys):
        t_val  = float(t_offset + t_idx)
        assign = step3_per_t[t_idx]
        hr_i   = hr_idx_per_t[t_idx]
        N_v    = hr_i.shape[0]
        dummy  = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0: dummy[t_idx - 1] = 1.0
        sim_vals = (field_flat[hr_i, t_idx]
                    + torch.randn(N_v, device=DEVICE, dtype=DTYPE) * nugget_std
                    ) if N_v > 0 else torch.zeros(0, device=DEVICE, dtype=DTYPE)
        rows = torch.full((N_grid, 11), float('nan'), device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords; rows[:, 3] = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid, -1)
        if N_v > 0:
            a = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            filled = a >= 0
            oob = filled & (a >= N_v)
            if oob.any():
                print(f"  [WARN] t_idx={t_idx}: {oob.sum().item()} assign idx "
                      f">= N_v={N_v} (max={a[filled].max().item()}), treating as missing")
                a = a.clone(); a[oob] = -1; filled = a >= 0
            rows[filled, 2] = sim_vals[a[filled]]
        reg_map[key] = rows.detach(); reg_list.append(rows.detach())
    return reg_map, torch.cat(reg_list, dim=0)


# ── Parameter utils ───────────────────────────────────────────────────────────

def backmap_params(out_params):
    p = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in out_params]
    phi2 = np.exp(p[1]); phi3 = np.exp(p[2]); phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {'sigmasq':    np.exp(p[0]) / phi2,
            'range_lat':  rlon / phi3**0.5,
            'range_lon':  rlon,
            'range_time': rlon / phi4**0.5,
            'advec_lat':  p[4],
            'advec_lon':  p[5],
            'nugget':     np.exp(p[6])}


def calculate_rmsre(out_params, true_dict, zero_thresh=0.01):
    """RMSRE over non-zero true params; zero-true params contribute MAE only (excluded from RMSRE)."""
    est  = backmap_params(out_params)
    keys = ['sigmasq','range_lat','range_lon','range_time','advec_lat','advec_lon','nugget']
    e    = np.array([est[k]       for k in keys])
    t    = np.array([true_dict[k] for k in keys])
    mask = np.abs(t) >= zero_thresh
    rmsre = float(np.sqrt(np.mean(((e[mask] - t[mask]) / np.abs(t[mask]))**2))) if mask.any() else float('nan')
    return rmsre, est


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(df, true_dict, plot_dir, n_iters):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    tv_list  = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                true_dict['nugget']]
    COLORS = {'raw': '#1976D2', 'filt_2110': '#E53935', 'lat1': '#43A047', 'lon1': '#FF6F00'}
    iters  = df['iter'].values

    # ── 1. KDE per parameter (4 models overlaid) ─────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    fig.suptitle(f'4-Model DW (advec_lat=0.2) — Parameter Distributions  ({n_iters} iters)',
                 fontsize=13, fontweight='bold')
    for idx, (lbl, tv) in enumerate(zip(P_LABELS, tv_list)):
        ax = axes[idx]
        for m, ml in zip(MODELS, MODEL_LABELS):
            col = f'range_t_est_{m}' if lbl == 'range_t' else f'{lbl}_est_{m}'
            if col not in df.columns: continue
            vals = df[col].dropna().values
            if len(vals) < 3: continue
            try:
                kde = gaussian_kde(vals)
                lo  = vals.min() - 0.15*(vals.max()-vals.min())
                hi  = vals.max() + 0.15*(vals.max()-vals.min())
                xs  = np.linspace(lo, hi, 300)
                ax.plot(xs, kde(xs), color=COLORS[m], lw=2.0, label=ml)
                ax.fill_between(xs, kde(xs), alpha=0.07, color=COLORS[m])
            except Exception:
                ax.hist(vals, bins=25, color=COLORS[m], alpha=0.3, label=ml)
        ax.axvline(tv, color='black', ls='--', lw=1.5, label=f'true={tv:.4f}')
        ax.set_title(f'{lbl}  (true={tv:.4f})', fontsize=10)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    if len(P_LABELS) < len(axes): axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'dw_adveclat_4model_kde.png', dpi=130, bbox_inches='tight')
    plt.close(); print(f"  Saved: dw_adveclat_4model_kde.png")

    # ── 2. Scatter + cumulative mean per parameter, per model ─────────────────
    for m, ml in zip(MODELS, MODEL_LABELS):
        fig, axes = plt.subplots(4, 2, figsize=(14, 18))
        axes = axes.flatten()
        fig.suptitle(f'{ml} (advec_lat=0.2) — Scatter + Cumulative Mean  ({n_iters} iters)',
                     fontsize=12, fontweight='bold')
        for idx, (lbl, tv) in enumerate(zip(P_LABELS, tv_list)):
            col = f'range_t_est_{m}' if lbl == 'range_t' else f'{lbl}_est_{m}'
            ax = axes[idx]
            if col not in df.columns: ax.set_title(lbl); continue
            vals = df[col].values
            ax.scatter(iters, vals, c=COLORS[m], s=16, alpha=0.5, linewidths=0)
            ax.axhline(tv, color='black', ls='--', lw=1.5, label=f'true={tv:.4f}')
            cum = pd.Series(vals).expanding().mean().values
            ax.plot(iters, cum, color='black', lw=1.8, label='cum. mean')
            ax.set_xlabel('Iteration', fontsize=9); ax.set_ylabel(lbl, fontsize=9)
            ax.set_title(f'{lbl}  (true={tv:.4f})', fontsize=10)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        if len(P_LABELS) < len(axes): axes[-1].set_visible(False)
        plt.tight_layout()
        plt.savefig(plot_dir / f'dw_adveclat_{m}_scatter.png', dpi=130, bbox_inches='tight')
        plt.close(); print(f"  Saved: dw_adveclat_{m}_scatter.png")

    # ── 3. RMSRE comparison ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for m, ml in zip(MODELS, MODEL_LABELS):
        col = f'rmsre_{m}'
        if col not in df.columns: continue
        rv  = df[col].values
        ax.scatter(iters, rv,  c=COLORS[m], s=16, alpha=0.4, linewidths=0)
        cum = pd.Series(rv).expanding().mean().values
        ax.plot(iters, cum, color=COLORS[m], lw=2.0, label=f'{ml} (med={np.median(rv):.3f})')
    ax.set_xlabel('Iteration', fontsize=11); ax.set_ylabel('RMSRE', fontsize=11)
    ax.set_title(f'4-Model DW (advec_lat=0.2) — RMSRE  ({n_iters} iters)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'dw_adveclat_4model_rmsre.png', dpi=130, bbox_inches='tight')
    plt.close(); print(f"  Saved: dw_adveclat_4model_rmsre.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    num_iters:  int   = typer.Option(300,  help="Simulation iterations"),
    years:      str   = typer.Option("2022,2024,2025", help="Years to sample obs patterns"),
    month:      int   = typer.Option(7,    help="Reference month"),
    lat_range:  str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:  str   = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor: int   = typer.Option(100,  help="High-res lat multiplier"),
    lon_factor: int   = typer.Option(20,   help="High-res lon multiplier"),
    init_noise: float = typer.Option(0.7,  help="Uniform noise half-width in log space"),
    dw_steps:   int   = typer.Option(5,    help="LBFGS max steps per run"),
    seed:       int   = typer.Option(42,   help="Random seed"),
) -> None:

    rng = np.random.default_rng(seed)
    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device     : {DEVICE}")
    print(f"Models     : {MODEL_LABELS}")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"Scenario   : advec_lat=0.2, advec_lon=0.0")
    print(f"Years      : {years_list}  month={month}")
    print(f"Iterations : {num_iters}  |  DW steps/iter: {dw_steps}")
    print(f"Init noise : ±{init_noise} log-space")

    output_path = Path(config.amarel_estimates_day_path if is_amarel
                       else config.mac_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_dw_advec_lat_{date_tag}.csv"
    csv_summary = f"sim_dw_advec_lat_summary_{date_tag}.csv"

    # ── True parameters ───────────────────────────────────────────────────────
    true_dict = {
        'sigmasq':    13.059,
        'range_lat':  0.154,
        'range_lon':  0.195,
        'range_time': 1.0,
        'advec_lat':  0.2,
        'advec_lon':  0.0,
        'nugget':     0.247,
    }

    phi2     = 1.0 / true_dict['range_lon']
    phi1     = true_dict['sigmasq'] * phi2
    phi3     = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4     = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    TRUE_VALS = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                 true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                 true_dict['nugget']]

    def make_random_init(rng):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            scale    = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng.uniform(-2*scale, 2*scale)
        return noisy

    # ── Load GEMS obs patterns ────────────────────────────────────────────────
    print("\n[Setup 1/4] Loading GEMS obs patterns...")
    data_path   = config.amarel_data_load_path if is_amarel else config.mac_data_load_path
    data_loader = load_data_dynamic_processed(data_path)
    all_day_mappings = []
    year_dfmaps, year_means, year_tco_maps = {}, {}, {}

    for yr in years_list:
        df_map_yr, _, _, mm_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1], mm_cond_number=8,
            years_=[yr], months_=[month],
            lat_range=lat_r, lon_range=lon_r, is_whittle=False)
        year_dfmaps[yr] = df_map_yr; year_means[yr] = mm_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots")

        yr2      = str(yr)[2:]
        tco_path = Path(data_path) / f"pickle_{yr}" / f"tco_grid_{yr2}_{month:02d}.pkl"
        if tco_path.exists():
            with open(tco_path, 'rb') as f:
                year_tco_maps[yr] = pickle.load(f)
            print(f"  tco_grid: {len(year_tco_maps[yr])} slots loaded")
        else:
            year_tco_maps[yr] = {}
            print(f"  [WARN] tco_grid not found: {tco_path}")

    # ── Build regular target grid ─────────────────────────────────────────────
    print("[Setup 2/4] Building regular target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid       = grid_coords.shape[0]
    n_lat, n_lon = len(lats_grid), len(lons_grid)
    print(f"  Grid: {n_lat} lat × {n_lon} lon = {N_grid} cells  "
          f"(lat {lats_grid[0].item():.3f}→{lats_grid[-1].item():.3f})")

    # ── High-res grid & obs mappings ──────────────────────────────────────────
    print("[Setup 3/4] Building high-res grid and precomputing obs mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r,
                                                               lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)}×{len(lons_hr)} = {len(lats_hr)*len(lons_hr):,} cells")

    DUMMY_KEYS = [f't{i}' for i in range(8)]
    for yr in years_list:
        all_sorted = sorted(year_dfmaps[yr].keys())
        n_days_yr  = len(all_sorted) // 8
        print(f"  {yr}: precomputing {n_days_yr} days...", flush=True)
        for d_idx in range(n_days_yr):
            day_keys = all_sorted[d_idx * 8 : (d_idx + 1) * 8]
            if len(day_keys) < 8: continue
            ref_day = {k: year_tco_maps[yr].get(k.split('_', 2)[-1], pd.DataFrame())
                       for k in day_keys}
            s3, hr_i, src = precompute_mapping_indices(
                ref_day, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_i, src))
    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    # ── Verify ────────────────────────────────────────────────────────────────
    print("[Setup 4/4] Verifying dataset structure...")
    _yr0, _d0, _s3_0, _hr0, _src0 = all_day_mappings[0]
    f0 = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    r0, _ = assemble_reg_dataset(f0, _s3_0, _hr0, _src0, DUMMY_KEYS, grid_coords, true_params)
    del f0
    fv = (~torch.isnan(list(r0.values())[0][:, 2])).sum().item()
    print(f"  reg_map first step: {fv}/{N_grid} valid")

    # ── DW likelihood objects ─────────────────────────────────────────────────
    dwl_raw  = dw_raw.debiased_whittle_likelihood()
    dwl_2110 = dw_2110.debiased_whittle_likelihood()
    dwl_lat1 = dw_lat1.debiased_whittle_likelihood()
    dwl_lon1 = dw_lon1.debiased_whittle_likelihood()

    LC, NC, VC, TC = 0, 1, 2, 3

    records = []
    skipped = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iter {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*60}")

        yr_it, d_it, step3_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        initial_vals = make_random_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Obs: {yr_it} day {d_it}  |  init sig={init_orig['sigmasq']:.3f} "
              f"rl={init_orig['range_lon']:.3f} nug={init_orig['nugget']:.3f}")

        rec = {'iter': it+1, 'obs_year': yr_it, 'obs_day': d_it,
               'init_sigmasq': round(init_orig['sigmasq'], 4),
               'init_range_lon': round(init_orig['range_lon'], 4)}

        try:
            # ── Generate field and reg_map once ───────────────────────────────
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
            reg_map, reg_agg = assemble_reg_dataset(
                field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field

            params_kwargs = dict(
                params_list=[true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                             true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                             true_dict['nugget']],
                lat_range=lat_r, lon_range=lon_r)

            # ── Raw preprocessing ─────────────────────────────────────────────
            db_raw  = dw_raw.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                          **params_kwargs)
            cur_raw = db_raw.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
            sl_raw  = [cur_raw[cur_raw[:, TC]==t] for t in torch.unique(cur_raw[:, TC])]

            J_raw, n1, n2, p_time, tap_raw, om_raw = dwl_raw.generate_Jvector_tapered_mv(
                sl_raw, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
            I_raw  = dwl_raw.calculate_sample_periodogram_vectorized(J_raw)
            ta_raw = dwl_raw.calculate_taper_autocorrelation_multivariate(
                tap_raw, om_raw, n1, n2, DEVICE_DW)
            del om_raw

            # ── 2-1-1-0 preprocessing ─────────────────────────────────────────
            db_2110  = dw_2110.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                            **params_kwargs)
            cur_2110 = db_2110.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
            sl_2110  = [cur_2110[cur_2110[:, TC]==t] for t in torch.unique(cur_2110[:, TC])]

            J_2110, n1_2, n2_2, _, tap_2110, om_2110 = dwl_raw.generate_Jvector_tapered_mv(
                sl_2110, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
            I_2110  = dwl_raw.calculate_sample_periodogram_vectorized(J_2110)
            ta_2110 = dwl_raw.calculate_taper_autocorrelation_multivariate(
                tap_2110, om_2110, n1_2, n2_2, DEVICE_DW)
            del om_2110

            # ── Lat-1 preprocessing ───────────────────────────────────────────
            db_lat1  = dw_lat1.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                            **params_kwargs)
            cur_lat1 = db_lat1.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
            sl_lat1  = [cur_lat1[cur_lat1[:, TC]==t] for t in torch.unique(cur_lat1[:, TC])]

            J_lat1, n1_l, n2_l, _, tap_lat1, om_lat1 = dwl_raw.generate_Jvector_tapered_mv(
                sl_lat1, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
            I_lat1  = dwl_raw.calculate_sample_periodogram_vectorized(J_lat1)
            ta_lat1 = dwl_raw.calculate_taper_autocorrelation_multivariate(
                tap_lat1, om_lat1, n1_l, n2_l, DEVICE_DW)
            del om_lat1

            # ── Lon-1 preprocessing ───────────────────────────────────────────
            db_lon1  = dw_lon1.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                            **params_kwargs)
            cur_lon1 = db_lon1.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
            sl_lon1  = [cur_lon1[cur_lon1[:, TC]==t] for t in torch.unique(cur_lon1[:, TC])]

            J_lon1, n1_o, n2_o, _, tap_lon1, om_lon1 = dwl_raw.generate_Jvector_tapered_mv(
                sl_lon1, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
            I_lon1  = dwl_raw.calculate_sample_periodogram_vectorized(J_lon1)
            ta_lon1 = dwl_raw.calculate_taper_autocorrelation_multivariate(
                tap_lon1, om_lon1, n1_o, n2_o, DEVICE_DW)
            del om_lon1

            print(f"  Raw:{n1}×{n2}  2110:{n1_2}×{n2_2}  lat1:{n1_l}×{n2_l}  lon1:{n1_o}×{n2_o}  p={p_time}")

            # ── Model 1: Raw ──────────────────────────────────────────────────
            p1 = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                  for v in initial_vals]
            opt1 = torch.optim.LBFGS(p1, lr=1.0, max_iter=20, max_eval=100,
                                      history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss1, _ = dwl_raw.run_lbfgs_tapered(
                params_list=p1, optimizer=opt1, I_sample=I_raw,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=ta_raw,
                max_steps=dw_steps, device=DEVICE_DW)
            t1 = time.time() - t0
            rmsre1, est1 = calculate_rmsre([p.item() for p in p1], true_dict)
            print(f"  [raw]      RMSRE={rmsre1:.4f}  ({t1:.1f}s)")

            # ── Model 2: 2-1-1-0 ─────────────────────────────────────────────
            p2 = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                  for v in initial_vals]
            opt2 = torch.optim.LBFGS(p2, lr=1.0, max_iter=20, max_eval=100,
                                      history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss2, _ = dwl_2110.run_lbfgs_tapered(
                params_list=p2, optimizer=opt2, I_sample=I_2110,
                n1=n1_2, n2=n2_2, p_time=p_time, taper_autocorr_grid=ta_2110,
                max_steps=dw_steps, device=DEVICE_DW)
            t2 = time.time() - t0
            rmsre2, est2 = calculate_rmsre([p.item() for p in p2], true_dict)
            print(f"  [filt_2110] RMSRE={rmsre2:.4f}  ({t2:.1f}s)")

            # ── Model 3: Lat-1 ────────────────────────────────────────────────
            p3 = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                  for v in initial_vals]
            opt3 = torch.optim.LBFGS(p3, lr=1.0, max_iter=20, max_eval=100,
                                      history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss3, _ = dwl_lat1.run_lbfgs_tapered(
                params_list=p3, optimizer=opt3, I_sample=I_lat1,
                n1=n1_l, n2=n2_l, p_time=p_time, taper_autocorr_grid=ta_lat1,
                max_steps=dw_steps, device=DEVICE_DW)
            t3 = time.time() - t0
            rmsre3, est3 = calculate_rmsre([p.item() for p in p3], true_dict)
            print(f"  [lat1]     RMSRE={rmsre3:.4f}  ({t3:.1f}s)")

            # ── Model 4: Lon-1 ────────────────────────────────────────────────
            p4 = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                  for v in initial_vals]
            opt4 = torch.optim.LBFGS(p4, lr=1.0, max_iter=20, max_eval=100,
                                      history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss4, _ = dwl_lon1.run_lbfgs_tapered(
                params_list=p4, optimizer=opt4, I_sample=I_lon1,
                n1=n1_o, n2=n2_o, p_time=p_time, taper_autocorr_grid=ta_lon1,
                max_steps=dw_steps, device=DEVICE_DW)
            t4 = time.time() - t0
            rmsre4, est4 = calculate_rmsre([p.item() for p in p4], true_dict)
            print(f"  [lon1]     RMSRE={rmsre4:.4f}  ({t4:.1f}s)")

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ───────────────────────────────────────────────────────────
        for m, est, rmsre, loss, t_s, n1_, n2_ in [
            ('raw',      est1, rmsre1, loss1, t1, n1,    n2),
            ('filt_2110',est2, rmsre2, loss2, t2, n1_2,  n2_2),
            ('lat1',     est3, rmsre3, loss3, t3, n1_l,  n2_l),
            ('lon1',     est4, rmsre4, loss4, t4, n1_o,  n2_o),
        ]:
            rec[f'rmsre_{m}']         = round(rmsre,        6)
            rec[f'time_s_{m}']        = round(t_s,          2)
            rec[f'loss_{m}']          = round(float(loss),  6)
            rec[f'sigmasq_est_{m}']   = round(est['sigmasq'],    6)
            rec[f'range_lat_est_{m}'] = round(est['range_lat'],  6)
            rec[f'range_lon_est_{m}'] = round(est['range_lon'],  6)
            rec[f'range_t_est_{m}']   = round(est['range_time'], 6)
            rec[f'advec_lat_est_{m}'] = round(est['advec_lat'],  6)
            rec[f'advec_lon_est_{m}'] = round(est['advec_lon'],  6)
            rec[f'nugget_est_{m}']    = round(est['nugget'],     6)
            rec[f'n1_{m}'] = n1_; rec[f'n2_{m}'] = n2_

        records.append(rec)
        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)

        # ── Running summary ───────────────────────────────────────────────────
        n_done = len(records)
        print(f"\n  ── Running summary ({n_done} done / {it+1} attempted) ──")
        cw = 9

        for m, ml in zip(MODELS, MODEL_LABELS):
            print(f"\n  [{ml}]")
            print(f"  {'param':<12} {'true':>{cw}}  {'mean':>{cw}}  {'median':>{cw}}  "
                  f"{'bias':>{cw}}  {'RMSRE':>{cw}}  {'RMSRE_med':>{cw}}  {'P90-P10':>{cw}}")
            print(f"  {'-'*94}")
            for lbl, tv in zip(P_LABELS, TRUE_VALS):
                col  = f'range_t_est_{m}' if lbl == 'range_t' else f'{lbl}_est_{m}'
                vals = np.array([r[col] for r in records])
                cm   = float(np.mean(vals))
                med  = float(np.median(vals))
                bi   = cm - tv
                p9p1 = float(np.percentile(vals, 90) - np.percentile(vals, 10))
                if abs(tv) >= 0.01:
                    rm  = float(np.sqrt(np.mean(((vals - tv) / abs(tv)) ** 2)))
                    rmd = float(np.median(np.abs((vals - tv) / abs(tv))))
                    print(f"  {lbl:<12} {tv:>{cw}.4f}  {cm:>{cw}.4f}  {med:>{cw}.4f}  "
                          f"{bi:>{cw}.4f}  {rm:>{cw}.4f}  {rmd:>{cw}.4f}  {p9p1:>{cw}.4f}")
                else:
                    mae = float(np.mean(np.abs(vals - tv)))
                    print(f"  {lbl:<12} {tv:>{cw}.4f}  {cm:>{cw}.4f}  {med:>{cw}.4f}  "
                          f"{bi:>{cw}.4f}  {'MAE':>{cw}}  {mae:>{cw}.4f}  {p9p1:>{cw}.4f}")
            rv = np.array([r[f'rmsre_{m}'] for r in records])
            print(f"  {'-'*94}")
            print(f"  {'Overall':<12} {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
                  f"{np.mean(rv):>{cw}.4f}  {np.median(rv):>{cw}.4f}  "
                  f"{np.percentile(rv, 90) - np.percentile(rv, 10):>{cw}.4f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE: {len(records)} completed, {skipped} skipped")
    print(f"{'='*60}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_raw, index=False)

    cw = 9
    summary_rows = []
    for m, ml in zip(MODELS, MODEL_LABELS):
        print(f"\n  ── FINAL [{ml}] ({len(records)} completed) ──")
        print(f"  {'param':<12} {'true':>{cw}}  {'mean':>{cw}}  {'median':>{cw}}  {'bias':>{cw}}  "
              f"{'RMSRE':>{cw}}  {'RMSRE_med':>{cw}}  {'P90-P10':>{cw}}")
        print(f"  {'-'*95}")
        for lbl, tv in zip(P_LABELS, TRUE_VALS):
            col  = f'range_t_est_{m}' if lbl == 'range_t' else f'{lbl}_est_{m}'
            vals = df_final[col].values
            mean_  = float(np.mean(vals))
            med_   = float(np.median(vals))
            bias_  = mean_ - tv
            p10_   = float(np.percentile(vals, 10))
            p90_   = float(np.percentile(vals, 90))
            p9p1_  = p90_ - p10_
            if abs(tv) >= 0.01:
                rmsre_  = float(np.sqrt(np.mean(((vals-tv)/abs(tv))**2)))
                rmsre_m = float(np.median(np.abs((vals-tv)/abs(tv))))
                print(f"  {lbl:<12} {tv:>{cw}.4f}  {mean_:>{cw}.4f}  {med_:>{cw}.4f}  {bias_:>{cw}.4f}  "
                      f"{rmsre_:>{cw}.4f}  {rmsre_m:>{cw}.4f}  {p9p1_:>{cw}.4f}")
            else:
                mae    = float(np.mean(np.abs(vals - tv)))
                rmsre_ = float('nan'); rmsre_m = float('nan')
                print(f"  {lbl:<12} {tv:>{cw}.4f}  {mean_:>{cw}.4f}  {med_:>{cw}.4f}  {bias_:>{cw}.4f}  "
                      f"  {'MAE':>{cw-2}}  {mae:>{cw}.4f}  {p9p1_:>{cw}.4f}")
            summary_rows.append({
                'model': ml, 'param': lbl, 'true': tv,
                'mean': round(mean_, 6), 'median': round(med_, 6),
                'bias': round(bias_, 6), 'std': round(float(np.std(vals)), 6),
                'RMSRE': round(rmsre_, 6) if not np.isnan(rmsre_) else float('nan'),
                'RMSRE_median': round(rmsre_m, 6) if not np.isnan(rmsre_m) else float('nan'),
                'P10': round(p10_, 6), 'P90': round(p90_, 6),
                'P90_P10': round(p9p1_, 6),
            })
        rv = df_final[f'rmsre_{m}'].values
        p9p1_rv = float(np.percentile(rv,90) - np.percentile(rv,10))
        print(f"  {'-'*95}")
        print(f"  {'Overall':<12} {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
              f"{np.mean(rv):>{cw}.4f}  {np.median(rv):>{cw}.4f}  {p9p1_rv:>{cw}.4f}")
        summary_rows.append({
            'model': ml, 'param': 'Overall', 'true': float('nan'),
            'mean': float('nan'), 'median': float('nan'), 'bias': float('nan'),
            'std': float('nan'), 'RMSRE': round(float(np.mean(rv)), 6),
            'RMSRE_median': round(float(np.median(rv)), 6),
            'P10': float('nan'), 'P90': float('nan'), 'P90_P10': round(p9p1_rv, 6),
        })

    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\n  Summary saved: {csv_summary}")

    try:
        plot_dir = output_path / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        make_plots(df_final, true_dict, plot_dir, len(records))
    except Exception as e:
        print(f"  [WARN] Plotting failed: {e}")


if __name__ == "__main__":
    app()
