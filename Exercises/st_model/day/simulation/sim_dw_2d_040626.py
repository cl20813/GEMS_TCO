"""
sim_dw_2d_040626.py

Two-model DW simulation: gridification bias test.

  DW_2d_loc : FFT high-res → real GEMS obs locations → step3 (obs→cell, 1:1)
              → 2D-diff DW on step3-gridified data.
              (same pipeline as sim_three_model_comparison_031926.py DW column)

  DW_2d_grid: FFT high-res → sample directly at each target grid cell
              (skip real location / step3 entirely)
              → 2D-diff DW on complete regular grid data.

Research question
-----------------
Does the step3 gridification process (irregular obs → regular grid cells)
introduce bias or excess variance in DW parameter estimates?
  → Compare DW_2d_loc vs DW_2d_grid with identical FFT fields and init values.

Both models use debiased_whittle_2d_conv (2D separable filter [-1,1;1,-1]).
Same true parameters (Scenario D), same random init scheme as the
three-model comparison script.

Usage:
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation
  python sim_dw_2d_040626.py --num-iters 1 --lat-factor 10 --lon-factor 4
  python sim_dw_2d_040626.py --num-iters 300
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
from sklearn.neighbors import BallTree

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")

from GEMS_TCO import debiased_whittle_2d_conv as debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

MODELS     = ['DW_2d_loc', 'DW_2d_grid']
P_COLS     = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
              'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
P_LABELS   = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
              'advec_lat', 'advec_lon', 'nugget']


# ── Covariance kernel for FFT field generation ────────────────────────────────

def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2)*phi3 + u_lon.pow(2) + lt.pow(2)*phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def build_high_res_grid(lat_range, lon_range, lat_factor=100, lon_factor=20):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lats = torch.arange(min(lat_range)-0.1, max(lat_range)+0.1, dlat, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0]-0.1,   lon_range[1]+0.1,  dlon, device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    CPU = torch.device("cpu"); F32 = torch.float32
    Nx, Ny, Nt = len(lats_hr), len(lons_hr), t_steps
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
    lx = torch.arange(Px,device=CPU,dtype=F32)*dlat; lx[Px//2:] -= Px*dlat
    ly = torch.arange(Py,device=CPU,dtype=F32)*dlon; ly[Py//2:] -= Py*dlon
    lt = torch.arange(Pt,device=CPU,dtype=F32);      lt[Pt//2:] -= Pt
    pc = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid(Lx, Ly, Lt, pc)
    S = torch.fft.fftn(C); S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px,Py,Pt,device=CPU,dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real)*noise).real[:Nx,:Ny,:Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Obs → grid mapping (for DW_2d_loc) ───────────────────────────────────────

def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N, -1, dtype=np.int64)
    dist, cell = grid_tree.query(np.radians(src_np_valid), k=1)
    dist = dist.flatten(); cell = cell.flatten()
    asgn = np.full(N,-1,dtype=np.int64); best = np.full(N, np.inf)
    for oi, (cj, d) in enumerate(zip(cell, dist)):
        if d < best[cj]:
            asgn[cj] = oi; best[cj] = d
    return asgn


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    lg, ng = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_np   = torch.stack([lg.flatten(), ng.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_np), metric='haversine')
    gn      = grid_coords.cpu().numpy()
    gt      = BallTree(np.radians(gn), metric='haversine')
    s3, hi, sl = [], [], []
    for key in sorted_keys:
        ref_t = ref_day_map[key].to(DEVICE)
        slocs = ref_t[:, :2]; snp = slocs.cpu().numpy()
        valid = ~np.isnan(snp).any(axis=1)
        s3.append(apply_step3_1to1(snp[valid], gn, gt))
        if valid.sum() > 0:
            _, hri = hr_tree.query(np.radians(snp[valid]), k=1)
            hi.append(torch.tensor(hri.flatten(), device=DEVICE))
        else:
            hi.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
        sl.append(slocs[valid])
    return s3, hi, sl


def precompute_grid_to_hr_idx(grid_coords, lats_hr, lons_hr):
    """
    For DW_2d_grid: precompute nearest high-res index for each target grid cell.
    This mapping is fixed (does not depend on the obs pattern).
    Returns: (N_grid,) long tensor of high-res flat indices.
    """
    lg, ng  = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_np   = torch.stack([lg.flatten(), ng.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_np), metric='haversine')
    _, hi   = hr_tree.query(np.radians(grid_coords.cpu().numpy()), k=1)
    return torch.tensor(hi.flatten(), device=DEVICE)


# ── Dataset assembly ───────────────────────────────────────────────────────────

def assemble_loc_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params, t_offset=21.0):
    """
    DW_2d_loc pipeline: FFT field → real obs locations → step3 grid assignment.
    Some grid cells are NaN (unobserved) after step3.
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    ff         = field.reshape(-1, 8)
    reg_map, reg_list = {}, []
    for t_idx, key in enumerate(sorted_keys):
        t_val  = float(t_offset + t_idx)
        assign = step3_per_t[t_idx]; hi = hr_idx_per_t[t_idx]; Nv = hi.shape[0]
        dummy  = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0: dummy[t_idx-1] = 1.0
        sim_vals = (ff[hi, t_idx] + torch.randn(Nv, device=DEVICE, dtype=DTYPE)*nugget_std
                    if Nv > 0 else torch.zeros(0, device=DEVICE, dtype=DTYPE))
        rows = torch.full((N_grid,11), float('nan'), device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords; rows[:, 3] = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid,-1)
        if Nv > 0:
            at = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            f_ = at >= 0; rows[f_, 2] = sim_vals[at[f_]]
        reg_map[key] = rows.detach(); reg_list.append(rows.detach())
    return reg_map, torch.cat(reg_list, dim=0)


def assemble_grid_dataset_direct(field, hr_idx_grid, sorted_keys,
                                  grid_coords, true_params, t_offset=21.0):
    """
    DW_2d_grid pipeline: FFT field sampled directly at target grid cells.
    Skips real observation locations and step3 entirely.
    All N_grid cells are observed — no NaN in the value column.

    hr_idx_grid: precomputed (N_grid,) tensor mapping each grid cell to its
                 nearest high-res flat index (see precompute_grid_to_hr_idx).
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    ff         = field.reshape(-1, 8)
    reg_map, reg_list = {}, []
    for t_idx, key in enumerate(sorted_keys):
        t_val = float(t_offset + t_idx)
        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0: dummy[t_idx-1] = 1.0
        # Sample field at every grid cell directly (no obs location step)
        gp_vals  = ff[hr_idx_grid, t_idx]
        sim_vals = gp_vals + torch.randn(N_grid, device=DEVICE, dtype=DTYPE)*nugget_std
        rows = torch.zeros((N_grid,11), device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords; rows[:, 2] = sim_vals; rows[:, 3] = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid,-1)
        reg_map[key] = rows.detach(); reg_list.append(rows.detach())
    return reg_map, torch.cat(reg_list, dim=0)


# ── Parameter back-mapping ─────────────────────────────────────────────────────

def backmap_params(out_params):
    p = out_params
    if isinstance(p[0], torch.Tensor):
        p = [x.item() if x.numel()==1 else x[0].item() for x in p]
    p  = [float(x) for x in p]
    p2 = np.exp(p[1]); p3 = np.exp(p[2]); p4 = np.exp(p[3])
    rn = 1./p2
    return {'sigmasq': np.exp(p[0])/p2, 'range_lat': rn/p3**0.5,
            'range_lon': rn, 'range_time': rn/p4**0.5,
            'advec_lat': p[4], 'advec_lon': p[5], 'nugget': np.exp(p[6])}


def calculate_rmsre(out_params, true_dict):
    est  = backmap_params(out_params)
    ea   = np.array([est['sigmasq'],  est['range_lat'],  est['range_lon'],
                     est['range_time'], est['advec_lat'], est['advec_lon'], est['nugget']])
    ta   = np.array([true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                     true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                     true_dict['nugget']])
    return float(np.sqrt(np.mean(((ea-ta)/np.abs(ta))**2))), est


# ── Summary printers ───────────────────────────────────────────────────────────

def print_running_summary(records, true_dict, it):
    cw      = 13
    n_done  = len([r for r in records if r['model'] == MODELS[0]])
    tv_list = [true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]

    print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
    # Mean table
    hdr = f"  {'param':<13} {'true':>{cw}}" + "".join(f"  {m:>{cw}}" for m in MODELS)
    print(hdr); print(f"  {'-'*65}")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        row = f"  {lbl:<13} {tv:>{cw}.4f}"
        for m in MODELS:
            vals = [r[col] for r in records if r['model'] == m]
            row += f"  {np.mean(vals):>{cw}.4f}"
        print(row)
    print(f"  {'-'*65}")

    # Min|Q1|Med|Q3|Max
    print(f"\n  [Min | Q1 | Median | Q3 | Max]")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        print(f"  {lbl} (true={tv:.4f})")
        for m in MODELS:
            v = np.array([r[col] for r in records if r['model'] == m])
            vmin = v.min(); q1,q2,q3 = np.percentile(v,[25,50,75]); vmax = v.max()
            print(f"    {m:<14} {vmin:.4f} | {q1:.4f} | {q2:.4f} | {q3:.4f} | {vmax:.4f}")

    # RMSRE / MdARE / P90-P10 per param
    def _ppv(m, fn):
        return [fn([r[col] for r in records if r['model'] == m], tv)
                for col, tv in zip(P_COLS, tv_list)]
    rmsre_m = {m: _ppv(m, lambda v,tv: float(np.sqrt(np.mean(((np.array(v)-tv)/abs(tv))**2)))) for m in MODELS}
    mdare_m = {m: _ppv(m, lambda v,tv: float(np.median(np.abs((np.array(v)-tv)/abs(tv))))) for m in MODELS}
    p9010_m = {m: _ppv(m, lambda v,tv: float(np.percentile(v,90)-np.percentile(v,10))) for m in MODELS}

    for metric, d in [('RMSRE', rmsre_m), ('MdARE', mdare_m), ('P90-P10', p9010_m)]:
        print(f"\n  [{metric} per param]")
        for lbl, idx in zip(P_LABELS, range(len(P_LABELS))):
            row = f"  {lbl:<13} {'':>{cw}}"
            for m in MODELS: row += f"  {d[m][idx]:>{cw}.4f}"
            print(row)
        overall = f"  {'Overall':<13} {'':>{cw}}"
        for m in MODELS: overall += f"  {np.mean(d[m]):>{cw}.4f}"
        print(overall)


def print_final_summary(df_final, true_dict, num_iters):
    tv_list = [true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]
    cw = 14

    def rmsre(sub, col, tv):
        return float(np.sqrt(np.mean(((sub[col].values - tv) / abs(tv))**2)))
    def mdare(sub, col, tv):
        return float(np.median(np.abs((sub[col].values - tv) / abs(tv))))
    def p9010(sub, col, tv):
        return float(np.percentile(sub[col].values, 90) - np.percentile(sub[col].values, 10))

    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — Per-parameter RMSRE  ({num_iters} iterations)")
    print(f"{'='*75}")
    hdr = f"  {'Parameter':<14} {'True':>10}" + "".join(f"  {m:>{cw}}" for m in MODELS)
    print(hdr); print(f"  {'-'*73}")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        row = f"  {lbl:<14} {tv:>10.4f}"
        for m in MODELS:
            sub = df_final[df_final['model']==m]
            row += f"  {rmsre(sub, col, tv):>{cw}.4f}"
        print(row)
    print(f"  {'-'*73}")
    ovr = f"  {'Overall RMSRE':<14} {'':>10}"
    for m in MODELS:
        sub = df_final[df_final['model']==m]
        ovr += f"  {np.mean([rmsre(sub,c,tv) for c,tv in zip(P_COLS,tv_list)]):>{cw}.4f}"
    print(ovr)
    ovr2 = f"  {'Overall MdARE':<14} {'':>10}"
    for m in MODELS:
        sub = df_final[df_final['model']==m]
        ovr2 += f"  {np.mean([mdare(sub,c,tv) for c,tv in zip(P_COLS,tv_list)]):>{cw}.4f}"
    print(ovr2)
    ovr3 = f"  {'Overall P90-P10':<14} {'':>10}"
    for m in MODELS:
        sub = df_final[df_final['model']==m]
        ovr3 += f"  {np.mean([p9010(sub,c,tv) for c,tv in zip(P_COLS,tv_list)]):>{cw}.4f}"
    print(ovr3)

    print(f"\n  Mean estimate (SD) across {num_iters} iterations")
    msd_hdr = f"  {'Parameter':<14} {'True':>10}" + "".join(f"  {'mean(SD)':>{cw}}" for _ in MODELS)
    print(msd_hdr); print(f"  {'-'*73}")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        row = f"  {lbl:<14} {tv:>10.4f}"
        for m in MODELS:
            sub = df_final[df_final['model']==m]
            me, sd = sub[col].mean(), sub[col].std()
            row += f"  {me:>6.3f}({sd:.3f})"
        print(row)

    print(f"\n{'='*75}")
    print(f"  5-NUMBER SUMMARY  (Min | Q1 | Median | Q3 | Max)")
    print(f"{'='*75}")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        print(f"\n  {lbl}  (true = {tv:.4f})")
        print(f"    {'Model':<12}  {'Min':>8}  {'Q1':>8}  {'Median':>8}  {'Q3':>8}  {'Max':>8}")
        print(f"    {'-'*58}")
        for m in MODELS:
            sub = df_final[df_final['model']==m][col].dropna().values
            q1, q2, q3 = np.percentile(sub, [25, 50, 75])
            print(f"    {m:<12}  {sub.min():>8.4f}  {q1:>8.4f}  {q2:>8.4f}  {q3:>8.4f}  {sub.max():>8.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v: float            = typer.Option(0.5,    help="(unused, API parity)"),
    mm_cond_number: int = typer.Option(100,    help="(unused, API parity)"),
    num_iters: int      = typer.Option(1000),
    years: str          = typer.Option("2022,2024,2025"),
    month: int          = typer.Option(7),
    lat_range: str      = typer.Option("-3,2"),
    lon_range: str      = typer.Option("121,131"),
    lat_factor: int     = typer.Option(100),
    lon_factor: int     = typer.Option(20),
    init_noise: float   = typer.Option(0.7),
    seed: int           = typer.Option(42),
) -> None:

    rng        = np.random.default_rng(seed)
    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device     : {DEVICE}")
    print(f"Models     : DW_2d_loc (step3 gridify) vs DW_2d_grid (direct grid sample)")
    print(f"Filter     : 2D separable  Z(i,j) = -X(i,j)+X(i,j+1)+X(i+1,j)-X(i+1,j+1)")
    print(f"Purpose    : Gridification bias test")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"Years      : {years_list}  month={month}")
    print(f"Iterations : {num_iters}")
    print(f"Init noise : ±{init_noise} log-space")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_dw_2d_{date_tag}.csv"
    csv_summary = f"sim_dw_2d_summary_{date_tag}.csv"

    # ── True parameters ──────────────────────────────────────────────────────
    true_dict = {
        'sigmasq': 13.059, 'range_lat': 0.154, 'range_lon': 0.195,
        'range_time': 1.0, 'advec_lat': 0.0218, 'advec_lon': -0.1689, 'nugget': 0.247,
    }
    phi2     = 1.0 / true_dict['range_lon']
    phi1     = true_dict['sigmasq'] * phi2
    phi3     = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4     = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    def make_init(rng):
        noisy = list(true_log)
        for i in [0,1,2,3,6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4,5]:
            s = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng.uniform(-2*s, 2*s)
        return noisy

    # ── [Setup 1/5] Load GEMS obs patterns ────────────────────────────────────
    print("\n[Setup 1/5] Loading GEMS obs patterns...")
    data_loader = load_data_dynamic_processed(config.amarel_data_load_path)
    all_day_mappings = []
    year_dfmaps, year_means = {}, {}
    for yr in years_list:
        df_map_yr, _, _, mm_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1,1], mm_cond_number=mm_cond_number,
            years_=[yr], months_=[month], lat_range=lat_r, lon_range=lon_r, is_whittle=False)
        year_dfmaps[yr] = df_map_yr; year_means[yr] = mm_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots")

    # ── [Setup 2/5] Build regular target grid ─────────────────────────────────
    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.arange(max(lat_r), min(lat_r)-0.0001, -DELTA_LAT_BASE, device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1] +0.0001,  DELTA_LON_BASE,  device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid*10000)/10000
    lons_grid = torch.round(lons_grid*10000)/10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]; n_lat = len(lats_grid); n_lon = len(lons_grid)
    print(f"  Grid: {n_lat}×{n_lon} = {N_grid} cells")
    print(f"  After 2D diff: ({n_lat-1})×({n_lon-1}) = {(n_lat-1)*(n_lon-1)} cells per step")

    # ── [Setup 3/5] Build high-res grid ──────────────────────────────────────
    print("[Setup 3/5] Building high-res grid...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)}×{len(lons_hr)} = {len(lats_hr)*len(lons_hr):,} cells")

    # ── [Setup 4/5] Precompute direct grid→hr mapping (for DW_2d_grid) ───────
    print("[Setup 4/5] Precomputing direct grid→HR mapping (DW_2d_grid)...")
    hr_idx_grid = precompute_grid_to_hr_idx(grid_coords, lats_hr, lons_hr)
    print(f"  hr_idx_grid: {hr_idx_grid.shape[0]} grid cells mapped to HR points")

    # ── [Setup 5/5] Precompute obs mappings (for DW_2d_loc) ──────────────────
    print("[Setup 5/5] Precomputing obs mappings (DW_2d_loc)...")
    DUMMY_KEYS = [f't{i}' for i in range(8)]
    for yr in years_list:
        df_map_yr = year_dfmaps[yr]; mm_yr = year_means[yr]
        all_sorted = sorted(df_map_yr.keys()); nd = len(all_sorted)//8
        print(f"  {yr}: {nd} days...", flush=True)
        for d_idx in range(nd):
            ref, _ = data_loader.load_working_data(
                df_map_yr, mm_yr, [d_idx*8,(d_idx+1)*8], ord_mm=None, dtype=DTYPE, keep_ori=True)
            dk = sorted(ref.keys())[:8]
            if len(dk) < 8: continue
            s3, hi, src = precompute_mapping_indices(ref, lats_hr, lons_hr, grid_coords, dk)
            all_day_mappings.append((yr, d_idx, s3, hi, src))
    print(f"  Total day-patterns: {len(all_day_mappings)}")

    # ── DW settings ──────────────────────────────────────────────────────────
    DWL_STEPS = 5
    LC, NC, VC, TC = 0, 1, 2, 3
    dwl = debiased_whittle.debiased_whittle_likelihood()

    records = []; skipped = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iter {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*60}")

        yr_it, d_it, s3, hi_t, src_t = all_day_mappings[rng.integers(len(all_day_mappings))]
        init_vals = make_init(rng)
        init_orig = backmap_params(init_vals)
        print(f"  Obs: {yr_it} day {d_it}  "
              f"init sig={init_orig['sigmasq']:.3f} "
              f"rn={init_orig['range_lon']:.3f} "
              f"nug={init_orig['nugget']:.3f}")

        try:
            # ── Generate ONE shared FFT field ─────────────────────────────────
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)

            # ── DW_2d_loc: real location → step3 gridify ──────────────────────
            reg_map_loc, reg_agg_loc = assemble_loc_dataset(
                field, s3, hi_t, src_t, DUMMY_KEYS, grid_coords, true_params)

            # ── DW_2d_grid: direct grid sampling (skip step3) ─────────────────
            reg_map_grid, reg_agg_grid = assemble_grid_dataset_direct(
                field, hr_idx_grid, DUMMY_KEYS, grid_coords, true_params)

            del field

            results = {}

            for tag, reg_map, reg_agg in [
                ('DW_2d_loc',  reg_map_loc,  reg_agg_loc),
                ('DW_2d_grid', reg_map_grid, reg_agg_grid),
            ]:
                print(f"--- {tag} ---")
                p_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                        for val in init_vals]
                db = debiased_whittle.debiased_whittle_preprocess(
                    [reg_agg], [reg_map], day_idx=0,
                    params_list=[true_dict['sigmasq'],   true_dict['range_lat'],
                                 true_dict['range_lon'],  true_dict['range_time'],
                                 true_dict['advec_lat'],  true_dict['advec_lon'],
                                 true_dict['nugget']],
                    lat_range=lat_r, lon_range=lon_r)
                cur_df = db.generate_spatially_filtered_days(
                    lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
                unique_t    = torch.unique(cur_df[:, TC])
                time_slices = [cur_df[cur_df[:, TC]==t] for t in unique_t]

                J_vec, n1, n2, p_time, taper, obs_masks = dwl.generate_Jvector_tapered_mv(
                    time_slices, dwl.cgn_hamming, LC, NC, VC, DEVICE)
                I_samp = dwl.calculate_sample_periodogram_vectorized(J_vec)
                t_auto = dwl.calculate_taper_autocorrelation_multivariate(
                    taper, obs_masks, n1, n2, DEVICE)
                del obs_masks

                opt = torch.optim.LBFGS(
                    p_dw, lr=1.0, max_iter=20, max_eval=100,
                    history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5)
                t0 = time.time()
                _, _, _, loss_dw, _ = dwl.run_lbfgs_tapered(
                    params_list=p_dw, optimizer=opt, I_sample=I_samp,
                    n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
                    max_steps=DWL_STEPS, device=DEVICE)
                elapsed = time.time() - t0
                out_dw  = [p.item() for p in p_dw]
                rmsre_dw, est_dw = calculate_rmsre(out_dw, true_dict)
                print(f"  RMSRE={rmsre_dw:.4f}  ({elapsed:.1f}s)  "
                      f"grid:{n1}×{n2} p={p_time}")
                results[tag] = (est_dw, rmsre_dw, elapsed, float(loss_dw), n1, n2)

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ────────────────────────────────────────────────────────────
        for tag, (est, rmsre_val, elapsed, loss_val, n1, n2) in results.items():
            records.append({
                'iter':          it + 1,
                'obs_year':      yr_it,
                'obs_day':       d_it,
                'model':         tag,
                'rmsre':         round(rmsre_val,  6),
                'time_s':        round(elapsed,    2),
                'loss_dw':       round(loss_val,   6),
                'sigmasq_est':   round(est['sigmasq'],    6),
                'range_lat_est': round(est['range_lat'],  6),
                'range_lon_est': round(est['range_lon'],  6),
                'range_t_est':   round(est['range_time'], 6),
                'advec_lat_est': round(est['advec_lat'],  6),
                'advec_lon_est': round(est['advec_lon'],  6),
                'nugget_est':    round(est['nugget'],     6),
                'init_sigmasq':  round(init_orig['sigmasq'],   4),
                'init_range_lon':round(init_orig['range_lon'], 4),
                'n1': n1, 'n2': n2,
            })

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)
        print_running_summary(records, true_dict, it)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE: {len(records)//2} iters completed, {skipped} skipped")
    print(f"{'='*60}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_raw, index=False)
    print_final_summary(df_final, true_dict, num_iters)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    tv_list = [true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]
    summary_rows = []
    for m in MODELS:
        sub = df_final[df_final['model'] == m]
        for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
            v = sub[col].values
            summary_rows.append({
                'model':   m, 'param': lbl, 'true': tv,
                'mean':    round(float(np.mean(v)),   6),
                'median':  round(float(np.median(v)), 6),
                'bias':    round(float(np.mean(v)-tv), 6),
                'std':     round(float(np.std(v)),    6),
                'RMSRE':   round(float(np.sqrt(np.mean(((v-tv)/abs(tv))**2))), 6),
                'MdARE':   round(float(np.median(np.abs((v-tv)/abs(tv)))), 6),
                'P10':     round(float(np.percentile(v,10)), 6),
                'P90':     round(float(np.percentile(v,90)), 6),
            })
    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\nSaved:\n  {output_path/csv_raw}\n  {output_path/csv_summary}")

    # ── Distribution plots ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        plot_dir = output_path / "plots"
        plot_dir.mkdir(exist_ok=True)
        MODEL_COLORS  = {'DW_2d_loc': '#2196F3', 'DW_2d_grid': '#FF9800'}

        # ── 1. Per-parameter plot: panels per model, hist + KDE ───────────────
        for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
            fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 4))
            if len(MODELS) == 1:
                axes = [axes]
            fig.suptitle(f"Distribution of estimates: {lbl}  (true = {tv:.4f})",
                         fontsize=11, fontweight='bold')
            for ax, m in zip(axes, MODELS):
                sub = df_final[df_final['model'] == m][col].dropna().values
                c   = MODEL_COLORS[m]
                n_b = max(5, min(20, len(sub) // 3 + 1))
                ax.hist(sub, bins=n_b, alpha=0.35, color=c, density=True,
                        edgecolor='white', linewidth=0.5)
                if len(sub) >= 3:
                    try:
                        kde = gaussian_kde(sub)
                        xs  = np.linspace(sub.min(), sub.max(), 300)
                        ax.plot(xs, kde(xs), color=c, lw=2.0)
                    except Exception:
                        pass
                ax.axvline(tv, color='black', lw=1.5, ls='--', label=f'true={tv:.3f}')
                ax.axvline(np.median(sub), color=c, lw=1.5, ls=':',
                           label=f'median={np.median(sub):.3f}')
                q1, q3 = np.percentile(sub, [25, 75])
                ax.axvspan(q1, q3, alpha=0.10, color=c)
                ax.set_title(m, fontsize=10)
                ax.set_xlabel(lbl, fontsize=9)
                ax.legend(fontsize=8, framealpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(plot_dir / f"dw_2d_{col}_dist.png", dpi=130, bbox_inches='tight')
            plt.close()

        # ── 2. Overview: all 7 params, models overlaid per panel ─────────────
        n_params = len(P_LABELS)
        n_cols_p = 2
        n_rows_p = (n_params + 1) // n_cols_p
        fig, axes = plt.subplots(n_rows_p, n_cols_p, figsize=(14, 4 * n_rows_p))
        axes = axes.flatten()
        for i, (lbl, col, tv) in enumerate(zip(P_LABELS, P_COLS, tv_list)):
            ax = axes[i]
            for m in MODELS:
                sub = df_final[df_final['model'] == m][col].dropna().values
                c   = MODEL_COLORS[m]
                if len(sub) >= 3:
                    try:
                        kde = gaussian_kde(sub)
                        xs  = np.linspace(sub.min(), sub.max(), 300)
                        ax.plot(xs, kde(xs), color=c, lw=2.0, label=m)
                        ax.fill_between(xs, kde(xs), alpha=0.10, color=c)
                    except Exception:
                        ax.hist(sub, bins=10, alpha=0.3, color=c, density=True, label=m)
                else:
                    ax.hist(sub, bins=5, alpha=0.3, color=c, density=True, label=m)
            ax.axvline(tv, color='black', lw=1.5, ls='--', label=f'true={tv:.3f}')
            ax.set_title(f"{lbl}  (true={tv:.3f})", fontsize=10)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"DW_2d — Parameter Estimate Distributions  ({num_iters} iterations)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "dw_2d_all_params_overview.png", dpi=130, bbox_inches='tight')
        plt.close()

        # ── 3. Boxplot comparison ─────────────────────────────────────────────
        fig, axes = plt.subplots(n_rows_p, n_cols_p, figsize=(14, 4 * n_rows_p))
        axes = axes.flatten()
        for i, (lbl, col, tv) in enumerate(zip(P_LABELS, P_COLS, tv_list)):
            ax = axes[i]
            data_bp = [df_final[df_final['model'] == m][col].dropna().values for m in MODELS]
            bp = ax.boxplot(data_bp, labels=MODELS, patch_artist=True, widths=0.5,
                            medianprops={'color': 'black', 'lw': 2})
            for patch, m in zip(bp['boxes'], MODELS):
                patch.set_facecolor(MODEL_COLORS[m])
                patch.set_alpha(0.6)
            ax.axhline(tv, color='black', lw=1.5, ls='--', label=f'true={tv:.3f}')
            ax.set_title(f"{lbl}  (true={tv:.3f})", fontsize=10)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"DW_2d — Boxplot of Parameter Estimates  ({num_iters} iterations)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "dw_2d_all_params_boxplot.png", dpi=130, bbox_inches='tight')
        plt.close()

        print(f"\n  Plots saved → {plot_dir}/")
        print(f"  - dw_2d_{{param}}_dist.png  × {n_params}  (per-param, per-model hist+KDE)")
        print(f"  - dw_2d_all_params_overview.png  (KDE overlay, all params)")
        print(f"  - dw_2d_all_params_boxplot.png   (boxplot comparison)")

    except ImportError as ie:
        print(f"\n  [Plot skipped — missing library: {ie}]")
    except Exception as pe:
        import traceback
        print(f"\n  [Plot generation failed: {pe}]")
        traceback.print_exc()


if __name__ == "__main__":
    app()
