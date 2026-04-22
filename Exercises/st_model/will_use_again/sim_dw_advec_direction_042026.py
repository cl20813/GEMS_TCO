"""
sim_dw_advec_direction_042026.py

Tests whether differencing direction interacts with advection direction.

Two scenarios (60 iters each, 120 total):
  Scenario A — pure lon advection : advec_lat=0,   advec_lon=-0.2
  Scenario B — pure lat advection : advec_lat=0.2, advec_lon=0

Four filters per scenario:
  Model 1 — Raw    : no spatial filter       (debiased_whittle_raw)
  Model 2 — Lat-1  : Z=X(i+1,j)-X(i,j)      (debiased_whittle_lat1)
  Model 3 — Lon-1  : Z=X(i,j+1)-X(i,j)      (debiased_whittle_lon1)
  Model 4 — 2-1-1-0: Z=-2X+X(i+1,j)+X(i,j+1)(debiased_whittle_2110)

Other true params fixed to July 2024 fit values.

Usage (local test):
  python sim_dw_advec_direction_042026.py --num-iters 3 --lat-factor 10 --lon-factor 4

Usage (Amarel):
  python sim_dw_advec_direction_042026.py --num-iters 60
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
from GEMS_TCO import debiased_whittle_lat1 as dw_lat1
from GEMS_TCO import debiased_whittle_lon1 as dw_lon1
from GEMS_TCO import debiased_whittle_2110 as dw_2110
from GEMS_TCO.data_loader import load_data_dynamic_processed

is_amarel = os.path.exists(config.amarel_data_load_path)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_DW = torch.device("cpu")
DTYPE     = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

MODELS       = ['raw', 'lat1', 'lon1', 'filt_2110']
MODEL_LABELS = ['Raw', 'Lat-1-1', 'Lon-1-1', '2-1-1-0']

SCENARIOS = [
    {'name': 'lon_advec', 'label': 'advec_lat=0 / advec_lon=-0.2',
     'advec_lat': 0.0, 'advec_lon': -0.2},
    {'name': 'lat_advec', 'label': 'advec_lat=0.2 / advec_lon=0',
     'advec_lat': 0.2, 'advec_lon':  0.0},
]

P_LABELS = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
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

    LX, LY, LT = torch.meshgrid(lx, ly, lt, indexing='ij')
    params_cpu = params.to(CPU).to(F32)
    S = get_covariance_on_grid(LX, LY, LT, params_cpu)

    S_fft  = torch.fft.fftn(S)
    S_real = S_fft.real.clamp(min=0)
    noise  = torch.randn(Px, Py, Pt, device=CPU, dtype=F32)
    Z_fft  = torch.fft.fftn(noise) * torch.sqrt(S_real + 1e-12)
    Z      = torch.fft.ifftn(Z_fft).real[:Nx, :Ny, :Nt]

    nugget   = torch.exp(params_cpu[6])
    noise_nug = torch.randn(Nx, Ny, Nt, device=CPU, dtype=F32) * nugget.sqrt()
    return (Z + noise_nug).to(DTYPE).to(DEVICE)


def precompute_mapping_indices(ref_day, lats_hr, lons_hr, grid_coords, day_keys):
    CPU = torch.device("cpu"); F32 = torch.float32
    g_lat, g_lon = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_pts_cpu = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1).to(CPU).to(F32)

    step3_per_t, hr_idx_per_t, src_locs_per_t = {}, {}, {}
    for k in day_keys:
        df = ref_day.get(k, pd.DataFrame())
        if df.empty or len(df) < 2: continue
        # tco_grid uses Source_Latitude/Source_Longitude; fallback to Latitude/Longitude
        if 'Source_Latitude' in df.columns:
            obs_np = df[['Source_Latitude', 'Source_Longitude']].to_numpy(dtype=np.float32)
        else:
            obs_np = df[['Latitude', 'Longitude']].to_numpy(dtype=np.float32)
        obs_np = obs_np[~np.isnan(obs_np).any(axis=1)]
        if len(obs_np) < 2: continue
        obs_t  = torch.tensor(obs_np, device=CPU, dtype=F32)
        bt = BallTree(np.deg2rad(obs_np), metric='haversine')
        dists, idx = bt.query(np.deg2rad(hr_pts_cpu.numpy()), k=1)
        idx = idx.flatten(); dists = dists.flatten()
        step3_per_t[k]  = idx
        hr_idx_per_t[k] = torch.arange(hr_pts_cpu.shape[0], device=CPU)
        src_locs_per_t[k] = obs_t
    return step3_per_t, hr_idx_per_t, src_locs_per_t


def assemble_reg_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                         day_keys, grid_coords, params):
    CPU = torch.device("cpu"); F32 = torch.float32
    Nx, Ny = field.shape[0], field.shape[1]
    lats_hr_flat = torch.linspace(grid_coords[:,0].min(), grid_coords[:,0].max(),
                                  Nx * Ny, device=CPU, dtype=F32)
    lons_hr_flat = torch.linspace(grid_coords[:,1].min(), grid_coords[:,1].max(),
                                  Nx * Ny, device=CPU, dtype=F32)

    reg_map = {}; tensors_for_agg = []
    for t_idx, k in enumerate(day_keys):
        if k not in step3_per_t: continue
        vals = field[:, :, t_idx].flatten().to(CPU).to(F32)
        nn_idx = step3_per_t[k]
        src    = src_locs_per_t[k].to(CPU).to(F32)

        mapped_lats = src[nn_idx, 0]
        mapped_lons = src[nn_idx, 1]
        t_col = torch.full((len(vals),), float(t_idx), device=CPU, dtype=F32)
        tensor = torch.stack([mapped_lats, mapped_lons, vals, t_col], dim=1).to(DTYPE)
        reg_map[k] = tensor
        tensors_for_agg.append(tensor)

    reg_agg = torch.cat(tensors_for_agg, dim=0) if tensors_for_agg else torch.empty(0, 4, dtype=DTYPE)
    return reg_map, reg_agg


def backmap_params(log_params):
    lp = log_params
    phi2 = np.exp(lp[1])
    phi1 = np.exp(lp[0])
    phi3 = np.exp(lp[2])
    phi4 = np.exp(lp[3])
    return {
        'sigmasq':    round(phi1 / phi2, 6),
        'range_lon':  round(1.0 / phi2,  6),
        'range_lat':  round(1.0 / phi2 / np.sqrt(phi3), 6) if phi3 > 0 else float('nan'),
        'range_time': round(1.0 / phi2 / np.sqrt(phi4), 6) if phi4 > 0 else float('nan'),
        'advec_lat':  round(lp[4], 6),
        'advec_lon':  round(lp[5], 6),
        'nugget':     round(np.exp(lp[6]), 6),
    }


def calculate_rmsre(log_params, true_dict):
    est = backmap_params(log_params)
    keys = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
            'advec_lat', 'advec_lon', 'nugget']
    sq_sum = 0.0
    for k in keys:
        tv = true_dict[k]
        ev = est.get(k, float('nan'))
        if abs(tv) > 1e-10:
            sq_sum += ((ev - tv) / abs(tv)) ** 2
    return float(np.sqrt(sq_sum / len(keys))), est


@app.command()
def main(
    num_iters:  int   = typer.Option(60,   help="Iterations per scenario"),
    lat_range:  str   = typer.Option("-3.0,2.0",    help="lat min,max"),
    lon_range:  str   = typer.Option("121.0,131.0", help="lon min,max"),
    years:      str   = typer.Option("2022,2024,2025", help="years comma-separated"),
    month:      int   = typer.Option(7),
    lat_factor: int   = typer.Option(100),
    lon_factor: int   = typer.Option(10),
    dw_steps:   int   = typer.Option(5),
    init_noise: float = typer.Option(0.7),
    seed:       int   = typer.Option(42),
):
    rng        = np.random.default_rng(seed)
    lat_r      = [float(x) for x in lat_range.split(",")]
    lon_r      = [float(x) for x in lon_range.split(",")]
    years_list = [int(y) for y in years.split(",")]

    print(f"Device     : {DEVICE}")
    print(f"Models     : {MODEL_LABELS}")
    print(f"Scenarios  : {[s['name'] for s in SCENARIOS]}")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"Years      : {years_list}  month={month}")
    print(f"Iters/scen : {num_iters}  |  DW steps/iter: {dw_steps}")
    print(f"Init noise : ±{init_noise} log-space")

    output_path = Path(config.amarel_estimates_day_path if is_amarel
                       else config.mac_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_dw_advec_dir_{date_tag}.csv"
    csv_summary = f"sim_dw_advec_dir_summary_{date_tag}.csv"

    # ── Base true parameters (advec will be overridden per scenario) ──────────
    base_true = {
        'sigmasq':    13.059,
        'range_lat':  0.154,
        'range_lon':  0.195,
        'range_time': 1.0,
        'nugget':     0.247,
    }

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
    print(f"  Grid: {n_lat} lat × {n_lon} lon = {N_grid} cells")

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

    # ── DW likelihood objects ─────────────────────────────────────────────────
    print("[Setup 4/4] Building DW likelihood objects...")
    dwl_raw  = dw_raw.debiased_whittle_likelihood()
    dwl_lat1 = dw_lat1.debiased_whittle_likelihood()
    dwl_lon1 = dw_lon1.debiased_whittle_likelihood()
    dwl_2110 = dw_2110.debiased_whittle_likelihood()

    LC, NC, VC, TC = 0, 1, 2, 3

    all_records = []
    skipped = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for scen in SCENARIOS:
        true_dict = {**base_true,
                     'advec_lat': scen['advec_lat'],
                     'advec_lon': scen['advec_lon']}

        phi2     = 1.0 / true_dict['range_lon']
        phi1     = true_dict['sigmasq'] * phi2
        phi3     = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
        phi4     = (true_dict['range_lon'] / true_dict['range_time']) ** 2
        true_log = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                    true_dict['advec_lat'], true_dict['advec_lon'],
                    np.log(true_dict['nugget'])]
        true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

        TRUE_VALS = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                     true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                     true_dict['nugget']]

        def make_random_init(rng_):
            noisy = list(true_log)
            for i in [0, 1, 2, 3, 6]:
                noisy[i] = true_log[i] + rng_.uniform(-init_noise, init_noise)
            for i in [4, 5]:
                scale    = max(abs(true_log[i]), 0.05)
                noisy[i] = true_log[i] + rng_.uniform(-2*scale, 2*scale)
            return noisy

        records = []
        scen_skipped = 0

        print(f"\n{'#'*60}")
        print(f"  SCENARIO: {scen['name']}  ({scen['label']})")
        print(f"{'#'*60}")

        for it in range(num_iters):
            print(f"\n{'='*60}")
            print(f"  [{scen['name']}] Iter {it+1}/{num_iters}")
            print(f"{'='*60}")

            yr_it, d_it, step3_per_t, hr_idx_per_t, src_locs_per_t = \
                all_day_mappings[rng.integers(len(all_day_mappings))]
            initial_vals = make_random_init(rng)
            init_orig    = backmap_params(initial_vals)
            print(f"  Obs: {yr_it} day {d_it}  |  init sig={init_orig['sigmasq']:.3f} "
                  f"rl={init_orig['range_lon']:.3f} nug={init_orig['nugget']:.3f}")

            rec = {'scenario': scen['name'], 'iter': it+1,
                   'obs_year': yr_it, 'obs_day': d_it,
                   'true_advec_lat': true_dict['advec_lat'],
                   'true_advec_lon': true_dict['advec_lon'],
                   'init_sigmasq': round(init_orig['sigmasq'], 4),
                   'init_range_lon': round(init_orig['range_lon'], 4)}

            try:
                # ── Generate field and reg_map ────────────────────────────────
                field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
                reg_map, reg_agg = assemble_reg_dataset(
                    field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                    DUMMY_KEYS, grid_coords, true_params)
                del field

                params_kwargs = dict(
                    params_list=[true_dict['sigmasq'], true_dict['range_lat'],
                                 true_dict['range_lon'], true_dict['range_time'],
                                 true_dict['advec_lat'], true_dict['advec_lon'],
                                 true_dict['nugget']],
                    lat_range=lat_r, lon_range=lon_r)

                # ── Raw ───────────────────────────────────────────────────────
                db_raw = dw_raw.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
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

                # ── Lat-1-1 ───────────────────────────────────────────────────
                db_lat = dw_lat1.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                              **params_kwargs)
                cur_lat = db_lat.generate_spatially_filtered_days(
                    lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
                sl_lat  = [cur_lat[cur_lat[:, TC]==t] for t in torch.unique(cur_lat[:, TC])]
                J_lat, n1_la, n2_la, _, tap_lat, om_lat = dwl_raw.generate_Jvector_tapered_mv(
                    sl_lat, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
                I_lat  = dwl_raw.calculate_sample_periodogram_vectorized(J_lat)
                ta_lat = dwl_raw.calculate_taper_autocorrelation_multivariate(
                    tap_lat, om_lat, n1_la, n2_la, DEVICE_DW)
                del om_lat

                # ── Lon-1-1 ───────────────────────────────────────────────────
                db_lon = dw_lon1.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                              **params_kwargs)
                cur_lon = db_lon.generate_spatially_filtered_days(
                    lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
                sl_lon  = [cur_lon[cur_lon[:, TC]==t] for t in torch.unique(cur_lon[:, TC])]
                J_lon, n1_lo, n2_lo, _, tap_lon, om_lon = dwl_raw.generate_Jvector_tapered_mv(
                    sl_lon, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
                I_lon  = dwl_raw.calculate_sample_periodogram_vectorized(J_lon)
                ta_lon = dwl_raw.calculate_taper_autocorrelation_multivariate(
                    tap_lon, om_lon, n1_lo, n2_lo, DEVICE_DW)
                del om_lon

                # ── 2-1-1-0 ───────────────────────────────────────────────────
                db_old = dw_2110.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                              **params_kwargs)
                cur_old = db_old.generate_spatially_filtered_days(
                    lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
                sl_old  = [cur_old[cur_old[:, TC]==t] for t in torch.unique(cur_old[:, TC])]
                J_old, n1o, n2o, _, tap_old, om_old = dwl_raw.generate_Jvector_tapered_mv(
                    sl_old, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
                I_old  = dwl_raw.calculate_sample_periodogram_vectorized(J_old)
                ta_old = dwl_raw.calculate_taper_autocorrelation_multivariate(
                    tap_old, om_old, n1o, n2o, DEVICE_DW)
                del om_old

                print(f"  Raw:{n1}×{n2}  Lat1:{n1_la}×{n2_la}  "
                      f"Lon1:{n1_lo}×{n2_lo}  2110:{n1o}×{n2o}  p={p_time}")

                # ── Optimize: Raw ─────────────────────────────────────────────
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
                print(f"  [raw]     RMSRE={rmsre1:.4f}  ({t1:.1f}s)")

                # ── Optimize: Lat-1-1 ─────────────────────────────────────────
                p2 = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                      for v in initial_vals]
                opt2 = torch.optim.LBFGS(p2, lr=1.0, max_iter=20, max_eval=100,
                                          history_size=10, line_search_fn="strong_wolfe",
                                          tolerance_grad=1e-5)
                t0 = time.time()
                _, _, _, loss2, _ = dwl_lat1.run_lbfgs_tapered(
                    params_list=p2, optimizer=opt2, I_sample=I_lat,
                    n1=n1_la, n2=n2_la, p_time=p_time, taper_autocorr_grid=ta_lat,
                    max_steps=dw_steps, device=DEVICE_DW)
                t2 = time.time() - t0
                rmsre2, est2 = calculate_rmsre([p.item() for p in p2], true_dict)
                print(f"  [lat1]    RMSRE={rmsre2:.4f}  ({t2:.1f}s)")

                # ── Optimize: Lon-1-1 ─────────────────────────────────────────
                p3 = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                      for v in initial_vals]
                opt3 = torch.optim.LBFGS(p3, lr=1.0, max_iter=20, max_eval=100,
                                          history_size=10, line_search_fn="strong_wolfe",
                                          tolerance_grad=1e-5)
                t0 = time.time()
                _, _, _, loss3, _ = dwl_lon1.run_lbfgs_tapered(
                    params_list=p3, optimizer=opt3, I_sample=I_lon,
                    n1=n1_lo, n2=n2_lo, p_time=p_time, taper_autocorr_grid=ta_lon,
                    max_steps=dw_steps, device=DEVICE_DW)
                t3 = time.time() - t0
                rmsre3, est3 = calculate_rmsre([p.item() for p in p3], true_dict)
                print(f"  [lon1]    RMSRE={rmsre3:.4f}  ({t3:.1f}s)")

                # ── Optimize: 2-1-1-0 ─────────────────────────────────────────
                p4 = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                      for v in initial_vals]
                opt4 = torch.optim.LBFGS(p4, lr=1.0, max_iter=20, max_eval=100,
                                          history_size=10, line_search_fn="strong_wolfe",
                                          tolerance_grad=1e-5)
                t0 = time.time()
                _, _, _, loss4, _ = dwl_2110.run_lbfgs_tapered(
                    params_list=p4, optimizer=opt4, I_sample=I_old,
                    n1=n1o, n2=n2o, p_time=p_time, taper_autocorr_grid=ta_old,
                    max_steps=dw_steps, device=DEVICE_DW)
                t4 = time.time() - t0
                rmsre4, est4 = calculate_rmsre([p.item() for p in p4], true_dict)
                print(f"  [2-1-1-0] RMSRE={rmsre4:.4f}  ({t4:.1f}s)")

            except Exception as e:
                scen_skipped += 1; skipped += 1
                import traceback
                print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            # ── Record ───────────────────────────────────────────────────────
            for m, est, rmsre, loss, t_s, n1_, n2_ in [
                ('raw',      est1, rmsre1, loss1, t1, n1,    n2),
                ('lat1',     est2, rmsre2, loss2, t2, n1_la, n2_la),
                ('lon1',     est3, rmsre3, loss3, t3, n1_lo, n2_lo),
                ('filt_2110',est4, rmsre4, loss4, t4, n1o,   n2o),
            ]:
                rec[f'rmsre_{m}']         = round(rmsre,       6)
                rec[f'time_s_{m}']        = round(t_s,         2)
                rec[f'loss_{m}']          = round(float(loss), 6)
                rec[f'sigmasq_est_{m}']   = round(est['sigmasq'],    6)
                rec[f'range_lat_est_{m}'] = round(est['range_lat'],  6)
                rec[f'range_lon_est_{m}'] = round(est['range_lon'],  6)
                rec[f'range_t_est_{m}']   = round(est['range_time'], 6)
                rec[f'advec_lat_est_{m}'] = round(est['advec_lat'],  6)
                rec[f'advec_lon_est_{m}'] = round(est['advec_lon'],  6)
                rec[f'nugget_est_{m}']    = round(est['nugget'],     6)
                rec[f'n1_{m}'] = n1_; rec[f'n2_{m}'] = n2_

            records.append(rec)
            all_records.append(rec)
            pd.DataFrame(all_records).to_csv(output_path / csv_raw, index=False)

            # ── Running summary ───────────────────────────────────────────────
            n_done = len(records)
            if n_done % 10 == 0 or n_done == 1:
                print(f"\n  ── Running summary [{scen['name']}] "
                      f"({n_done} done / {it+1} attempted) ──")
                cw = 9
                for m, ml in zip(MODELS, MODEL_LABELS):
                    rv = np.array([r[f'rmsre_{m}'] for r in records])
                    print(f"  [{ml}]  RMSRE mean={np.mean(rv):.4f}  median={np.median(rv):.4f}")

        print(f"\n  [{scen['name']}] DONE: {len(records)} completed, {scen_skipped} skipped")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OVERALL: {len(all_records)} completed, {skipped} skipped")
    print(f"{'='*60}")

    df_final = pd.DataFrame(all_records)
    df_final.to_csv(output_path / csv_raw, index=False)

    cw = 9
    summary_rows = []
    for scen in SCENARIOS:
        df_s = df_final[df_final['scenario'] == scen['name']]
        if df_s.empty: continue
        print(f"\n{'='*60}")
        print(f"  SCENARIO: {scen['name']}  ({scen['label']})")
        print(f"{'='*60}")
        true_dict_s = {**base_true, 'advec_lat': scen['advec_lat'],
                       'advec_lon': scen['advec_lon']}
        TRUE_VALS_S = [true_dict_s['sigmasq'],    true_dict_s['range_lat'],
                       true_dict_s['range_lon'],  true_dict_s['range_time'],
                       true_dict_s['advec_lat'],  true_dict_s['advec_lon'],
                       true_dict_s['nugget']]
        for m, ml in zip(MODELS, MODEL_LABELS):
            print(f"\n  [{ml}]")
            print(f"  {'param':<12} {'true':>{cw}}  {'mean':>{cw}}  {'median':>{cw}}  "
                  f"{'bias':>{cw}}  {'RMSRE':>{cw}}  {'RMSRE_med':>{cw}}  {'P90-P10':>{cw}}")
            print(f"  {'-'*94}")
            for lbl, tv in zip(P_LABELS, TRUE_VALS_S):
                col  = f'range_t_est_{m}' if lbl == 'range_t' else f'{lbl}_est_{m}'
                if col not in df_s.columns: continue
                vals = df_s[col].values
                cm   = float(np.mean(vals))
                med  = float(np.median(vals))
                bi   = cm - tv
                denom = max(abs(tv), 1e-10)
                rm   = float(np.sqrt(np.mean(((vals - tv) / denom) ** 2)))
                rmd  = float(np.median(np.abs((vals - tv) / denom)))
                p9p1 = float(np.percentile(vals, 90) - np.percentile(vals, 10))
                print(f"  {lbl:<12} {tv:>{cw}.4f}  {cm:>{cw}.4f}  {med:>{cw}.4f}  "
                      f"{bi:>{cw}.4f}  {rm:>{cw}.4f}  {rmd:>{cw}.4f}  {p9p1:>{cw}.4f}")
                summary_rows.append({
                    'scenario': scen['name'], 'model': ml, 'param': lbl,
                    'true': tv, 'mean': round(cm, 6), 'median': round(med, 6),
                    'bias': round(bi, 6), 'std': round(float(np.std(vals)), 6),
                    'RMSRE': round(rm, 6), 'RMSRE_median': round(rmd, 6),
                    'P10': round(float(np.percentile(vals, 10)), 6),
                    'P90': round(float(np.percentile(vals, 90)), 6),
                    'P90_P10': round(p9p1, 6),
                })
            rv = df_s[f'rmsre_{m}'].values
            p9p1_rv = float(np.percentile(rv, 90) - np.percentile(rv, 10))
            print(f"  {'-'*94}")
            print(f"  {'Overall':<12} {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
                  f"{np.mean(rv):>{cw}.4f}  {np.median(rv):>{cw}.4f}  {p9p1_rv:>{cw}.4f}")
            summary_rows.append({
                'scenario': scen['name'], 'model': ml, 'param': 'Overall',
                'true': float('nan'), 'mean': float('nan'), 'median': float('nan'),
                'bias': float('nan'), 'std': float('nan'),
                'RMSRE': round(float(np.mean(rv)), 6),
                'RMSRE_median': round(float(np.median(rv)), 6),
                'P10': float('nan'), 'P90': float('nan'),
                'P90_P10': round(p9p1_rv, 6),
            })

        # ── Distribution: Q1 / Median / Q3 / Min / Max ───────────────────────
        print(f"\n  ── DISTRIBUTION [{scen['name']}] — Q1 / Median / Q3 / Min / Max ──")
        cw2 = 9
        for m, ml in zip(MODELS, MODEL_LABELS):
            print(f"\n  [{ml}]")
            print(f"  {'param':<12} {'true':>{cw2}}  {'Q1':>{cw2}}  {'median':>{cw2}}  "
                  f"{'Q3':>{cw2}}  {'min':>{cw2}}  {'max':>{cw2}}")
            print(f"  {'-'*80}")
            for lbl, tv in zip(P_LABELS, TRUE_VALS_S):
                col = f'range_t_est_{m}' if lbl == 'range_t' else f'{lbl}_est_{m}'
                if col not in df_s.columns: continue
                vals = df_s[col].values
                q1   = float(np.percentile(vals, 25))
                med  = float(np.median(vals))
                q3   = float(np.percentile(vals, 75))
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                print(f"  {lbl:<12} {tv:>{cw2}.4f}  {q1:>{cw2}.4f}  {med:>{cw2}.4f}  "
                      f"{q3:>{cw2}.4f}  {vmin:>{cw2}.4f}  {vmax:>{cw2}.4f}")

    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\nSaved: {output_path / csv_raw}")
    print(f"Saved: {output_path / csv_summary}")


if __name__ == "__main__":
    app()
