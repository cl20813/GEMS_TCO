"""
sim_dw_mixed_040626.py

Simulation study: Mixed-frequency Debiased Whittle estimator.

Splits the Whittle objective by spatial frequency:

  L_mixed(θ) = Σ_{ω∈Ω_L} ℓ_raw(ω; θ)  +  Σ_{ω∈Ω_H} ℓ_diff(ω; θ)

  Ω_L = { (k1,k2) : k1 ≤ K1, k2 ≤ K2 } \\ {(0,0)}
        → raw (no-filter) periodogram.  Cov: C_X(u·δ1, v·δ2, τ) directly.
  Ω_H = complement on 2D-diff grid
        → 2D-filter periodogram. Cov: ΣΣ h·h·C_X with filter cross-terms.
  K1 = floor(n1·α),  K2 = floor(n2·α)   (--freq-alpha, default 0.20)

Hypothesis: low-freq raw contribution recovers advection information lost
by spatial differencing, while high-freq diff contribution suppresses
spatial non-stationarity effects.

Statistical validity: composite likelihood estimating equation is unbiased
at θ_true → consistent. Variance: Godambe / sandwich matrix.

Comparison targets:
  sim_three_model_comparison_031926.py  → DW_2d  (full diff)
  sim_dw_lat1d_040526.py               → DW_lat1d
  sim_dw_raw_040626.py                 → DW_raw  (full raw)

Usage:
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation
  python sim_dw_mixed_040626.py --num-iters 1 --lat-factor 10 --lon-factor 4
  python sim_dw_mixed_040626.py --num-iters 1000
  python sim_dw_mixed_040626.py --num-iters 1000 --freq-alpha 0.15
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

from GEMS_TCO import debiased_whittle_mixed as dwm
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


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
    Nx,Ny,Nt = len(lats_hr), len(lons_hr), t_steps
    Px,Py,Pt = 2*Nx, 2*Ny, 2*Nt
    lx = torch.arange(Px,device=CPU,dtype=F32)*dlat; lx[Px//2:] -= Px*dlat
    ly = torch.arange(Py,device=CPU,dtype=F32)*dlon; ly[Py//2:] -= Py*dlon
    lt = torch.arange(Pt,device=CPU,dtype=F32);      lt[Pt//2:] -= Pt
    pc = params.cpu().float()
    Lx,Ly,Lt = torch.meshgrid(lx,ly,lt,indexing='ij')
    C = get_covariance_on_grid(Lx,Ly,Lt,pc)
    S = torch.fft.fftn(C); S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px,Py,Pt,device=CPU,dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real)*noise).real[:Nx,:Ny,:Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Obs → grid mapping ────────────────────────────────────────────────────────

def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N, -1, dtype=np.int64)
    dist, cell = grid_tree.query(np.radians(src_np_valid), k=1)
    dist = dist.flatten(); cell = cell.flatten()
    asgn = np.full(N,-1,dtype=np.int64); best = np.full(N,np.inf)
    for oi,(cj,d) in enumerate(zip(cell,dist)):
        if d < best[cj]:
            asgn[cj] = oi; best[cj] = d
    return asgn


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    lg,ng = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_np = torch.stack([lg.flatten(), ng.flatten()],dim=1).cpu().numpy()
    hr_tree   = BallTree(np.radians(hr_np),                  metric='haversine')
    grid_tree = BallTree(np.radians(grid_coords.cpu().numpy()), metric='haversine')
    s3, hr_idx, src_locs = [], [], []
    for key in sorted_keys:
        ref_t    = ref_day_map[key].to(DEVICE)
        slocs    = ref_t[:,:2]
        snp      = slocs.cpu().numpy()
        valid    = ~np.isnan(snp).any(axis=1)
        snp_v    = snp[valid]
        asgn     = apply_step3_1to1(snp_v, grid_coords.cpu().numpy(), grid_tree)
        s3.append(asgn)
        if valid.sum() > 0:
            _, hi = hr_tree.query(np.radians(snp_v), k=1)
            hr_idx.append(torch.tensor(hi.flatten(), device=DEVICE))
        else:
            hr_idx.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
        src_locs.append(slocs[valid])
    return s3, hr_idx, src_locs


# ── Dataset assembly ───────────────────────────────────────────────────────────

def assemble_reg_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params, t_offset=21.0):
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    ff         = field.reshape(-1, 8)
    reg_map, reg_list = {}, []
    for t_idx, key in enumerate(sorted_keys):
        t_val  = float(t_offset + t_idx)
        assign = step3_per_t[t_idx]
        hi     = hr_idx_per_t[t_idx]
        Nv     = hi.shape[0]
        dummy  = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0: dummy[t_idx-1] = 1.0
        sim_vals = (ff[hi, t_idx] + torch.randn(Nv,device=DEVICE,dtype=DTYPE)*nugget_std
                    if Nv > 0 else torch.zeros(0,device=DEVICE,dtype=DTYPE))
        rows = torch.full((N_grid,11), float('nan'), device=DEVICE, dtype=DTYPE)
        rows[:,:2] = grid_coords
        rows[:,3]  = t_val
        rows[:,4:] = dummy.unsqueeze(0).expand(N_grid,-1)
        if Nv > 0:
            at = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            f_ = at >= 0
            rows[f_,2] = sim_vals[at[f_]]
        reg_map[key] = rows.detach()
        reg_list.append(rows.detach())
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


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v: float              = typer.Option(0.5,    help="(unused, API parity)"),
    mm_cond_number: int   = typer.Option(100,    help="(unused, API parity)"),
    num_iters: int        = typer.Option(1000,   help="Simulation iterations"),
    years: str            = typer.Option("2022,2024,2025"),
    month: int            = typer.Option(7),
    lat_range: str        = typer.Option("-3,2"),
    lon_range: str        = typer.Option("121,131"),
    lat_factor: int       = typer.Option(100),
    lon_factor: int       = typer.Option(20),
    init_noise: float     = typer.Option(0.7),
    seed: int             = typer.Option(42),
    freq_alpha: float     = typer.Option(0.20,
        help="Low-freq cutoff fraction α: K1=floor(n1·α), K2=floor(n2·α). "
             "Frequencies (k1≤K1,k2≤K2) use raw periodogram; rest use 2D-diff."),
) -> None:

    rng        = np.random.default_rng(seed)
    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device     : {DEVICE}")
    print(f"Model      : DW_mixed  (composite-likelihood frequency split)")
    print(f"Filter     : low-freq (α≤{freq_alpha}) → raw C_X  |  high-freq → 2D diff")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"Years      : {years_list}  month={month}")
    print(f"Iterations : {num_iters}")
    print(f"Init noise : ±{init_noise} log-space")
    print(f"freq_alpha : {freq_alpha}  (K1=floor(n1·α), K2=floor(n2·α))")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    alpha_tag   = str(freq_alpha).replace('.', 'p')
    csv_raw     = f"sim_dw_mixed_a{alpha_tag}_{date_tag}.csv"
    csv_summary = f"sim_dw_mixed_a{alpha_tag}_summary_{date_tag}.csv"

    # ── True parameters: Scenario D ──────────────────────────────────────────
    true_dict = {
        'sigmasq': 10.0, 'range_lat': 0.5,  'range_lon':  0.6,
        'range_time': 2.5, 'advec_lat': 0.25, 'advec_lon': -0.16, 'nugget': 1.2,
    }
    phi2      = 1.0 / true_dict['range_lon']
    phi1      = true_dict['sigmasq'] * phi2
    phi3      = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4      = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log  = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
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

    # ── Load GEMS obs patterns ────────────────────────────────────────────────
    print("\n[Setup 1/4] Loading GEMS obs patterns...")
    data_loader = load_data_dynamic_processed(config.amarel_data_load_path)
    all_day_mappings = []
    year_dfmaps, year_means = {}, {}
    for yr in years_list:
        df_map_yr, _, _, mm_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1,1], mm_cond_number=mm_cond_number,
            years_=[yr], months_=[month], lat_range=lat_r, lon_range=lon_r, is_whittle=False)
        year_dfmaps[yr] = df_map_yr; year_means[yr] = mm_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots")

    # ── Build regular target grid ─────────────────────────────────────────────
    print("[Setup 2/4] Building regular target grid...")
    lats_grid = torch.arange(max(lat_r), min(lat_r)-0.0001, -DELTA_LAT_BASE, device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1] +0.0001,  DELTA_LON_BASE,  device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid*10000)/10000
    lons_grid = torch.round(lons_grid*10000)/10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    n_lat  = len(lats_grid); n_lon = len(lons_grid)
    print(f"  Grid: {n_lat}×{n_lon} = {N_grid} cells")
    print(f"  Raw grid: {n_lat}×{n_lon}   Diff grid: {n_lat-1}×{n_lon-1}")

    # ── High-res grid & precompute mappings ───────────────────────────────────
    print("[Setup 3/4] High-res grid and obs mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)}×{len(lons_hr)} = {len(lats_hr)*len(lons_hr):,} cells")
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

    # ── Verify structure ──────────────────────────────────────────────────────
    print("[Setup 4/4] Verifying...")
    _yr0, _d0, _s3, _hi, _src = all_day_mappings[0]
    f0 = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    r0, _ = assemble_reg_dataset(f0, _s3, _hi, _src, DUMMY_KEYS, grid_coords, true_params)
    del f0
    fst = list(r0.values())[0]
    print(f"  First time step: {(~torch.isnan(fst[:,2])).sum().item()}/{N_grid} valid cells")

    # ── DW settings ──────────────────────────────────────────────────────────
    DWL_STEPS = 5
    LC, NC, VC, TC = 0, 1, 2, 3
    dwl = dwm.debiased_whittle_likelihood()

    P_COLS   = ['sigmasq_est','range_lat_est','range_lon_est',
                'range_t_est','advec_lat_est','advec_lon_est','nugget_est']
    P_LABELS = ['sigmasq','range_lat','range_lon','range_t','advec_lat','advec_lon','nugget']
    TRUE_VALS = [true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                 true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                 true_dict['nugget']]

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
            # ── Generate field ────────────────────────────────────────────────
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
            reg_map, reg_agg = assemble_reg_dataset(
                field, s3, hi_t, src_t, DUMMY_KEYS, grid_coords, true_params)
            del field

            # ── Raw preprocessing (no filter, demean) ─────────────────────────
            db_raw = dwm.debiased_whittle_preprocess_raw(
                [reg_agg], [reg_map], day_idx=0,
                params_list=[true_dict['sigmasq'],   true_dict['range_lat'],
                             true_dict['range_lon'],  true_dict['range_time'],
                             true_dict['advec_lat'],  true_dict['advec_lon'],
                             true_dict['nugget']],
                lat_range=lat_r, lon_range=lon_r)
            cur_raw = db_raw.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
            ut_raw  = torch.unique(cur_raw[:, TC])
            sl_raw  = [cur_raw[cur_raw[:, TC]==t] for t in ut_raw]

            J_raw, n1, n2, p_time, tap_raw, om_raw = dwl.generate_Jvector_tapered_mv(
                sl_raw, dwl.cgn_hamming, LC, NC, VC, DEVICE)
            I_raw  = dwl.calculate_sample_periodogram_vectorized(J_raw)
            ta_raw = dwl.calculate_taper_autocorrelation_multivariate(tap_raw, om_raw, n1, n2, DEVICE)
            del om_raw

            # ── Diff preprocessing (2D filter) ────────────────────────────────
            db_diff = dwm.debiased_whittle_preprocess_diff(
                [reg_agg], [reg_map], day_idx=0,
                params_list=[true_dict['sigmasq'],   true_dict['range_lat'],
                             true_dict['range_lon'],  true_dict['range_time'],
                             true_dict['advec_lat'],  true_dict['advec_lon'],
                             true_dict['nugget']],
                lat_range=lat_r, lon_range=lon_r)
            cur_diff = db_diff.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
            ut_diff  = torch.unique(cur_diff[:, TC])
            sl_diff  = [cur_diff[cur_diff[:, TC]==t] for t in ut_diff]

            J_diff, n1d, n2d, _, tap_diff, om_diff = dwl.generate_Jvector_tapered_mv(
                sl_diff, dwl.cgn_hamming, LC, NC, VC, DEVICE)
            I_diff  = dwl.calculate_sample_periodogram_vectorized(J_diff)
            ta_diff = dwl.calculate_taper_autocorrelation_multivariate(
                tap_diff, om_diff, n1d, n2d, DEVICE)
            del om_diff

            # ── Frequency cutoffs ─────────────────────────────────────────────
            K1 = int(n1  * freq_alpha)
            K2 = int(n2  * freq_alpha)
            n_low  = (K1+1)*(K2+1) - 1           # exclude DC
            n_high = n1d*n2d - (K1+1)*(K2+1)     # diff grid minus low-freq corner
            print(f"  Raw grid: {n1}×{n2}  Diff grid: {n1d}×{n2d}  "
                  f"K1={K1} K2={K2}  "
                  f"|Ω_L|={n_low}  |Ω_H|={n_high}  total={n_low+n_high}")

            # ── Optimize ──────────────────────────────────────────────────────
            p_dw = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True)
                    for v in init_vals]
            opt  = torch.optim.LBFGS(p_dw, lr=1.0, max_iter=20, max_eval=100,
                                     history_size=10, line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-5)

            t0 = time.time()
            _, _, _, loss_mixed, _ = dwl.run_lbfgs_mixed(
                params_list=p_dw, optimizer=opt,
                I_samp_raw=I_raw, I_samp_diff=I_diff,
                n1=n1, n2=n2, n1d=n1d, n2d=n2d,
                p_time=p_time,
                taper_auto_raw=ta_raw, taper_auto_diff=ta_diff,
                K1=K1, K2=K2,
                max_steps=DWL_STEPS, device=DEVICE)
            t_elapsed = time.time() - t0

            out_dw = [p.item() for p in p_dw]
            rmsre_dw, est_dw = calculate_rmsre(out_dw, true_dict)
            print(f"  RMSRE={rmsre_dw:.4f}  ({t_elapsed:.1f}s)")

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ────────────────────────────────────────────────────────────
        records.append({
            'iter':          it + 1,
            'obs_year':      yr_it,
            'obs_day':       d_it,
            'model':         f'DW_mixed_a{freq_alpha}',
            'freq_alpha':    freq_alpha,
            'K1':            K1, 'K2': K2,
            'n_low':         n_low, 'n_high': n_high,
            'rmsre':         round(rmsre_dw,      6),
            'time_s':        round(t_elapsed,     2),
            'loss_mixed':    round(float(loss_mixed), 6),
            'sigmasq_est':   round(est_dw['sigmasq'],    6),
            'range_lat_est': round(est_dw['range_lat'],  6),
            'range_lon_est': round(est_dw['range_lon'],  6),
            'range_t_est':   round(est_dw['range_time'], 6),
            'advec_lat_est': round(est_dw['advec_lat'],  6),
            'advec_lon_est': round(est_dw['advec_lon'],  6),
            'nugget_est':    round(est_dw['nugget'],     6),
            'init_sigmasq':  round(init_orig['sigmasq'],   4),
            'init_range_lon':round(init_orig['range_lon'], 4),
            'n1': n1, 'n2': n2, 'n1d': n1d, 'n2d': n2d,
        })

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)

        # ── Running summary ───────────────────────────────────────────────────
        nd_ = len(records)
        print(f"\n  ── Running summary ({nd_} done / {it+1} attempted) ──")
        cw = 10
        print(f"  {'param':<12} {'true':>{cw}}  {'mean':>{cw}}  {'bias':>{cw}}  "
              f"{'RMSRE':>{cw}}  {'MdARE':>{cw}}  {'P90-P10':>{cw}}")
        print(f"  {'-'*80}")
        for lbl, col, tv in zip(P_LABELS, P_COLS, TRUE_VALS):
            vals    = np.array([r[col] for r in records])
            mn_     = np.mean(vals)
            bi_     = mn_ - tv
            rm_     = np.sqrt(np.mean(((vals-tv)/abs(tv))**2))
            md_     = np.median(np.abs((vals-tv)/abs(tv)))
            pp_     = np.percentile(vals,90) - np.percentile(vals,10)
            print(f"  {lbl:<12} {tv:>{cw}.4f}  {mn_:>{cw}.4f}  {bi_:>{cw}.4f}  "
                  f"{rm_:>{cw}.4f}  {md_:>{cw}.4f}  {pp_:>{cw}.4f}")
        ra = np.array([r['rmsre'] for r in records])
        print(f"  {'-'*80}")
        print(f"  {'Overall':<12} {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
              f"{np.mean(ra):>{cw}.4f}  {np.median(ra):>{cw}.4f}  "
              f"{np.percentile(ra,90)-np.percentile(ra,10):>{cw}.4f}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE: {len(records)} completed, {skipped} skipped")
    print(f"{'='*60}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_raw, index=False)

    summary_rows = []
    for lbl, col, tv in zip(P_LABELS, P_COLS, TRUE_VALS):
        vals = df_final[col].values
        summary_rows.append({
            'param':   lbl, 'true': tv,
            'mean':    round(float(np.mean(vals)),   6),
            'median':  round(float(np.median(vals)), 6),
            'bias':    round(float(np.mean(vals)-tv), 6),
            'std':     round(float(np.std(vals)),    6),
            'RMSRE':   round(float(np.sqrt(np.mean(((vals-tv)/abs(tv))**2))), 6),
            'MdARE':   round(float(np.median(np.abs((vals-tv)/abs(tv)))), 6),
            'P10':     round(float(np.percentile(vals,10)), 6),
            'P90':     round(float(np.percentile(vals,90)), 6),
            'P90_P10': round(float(np.percentile(vals,90)-np.percentile(vals,10)), 6),
        })
    summary_rows.append({
        'param':'Overall','true':float('nan'),'mean':float('nan'),'median':float('nan'),
        'bias':float('nan'),'std':float('nan'),
        'RMSRE':round(float(np.mean(df_final['rmsre'].values)),6),
        'MdARE':round(float(np.median(df_final['rmsre'].values)),6),
        'P10':float('nan'),'P90':float('nan'),'P90_P10':float('nan'),
    })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(output_path / csv_summary, index=False)
    print(f"\nSaved:\n  {output_path/csv_raw}\n  {output_path/csv_summary}")
    print(f"\nFinal summary:\n{df_sum[['param','true','mean','bias','RMSRE','MdARE']].to_string(index=False)}")


if __name__ == "__main__":
    app()
