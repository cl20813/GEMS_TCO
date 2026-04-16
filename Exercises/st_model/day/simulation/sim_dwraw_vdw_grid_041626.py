"""
sim_vdw_grid_040626.py

Simulation study: Vecchia vs Debiased Whittle (raw) on a complete regular grid.

Design
------
Data generated directly on the target grid — NO high-res FFT → real obs locations
→ step3 gridification.  Every grid cell is observed at every time step (no NaN).

  Vecchia  : exact GMRF likelihood (Vecchia approximation) on complete regular grid.
             Uses Vecchia-irr API; since data is on the grid, irr ≡ reg.

  DW_raw   : Debiased Whittle with identity filter (H=1), per-slice demean,
             DC excluded.  Same raw likelihood as used in real data fitting.

Research question
-----------------
What is the pure DW spectral approximation error relative to near-exact Vecchia,
on ideal (complete, stationary, perfectly gridded) data with no gridification?

Interpreting together with sim_dw_raw_040626.py:
  DW_raw error here          = DW spectral approximation bias (no gridification)
  DW_raw_loc - DW_raw_grid   = pure gridification bias   (from sim_dw_raw)
  DW_raw_loc - Vecchia(grid) = total bias vs near-exact reference

Usage:
  python sim_vdw_grid_040626.py --num-iters 1
  python sim_vdw_grid_040626.py --num-iters 500
  python sim_vdw_grid_040626.py --num-iters 500 --mm-cond-number 150
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

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle_raw as dw_raw_module
from GEMS_TCO import configuration as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

MODELS   = ['Vecchia', 'DW_raw']
P_COLS   = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
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


# ── FFT field on target grid directly ─────────────────────────────────────────

def generate_field_on_grid(lats_grid, lons_grid, t_steps, params, dlat, dlon):
    """Generate one FFT circulant embedding realization directly on the target grid.

    No high-res intermediate step — field is sampled on the exact target grid cells.
    Returns field [N_lat, N_lon, T] on DEVICE in DTYPE.
    """
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
    return field.to(dtype=DTYPE, device=DEVICE)   # [N_lat, N_lon, T]


# ── Assemble complete-grid dataset ────────────────────────────────────────────

def assemble_grid_dataset(field, grid_coords, true_params, t_offset=21.0):
    """Build dataset from FFT field sampled at every grid cell.

    No step3, no NaN in value column — all N_grid cells are observed at each step.

    Returns:
      grid_map : dict  key → [N_grid, 11] tensor  (no NaN in col 2)
      grid_agg : [N_grid * T, 11] tensor
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_lat, N_lon, T = field.shape
    N_grid = N_lat * N_lon

    grid_map  = {}
    grid_list = []
    field_flat = field.reshape(N_grid, T)   # [N_grid, T]

    for t_idx in range(T):
        key   = f't{t_idx}'
        t_val = float(t_offset + t_idx)

        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0
        dummy_row = dummy.unsqueeze(0).expand(N_grid, -1)   # [N_grid, 7]

        gp_vals  = field_flat[:, t_idx]
        sim_vals = gp_vals + torch.randn(N_grid, device=DEVICE, dtype=DTYPE) * nugget_std

        rows = torch.zeros(N_grid, 11, device=DEVICE, dtype=DTYPE)
        rows[:, :2] = grid_coords          # lat, lon
        rows[:, 2]  = sim_vals             # value  (no NaN)
        rows[:, 3]  = t_val               # time
        rows[:, 4:] = dummy_row           # time dummies

        grid_map[key] = rows.detach()
        grid_list.append(rows.detach())

    return grid_map, torch.cat(grid_list, dim=0)


# ── Metrics ────────────────────────────────────────────────────────────────────

def backmap_params(out_params):
    p = out_params
    if isinstance(p[0], torch.Tensor):
        p = [x.item() if x.numel() == 1 else x[0].item() for x in p]
    p = [float(x) for x in p]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
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


# ── Summary printers ───────────────────────────────────────────────────────────

def print_running_summary(records, true_dict, it):
    n_done  = len([r for r in records if r['model'] == MODELS[0]])
    tv_list = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]
    cw = 14

    print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
    hdr = f"  {'param':<13} {'true':>{cw}}" + "".join(f"  {m:>{cw}}" for m in MODELS)
    print(hdr)
    print(f"  {'-'*(13 + (cw+2)*3)}")

    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        row = f"  {lbl:<13} {tv:>{cw}.4f}"
        for m in MODELS:
            vals = [r[col] for r in records if r['model'] == m]
            row += f"  {np.mean(vals):>{cw}.4f}" if vals else f"  {'—':>{cw}}"
        print(row)

    print(f"\n  [Min | Q1 | Q2(med) | Q3 | Max  — per param]")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        print(f"  {lbl} (true={tv:.4f})")
        for m in MODELS:
            vals = np.array([r[col] for r in records if r['model'] == m])
            if len(vals) < 2:
                print(f"    {m:<14} (insufficient data)")
                continue
            q1, q2, q3 = np.percentile(vals, [25, 50, 75])
            print(f"    {m:<14} {vals.min():.4f} | {q1:.4f} | {q2:.4f} | {q3:.4f} | {vals.max():.4f}")

    for metric_lbl, fn in [
        ('RMSRE',   lambda v, tv: float(np.sqrt(np.mean(((v - tv) / abs(tv)) ** 2)))),
        ('MdARE',   lambda v, tv: float(np.median(np.abs((v - tv) / abs(tv))))),
        ('P90-P10', lambda v, tv: float(np.percentile(v, 90) - np.percentile(v, 10))),
    ]:
        print(f"\n  [{metric_lbl} per param]")
        hdr2 = f"  {'param':<13} {'true':>{cw}}" + "".join(f"  {m:>{cw}}" for m in MODELS)
        print(hdr2)
        for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
            row = f"  {lbl:<13} {tv:>{cw}.4f}"
            for m in MODELS:
                vals = np.array([r[col] for r in records if r['model'] == m])
                row += f"  {fn(vals, tv):>{cw}.4f}" if len(vals) >= 2 else f"  {'—':>{cw}}"
            print(row)
        overall = f"  {'Overall':<13} {'':>{cw}}"
        for m in MODELS:
            per_p = []
            for col, tv in zip(P_COLS, tv_list):
                vals = np.array([r[col] for r in records if r['model'] == m])
                if len(vals) >= 2:
                    per_p.append(fn(vals, tv))
            overall += f"  {np.mean(per_p):>{cw}.4f}" if per_p else f"  {'—':>{cw}}"
        print(overall)

    print(f"\n  [Overall RMSRE per iter — last 5]")
    for m in MODELS:
        rmsres = [r['rmsre'] for r in records if r['model'] == m][-5:]
        print(f"    {m:<14} {[f'{v:.4f}' for v in rmsres]}")


def print_final_summary(df_final, true_dict, num_iters):
    tv_list = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]
    cw = 15

    def rmsre(sub, col, tv):
        return float(np.sqrt(np.mean(((sub[col].values - tv) / abs(tv)) ** 2)))
    def mdare(sub, col, tv):
        return float(np.median(np.abs((sub[col].values - tv) / abs(tv))))
    def p90p10(sub, col, tv):
        return float(np.percentile(sub[col].values, 90) - np.percentile(sub[col].values, 10))

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — Vecchia vs DW_raw on complete grid  ({num_iters} iters)")
    print(f"{'='*70}")

    for metric_lbl, fn in [('RMSRE', rmsre), ('MdARE', mdare), ('P90-P10', p90p10)]:
        print(f"\n  [{metric_lbl} per parameter]")
        hdr = f"  {'param':<14} {'true':>10}" + "".join(f"  {m:>{cw}}" for m in MODELS)
        print(hdr)
        print(f"  {'-'*(14 + 12 + (cw+2)*len(MODELS))}")
        for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
            row = f"  {lbl:<14} {tv:>10.4f}"
            for m in MODELS:
                sub = df_final[df_final['model'] == m]
                row += f"  {fn(sub, col, tv):>{cw}.4f}"
            print(row)
        print(f"  {'-'*(14 + 12 + (cw+2)*len(MODELS))}")
        overall = f"  {'Overall':<14} {'':>10}"
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            per_p = [fn(sub, col, tv) for col, tv in zip(P_COLS, tv_list)]
            overall += f"  {np.mean(per_p):>{cw}.4f}"
        print(overall)

    print(f"\n  [Mean estimate (SD)]")
    hdr = f"  {'param':<14} {'true':>10}" + "".join(f"  {'mean(SD)':>{cw}}" for _ in MODELS)
    print(hdr)
    print(f"  {'-'*(14 + 12 + (cw+2)*len(MODELS))}")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        row = f"  {lbl:<14} {tv:>10.4f}"
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            me, sd = sub[col].mean(), sub[col].std()
            row += f"  {me:>6.3f}({sd:.3f})"
        print(row)

    print(f"\n  [5-Number Summary  (Min | Q1 | Median | Q3 | Max)]")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        print(f"\n  {lbl}  (true = {tv:.4f})")
        print(f"    {'Model':<14}  {'Min':>8}  {'Q1':>8}  {'Median':>8}  {'Q3':>8}  {'Max':>8}")
        print(f"    {'-'*60}")
        for m in MODELS:
            vals = df_final[df_final['model'] == m][col].dropna().values
            q1, q2, q3 = np.percentile(vals, [25, 50, 75])
            print(f"    {m:<14}  {vals.min():>8.4f}  {q1:>8.4f}  {q2:>8.4f}  {q3:>8.4f}  {vals.max():>8.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100, help="Vecchia neighbors"),
    nheads: int = typer.Option(0, help="Vecchia head points per time step"),
    limit_a: int = typer.Option(20, help="Set A neighbors"),
    limit_b: int = typer.Option(20, help="Set B neighbors"),
    limit_c: int = typer.Option(20, help="Set C neighbors"),
    daily_stride: int = typer.Option(8, help="Daily stride for Set C"),
    num_iters: int = typer.Option(10, help="Simulation iterations"),
    lat_range: str = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range: str = typer.Option("121,131", help="lon_min,lon_max"),
    init_noise: float = typer.Option(0.7, help="Uniform noise half-width in log space for init"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:

    rng = np.random.default_rng(seed)
    lat_r = [float(x) for x in lat_range.split(',')]
    lon_r = [float(x) for x in lon_range.split(',')]

    print(f"Device  : {DEVICE}")
    print(f"Region  : lat {lat_r}, lon {lon_r}")
    print(f"Models  : {MODELS}")
    print(f"Vecchia : mm_cond={mm_cond_number}, nheads={nheads}, A={limit_a}, B={limit_b}, C={limit_c}")
    print(f"DW_raw  : identity filter, per-slice demean, DC excluded")
    print(f"Data    : direct grid sampling — NO step3, NO NaN values")
    print(f"Init noise: ±{init_noise} log-space  (×{np.exp(init_noise):.2f} original scale)")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_vdw_grid_{date_tag}.csv"
    csv_summary = f"sim_vdw_grid_summary_{date_tag}.csv"

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
    phi2     = 1.0 / true_dict['range_lon']
    phi1     = true_dict['sigmasq'] * phi2
    phi3     = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4     = (true_dict['range_lon'] / true_dict['range_time']) ** 2
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
    # IMPORTANT: lats_grid must be INCREASING (min→max) to match the FFT field
    # convention where field[i] corresponds to lx = i*dlat (increasing position).
    # Using decreasing lat would flip the sign of u_lat = lat_i - lat_j relative
    # to the FFT lag, causing advec_lat estimates to have the wrong sign.
    print("\n[Setup 1/3] Building target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon  = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords   = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    dlat   = DELTA_LAT_BASE
    dlon   = DELTA_LON_BASE
    print(f"  Grid: {len(lats_grid)} lat × {len(lons_grid)} lon = {N_grid} cells  "
          f"(δlat={dlat:.4f}°, δlon={dlon:.4f}°)")
    print(f"  ALL {N_grid} cells observed at every time step — no NaN, no missing")

    # ── Vecchia ordering (computed once, shared across iterations) ────────────
    print("[Setup 2/3] Computing maxmin ordering for Vecchia...")
    coords_np = grid_coords.cpu().numpy()
    ord_mm    = _orderings.maxmin_cpp(coords_np)
    nns_grid  = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    print(f"  Ordering: N_grid={N_grid}, mm_cond_number={mm_cond_number}")

    # ── Verify dataset structure ──────────────────────────────────────────────
    print("[Setup 3/3] Verifying dataset structure with sample field...")
    _field0 = generate_field_on_grid(lats_grid, lons_grid, 8, true_params, dlat, dlon)
    _gmap0, _gagg0 = assemble_grid_dataset(_field0, grid_coords, true_params)
    _first = list(_gmap0.values())[0]
    n_valid = (~torch.isnan(_first[:, 2])).sum().item()
    print(f"  Sample: {n_valid}/{N_grid} valid (should be {N_grid})")
    del _field0, _gmap0, _gagg0

    # ── Shared settings ───────────────────────────────────────────────────────
    LBFGS_LR    = 1.0
    LBFGS_STEPS = 5
    LBFGS_HIST  = 10
    LBFGS_EVAL  = 20
    DWL_STEPS   = 5
    LC, NC, VC, TC = 0, 1, 2, 3
    dwl = dw_raw_module.debiased_whittle_likelihood()

    records = []
    skipped = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iteration {it+1}/{num_iters}  (skipped so far: {skipped})")
        print(f"{'='*60}")

        initial_vals = make_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Init: sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  "
              f"nugget={init_orig['nugget']:.3f}")

        try:
            # Generate field directly on target grid
            field = generate_field_on_grid(lats_grid, lons_grid, 8, true_params, dlat, dlon)
            grid_map, grid_agg = assemble_grid_dataset(field, grid_coords, true_params)
            del field

            # Apply maxmin ordering to Vecchia input
            grid_map_ord = {k: v[ord_mm] for k, v in grid_map.items()}

            results = {}

            # ──────────────────────────────────────────────────────────────────
            # Model 1: Vecchia  (irr API, complete grid — irr ≡ reg here)
            # ──────────────────────────────────────────────────────────────────
            print("--- Model 1: Vecchia ---")
            p_vecc = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                      for val in initial_vals]
            model_vecc = kernels_vecchia.fit_vecchia_lbfgs(
                smooth=v, input_map=grid_map_ord,
                nns_map=nns_grid, mm_cond_number=mm_cond_number, nheads=nheads,
                limit_A=limit_a, limit_B=limit_b, limit_C=limit_c,
                daily_stride=daily_stride
            )
            model_vecc.precompute_conditioning_sets()
            opt_vecc = model_vecc.set_optimizer(
                p_vecc, lr=LBFGS_LR, max_iter=LBFGS_EVAL, history_size=LBFGS_HIST)
            t0 = time.time()
            out_vecc, _ = model_vecc.fit_vecc_lbfgs(
                p_vecc, opt_vecc, max_steps=LBFGS_STEPS, grad_tol=1e-5)
            t_vecc = time.time() - t0
            rmsre_vecc, est_vecc = calculate_rmsre(out_vecc, true_dict)
            print(f"  RMSRE = {rmsre_vecc:.4f}  ({t_vecc:.1f}s)")
            results['Vecchia'] = (est_vecc, rmsre_vecc, t_vecc)

            # ──────────────────────────────────────────────────────────────────
            # Model 2: DW_raw  (identity filter, per-slice demean, DC excluded)
            # ──────────────────────────────────────────────────────────────────
            print("--- Model 2: DW_raw ---")
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
            cur_df      = db.generate_spatially_filtered_days(
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
            _, _, _, loss_dw, _ = dwl.run_lbfgs_tapered(
                params_list=p_dw, optimizer=opt_dw, I_sample=I_samp,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
                max_steps=DWL_STEPS, device=DEVICE)
            t_dw = time.time() - t0
            out_dw = [p.item() for p in p_dw]
            rmsre_dw, est_dw = calculate_rmsre(out_dw, true_dict)
            print(f"  RMSRE = {rmsre_dw:.4f}  ({t_dw:.1f}s)  grid: {n1}×{n2}, p={p_time}")
            results['DW_raw'] = (est_dw, rmsre_dw, t_dw)

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ────────────────────────────────────────────────────────────
        for model_name in MODELS:
            est_d, rmsre_val, elapsed = results[model_name]
            records.append({
                'iter':          it + 1,
                'model':         model_name,
                'rmsre':         round(rmsre_val,          6),
                'time_s':        round(elapsed,            2),
                'sigmasq_est':   round(est_d['sigmasq'],   6),
                'range_lat_est': round(est_d['range_lat'], 6),
                'range_lon_est': round(est_d['range_lon'], 6),
                'range_t_est':   round(est_d['range_time'],6),
                'advec_lat_est': round(est_d['advec_lat'], 6),
                'advec_lon_est': round(est_d['advec_lon'], 6),
                'nugget_est':    round(est_d['nugget'],    6),
                'init_sigmasq':  round(init_orig['sigmasq'],  4),
                'init_rlon':     round(init_orig['range_lon'],4),
            })

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)
        print_running_summary(records, true_dict, it)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Simulation complete: {num_iters} iters, {skipped} skipped")
    print(f"{'='*60}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_raw, index=False)
    print_final_summary(df_final, true_dict, num_iters)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    tv_list = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]
    summary_rows = []
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        row = {'parameter': lbl, 'true': tv}
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            row[f'{m}_rmsre'] = round(
                float(np.sqrt(np.mean(((sub[col].values - tv) / abs(tv)) ** 2))), 6)
            row[f'{m}_mdare'] = round(
                float(np.median(np.abs((sub[col].values - tv) / abs(tv)))), 6)
            row[f'{m}_p90p10'] = round(
                float(np.percentile(sub[col].values, 90) - np.percentile(sub[col].values, 10)), 6)
            row[f'{m}_mean'] = round(float(sub[col].mean()), 6)
            row[f'{m}_sd']   = round(float(sub[col].std()),  6)
        summary_rows.append(row)
    overall_row = {'parameter': 'Overall', 'true': float('nan')}
    for m in MODELS:
        sub = df_final[df_final['model'] == m]
        rmsres  = [float(np.sqrt(np.mean(((sub[c].values - tv) / abs(tv))**2)))
                   for c, tv in zip(P_COLS, tv_list)]
        mdares  = [float(np.median(np.abs((sub[c].values - tv) / abs(tv))))
                   for c, tv in zip(P_COLS, tv_list)]
        p90p10s = [float(np.percentile(sub[c].values, 90) - np.percentile(sub[c].values, 10))
                   for c, tv in zip(P_COLS, tv_list)]
        overall_row[f'{m}_rmsre']  = round(np.mean(rmsres),  6)
        overall_row[f'{m}_mdare']  = round(np.mean(mdares),  6)
        overall_row[f'{m}_p90p10'] = round(np.mean(p90p10s), 6)
        overall_row[f'{m}_mean']   = float('nan')
        overall_row[f'{m}_sd']     = float('nan')
    summary_rows.append(overall_row)
    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\n  Saved: {csv_raw}  (all iters, raw)")
    print(f"  Saved: {csv_summary}  (per-parameter summary)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        plot_dir = output_path / "plots" / "vdw"
        plot_dir.mkdir(parents=True, exist_ok=True)

        MODEL_COLORS = {'Vecchia': '#2196F3', 'DW_raw': '#E91E63'}

        # 1. Per-parameter: 2 panels (one per model), hist + KDE
        n_params = len(P_LABELS)
        for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            fig.suptitle(f"Distribution of estimates: {lbl}  (true = {tv:.4f})",
                         fontsize=12, fontweight='bold')
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
                           label=f'med={np.median(sub):.3f}')
                q1, q3 = np.percentile(sub, [25, 75])
                ax.axvspan(q1, q3, alpha=0.10, color=c)
                ax.set_title(m, fontsize=11)
                ax.set_xlabel(lbl, fontsize=9)
                ax.legend(fontsize=8, framealpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(plot_dir / f"vdw_{col}_dist.png", dpi=130, bbox_inches='tight')
            plt.close()

        # 2. Overview: all 7 params, 2 models overlaid per panel
        n_cols_p = 2
        n_rows_p = (n_params + 1) // n_cols_p
        fig, axes = plt.subplots(n_rows_p, n_cols_p, figsize=(13, 4 * n_rows_p))
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
        fig.suptitle(f"Vecchia vs DW_raw — Parameter Estimate Distributions  ({num_iters} iters)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "vdw_all_params_overview.png", dpi=130, bbox_inches='tight')
        plt.close()

        # 3. Boxplot comparison
        fig, axes = plt.subplots(n_rows_p, n_cols_p, figsize=(13, 4 * n_rows_p))
        axes = axes.flatten()
        for i, (lbl, col, tv) in enumerate(zip(P_LABELS, P_COLS, tv_list)):
            ax = axes[i]
            data  = [df_final[df_final['model'] == m][col].dropna().values for m in MODELS]
            bp    = ax.boxplot(data, labels=MODELS, patch_artist=True,
                               medianprops={'color': 'black', 'lw': 2})
            for patch, m in zip(bp['boxes'], MODELS):
                patch.set_facecolor(MODEL_COLORS[m])
                patch.set_alpha(0.5)
            ax.axhline(tv, color='black', lw=1.5, ls='--', label=f'true={tv:.3f}')
            ax.set_title(f"{lbl}  (true={tv:.3f})", fontsize=10)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Vecchia vs DW_raw — Boxplot  ({num_iters} iters)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "vdw_all_params_boxplot.png", dpi=130, bbox_inches='tight')
        plt.close()

        print(f"\n  Plots saved → {plot_dir}/")
        print(f"  - vdw_{{param}}_dist.png  × {n_params}  (per-param, 2-panel hist+KDE)")
        print(f"  - vdw_all_params_overview.png  (KDE overlay, all params)")
        print(f"  - vdw_all_params_boxplot.png   (boxplot comparison)")

    except ImportError as ie:
        print(f"\n  [Plot skipped — missing library: {ie}]")
    except Exception as pe:
        import traceback
        print(f"\n  [Plot error: {pe}]")
        traceback.print_exc()


if __name__ == "__main__":
    app()
