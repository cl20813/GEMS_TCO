"""
sim_dw_raw_complex.py

Simulation study: Debiased Whittle — COMPLEX version (no spatial differencing).
Purpose: verify that complex Whittle resolves the advection sign ambiguity.

Pipeline (single model, no gridification):
  FFT field generated directly on the target regular grid
  → complex DW likelihood → parameter estimation

Research question
-----------------
Does the complex Whittle recover the correct sign of advec_lon / advec_lat?
  Real DW:    L(advec) = L(-advec)  (cos symmetry → ~50% sign recovery)
  Complex DW: L(advec) ≠ L(-advec)  (Im retained → ~100% sign recovery)

Mathematical model:
  Z(i,j,t) = X(i,j,t)  (identity filter, H(ω) = 1)
  E[I(ω)] = F(ω; α)  COMPLEX Hermitian  (no .real taken)
  F(ω; -α) = F(ω; α)*  ≠  F(ω; α)   → L(-α) ≠ L(α)

DC handling: per-slice spatial demean + ω=(0,0) excluded from likelihood.

Usage:
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation
  python sim_dw_raw_complex.py --num-iters 1
  python sim_dw_raw_complex.py --num-iters 300
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

from GEMS_TCO import debiased_whittle_raw as debiased_whittle
from GEMS_TCO import configuration as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT = 0.044
DELTA_LON = 0.063

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


# ── Field generation directly on target grid ───────────────────────────────────

def generate_field_on_grid(n_lat, n_lon, t_steps, params, dlat, dlon):
    """
    FFT circulant embedding realization directly on the target regular grid.
    No high-res upsampling needed — field is generated at grid resolution.
    Shape: (n_lat, n_lon, t_steps).
    """
    CPU = torch.device("cpu")
    F32 = torch.float32
    Px, Py, Pt = 2 * n_lat, 2 * n_lon, 2 * t_steps

    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt

    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid(Lx, Ly, Lt, params_cpu)
    S = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:n_lat, :n_lon, :t_steps]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Dataset assembly from grid field ──────────────────────────────────────────

def assemble_grid_dataset(field, lats_grid, lons_grid, true_params, t_steps, t_offset=21.0):
    """
    Build tensor_list (one per time step) from field sampled on the regular grid.
    Each tensor: (N_grid, 4) with columns [lat, lon, value, time].
    Adds nugget noise. Returns tensor_list and reg_map dict.
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    n_lat, n_lon = len(lats_grid), len(lons_grid)
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    coords = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)  # (N_grid, 2)
    N_grid = coords.shape[0]

    tensor_list = []
    reg_map = {}

    for t_idx in range(t_steps):
        t_val    = float(t_offset + t_idx)
        gp_vals  = field[:, :, t_idx].flatten()
        sim_vals = gp_vals + torch.randn(N_grid, device=DEVICE, dtype=DTYPE) * nugget_std

        rows = torch.zeros((N_grid, 4), device=DEVICE, dtype=DTYPE)
        rows[:, 0] = coords[:, 0]
        rows[:, 1] = coords[:, 1]
        rows[:, 2] = sim_vals
        rows[:, 3] = t_val

        key = f't{t_idx}'
        reg_map[key] = rows
        tensor_list.append(rows)

    return tensor_list, reg_map


# ── Parameter back-mapping ─────────────────────────────────────────────────────

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
    est      = backmap_params(out_params)
    est_arr  = np.array([est['sigmasq'],       est['range_lat'],  est['range_lon'],
                         est['range_time'],    est['advec_lat'],  est['advec_lon'],
                         est['nugget']])
    true_arr = np.array([true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                         true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                         true_dict['nugget']])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


# ── Summary printers ───────────────────────────────────────────────────────────

def print_running_summary(records, true_dict, it):
    n_done  = len(records)
    tv_list = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]

    print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
    print(f"  {'param':<13} {'true':>12} {'mean_est':>12} {'bias':>10}")
    print(f"  {'-'*50}")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        vals = [r[col] for r in records]
        print(f"  {lbl:<13} {tv:>12.4f} {np.mean(vals):>12.4f} {np.mean(vals)-tv:>10.4f}")
    print(f"  {'-'*50}")

    # Sign recovery — the key metric
    lat_rate = np.mean([np.sign(r['advec_lat_est']) == np.sign(true_dict['advec_lat'])
                        for r in records])
    lon_rate = np.mean([np.sign(r['advec_lon_est']) == np.sign(true_dict['advec_lon'])
                        for r in records])
    print(f"\n  [Sign Recovery Rate — key metric for complex DW]")
    print(f"  advec_lat sign correct: {lat_rate:.3f}  (expect ~1.0 for complex, ~0.5 for real)")
    print(f"  advec_lon sign correct: {lon_rate:.3f}  (expect ~1.0 for complex, ~0.5 for real)")


def print_final_summary(df_final, true_dict, num_iters):
    tv_list = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
               true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
               true_dict['nugget']]

    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY — {num_iters} iterations")
    print(f"{'='*65}")
    print(f"  {'Parameter':<14} {'True':>10} {'Mean':>10} {'Bias':>10} {'RMSRE':>10}")
    print(f"  {'-'*60}")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        v = df_final[col].values
        rmsre = float(np.sqrt(np.mean(((v - tv) / abs(tv))**2)))
        print(f"  {lbl:<14} {tv:>10.4f} {v.mean():>10.4f} {v.mean()-tv:>10.4f} {rmsre:>10.4f}")

    print(f"\n{'='*65}")
    print(f"  SIGN RECOVERY RATE")
    print(f"{'='*65}")
    lat_rate = df_final['advec_lat_sign_ok'].mean()
    lon_rate = df_final['advec_lon_sign_ok'].mean()
    print(f"  advec_lat: {lat_rate:.4f}  (true={true_dict['advec_lat']:+.4f})")
    print(f"  advec_lon: {lon_rate:.4f}  (true={true_dict['advec_lon']:+.4f})")

    print(f"\n{'='*65}")
    print(f"  5-NUMBER SUMMARY")
    print(f"{'='*65}")
    for lbl, col, tv in zip(P_LABELS, P_COLS, tv_list):
        v = df_final[col].dropna().values
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        print(f"  {lbl:<14} (true={tv:.4f})  "
              f"{v.min():.4f} | {q1:.4f} | {q2:.4f} | {q3:.4f} | {v.max():.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    num_iters:   int   = typer.Option(300,      help="Simulation iterations"),
    t_steps:     int   = typer.Option(8,        help="Time steps per realization"),
    lat_range:   str   = typer.Option("-3,2",   help="lat_min,lat_max"),
    lon_range:   str   = typer.Option("121,131",help="lon_min,lon_max"),
    init_noise:  float = typer.Option(0.7,      help="Uniform noise half-width in log space"),
    seed:        int   = typer.Option(42,       help="Random seed"),
) -> None:

    rng   = np.random.default_rng(seed)
    lat_r = [float(x) for x in lat_range.split(',')]
    lon_r = [float(x) for x in lon_range.split(',')]

    print(f"Device     : {DEVICE}")
    print(f"Likelihood : COMPLEX Whittle — F(ω) complex Hermitian, sign ambiguity resolved")
    print(f"Pipeline   : FFT field → target regular grid (no high-res, no step3)")
    print(f"Purpose    : advec sign recovery rate test")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"t_steps    : {t_steps}")
    print(f"Iterations : {num_iters}")
    print(f"Init noise : ±{init_noise} log-space")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_dw_complex_{date_tag}.csv"
    csv_summary = f"sim_dw_complex_summary_{date_tag}.csv"

    # ── True parameters ───────────────────────────────────────────────────────
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

    def make_random_init(rng):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        # advec_lat, advec_lon: use fixed ±0.3 range so ~50% of inits have wrong sign.
        # scale based on 0.3 regardless of true magnitude — ensures the sign test
        # is informative (init frequently starts on the wrong side of 0).
        for i in [4, 5]:
            noisy[i] = true_log[i] + rng.uniform(-0.3, 0.3)
        return noisy

    # ── Build target regular grid ─────────────────────────────────────────────
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    n_lat     = len(lats_grid)
    n_lon     = len(lons_grid)
    print(f"\nTarget grid: {n_lat} lat × {n_lon} lon = {n_lat*n_lon} cells")

    # ── DW setup ──────────────────────────────────────────────────────────────
    DWL_STEPS = 5
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3
    dwl = debiased_whittle.debiased_whittle_likelihood()

    records = []
    skipped = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iter {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*60}")

        initial_vals = make_random_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  init sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  "
              f"advec_lat={init_orig['advec_lat']:+.4f}  "
              f"advec_lon={init_orig['advec_lon']:+.4f}  "
              f"nugget={init_orig['nugget']:.3f}")

        try:
            # ── Generate field directly on target grid ────────────────────────
            field = generate_field_on_grid(n_lat, n_lon, t_steps, true_params,
                                           DELTA_LAT, DELTA_LON)

            # ── Build time-slice tensor list ──────────────────────────────────
            tensor_list, _ = assemble_grid_dataset(
                field, lats_grid, lons_grid, true_params, t_steps)
            del field

            # ── Demean each time slice (DC removal) ───────────────────────────
            # Removes spatial mean per slice to zero the DC frequency (ω=0).
            # The ω=(0,0) exclusion in the likelihood is a second safeguard.
            for i in range(len(tensor_list)):
                vals = tensor_list[i][:, VAL_COL]
                tensor_list[i] = tensor_list[i].clone()
                tensor_list[i][:, VAL_COL] = vals - vals.mean()

            # ── DFT & periodogram ─────────────────────────────────────────────
            J_vec, n1, n2, p_time, taper, obs_masks = dwl.generate_Jvector_tapered_mv(
                tensor_list, dwl.cgn_hamming, LAT_COL, LON_COL, VAL_COL, DEVICE)
            I_samp = dwl.calculate_sample_periodogram_vectorized(J_vec)
            t_auto = dwl.calculate_taper_autocorrelation_multivariate(
                taper, obs_masks, n1, n2, DEVICE)
            del obs_masks

            # ── Optimize ──────────────────────────────────────────────────────
            p_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                    for val in initial_vals]
            opt_dw = torch.optim.LBFGS(
                p_dw, lr=1.0, max_iter=20, max_eval=100,
                history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5)

            t0 = time.time()
            _, _, _, loss_dw, _ = dwl.run_lbfgs_tapered(
                params_list=p_dw, optimizer=opt_dw, I_sample=I_samp,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
                max_steps=DWL_STEPS, device=DEVICE)
            elapsed = time.time() - t0

            out_dw = [p.item() for p in p_dw]
            rmsre_dw, est_dw = calculate_rmsre(out_dw, true_dict)

            lat_sign_ok = int(np.sign(est_dw['advec_lat']) == np.sign(true_dict['advec_lat']))
            lon_sign_ok = int(np.sign(est_dw['advec_lon']) == np.sign(true_dict['advec_lon']))
            print(f"  RMSRE={rmsre_dw:.4f}  ({elapsed:.1f}s)  grid:{n1}×{n2}  "
                  f"advec_lat={est_dw['advec_lat']:+.4f}({'OK' if lat_sign_ok else 'WRONG'})  "
                  f"advec_lon={est_dw['advec_lon']:+.4f}({'OK' if lon_sign_ok else 'WRONG'})")

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        records.append({
            'iter':              it + 1,
            'rmsre':             round(rmsre_dw,  6),
            'time_s':            round(elapsed,    2),
            'loss_dw':           round(float(loss_dw), 6),
            'sigmasq_est':       round(est_dw['sigmasq'],    6),
            'range_lat_est':     round(est_dw['range_lat'],  6),
            'range_lon_est':     round(est_dw['range_lon'],  6),
            'range_t_est':       round(est_dw['range_time'], 6),
            'advec_lat_est':     round(est_dw['advec_lat'],  6),
            'advec_lon_est':     round(est_dw['advec_lon'],  6),
            'nugget_est':        round(est_dw['nugget'],     6),
            'advec_lat_sign_ok': lat_sign_ok,
            'advec_lon_sign_ok': lon_sign_ok,
            'init_sigmasq':      round(init_orig['sigmasq'],   4),
            'init_range_lon':    round(init_orig['range_lon'], 4),
            'init_advec_lat':    round(init_orig['advec_lat'], 4),
            'init_advec_lon':    round(init_orig['advec_lon'], 4),
            'n1': n1, 'n2': n2,
        })

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)
        print_running_summary(records, true_dict, it)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE: {len(records)} iters completed, {skipped} skipped")
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
        v = df_final[col].values
        summary_rows.append({
            'param':  lbl, 'true': tv,
            'mean':   round(float(np.mean(v)),   6),
            'median': round(float(np.median(v)), 6),
            'bias':   round(float(np.mean(v)-tv), 6),
            'std':    round(float(np.std(v)),    6),
            'RMSRE':  round(float(np.sqrt(np.mean(((v-tv)/abs(tv))**2))), 6),
            'MdARE':  round(float(np.median(np.abs((v-tv)/abs(tv)))), 6),
            'P10':    round(float(np.percentile(v, 10)), 6),
            'P90':    round(float(np.percentile(v, 90)), 6),
        })
    summary_rows.append({
        'param': 'advec_lat_sign_rate', 'true': 1.0,
        'mean':  round(float(df_final['advec_lat_sign_ok'].mean()), 4),
        'median': float('nan'), 'bias': float('nan'), 'std': float('nan'),
        'RMSRE': float('nan'), 'MdARE': float('nan'),
        'P10': float('nan'), 'P90': float('nan'),
    })
    summary_rows.append({
        'param': 'advec_lon_sign_rate', 'true': 1.0,
        'mean':  round(float(df_final['advec_lon_sign_ok'].mean()), 4),
        'median': float('nan'), 'bias': float('nan'), 'std': float('nan'),
        'RMSRE': float('nan'), 'MdARE': float('nan'),
        'P10': float('nan'), 'P90': float('nan'),
    })

    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\nSaved:\n  {output_path / csv_raw}\n  {output_path / csv_summary}")


if __name__ == "__main__":
    app()
