"""
sim_gradient_test_dw_gridding_033126.py  —  LOCAL VERSION

Disentangle gridding (high-res → location matching) vs missing data.

Three variants on the SAME DGP realization:
  DW_grid         : Field directly on target grid, no missing      (baseline)
  DW_matched_full : High-res field → random irregular obs → match ALL target cells
                    to nearest obs regardless of distance (no missing)
  DW_matched_miss : Same, but cells with nearest obs > threshold → NaN (original approach)

Comparisons:
  DW_grid vs DW_matched_full  →  effect of location-snapping / gridding
  DW_matched_full vs DW_matched_miss  →  effect of threshold-based missing

Run:
  conda activate faiss_env
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
   
"""

import sys
import gc
import time
from datetime import datetime
import numpy as np
import torch
import torch.fft
import pandas as pd
import scipy.spatial
import typer
from pathlib import Path

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1")

from GEMS_TCO import debiased_whittle

app    = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

P_NAMES = ["SigmaSq", "RangeLat", "RangeLon", "RangeTime", "AdvecLat", "AdvecLon", "Nugget"]


# ── DGP covariance: Matérn ν=0.5 ─────────────────────────────────────────────

def get_covariance_on_grid_matern(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


# ── FFT circulant embedding (shared core) ────────────────────────────────────

def _fft_field(Nx, Ny, Nt, delta_lat, delta_lon, params):
    """Returns (Nx, Ny, Nt) float64 numpy array via circulant embedding."""
    Px, Py, Pt = 2 * Nx, 2 * Ny, 2 * Nt
    CPU, F32 = torch.device("cpu"), torch.float32

    lx = torch.arange(Px, dtype=F32) * delta_lat
    lx[Px // 2:] -= Px * delta_lat
    ly = torch.arange(Py, dtype=F32) * delta_lon
    ly[Py // 2:] -= Py * delta_lon
    lt = torch.arange(Pt, dtype=F32)
    lt[Pt // 2:] -= Pt

    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid_matern(Lx, Ly, Lt, params.cpu().float())
    S = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=torch.float64).numpy()


def generate_field_on_target_grid(lats_grid, lons_grid, t_steps, params):
    """Field on perfect target grid. Returns (N_lat, N_lon, T) numpy."""
    return _fft_field(len(lats_grid), len(lons_grid), t_steps,
                      DELTA_LAT_BASE, DELTA_LON_BASE, params)


def generate_field_on_hires_grid(lats_grid, lons_grid, lat_factor, lon_factor, t_steps, params):
    """Field on lat_factor × lon_factor finer grid.
    Returns (field_np, lats_h, lons_h) — all numpy."""
    lf, nf = int(lat_factor), int(lon_factor)
    delta_lat_h = DELTA_LAT_BASE / lf
    delta_lon_h = DELTA_LON_BASE / nf
    Nx = len(lats_grid) * lf
    Ny = len(lons_grid) * nf

    field = _fft_field(Nx, Ny, t_steps, delta_lat_h, delta_lon_h, params)

    lat_start = lats_grid[0].item()   # lats_grid is descending
    lon_start = lons_grid[0].item()   # lons_grid is ascending
    lats_h = lat_start - delta_lat_h * np.arange(Nx)
    lons_h = lon_start + delta_lon_h * np.arange(Ny)

    return field, lats_h, lons_h


# ── Irregular obs: generate and match to target grid ─────────────────────────

def build_irregular_obs(field_h, lats_h, lons_h, n_locs, nugget_std, rng):
    """
    Simulate n_locs random obs locations, snap to hires grid, add nugget noise.
    Returns obs_coords (n_locs, 2), obs_vals (n_locs, T).
    """
    lat_min, lat_max = lats_h.min(), lats_h.max()
    lon_min, lon_max = lons_h.min(), lons_h.max()

    obs_lats = rng.uniform(lat_min, lat_max, n_locs)
    obs_lons = rng.uniform(lon_min, lon_max, n_locs)
    obs_coords = np.stack([obs_lats, obs_lons], axis=1)   # (n_locs, 2)

    Nx, Ny, T = field_h.shape
    g_lat_h, g_lon_h = np.meshgrid(lats_h, lons_h, indexing='ij')
    hires_flat = np.stack([g_lat_h.ravel(), g_lon_h.ravel()], axis=1)  # (Nx*Ny, 2)

    tree = scipy.spatial.cKDTree(hires_flat)
    _, hires_idx = tree.query(obs_coords)
    h_r = hires_idx // Ny
    h_c = hires_idx % Ny

    obs_vals = field_h[h_r, h_c, :].copy()               # (n_locs, T)
    obs_vals += rng.standard_normal((n_locs, T)) * nugget_std

    return obs_coords, obs_vals


def match_obs_to_grid(obs_coords, obs_vals, target_coords_np, threshold):
    """
    Match obs locations to target grid cells.

    DW_matched_full : always assign nearest obs (no threshold)
    DW_matched_miss : NaN if nearest obs distance > threshold

    Returns gridded_full (N_grid, T), gridded_miss (N_grid, T), p_hat.
    """
    obs_tree = scipy.spatial.cKDTree(obs_coords)
    dists, nearest_idx = obs_tree.query(target_coords_np)

    gridded_full = obs_vals[nearest_idx, :]     # (N_grid, T)
    gridded_miss = gridded_full.copy()
    within = dists <= threshold
    gridded_miss[~within, :] = np.nan

    return gridded_full, gridded_miss, float(within.mean())


# ── Build DW hourly_map ───────────────────────────────────────────────────────

def build_dw_hourly_map_from_field(field_np, grid_coords, true_params, t_offset=21.0):
    """DW_grid: field (N_lat, N_lon, T) numpy, all cells observed, nugget added here."""
    nugget_std = float(torch.sqrt(torch.exp(true_params[6])).item())
    N_lat, N_lon, T = field_np.shape
    N_grid = N_lat * N_lon
    field_flat = field_np.reshape(N_grid, T)
    hourly_map = {}
    rng_np = np.random.default_rng()   # fresh for nugget
    for t_idx in range(T):
        key   = f't{t_idx}'
        t_val = float(t_offset + t_idx)
        dummy = torch.zeros(N_grid, 7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[:, t_idx - 1] = 1.0

        tensor_t = torch.full((N_grid, 11), float('nan'), device=DEVICE, dtype=DTYPE)
        tensor_t[:, 0] = grid_coords[:, 0]
        tensor_t[:, 1] = grid_coords[:, 1]
        tensor_t[:, 3] = t_val
        tensor_t[:, 4:] = dummy

        vals = field_flat[:, t_idx] + rng_np.standard_normal(N_grid) * nugget_std
        tensor_t[:, 2] = torch.tensor(vals, device=DEVICE, dtype=DTYPE)

        hourly_map[key] = tensor_t.detach()
    return hourly_map


def build_dw_hourly_map_from_array(gridded_vals_np, grid_coords, t_offset=21.0):
    """DW_matched: (N_grid, T) numpy array with NaN = unobserved (nugget already added)."""
    vals_t = torch.tensor(gridded_vals_np, device=DEVICE, dtype=DTYPE)
    N_grid, T = vals_t.shape
    hourly_map = {}
    for t_idx in range(T):
        key   = f't{t_idx}'
        t_val = float(t_offset + t_idx)
        dummy = torch.zeros(N_grid, 7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[:, t_idx - 1] = 1.0

        tensor_t = torch.full((N_grid, 11), float('nan'), device=DEVICE, dtype=DTYPE)
        tensor_t[:, 0] = grid_coords[:, 0]
        tensor_t[:, 1] = grid_coords[:, 1]
        tensor_t[:, 3] = t_val
        tensor_t[:, 4:] = dummy

        col  = vals_t[:, t_idx]
        mask = ~torch.isnan(col)
        if mask.any():
            tensor_t[mask, 2] = col[mask]

        hourly_map[key] = tensor_t.detach()
    return hourly_map


# ── Build DW precomputed objects ──────────────────────────────────────────────

def build_dw_precomputed(hourly_map, true_phys_list, lat_r, lon_r, dwl):
    db = debiased_whittle.debiased_whittle_preprocess(
        [None], [hourly_map], day_idx=0,
        params_list=true_phys_list, lat_range=lat_r, lon_range=lon_r,
    )
    cur_df       = db.generate_spatially_filtered_days(
        lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
    unique_times = torch.unique(cur_df[:, 3])
    time_slices  = [cur_df[cur_df[:, 3] == t] for t in unique_times]
    J_vec, n1, n2, p_time, taper_grid, obs_masks = dwl.generate_Jvector_tapered_mv(
        time_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE)
    I_obs  = dwl.calculate_sample_periodogram_vectorized(J_vec)
    t_auto = dwl.calculate_taper_autocorrelation_multivariate(taper_grid, obs_masks, n1, n2, DEVICE)
    del cur_df, time_slices, taper_grid, obs_masks
    return I_obs, n1, n2, p_time, t_auto


# ── Gradient helper ───────────────────────────────────────────────────────────

def compute_gradient_at(nll_fn, log_phi_val):
    p    = log_phi_val.detach().clone().requires_grad_(True)
    loss = nll_fn(p)
    grad = torch.autograd.grad(loss, p)[0].detach()
    return grad, loss.detach().item()


# ── Print helpers ─────────────────────────────────────────────────────────────

def _grad_table_row(tag, grad, loss):
    g = grad.cpu().numpy()
    print(f"  {tag:<24}  loss={loss:>12.4f}  ||∇||∞={np.abs(g).max():>10.6f}")
    print(f"    per-param: " +
          "  ".join(f"{n[:6]}={v:+.4f}" for n, v in zip(P_NAMES, g)))


def _print_running_summary(records):
    if not records:
        return
    n = len(records)
    print(f"\n  ── Running summary ({n} iters) ──")
    print(f"  {'variant':<22} {'mean ||∇||∞ @θ₀':>16} {'mean ||∇||∞ @pert':>18} {'ratio':>8}")
    print(f"  {'-'*67}")
    for m in ['DW_grid', 'DW_matched_full', 'DW_matched_miss']:
        g_true = np.mean([r[f'{m}_grad_inf_true'] for r in records])
        g_pert = np.mean([r[f'{m}_grad_inf_pert'] for r in records])
        ratio  = g_true / (g_pert + 1e-15)
        print(f"  {m:<22} {g_true:>16.6f} {g_pert:>18.6f} {ratio:>8.4f}")
    print(f"\n  Per-parameter mean |∇| at θ_true:")
    print(f"  {'param':<12} {'DW_grid':>12} {'DW_match_full':>14} {'DW_match_miss':>14}")
    print(f"  {'-'*54}")
    for p_name in P_NAMES:
        v1 = np.mean([abs(r[f'DW_grid_grad_{p_name}_true']) for r in records])
        v2 = np.mean([abs(r[f'DW_matched_full_grad_{p_name}_true']) for r in records])
        v3 = np.mean([abs(r[f'DW_matched_miss_grad_{p_name}_true']) for r in records])
        print(f"  {p_name:<12} {v1:>12.6f} {v2:>14.6f} {v3:>14.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    num_iters:      int   = typer.Option(10,        help="Simulation iterations"),
    p_obs:          float = typer.Option(0.7,       help="Obs density: n_locs = p_obs × N_grid"),
    lat_range:      str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:      str   = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor:     int   = typer.Option(5,         help="High-res upsampling factor (lat)"),
    lon_factor:     int   = typer.Option(5,         help="High-res upsampling factor (lon)"),
    threshold_frac: float = typer.Option(0.5,       help="Missing threshold = frac × max(Δlat,Δlon)"),
    pert_scale:     float = typer.Option(0.3,       help="Log-space perturbation half-width"),
    seed:           int   = typer.Option(42,        help="Random seed"),
) -> None:

    rng       = np.random.default_rng(seed)
    lat_r     = [float(x) for x in lat_range.split(',')]
    lon_r     = [float(x) for x in lon_range.split(',')]
    threshold = threshold_frac * max(DELTA_LAT_BASE, DELTA_LON_BASE)

    print(f"Device           : {DEVICE}")
    print(f"DGP              : Matérn ν=0.5")
    print(f"DW_grid          : perfect target grid, no missing")
    print(f"DW_matched_full  : high-res ×{lat_factor}/×{lon_factor} → match all cells (no missing)")
    print(f"DW_matched_miss  : same + distance threshold = {threshold:.4f}°")
    print(f"p_obs / n_locs   : {p_obs} / (p_obs × N_grid)")
    print(f"Iters            : {num_iters}")

    output_path = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/gradient_test")
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%m%d%y")
    csv_out  = output_path / f"gradient_test_dw_gridding_{date_tag}.csv"

    # ── True DGP parameters ───────────────────────────────────────────────────
    true_dict = {
        'sigmasq':    10.0,
        'range_lat':   0.2,
        'range_lon':   0.25,
        'range_time':  1.5,
        'advec_lat':   0.02,
        'advec_lon':  -0.17,
        'nugget':      0.25,
    }
    phi2 = 1.0 / true_dict['range_lon']
    phi1 = true_dict['sigmasq'] * phi2
    phi3 = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4 = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params    = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)
    true_phys_list = [true_dict[k] for k in
                      ['sigmasq', 'range_lat', 'range_lon', 'range_time',
                       'advec_lat', 'advec_lon', 'nugget']]
    nugget_std_np = float(np.sqrt(true_dict['nugget']))

    print(f"True params      : {true_dict}")

    # ── Target grid ───────────────────────────────────────────────────────────
    lats_grid = torch.round(
        torch.arange(min(lat_r), max(lat_r) + 1e-4,  DELTA_LAT_BASE,  device=DEVICE, dtype=DTYPE)
        * 10000) / 10000
    lons_grid = torch.round(
        torch.arange(lon_r[0], lon_r[1] + 1e-4, DELTA_LON_BASE, device=DEVICE, dtype=DTYPE)
        * 10000) / 10000
    g_lat, g_lon     = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords      = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    target_coords_np = grid_coords.cpu().numpy()
    N_lat, N_lon     = len(lats_grid), len(lons_grid)
    N_grid           = N_lat * N_lon
    n_locs           = max(1, int(p_obs * N_grid))

    print(f"Target grid      : {N_lat} lat × {N_lon} lon = {N_grid} cells")
    print(f"n_locs           : {n_locs}  ({p_obs:.0%} × {N_grid})")

    dwl     = debiased_whittle.debiased_whittle_likelihood()
    records = []

    # ── Simulation loop ───────────────────────────────────────────────────────
    for it in range(num_iters):
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"  Iteration {it+1}/{num_iters}")
        print(f"{'='*70}")

        pert_log = true_params.clone()
        for i in [0, 1, 2, 3, 6]:
            pert_log[i] = pert_log[i] + float(rng.uniform(-pert_scale, pert_scale))

        try:
            # ── [1/3] DW_grid — field on perfect target grid ───────────────────
            print("  [1/3] DW_grid ...")
            field_np = generate_field_on_target_grid(lats_grid, lons_grid, 8, true_params)
            map_grid = build_dw_hourly_map_from_field(field_np, grid_coords, true_params)
            del field_np
            I_grid, n1, n2, p_time, t_auto_grid = build_dw_precomputed(
                map_grid, true_phys_list, lat_r, lon_r, dwl)
            del map_grid

            def nll_grid(p):
                loss = dwl.whittle_likelihood_loss_tapered(
                    p, I_grid, n1, n2, p_time, t_auto_grid, DELTA_LAT_BASE, DELTA_LON_BASE)
                return loss[0] if isinstance(loss, tuple) else loss

            grad_grid_true, loss_grid_true = compute_gradient_at(nll_grid, true_params)
            grad_grid_pert, loss_grid_pert = compute_gradient_at(nll_grid, pert_log)
            _grad_table_row("DW_grid@θ₀",   grad_grid_true, loss_grid_true)
            _grad_table_row("DW_grid@pert", grad_grid_pert, loss_grid_pert)
            del I_grid, t_auto_grid; gc.collect()

            # ── [2+3/3] DW_matched — high-res field → irregular obs ────────────
            print(f"  [2+3/3] DW_matched (×{lat_factor}/×{lon_factor}) ...")
            field_h, lats_h, lons_h = generate_field_on_hires_grid(
                lats_grid, lons_grid, lat_factor, lon_factor, 8, true_params)
            obs_coords, obs_vals = build_irregular_obs(
                field_h, lats_h, lons_h, n_locs, nugget_std_np, rng)
            del field_h; gc.collect()

            gridded_full, gridded_miss, p_hat = match_obs_to_grid(
                obs_coords, obs_vals, target_coords_np, threshold)
            print(f"    coverage (DW_matched_miss) = {p_hat:.3f}")

            # DW_matched_full — all cells filled, no missing
            print("    DW_matched_full ...")
            map_mfull = build_dw_hourly_map_from_array(gridded_full, grid_coords)
            I_mfull, n1, n2, p_time, t_auto_mfull = build_dw_precomputed(
                map_mfull, true_phys_list, lat_r, lon_r, dwl)
            del map_mfull

            def nll_mfull(p):
                loss = dwl.whittle_likelihood_loss_tapered(
                    p, I_mfull, n1, n2, p_time, t_auto_mfull, DELTA_LAT_BASE, DELTA_LON_BASE)
                return loss[0] if isinstance(loss, tuple) else loss

            grad_mfull_true, loss_mfull_true = compute_gradient_at(nll_mfull, true_params)
            grad_mfull_pert, loss_mfull_pert = compute_gradient_at(nll_mfull, pert_log)
            _grad_table_row("DW_matched_full@θ₀",   grad_mfull_true, loss_mfull_true)
            _grad_table_row("DW_matched_full@pert", grad_mfull_pert, loss_mfull_pert)
            del I_mfull, t_auto_mfull; gc.collect()

            # DW_matched_miss — threshold-based missing
            print("    DW_matched_miss ...")
            map_mmiss = build_dw_hourly_map_from_array(gridded_miss, grid_coords)
            I_mmiss, n1, n2, p_time, t_auto_mmiss = build_dw_precomputed(
                map_mmiss, true_phys_list, lat_r, lon_r, dwl)
            del map_mmiss

            def nll_mmiss(p):
                loss = dwl.whittle_likelihood_loss_tapered(
                    p, I_mmiss, n1, n2, p_time, t_auto_mmiss, DELTA_LAT_BASE, DELTA_LON_BASE)
                return loss[0] if isinstance(loss, tuple) else loss

            grad_mmiss_true, loss_mmiss_true = compute_gradient_at(nll_mmiss, true_params)
            grad_mmiss_pert, loss_mmiss_pert = compute_gradient_at(nll_mmiss, pert_log)
            _grad_table_row("DW_matched_miss@θ₀",   grad_mmiss_true, loss_mmiss_true)
            _grad_table_row("DW_matched_miss@pert", grad_mmiss_pert, loss_mmiss_pert)
            del I_mmiss, t_auto_mmiss; gc.collect()

            # ── Record ─────────────────────────────────────────────────────────
            elapsed = time.time() - t0
            row = {
                'iter': it + 1, 'p_obs': p_obs, 'p_hat': round(p_hat, 4),
                'lat_factor': lat_factor, 'lon_factor': lon_factor,
                'threshold': round(threshold, 5), 'elapsed': round(elapsed, 1),
            }

            def _fill(tag, g_true, l_true, g_pert, l_pert):
                g_t = g_true.cpu().numpy()
                g_p = g_pert.cpu().numpy()
                row[f'{tag}_grad_inf_true'] = float(np.abs(g_t).max())
                row[f'{tag}_grad_inf_pert'] = float(np.abs(g_p).max())
                row[f'{tag}_ratio_inf']     = float(np.abs(g_t).max() /
                                                     (np.abs(g_p).max() + 1e-15))
                row[f'{tag}_loss_true']     = float(l_true)
                for pn, v in zip(P_NAMES, g_t):
                    row[f'{tag}_grad_{pn}_true'] = float(v)

            _fill('DW_grid',         grad_grid_true,  loss_grid_true,
                                     grad_grid_pert,  loss_grid_pert)
            _fill('DW_matched_full', grad_mfull_true, loss_mfull_true,
                                     grad_mfull_pert, loss_mfull_pert)
            _fill('DW_matched_miss', grad_mmiss_true, loss_mmiss_true,
                                     grad_mmiss_pert, loss_mmiss_pert)

            records.append(row)
            print(f"\n  Elapsed: {elapsed:.1f}s")
            print(f"  DW_grid={row['DW_grid_ratio_inf']:.4f}  "
                  f"DW_matched_full={row['DW_matched_full_ratio_inf']:.4f}  "
                  f"DW_matched_miss={row['DW_matched_miss_ratio_inf']:.4f}")

        except Exception as e:
            import traceback
            print(f"  [ERROR] iter {it+1}: {e}")
            traceback.print_exc()
            gc.collect()
            continue

        if (it + 1) % 5 == 0:
            _print_running_summary(records)

        pd.DataFrame(records).to_csv(csv_out, index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    _print_running_summary(records)
    pd.DataFrame(records).to_csv(csv_out, index=False)
    print(f"\n  Saved → {csv_out}")


if __name__ == "__main__":
    app()
