"""
sim_gradient_test_vecc_matern_cauchy_dw_033026.py  —  Amarel (cluster) version

Score/gradient test: verifies ||∇ℓ(θ_true)|| ≈ 0 at the true DGP parameters
for simulated data.

Methods tested (same 3 as sim_gim_vecc_cauchy_irr_dw_032826):
  - VM : Vecchia Matérn ν=0.5   (irregular obs locations)
  - VC : Vecchia Cauchy β=1.0   (irregular obs locations)
  - DW : Debiased Whittle        (regular gridded data)

DGP : Matérn ν=0.5 via FFT circulant embedding on high-res grid.
      VM is correctly specified; VC/DW are also expected to have ≈0 gradient
      at true params asymptotically (DW is consistent; VC pseudo-true ≈ true).

Simulation pipeline:
  1. FFT Matérn field on high-res grid (sim_heads_vs_limit logic)
  2. Real obs location matching → irregular dataset for VM / VC
  3. High-res field subsampled to regular grid → gridded dataset for DW
  4. Compute ∇ℓ(θ_true) for each method via torch.autograd.grad
  5. Compare ||∇ℓ(θ_true)|| vs ||∇ℓ(θ_perturbed)||  (ratio << 1 = pass)

Run via sbatch; see slurm_gradient_test_vecc_matern_cauchy_dw_033026.md for job script.

Output columns per method (VM/VC/DW):
  {M}_grad_inf_true   : L∞ norm of gradient at θ_true
  {M}_grad_l2_true    : L2 norm of gradient at θ_true
  {M}_grad_inf_pert   : L∞ norm of gradient at perturbed θ
  {M}_ratio_inf       : grad_inf_true / grad_inf_pert  (target: << 1)
  {M}_grad_{param}_true : per-parameter gradient component
"""

import sys
import gc
import time
from datetime import datetime
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
import pandas as pd
import typer
from pathlib import Path
from sklearn.neighbors import BallTree

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import kernels_vecchia_cauchy
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

app    = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
GC_BETA_DGP    = 1.0

P_NAMES = ["SigmaSq", "RangeLat", "RangeLon", "RangeTime", "AdvecLat", "AdvecLon", "Nugget"]


# ── DGP covariance: Matérn ν=0.5 (exponential) ───────────────────────────────

def get_covariance_on_grid_matern(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


# ── High-res grid & FFT field ─────────────────────────────────────────────────

def build_high_res_grid(lat_range, lon_range, lat_factor, lon_factor):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lats = torch.arange(min(lat_range) - 0.1, max(lat_range) + 0.1, dlat,
                        device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon,
                        device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    CPU, F32 = torch.device("cpu"), torch.float32
    Nx, Ny, Nt = len(lats_hr), len(lons_hr), t_steps
    Px, Py, Pt = 2 * Nx, 2 * Ny, 2 * Nt
    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px // 2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py // 2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt // 2:] -= Pt
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C     = get_covariance_on_grid_matern(Lx, Ly, Lt, params.cpu().float())
    S     = torch.fft.fftn(C);  S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Irregular obs pipeline (identical to sim_heads_vs_limit_032726) ───────────

def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N_grid = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N_grid, -1, dtype=np.int64)
    dist_to_cell, cell_for_obs = grid_tree.query(np.radians(src_np_valid), k=1)
    assignment = np.full(N_grid, -1, dtype=np.int64)
    best_dist  = np.full(N_grid, np.inf)
    for obs_i, (cell_j, d) in enumerate(zip(cell_for_obs.flatten(), dist_to_cell.flatten())):
        if d < best_dist[cell_j]:
            assignment[cell_j] = obs_i
            best_dist[cell_j]  = d
    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack([hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree      = BallTree(np.radians(hr_coords_np), metric='haversine')
    grid_coords_np = grid_coords.cpu().numpy()
    grid_tree    = BallTree(np.radians(grid_coords_np), metric='haversine')
    step3_per_t, hr_idx_per_t, src_locs_per_t = [], [], []
    for key in sorted_keys:
        ref_t    = ref_day_map[key].to(DEVICE)
        src_locs = ref_t[:, :2]
        src_np   = src_locs.cpu().numpy()
        valid    = ~np.isnan(src_np).any(axis=1)
        assignment = apply_step3_1to1(src_np[valid], grid_coords_np, grid_tree)
        step3_per_t.append(assignment)
        if valid.sum() > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np[valid]), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
        src_locs_per_t.append(src_locs[valid])
    return step3_per_t, hr_idx_per_t, src_locs_per_t


def assemble_irr_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params, t_offset=21.0):
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)
    irr_map = {}
    for t_idx, key in enumerate(sorted_keys):
        t_val    = float(t_offset + t_idx)
        hr_idx   = hr_idx_per_t[t_idx]
        src_locs = src_locs_per_t[t_idx]
        N_valid  = hr_idx.shape[0]
        dummy    = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0
        if N_valid > 0:
            gp_vals  = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(N_valid, device=DEVICE, dtype=DTYPE) * nugget_std
        else:
            sim_vals = torch.zeros(0, device=DEVICE, dtype=DTYPE)
        NaN      = float('nan')
        irr_rows = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
        irr_rows[:, 3]  = t_val
        irr_rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid, -1)
        if N_valid > 0:
            assign_t = torch.tensor(step3_per_t[t_idx], device=DEVICE, dtype=torch.long)
            filled   = assign_t >= 0
            win_obs  = assign_t[filled]
            irr_rows[filled, 0] = src_locs[win_obs, 0]
            irr_rows[filled, 1] = src_locs[win_obs, 1]
            irr_rows[filled, 2] = sim_vals[win_obs]
        irr_map[key] = irr_rows.detach()
    return irr_map


# ── Gridded dataset for DW (mirrors real data pipeline: NaN for unmatched cells) ─

def assemble_gridded_dataset_for_dw(field, grid_coords, step3_per_t,
                                     hr_idx_per_t, src_locs_per_t,
                                     true_params, sorted_keys, t_offset=21.0):
    """
    Build daily_hourly_map for debiased_whittle_preprocess.
    Mirrors real data pipeline exactly:
      - lat/lon always set from grid_coords (required by apply_first_difference_2d_tensor)
      - value column NaN for unmatched cells, filled only where step3_per_t >= 0
      - NaN propagates through spatial differencing, same as real data
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)
    sim_hourly_map = {}
    for t_idx, key in enumerate(sorted_keys):
        t_val  = float(t_offset + t_idx)
        dummy  = torch.zeros(N_grid, 7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[:, t_idx - 1] = 1.0
        # lat/lon always set; value starts as NaN (unobserved)
        tensor_t = torch.full((N_grid, 11), float('nan'), device=DEVICE, dtype=DTYPE)
        tensor_t[:, 0] = grid_coords[:, 0]   # lat always set
        tensor_t[:, 1] = grid_coords[:, 1]   # lon always set
        tensor_t[:, 3] = t_val
        tensor_t[:, 4:] = dummy
        # fill value only where real obs matched (same step3_per_t as Vecchia)
        hr_idx   = hr_idx_per_t[t_idx]
        N_valid  = hr_idx.shape[0]
        if N_valid > 0:
            gp_vals  = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(N_valid, device=DEVICE, dtype=DTYPE) * nugget_std
            assign_t = torch.tensor(step3_per_t[t_idx], device=DEVICE, dtype=torch.long)
            filled   = assign_t >= 0
            win_obs  = assign_t[filled]
            tensor_t[filled, 2] = sim_vals[win_obs]
        sim_hourly_map[key] = tensor_t.detach()
    return sim_hourly_map


# ── Gradient helper ───────────────────────────────────────────────────────────

def compute_gradient_at(nll_fn, log_phi_val):
    p    = log_phi_val.detach().clone().requires_grad_(True)
    loss = nll_fn(p)
    grad = torch.autograd.grad(loss, p)[0].detach()
    return grad, loss.detach().item()


# ── Summary helpers ───────────────────────────────────────────────────────────

def _grad_table_row(tag, grad, loss):
    g = grad.cpu().numpy()
    print(f"  {tag:<8}  loss={loss:>12.4f}  ||∇||∞={np.abs(g).max():>10.6f}  "
          f"||∇||2={np.linalg.norm(g):>10.6f}")
    print(f"           per-param: " +
          "  ".join(f"{n[:6]}={v:+.4f}" for n, v in zip(P_NAMES, g)))


def _print_summary(records, label):
    n = len(records)
    if n == 0:
        return
    print(f"\n  ── {label} ({n} iters) ──")
    cw = 10
    print(f"  {'method':<6} {'mean ||∇||∞ @θ₀':>{cw+4}} {'mean ||∇||∞ @pert':>{cw+4}} "
          f"{'ratio':>{cw}}")
    print(f"  {'-'*48}")
    for m in ['VM', 'VC', 'DW']:
        g_true = np.mean([r[f'{m}_grad_inf_true'] for r in records])
        g_pert = np.mean([r[f'{m}_grad_inf_pert'] for r in records])
        print(f"  {m:<6} {g_true:>{cw+4}.6f} {g_pert:>{cw+4}.6f} "
              f"{g_true/(g_pert+1e-15):>{cw}.4f}")
    print(f"\n  Per-parameter mean |∇| at θ_true:")
    print(f"  {'param':<12} {'VM':>{cw}} {'VC':>{cw}} {'DW':>{cw}}")
    print(f"  {'-'*42}")
    for p_name in P_NAMES:
        vm_g = np.mean([abs(r[f'VM_grad_{p_name}_true']) for r in records])
        vc_g = np.mean([abs(r[f'VC_grad_{p_name}_true']) for r in records])
        dw_g = np.mean([abs(r[f'DW_grad_{p_name}_true']) for r in records])
        print(f"  {p_name:<12} {vm_g:>{cw}.6f} {vc_g:>{cw}.6f} {dw_g:>{cw}.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    num_iters:    int   = typer.Option(20,      help="Iterations per job"),
    job_id:       int   = typer.Option(0,       help="Job index for parallel runs"),
    daily_stride: int   = typer.Option(2,       help="Set C daily stride"),
    years:        str   = typer.Option("2022,2023,2024,2025", help="Years for real obs patterns"),
    month:        int   = typer.Option(7,       help="Reference month"),
    lat_range:    str   = typer.Option("-3,2",  help="lat_min,lat_max"),
    lon_range:    str   = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor:   int   = typer.Option(100,     help="High-res lat multiplier"),
    lon_factor:   int   = typer.Option(10,      help="High-res lon multiplier"),
    nheads:       int   = typer.Option(0,        help="Vecchia head points"),
    limit:        int   = typer.Option(20,      help="Vecchia neighbor limit (A=B=C)"),
    mm_cond:      int   = typer.Option(30,      help="NNS map size"),
    pert_scale:   float = typer.Option(0.3,     help="Log-space perturbation half-width"),
    seed:         int   = typer.Option(42,      help="Base seed (actual = seed + job_id)"),
) -> None:

    actual_seed = seed + job_id
    rng = np.random.default_rng(actual_seed)
    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device   : {DEVICE}")
    print(f"Job      : {job_id}  (seed={actual_seed})")
    print(f"DGP      : Matérn ν=0.5")
    print(f"Methods  : VM (Vecchia Matérn), VC (Vecchia Cauchy β={GC_BETA_DGP}), DW")
    print(f"Iters    : {num_iters}  lat×{lat_factor}  lon×{lon_factor}")
    print(f"nheads={nheads}  limit={limit}  mm_cond={mm_cond}")

    output_path = Path(config.amarel_estimates_day_path) / "gradient_test"
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%m%d%y")
    csv_out  = output_path / f"gradient_test_{date_tag}_j{job_id}.csv"

    # ── True DGP parameters (Matérn ν=0.5) ────────────────────────────────────
    true_dict = {'sigmasq': 10.0, 'range_lat': 0.2, 'range_lon': 0.25, 'range_time': 1.5,
                 'advec_lat': 0.02, 'advec_lon': -0.17, 'nugget': 0.25}
    phi2 = 1.0 / true_dict['range_lon']
    phi1 = true_dict['sigmasq'] * phi2
    phi3 = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4 = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)
    true_phys_list = [true_dict[k] for k in
                      ['sigmasq', 'range_lat', 'range_lon', 'range_time',
                       'advec_lat', 'advec_lon', 'nugget']]

    print(f"\nTrue params (physical): {true_dict}")

    # ── Load real obs patterns ─────────────────────────────────────────────────
    print("\n[Setup 1/5] Loading real obs patterns...")
    data_loader   = load_data_dynamic_processed(config.amarel_data_load_path)
    year_dfmaps, year_means = {}, {}
    for yr in years_list:
        df_map_yr, _, _, monthly_mean_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1], mm_cond_number=mm_cond,
            years_=[yr], months_=[month], lat_range=lat_r, lon_range=lon_r,
            is_whittle=False)
        year_dfmaps[yr] = df_map_yr
        year_means[yr]  = monthly_mean_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots")

    # ── Regular target grid ────────────────────────────────────────────────────
    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.round(torch.arange(min(lat_r), max(lat_r) + 1e-4,  DELTA_LAT_BASE,
                                          device=DEVICE, dtype=DTYPE) * 10000) / 10000
    lons_grid = torch.round(torch.arange(lon_r[0], lon_r[1] + 1e-4, DELTA_LON_BASE,
                                          device=DEVICE, dtype=DTYPE) * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat × {len(lons_grid)} lon = {N_grid} cells")

    # ── High-res grid & obs-location mappings ─────────────────────────────────
    print("[Setup 3/5] Building high-res grid and precomputing obs mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat × {len(lons_hr)} lon = {len(lats_hr)*len(lons_hr):,}")

    DUMMY_KEYS  = [f't{i}' for i in range(8)]
    all_day_mappings = []
    for yr in years_list:
        all_sorted = sorted(year_dfmaps[yr].keys())
        n_days_yr  = len(all_sorted) // 8
        print(f"  {yr}: {n_days_yr} days...", flush=True)
        for d_idx in range(n_days_yr):
            ref_day_map, _ = data_loader.load_working_data(
                year_dfmaps[yr], year_means[yr], [d_idx * 8, (d_idx + 1) * 8],
                ord_mm=None, dtype=DTYPE, keep_ori=True)
            day_keys = sorted(ref_day_map.keys())[:8]
            if len(day_keys) < 8:
                continue
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day_map, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))
    print(f"  Total day-patterns: {len(all_day_mappings)}")

    # ── Maxmin ordering for Vecchia ────────────────────────────────────────────
    print("[Setup 4/5] Computing maxmin ordering...")
    ord_grid = _orderings.maxmin_cpp(grid_coords.cpu().numpy())
    nns_grid = _orderings.find_nns_l2(locs=grid_coords.cpu().numpy()[ord_grid], max_nn=mm_cond)

    print("[Setup 5/5] Done (DW uses same obs-location mapping as Vecchia).")

    dwl     = debiased_whittle.debiased_whittle_likelihood()
    records = []

    # ── SIMULATION LOOP ────────────────────────────────────────────────────────
    for it in range(num_iters):
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"  Iteration {it+1}/{num_iters}  [job {job_id}]")
        print(f"{'='*70}")

        yr_it, d_it, step3_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        print(f"  Obs pattern: {yr_it} day {d_it}")

        # Fixed perturbation for comparison (same across methods per iter)
        pert_log = true_params.clone()
        for i in [0, 1, 2, 3, 6]:
            pert_log[i] = pert_log[i] + float(rng.uniform(-pert_scale, pert_scale))

        try:
            # ── Generate FFT field ─────────────────────────────────────────────
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)

            # ── Irregular dataset (VM / VC) ────────────────────────────────────
            irr_map = assemble_irr_dataset(
                field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}

            # ── Gridded dataset (DW) ───────────────────────────────────────────
            # Same obs-location matching as Vecchia: lat/lon always set,
            # value NaN for unmatched cells — mirrors real data pipeline
            dw_hourly_map = assemble_gridded_dataset_for_dw(
                field, grid_coords, step3_per_t, hr_idx_per_t, src_locs_per_t,
                true_params, DUMMY_KEYS)
            del field;  gc.collect();  torch.cuda.empty_cache()

            # Build DW preprocess
            db = debiased_whittle.debiased_whittle_preprocess(
                [None], [dw_hourly_map], day_idx=0,
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
            del dw_hourly_map, cur_df, time_slices, taper_grid, obs_masks

            def nll_dw(p):
                loss = dwl.whittle_likelihood_loss_tapered(
                    p, I_obs, n1, n2, p_time, t_auto, DELTA_LAT_BASE, DELTA_LON_BASE)
                return loss[0] if isinstance(loss, tuple) else loss

            # ── [1/3] Vecchia Matérn ───────────────────────────────────────────
            print("  [1/3] Vecchia Matérn gradient...")
            model_vm = kernels_vecchia.fit_vecchia_lbfgs(
                smooth=0.5, input_map=irr_map_ord, nns_map=nns_grid,
                mm_cond_number=mm_cond, nheads=nheads,
                limit_A=limit, limit_B=limit, limit_C=limit,
                daily_stride=daily_stride,
            )
            model_vm.precompute_conditioning_sets()
            def nll_vm(p): return model_vm.vecchia_batched_likelihood(p)
            grad_vm_true, loss_vm_true = compute_gradient_at(nll_vm, true_params)
            grad_vm_pert, loss_vm_pert = compute_gradient_at(nll_vm, pert_log)
            _grad_table_row("VM@θ₀",   grad_vm_true, loss_vm_true)
            _grad_table_row("VM@pert", grad_vm_pert, loss_vm_pert)
            del model_vm;  gc.collect();  torch.cuda.empty_cache()

            # ── [2/3] Vecchia Cauchy ───────────────────────────────────────────
            print("  [2/3] Vecchia Cauchy gradient...")
            model_vc = kernels_vecchia_cauchy.fit_cauchy_vecchia_lbfgs(
                smooth=0.5, gc_beta=GC_BETA_DGP,
                input_map=irr_map_ord, nns_map=nns_grid,
                mm_cond_number=mm_cond, nheads=nheads,
                limit_A=limit, limit_B=limit, limit_C=limit,
                daily_stride=daily_stride,
            )
            model_vc.precompute_conditioning_sets()
            def nll_vc(p): return model_vc.vecchia_batched_likelihood(p)
            grad_vc_true, loss_vc_true = compute_gradient_at(nll_vc, true_params)
            grad_vc_pert, loss_vc_pert = compute_gradient_at(nll_vc, pert_log)
            _grad_table_row("VC@θ₀",   grad_vc_true, loss_vc_true)
            _grad_table_row("VC@pert", grad_vc_pert, loss_vc_pert)
            del model_vc;  gc.collect();  torch.cuda.empty_cache()

            # ── [3/3] Debiased Whittle ─────────────────────────────────────────
            print("  [3/3] Debiased Whittle gradient...")
            grad_dw_true, loss_dw_true = compute_gradient_at(nll_dw, true_params)
            grad_dw_pert, loss_dw_pert = compute_gradient_at(nll_dw, pert_log)
            _grad_table_row("DW@θ₀",   grad_dw_true, loss_dw_true)
            _grad_table_row("DW@pert", grad_dw_pert, loss_dw_pert)
            del I_obs, t_auto;  gc.collect();  torch.cuda.empty_cache()

            # ── Record ────────────────────────────────────────────────────────
            elapsed = time.time() - t0
            row = {'iter': it + 1, 'job_id': job_id, 'obs_year': yr_it,
                   'obs_day': d_it, 'elapsed': round(elapsed, 1)}
            for m, gt, gp, lt_val, lp_val in [
                ('VM', grad_vm_true, grad_vm_pert, loss_vm_true, loss_vm_pert),
                ('VC', grad_vc_true, grad_vc_pert, loss_vc_true, loss_vc_pert),
                ('DW', grad_dw_true, grad_dw_pert, loss_dw_true, loss_dw_pert),
            ]:
                gt_np = gt.cpu().numpy();  gp_np = gp.cpu().numpy()
                row[f'{m}_loss_true']     = round(float(lt_val), 6)
                row[f'{m}_loss_pert']     = round(float(lp_val), 6)
                row[f'{m}_grad_inf_true'] = round(float(np.abs(gt_np).max()), 8)
                row[f'{m}_grad_l2_true']  = round(float(np.linalg.norm(gt_np)), 8)
                row[f'{m}_grad_inf_pert'] = round(float(np.abs(gp_np).max()), 8)
                row[f'{m}_grad_l2_pert']  = round(float(np.linalg.norm(gp_np)), 8)
                row[f'{m}_ratio_inf']     = round(
                    float(np.abs(gt_np).max()) / (float(np.abs(gp_np).max()) + 1e-15), 6)
                for k, p_name in enumerate(P_NAMES):
                    row[f'{m}_grad_{p_name}_true'] = round(float(gt_np[k]), 8)
                    row[f'{m}_grad_{p_name}_pert'] = round(float(gp_np[k]), 8)

            records.append(row)
            pd.DataFrame(records).to_csv(csv_out, index=False)
            _print_summary(records, f"Running summary ({it+1} iters done)")

        except Exception as e:
            import traceback
            print(f"  [ERROR] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            gc.collect();  torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  FINAL — Gradient test  ({len(records)} iters, job {job_id})")
    print(f"{'='*75}")
    _print_summary(records, "Final summary")
    if records:
        print(f"\n  Saved: {csv_out}")
    print("\n[Done]")


if __name__ == "__main__":
    app()
