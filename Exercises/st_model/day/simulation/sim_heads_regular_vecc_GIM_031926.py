import sys
import random
import time
import json
import numpy as np
import torch
import typer
from typing import List
from pathlib import Path

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle
from GEMS_TCO import configuration as config

# --- GLOBAL SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
NUM_SIMS = 100

DELTA_LAT = 0.044
DELTA_LON = 0.063


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Seed set to {seed}")


# ── Covariance & FFT simulation ──────────────────────────────────────────────

def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
    params    = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    advec_lat, advec_lon   = params[4], params[5]
    sigmasq = phi1 / phi2

    u_lat_eff = lags_x - advec_lat * lags_t
    u_lon_eff = lags_y - advec_lon * lags_t
    dist_sq   = u_lat_eff.pow(2) * phi3 + u_lon_eff.pow(2) + lags_t.pow(2) * phi4
    return sigmasq * torch.exp(-torch.sqrt(dist_sq + 1e-8) * phi2)


def generate_exact_gems_field(lat_coords, lon_coords, t_steps, params):
    Nx, Ny, Nt = len(lat_coords), len(lon_coords), t_steps
    dlat = float(lat_coords[1] - lat_coords[0]) if Nx > 1 else DELTA_LAT
    dlon = float(lon_coords[1] - lon_coords[0]) if Ny > 1 else DELTA_LON
    dt   = 1.0
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt

    lags_x = torch.arange(Px, device=DEVICE, dtype=DTYPE) * dlat
    lags_x[Px//2:] -= Px * dlat
    lags_y = torch.arange(Py, device=DEVICE, dtype=DTYPE) * dlon
    lags_y[Py//2:] -= Py * dlon
    lags_t = torch.arange(Pt, device=DEVICE, dtype=DTYPE) * dt
    lags_t[Pt//2:] -= Pt * dt

    L_x, L_y, L_t = torch.meshgrid(lags_x, lags_y, lags_t, indexing='ij')
    C_vals = get_model_covariance_on_grid(L_x, L_y, L_t, params)

    S = torch.fft.fftn(C_vals)
    S.real = torch.clamp(S.real, min=0)
    noise   = torch.fft.fftn(torch.randn(Px, Py, Pt, device=DEVICE, dtype=DTYPE))
    field   = torch.fft.ifftn(torch.sqrt(S.real) * noise).real
    return field[:Nx, :Ny, :Nt]


def generate_regular_data(params_tensor, grid_config):
    lats_sim = grid_config['lats']
    lons_sim = grid_config['lons']
    t_def    = grid_config['t_def']
    ozone_mean = grid_config['mean']

    sim_field   = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_tensor)
    nugget_std  = torch.sqrt(torch.exp(params_tensor[6]))
    grid_lat, grid_lon = torch.meshgrid(lats_sim, lons_sim, indexing='ij')
    flat_lats, flat_lons = grid_lat.flatten(), grid_lon.flatten()

    input_map, agg_list = {}, []
    for t in range(t_def):
        obs_vals = sim_field[:, :, t].flatten() + torch.randn_like(flat_lats) * nugget_std + ozone_mean
        row = torch.stack([flat_lats, flat_lons, obs_vals,
                           torch.full_like(flat_lats, 21.0 + t)], dim=1).detach()
        input_map[f't_{t:02d}'] = row
        agg_list.append(row)
    return input_map, torch.cat(agg_list, dim=0)


# ── Spatial ordering ─────────────────────────────────────────────────────────

def get_spatial_ordering(input_map, mm_cond_number=10):
    data = list(input_map.values())[0]
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    coords = np.stack((data[:, 0], data[:, 1]), axis=-1)
    ord_mm = _orderings.maxmin_cpp(coords)
    coords_r = coords[ord_mm]
    nns_map  = _orderings.find_nns_l2(locs=coords_r, max_nn=mm_cond_number)
    return ord_mm, nns_map


# ── Result printing ───────────────────────────────────────────────────────────

def transform_model_to_physical(p):
    phi1, phi2, phi3, phi4 = (torch.exp(p[i]) for i in range(4))
    range_lon = 1.0 / phi2
    return torch.stack([
        phi1 / phi2,
        range_lon / torch.sqrt(phi3),
        range_lon,
        range_lon / torch.sqrt(phi4),
        p[4], p[5],
        torch.exp(p[6])
    ])


def print_results(label, true_p, est_p, se_p):
    names = ["log_phi1", "log_phi2", "log_phi3", "log_phi4",
             "advec_lat", "advec_lon", "log_nugget"]
    true_v = true_p.detach().cpu().numpy()
    est_v  = est_p.detach().cpu().numpy()
    se_v   = se_p.detach().cpu().numpy()
    print(f"\n=== {label} ===")
    print(f"{'Param':<15} | {'True':>8} | {'Est':>8} | {'SE':>8} | {'95% CI':>22} | Covered")
    print("-" * 85)
    covered = 0
    for i, name in enumerate(names):
        lo, hi = est_v[i] - 1.96*se_v[i], est_v[i] + 1.96*se_v[i]
        ok = true_v[i] >= lo and true_v[i] <= hi
        if ok: covered += 1
        print(f"{name:<15} | {true_v[i]:>8.4f} | {est_v[i]:>8.4f} | {se_v[i]:>8.4f} | "
              f"({lo:.4f}, {hi:.4f}) | {'YES' if ok else 'NO'}")
    print(f"Coverage: {covered}/{len(names)}")


# ── Main ──────────────────────────────────────────────────────────────────────

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness"),
    lr: float = typer.Option(1.0, help="Learning rate"),
    mm_cond_number: int = typer.Option(10, help="Vecchia neighbors"),
    epochs: int = typer.Option(20, help="Max LBFGS steps"),
    head_configs: List[str] = typer.Option(['300,500,1000'], help="Comma-separated nheads values to compare"),
    limit_a: int = typer.Option(8, help="Set A neighbors"),
    limit_b: int = typer.Option(8, help="Set B neighbors"),
    limit_c: int = typer.Option(8, help="Set C neighbors"),
    daily_stride: int = typer.Option(8, help="Set C stride"),
    num_sims: int = typer.Option(100, help="GIM bootstrap samples"),
) -> None:

    heads_list = [int(h.strip()) for h in head_configs[0].split(',')]

    set_seed(2025)
    print(f"Device: {DEVICE}")
    print(f"Heads to compare: {heads_list}")

    # True parameters (natural scale → log phi scale)
    init_sigmasq, init_range_lon, init_range_lat = 13.059, 0.195, 0.154
    init_range_time, init_advec_lat, init_advec_lon, init_nugget = 1.0, 0.0218, -0.1689, 0.247

    phi2 = 1.0 / init_range_lon
    phi1 = init_sigmasq * phi2
    phi3 = (init_range_lon / init_range_lat) ** 2
    phi4 = (init_range_lon / init_range_time) ** 2

    true_params = torch.tensor([
        np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
        init_advec_lat, init_advec_lon, np.log(init_nugget)
    ], device=DEVICE, dtype=DTYPE)

    grid_cfg = {
        'lats': torch.arange(5.0, -0.0001, -DELTA_LAT, device=DEVICE, dtype=DTYPE),
        'lons': torch.arange(123.0, 133.0 + 0.0001, DELTA_LON, device=DEVICE, dtype=DTYPE),
        't_def': 8,
        'mean': 260.0
    }

    # ── [Step 1] Generate observed data ──────────────────────────────────────
    print("\n[Step 1] Generating data (FFT circulant embedding)...")
    obs_input_map, obs_agg = generate_regular_data(true_params, grid_cfg)
    print(f"  Shape: {obs_agg.shape}")

    # ── [Step 2] Debiased Whittle ─────────────────────────────────────────────
    print("\n" + "="*50)
    print(" >>> MODEL 1: DEBIASED WHITTLE <<<")
    print("="*50)

    dwl = debiased_whittle.debiased_whittle_likelihood()
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3

    db = debiased_whittle.debiased_whittle_preprocess(
        [obs_agg], [obs_input_map], day_idx=0,
        params_list=[init_sigmasq, init_range_lat, init_range_lon, init_range_time,
                     init_advec_lat, init_advec_lon, init_nugget],
        lat_range=[0, 5], lon_range=[123.0, 133.0]
    )
    cur_df = db.generate_spatially_filtered_days(0, 5, 123, 133).to(DEVICE)
    unique_times = torch.unique(cur_df[:, TIME_COL])
    time_slices  = [cur_df[cur_df[:, TIME_COL] == t] for t in unique_times]

    J_vec, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered(
        time_slices, tapering_func=dwl.cgn_hamming,
        lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL, device=DEVICE
    )
    I_sample_obs = dwl.calculate_sample_periodogram_vectorized(J_vec)
    taper_autocorr = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

    params_dw = [true_params[i].clone().detach().requires_grad_(True) for i in range(7)]
    optimizer_dw = torch.optim.LBFGS(
        params_dw, lr=1.0, max_iter=20, history_size=100,
        line_search_fn="strong_wolfe", tolerance_grad=1e-7
    )
    print("Fitting DW...")
    dwl.run_lbfgs_tapered(
        params_list=params_dw, optimizer=optimizer_dw, I_sample=I_sample_obs,
        n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=taper_autocorr,
        max_steps=epochs, device=DEVICE
    )
    best_params_dw = torch.stack(params_dw).detach().requires_grad_(True)

    # Hessian
    def nll_dw(p):
        loss = dwl.whittle_likelihood_loss_tapered(
            p, I_sample_obs, n1, n2, p_time, taper_autocorr, DELTA_LAT, DELTA_LON)
        return loss[0] if isinstance(loss, tuple) else loss

    print("Computing Hessian (DW)...")
    H_dw     = torch.autograd.functional.hessian(nll_dw, best_params_dw)
    H_inv_dw = torch.linalg.inv(H_dw + torch.eye(7, device=DEVICE) * 1e-5)

    # GIM bootstrap
    print(f"GIM bootstrap (DW, {num_sims} sims)...")
    grads_dw = []
    for _ in range(num_sims):
        with torch.no_grad():
            _, boot_agg = generate_regular_data(best_params_dw, grid_cfg)
        boot_slices = [boot_agg[boot_agg[:, TIME_COL] == t] for t in unique_times]
        J_b, bn1, bn2, _, bt = dwl.generate_Jvector_tapered(
            boot_slices, dwl.cgn_hamming, LAT_COL, LON_COL, VAL_COL, DEVICE)
        I_b    = dwl.calculate_sample_periodogram_vectorized(J_b)
        t_auto = taper_autocorr if (bn1 == n1 and bn2 == n2) else \
                 dwl.calculate_taper_autocorrelation_fft(bt, bn1, bn2, DEVICE)
        if best_params_dw.grad is not None: best_params_dw.grad.zero_()
        loss = dwl.whittle_likelihood_loss_tapered(
            best_params_dw, I_b, bn1, bn2, p_time, t_auto, DELTA_LAT, DELTA_LON)
        (loss[0] if isinstance(loss, tuple) else loss).backward()
        grads_dw.append(best_params_dw.grad.detach().clone())

    J_dw     = torch.stack(grads_dw).T @ torch.stack(grads_dw) / num_sims
    GIM_dw   = H_inv_dw @ J_dw @ H_inv_dw
    SE_dw    = torch.sqrt(torch.diag(GIM_dw))
    print_results("DEBIASED WHITTLE GIM", true_params, best_params_dw, SE_dw)

    # ── [Step 3] Vecchia — loop over head_configs ─────────────────────────────
    ord_mm, nns_map = get_spatial_ordering(obs_input_map, mm_cond_number)
    mm_input_map = {k: v[ord_mm] for k, v in obs_input_map.items()}

    for nh in heads_list:
        print("\n" + "="*50)
        print(f" >>> MODEL 2: VECCHIA  (nheads={nh}) <<<")
        print("="*50)

        params_vecc = [true_params[i].clone().detach().requires_grad_(True) for i in range(7)]
        model_vecc = kernels_vecchia.fit_vecchia_lbfgs(
            smooth=v, input_map=mm_input_map,
            nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nh,
            limit_A=limit_a, limit_B=limit_b, limit_C=limit_c, daily_stride=daily_stride
        )
        optimizer_vecc = model_vecc.set_optimizer(params_vecc, lr=lr, max_iter=100, history_size=100)

        print("Fitting Vecchia...")
        out, _ = model_vecc.fit_vecc_lbfgs(params_vecc, optimizer_vecc, max_steps=epochs, grad_tol=1e-6)
        best_params_vecc = torch.tensor(out[:-1], device=DEVICE, dtype=DTYPE, requires_grad=True)

        # Hessian — from the same simulated observed data at θ̂ (orthodox)
        def nll_vecc(p): return model_vecc.vecchia_batched_likelihood(p)

        print("Computing Hessian (Vecchia)...")
        H_vecc     = torch.autograd.functional.hessian(nll_vecc, best_params_vecc)
        H_inv_vecc = torch.linalg.inv(H_vecc + torch.eye(7, device=DEVICE) * 1e-6)

        # GIM bootstrap — FFT sims at θ̂ (orthodox)
        print(f"GIM bootstrap (Vecchia nheads={nh}, {num_sims} sims)...")
        grads_vecc = []
        for _ in range(num_sims):
            with torch.no_grad():
                boot_map, _ = generate_regular_data(best_params_vecc, grid_cfg)
            model_vecc.input_map = {k: boot_map[k][ord_mm] for k in boot_map}
            model_vecc.precompute_conditioning_sets()
            if best_params_vecc.grad is not None: best_params_vecc.grad.zero_()
            model_vecc.vecchia_batched_likelihood(best_params_vecc).backward()
            grads_vecc.append(best_params_vecc.grad.detach().clone())
            del boot_map

        J_vecc   = torch.stack(grads_vecc).T @ torch.stack(grads_vecc) / num_sims
        GIM_vecc = H_inv_vecc @ J_vecc @ H_inv_vecc
        SE_vecc  = torch.sqrt(torch.diag(GIM_vecc))
        print_results(f"VECCHIA GIM (nheads={nh})", true_params, best_params_vecc, SE_vecc)


if __name__ == "__main__":
    app()
