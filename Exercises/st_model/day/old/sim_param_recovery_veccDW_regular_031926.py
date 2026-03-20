# Standard libraries
import sys
import time
import json
import numpy as np
import torch
import torch.fft
import typer
from pathlib import Path
from typing import List

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle
from GEMS_TCO import alg_optimization, BaseLogger

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64

DELTA_LAT = 0.044
DELTA_LON = 0.063


# --- Helper Functions ---

def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    advec_lat, advec_lon = params[4], params[5]
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
    noise  = torch.fft.fftn(torch.randn(Px, Py, Pt, device=DEVICE, dtype=DTYPE))
    field  = torch.fft.ifftn(torch.sqrt(S.real) * noise).real
    return field[:Nx, :Ny, :Nt]


def get_spatial_ordering(input_map, mm_cond_number=10):
    data = list(input_map.values())[0]
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    coords = np.stack((data[:, 0], data[:, 1]), axis=-1)
    ord_mm = _orderings.maxmin_cpp(coords)
    coords_r = coords[ord_mm]
    nns_map  = _orderings.find_nns_l2(locs=coords_r, max_nn=mm_cond_number)
    return ord_mm, nns_map


def calculate_original_scale_metrics(est_params, true_init_dict):
    if len(est_params) > 7:
        est_params = est_params[:7]

    est_t = torch.tensor(est_params, device='cpu', dtype=torch.float64).flatten()

    phi1_e, phi2_e = torch.exp(est_t[0]), torch.exp(est_t[1])
    phi3_e, phi4_e = torch.exp(est_t[2]), torch.exp(est_t[3])
    adv_lat_e, adv_lon_e = est_t[4], est_t[5]
    nugget_e = torch.exp(est_t[6])

    sigmasq_e   = phi1_e / phi2_e
    range_lon_e = 1.0 / phi2_e
    range_lat_e = range_lon_e / torch.sqrt(phi3_e)
    range_time_e = range_lon_e / torch.sqrt(phi4_e)

    est_array = torch.stack([sigmasq_e, range_lat_e, range_lon_e, range_time_e,
                              adv_lat_e, adv_lon_e, nugget_e])

    true_array = torch.tensor([
        true_init_dict['sigmasq'], true_init_dict['range_lat'], true_init_dict['range_lon'],
        true_init_dict['range_time'], true_init_dict['advec_lat'], true_init_dict['advec_lon'],
        true_init_dict['nugget']
    ], device='cpu', dtype=torch.float64)

    relative_error = (est_array - true_array) / true_array
    return torch.sqrt(torch.mean(relative_error ** 2)).item()


@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness"),
    lr: float = typer.Option(0.1, help="Learning rate"),
    mm_cond_number: int = typer.Option(8, help="Vecchia nearest neighbors"),
    epochs: int = typer.Option(20, help="Max LBFGS steps"),
    nheads: int = typer.Option(300, help="Vecchia head points"),
    limit_a: int = typer.Option(8, help="Set A neighbors"),
    limit_b: int = typer.Option(8, help="Set B neighbors"),
    limit_c: int = typer.Option(8, help="Set C neighbors"),
    daily_stride: int = typer.Option(8, help="Daily stride for Set C"),
    num_iters: int = typer.Option(100, help="Number of simulation iterations"),
) -> None:

    print(f"Device: {DEVICE}")
    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)

    LBFGS_LR         = 1.0
    LBFGS_MAX_STEPS  = epochs
    LBFGS_HISTORY    = 100
    LBFGS_MAX_EVAL   = 100
    DWL_MAX_STEPS    = 20
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3
    OZONE_MEAN = 260.0

    true_params_dict = {
        'sigmasq': 13.059, 'range_lat': 0.154, 'range_lon': 0.195,
        'range_time': 0.7,  'advec_lat': 0.0218, 'advec_lon': -0.1689, 'nugget': 0.247
    }

    init_sigmasq    = true_params_dict['sigmasq']
    init_range_lat  = true_params_dict['range_lat']
    init_range_lon  = true_params_dict['range_lon']
    init_range_time = true_params_dict['range_time']
    init_advec_lat  = true_params_dict['advec_lat']
    init_advec_lon  = true_params_dict['advec_lon']
    init_nugget     = true_params_dict['nugget']

    phi2 = 1.0 / init_range_lon
    phi1 = init_sigmasq * phi2
    phi3 = (init_range_lon / init_range_lat) ** 2
    phi4 = (init_range_lon / init_range_time) ** 2

    initial_vals = [
        np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
        init_advec_lat, init_advec_lon, np.log(init_nugget)
    ]

    # Simulation grid: lat 0→5 (step DELTA_LAT), lon 123→133 (step DELTA_LON)
    lats_sim = torch.arange(0, 5.0 + 0.001, DELTA_LAT, device=DEVICE, dtype=DTYPE)
    lons_sim = torch.arange(123.0, 133.0 + 0.001, DELTA_LON, device=DEVICE, dtype=DTYPE)
    lats_flip = torch.flip(lats_sim, dims=[0])
    lons_flip = torch.flip(lons_sim, dims=[0])
    grid_lat, grid_lon = torch.meshgrid(lats_flip, lons_flip, indexing='ij')
    flat_lats = grid_lat.flatten()
    flat_lons = grid_lon.flatten()

    dw_norm_list   = []
    vecc_norm_list = []

    params_gen_tensor = torch.tensor(initial_vals, device=DEVICE, dtype=DTYPE)

    for num_iter in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iteration {num_iter+1}/{num_iters} [REGULAR GRID SIMULATION]")
        print(f"{'='*60}")

        # ── [Step 1] Generate data ────────────────────────────────────────────
        params_list_dw   = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True)
                            for v in initial_vals]
        params_list_vecc = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True)
                            for v in initial_vals]

        t_def = 8
        sim_field = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_gen_tensor)

        input_map       = {}
        aggregated_list = []
        nugget_std = torch.sqrt(torch.exp(params_gen_tensor[6]))

        for t in range(t_def):
            field_t   = sim_field[:, :, t]
            field_flip = torch.flip(field_t, dims=[0, 1])
            flat_vals = field_flip.flatten()
            obs_vals  = flat_vals + torch.randn_like(flat_vals) * nugget_std + OZONE_MEAN

            time_val    = 21.0 + t
            flat_times  = torch.full_like(flat_lats, time_val)
            row_tensor  = torch.stack([flat_lats, flat_lons, obs_vals, flat_times], dim=1).detach()

            key_str = f'2024_07_y24m07day01_hm{t:02d}:53'
            input_map[key_str] = row_tensor
            aggregated_list.append(row_tensor)

        aggregated_data = torch.cat(aggregated_list, dim=0)

        # ── [Step 2] Debiased Whittle ─────────────────────────────────────────
        print("--- Debiased Whittle ---")
        dwl          = debiased_whittle.debiased_whittle_likelihood()
        TAPERING_FUNC = dwl.cgn_hamming

        raw_init_floats = [init_sigmasq, init_range_lat, init_range_lon, init_range_time,
                           init_advec_lat, init_advec_lon, init_nugget]

        db = debiased_whittle.debiased_whittle_preprocess(
            [aggregated_data], [input_map], day_idx=0,
            params_list=raw_init_floats,
            lat_range=[0, 5], lon_range=[123.0, 133.0]
        )
        cur_df = db.generate_spatially_filtered_days(0, 5, 123, 133).to(DEVICE)
        unique_times    = torch.unique(cur_df[:, TIME_COL])
        time_slices_list = [cur_df[cur_df[:, TIME_COL] == t_val] for t_val in unique_times]

        J_vec, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered(
            time_slices_list, tapering_func=TAPERING_FUNC,
            lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL, device=DEVICE
        )
        I_sample = dwl.calculate_sample_periodogram_vectorized(J_vec)
        taper_autocorr_grid = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

        optimizer_dw = torch.optim.LBFGS(
            params_list_dw, lr=1.0, max_iter=20, history_size=100,
            line_search_fn="strong_wolfe", tolerance_grad=1e-5
        )

        nat_str, phi_str, raw_str, loss_dw, steps = dwl.run_lbfgs_tapered(
            params_list=params_list_dw, optimizer=optimizer_dw, I_sample=I_sample,
            n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=taper_autocorr_grid,
            max_steps=DWL_MAX_STEPS, device=DEVICE
        )

        dw_estimates_values = [p.item() for p in params_list_dw]
        dw_estimates_loss   = dw_estimates_values + [loss_dw]
        rmsre_dw = calculate_original_scale_metrics(dw_estimates_values, true_params_dict)

        grid_res = int(aggregated_data.shape[0] / 8)
        input_filepath_dw = output_path / f"sim_reg_dw_031926_{grid_res}.json"

        res_dw = alg_optimization(
            day="2024-07-01", cov_name=f"DW_{num_iter}", space_size=grid_res,
            lr=lr, params=dw_estimates_loss, time=0, rmsre=rmsre_dw
        )
        current_data_dw = BaseLogger.load_list(input_filepath_dw)
        current_data_dw.append(res_dw.__dict__)
        with input_filepath_dw.open('w', encoding='utf-8') as f:
            json.dump(current_data_dw, f, separators=(",", ":"), indent=4)

        import pandas as pd
        pd.DataFrame(current_data_dw).to_csv(
            output_path / f"sim_reg_dw_v{int(v*100):03d}_031926_{grid_res}.csv", index=False
        )

        # ── [Step 3] Vecchia ──────────────────────────────────────────────────
        print("--- Vecchia ---")
        ord_mm, nns_map = get_spatial_ordering(input_map, mm_cond_number=mm_cond_number)
        mm_input_map = {key: val[ord_mm] for key, val in input_map.items()}

        model_vecc = kernels_vecchia.fit_vecchia_lbfgs(
            smooth=v, input_map=mm_input_map,
            nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nheads,
            limit_A=limit_a, limit_B=limit_b, limit_C=limit_c, daily_stride=daily_stride
        )

        optimizer_vecc = model_vecc.set_optimizer(
            params_list_vecc, lr=LBFGS_LR, max_iter=LBFGS_MAX_EVAL, history_size=LBFGS_HISTORY
        )

        start_time = time.time()
        out, steps_ran = model_vecc.fit_vecc_lbfgs(
            params_list_vecc, optimizer_vecc, max_steps=LBFGS_MAX_STEPS, grad_tol=1e-7
        )
        epoch_time = time.time() - start_time

        rmsre_vecc = calculate_original_scale_metrics(out, true_params_dict)
        print(f"Vecchia RMSRE: {rmsre_vecc:.4f}  ({epoch_time:.1f}s)")

        input_filepath_vecc = output_path / f"sim_reg_vecc_031926_{grid_res}.json"
        res_vecc = alg_optimization(
            day="2024-07-01", cov_name=f"Vecc_{num_iter}", space_size=grid_res,
            lr=lr, params=out, time=epoch_time, rmsre=rmsre_vecc
        )
        current_data_vecc = BaseLogger.load_list(input_filepath_vecc)
        current_data_vecc.append(res_vecc.__dict__)
        with input_filepath_vecc.open('w', encoding='utf-8') as f:
            json.dump(current_data_vecc, f, separators=(",", ":"), indent=4)

        pd.DataFrame(current_data_vecc).to_csv(
            output_path / f"sim_reg_vecc_v{int(v*100):03d}_031926_LBFGS_{grid_res}.csv", index=False
        )

        dw_norm_list.append(rmsre_dw)
        vecc_norm_list.append(rmsre_vecc)
        print(f"Average RMSRE — DW: {np.mean(dw_norm_list):.4f}  Vecc: {np.mean(vecc_norm_list):.4f}")


if __name__ == "__main__":
    app()
