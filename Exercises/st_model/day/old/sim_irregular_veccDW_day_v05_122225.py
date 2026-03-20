# Standard libraries
import sys
import os
import argparse 
import time
import copy
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

# Data manipulation and analysis
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.fft
from sklearn.neighbors import BallTree
import typer

# --- Custom Imports ---
sys.path.append("/cache/home/jl2815/tco") 

from GEMS_TCO import kernels_reparam_space_time_gpu_122225 as kernels_reparam_space_time
from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time_gpu
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2, exact_location_filter
from GEMS_TCO import debiased_whittle

# 🟢 [Import] GEMS_TCO classes
from GEMS_TCO import alg_optimization, BaseLogger

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

# --- Helper Functions ---
def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
    phi1, phi2 = torch.exp(params[0]), torch.exp(params[1])
    phi3, phi4 = torch.exp(params[2]), torch.exp(params[3])
    advec_lat, advec_lon = params[4], params[5]
    sigmasq = phi1 / phi2

    u_lat_eff = lags_x - advec_lat * lags_t
    u_lon_eff = lags_y - advec_lon * lags_t
    
    dist_sq = (u_lat_eff.pow(2) * phi3) + (u_lon_eff.pow(2)) + (lags_t.pow(2) * phi4)
    distance = torch.sqrt(dist_sq + 1e-8)
    return sigmasq * torch.exp(-distance * phi2)

def generate_exact_gems_field(lat_coords, lon_coords, t_steps, params, device, dtype):
    Nx, Ny, Nt = len(lat_coords), len(lon_coords), t_steps
    dlat = float(lat_coords[1] - lat_coords[0])
    dlon = float(lon_coords[1] - lon_coords[0])
    dt = 1.0 
    
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
    
    lags_x = torch.arange(Px, device=device, dtype=dtype) * dlat
    lags_x[Px//2:] -= (Px * dlat)
    lags_y = torch.arange(Py, device=device, dtype=dtype) * dlon
    lags_y[Py//2:] -= (Py * dlon)
    lags_t = torch.arange(Pt, device=device, dtype=dtype) * dt
    lags_t[Pt//2:] -= (Pt * dt)

    L_x, L_y, L_t = torch.meshgrid(lags_x, lags_y, lags_t, indexing='ij')
    C_vals = get_model_covariance_on_grid(L_x, L_y, L_t, params)

    S = torch.fft.fftn(C_vals)
    S.real = torch.clamp(S.real, min=0)
    random_phase = torch.fft.fftn(torch.randn(Px, Py, Pt, device=device, dtype=dtype))
    weighted_freq = torch.sqrt(S.real) * random_phase
    field_sim = torch.fft.ifftn(weighted_freq).real
    
    return field_sim[:Nx, :Ny, :Nt]

def make_target_grid(lat_start, lat_end, lat_step, lon_start, lon_end, lon_step, device, dtype):
    lats = torch.arange(lat_start, lat_end - 0.0001, lat_step, device=device, dtype=dtype)
    lats = torch.round(lats * 10000) / 10000
    lons = torch.arange(lon_start, lon_end - 0.0001, lon_step, device=device, dtype=dtype)
    lons = torch.round(lons * 10000) / 10000
    grid_lat, grid_lon = torch.meshgrid(lats, lons, indexing='ij')
    return torch.stack([grid_lat.flatten(), grid_lon.flatten()], dim=1), len(lats), len(lons)

def coarse_by_center_tensor(input_map_tensors, target_grid_tensor):
    coarse_map = {}
    query_points_rad = np.radians(target_grid_tensor.cpu().numpy())
    for key, val_tensor in input_map_tensors.items():
        source_locs_rad = np.radians(val_tensor[:, :2].cpu().numpy())
        tree = BallTree(source_locs_rad, metric='haversine')
        _, ind = tree.query(query_points_rad, k=1)
        indices_tensor = torch.tensor(ind.flatten(), device=val_tensor.device, dtype=torch.long)
        coarse_map[key] = torch.stack([
            target_grid_tensor[:, 0], target_grid_tensor[:, 1],
            val_tensor[indices_tensor, 2], val_tensor[indices_tensor, 3]
        ], dim=1)
    return coarse_map

def get_spatial_ordering(input_maps, mm_cond_number=10):
    key_list = list(input_maps.keys())
    data_for_coord = input_maps[key_list[0]]
    if isinstance(data_for_coord, torch.Tensor):
        data_for_coord = data_for_coord.cpu().numpy()
    
    coords1 = np.stack((data_for_coord[:, 0], data_for_coord[:, 1]), axis=-1)
    ord_mm = _orderings.maxmin_cpp(coords1)
    
    data_reordered = data_for_coord[ord_mm]
    coords_reordered = np.stack((data_reordered[:, 0], data_reordered[:, 1]), axis=-1)
    
    nns_map_dict = _orderings.find_nns_l2(locs=coords_reordered, max_nn=mm_cond_number)
    return ord_mm, [nns_map_dict[i] for i in range(len(nns_map_dict))]

def calculate_original_scale_metrics(est_params, true_init_dict):
    if len(est_params) > 7:
        est_params = est_params[:7]

    est_t = torch.tensor(est_params, device='cpu', dtype=torch.float64).flatten()
    
    # 🟢 [Fix] Variable naming consistency
    phi1_e, phi2_e = torch.exp(est_t[0]), torch.exp(est_t[1])
    phi3_e, phi4_e = torch.exp(est_t[2]), torch.exp(est_t[3])
    adv_lat_e, adv_lon_e = est_t[4], est_t[5]  
    nugget_e = torch.exp(est_t[6])

    sigmasq_e = phi1_e / phi2_e
    range_lon_e = 1.0 / phi2_e
    range_lat_e = range_lon_e / torch.sqrt(phi3_e)
    range_time_e = range_lon_e / torch.sqrt(phi4_e)

    est_array = torch.stack([sigmasq_e, range_lat_e, range_lon_e, range_time_e, adv_lat_e, adv_lon_e, nugget_e])
    
    true_array = torch.tensor([
        true_init_dict['sigmasq'], true_init_dict['range_lat'], true_init_dict['range_lon'],
        true_init_dict['range_time'], true_init_dict['advec_lat'], true_init_dict['advec_lon'], 
        true_init_dict['nugget']
    ], device='cpu', dtype=torch.float64)

    relative_error = (est_array - true_array) / true_array
    return torch.sqrt(torch.mean(relative_error ** 2)).item()

@app.command()
def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.1, help="learning rate"),
    mm_cond_number: int = typer.Option(8, help="Number of nearest neighbors in Vecchia approx."),
    epochs: int = typer.Option(120, help="Number of iterations in optimization"),
    nheads: int = typer.Option(300, help="Number of iterations in optimization"),
    keep_exact_loc: bool = typer.Option(True, help="whether to keep exact location data or not")
) -> None:

    output_path = input_path = Path(config.amarel_estimates_day_path)
    #DEVICE = torch.device("cpu")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE= torch.float32 if DEVICE.type == 'mps' else torch.float64

    LOC_ERR_STD = 0.03
    OZONE_MEAN = 260.0
    
    # 🟢 [Constants Defined]
    LBFGS_LR = 1.0
    LBFGS_MAX_STEPS = 10      
    LBFGS_HISTORY_SIZE = 100   
    LBFGS_MAX_EVAL = 100       
    DWL_MAX_STEPS = 20         
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3

    true_params_dict = {
        'sigmasq': 13.059, 'range_lat': 0.154, 'range_lon': 0.195,
        'range_time': 1.0, 'advec_lat': 0.042, 'advec_lon': -0.1689, 'nugget': 0.247
    }

    init_sigmasq   = 13.059
    init_range_lon = 0.195 
    init_range_lat = 0.154 
    init_advec_lat = 0.042
    init_range_time = 1.0
    init_advec_lon = -0.1689
    init_nugget    = 0.247

    init_phi2 = 1.0 / init_range_lon
    init_phi1 = init_sigmasq * init_phi2
    init_phi3 = (init_range_lon / init_range_lat)**2
    init_phi4 = (init_range_lon / init_range_time)**2

    initial_vals_numpy = [
        np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), np.log(init_phi4), 
        init_advec_lat, init_advec_lon, np.log(init_nugget)
    ]

    dw_norm_list = []
    vecc_norm_list = []

    # Simulation Grid
    lats_sim = torch.arange(0, 5.0 + 0.001, 0.044, device=DEVICE, dtype=DTYPE)
    lons_sim = torch.arange(123.0, 133.0 + 0.001, 0.063, device=DEVICE, dtype=DTYPE)
    lats_flip = torch.flip(lats_sim, dims=[0])
    lons_flip = torch.flip(lons_sim, dims=[0])
    grid_lat, grid_lon = torch.meshgrid(lats_flip, lons_flip, indexing='ij')
    flat_lats = grid_lat.flatten()
    flat_lons = grid_lon.flatten()
    
    # Target Grid
    step_lat = 0.044
    step_lon = 0.063
    target_grid, Nx_reg, Ny_reg = make_target_grid(
        lat_start=5.0, lat_end=0.0, lat_step=-step_lat,
        lon_start=133.0, lon_end=123.0, lon_step=-step_lon,
        device=DEVICE, dtype=DTYPE
    )

    num_iters = 100
    for num_iter in range(num_iters):
        print(f"\n================ Iteration {num_iter+1}/{num_iters} [IRREGULAR] ================")

        # 🟢 [Step 1] Fresh Parameters
        params_gen = [torch.tensor([val], device=DEVICE, dtype=DTYPE) for val in initial_vals_numpy]
        params_list_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True) for val in initial_vals_numpy]
        params_list_vecc = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True) for val in initial_vals_numpy]

        # 🟢 [Step 2] Data Generation (Irregular Data)
        t_def = 8
        print("1. Generating True Field...")
        sim_field = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_gen, DEVICE, DTYPE)
        
        input_map = {}
        aggregated_list = [] 
        nugget_std = torch.sqrt(torch.exp(params_gen[6])) 
        
        for t in range(t_def):
            field_t = sim_field[:, :, t] 
            field_t_flipped = torch.flip(field_t, dims=[0, 1]) 
            flat_vals = field_t_flipped.flatten()
            
            obs_vals = flat_vals + (torch.randn_like(flat_vals) * nugget_std) + OZONE_MEAN
            
            # 🟢 [Irregular Feature] Perturbation Added
            lat_noise = torch.randn_like(flat_lats) * LOC_ERR_STD
            lon_noise = torch.randn_like(flat_lons) * LOC_ERR_STD
            perturbed_lats = flat_lats + lat_noise
            perturbed_lons = flat_lons + lon_noise

            time_val = 21.0 + t
            flat_times = torch.full_like(flat_lats, time_val)
            
            row_tensor = torch.stack([perturbed_lats, perturbed_lons, obs_vals, flat_times], dim=1)
            key_str = f'2024_07_y24m07day01_hm{t:02d}:53'
            input_map[key_str] = row_tensor.detach()
            aggregated_list.append(input_map[key_str])

        aggregated_data = torch.cat(aggregated_list, dim=0)

        # 🟢 [Step 3] Regularization (Irregular -> Regular)
        coarse_map = coarse_by_center_tensor(input_map, target_grid)
        coarse_aggregated_data = torch.cat(list(coarse_map.values()), dim=0)

        input_map = coarse_map
        aggregated_data = coarse_aggregated_data
        
        # Ordering
        ord_mm, nns_map = get_spatial_ordering(input_map, mm_cond_number=mm_cond_number)
        mm_input_map = {key: val[ord_mm] for key, val in input_map.items()}

        # ------------------------------------------------------------------
        # 🟢 [TRY-EXCEPT Scope Extended]
        #    Covers DW Opt -> Save -> Vecc Opt -> Save
        # ------------------------------------------------------------------
        try:
            # ------------------------------------------------------------------
            # STEP 4: Debiased Whittle Optimization
            # ------------------------------------------------------------------
            print(f"\n--- Debiased Whittle Optimization ---")
            
            dwl = debiased_whittle.debiased_whittle_likelihood()
            TAPERING_FUNC = dwl.cgn_hamming 
            
            daily_aggregated_tensors_dw = [aggregated_data]
            daily_hourly_maps_dw = [input_map]
            
            db = debiased_whittle.debiased_whittle_preprocess(
                daily_aggregated_tensors_dw, daily_hourly_maps_dw, day_idx=0,
                params_list=params_list_dw, lat_range=[0,5], lon_range=[123.0, 133.0]
            )
            cur_df = db.generate_spatially_filtered_days(0, 5, 123, 133)
            time_slices_list = [cur_df[cur_df[:, TIME_COL] == t_val] for t_val in torch.unique(cur_df[:, TIME_COL])]

            J_vec, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered( 
                time_slices_list, tapering_func=TAPERING_FUNC, 
                lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL,
                device=DEVICE
            )
            
            I_sample = dwl.calculate_sample_periodogram_vectorized(J_vec)
            taper_autocorr_grid = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

            optimizer_dw = torch.optim.LBFGS(
                params_list_dw, lr=1.0, max_iter=20, history_size=100, 
                line_search_fn="strong_wolfe", tolerance_grad=1e-5
            )

            # [DW Run] Raises exception on numerical failure
            nat_str, phi_str, raw_str, loss, steps = dwl.run_lbfgs_tapered(
                params_list=params_list_dw,
                optimizer=optimizer_dw,
                I_sample=I_sample, n1=n1, n2=n2, p_time=p_time,
                taper_autocorr_grid=taper_autocorr_grid, max_steps=DWL_MAX_STEPS, device=DEVICE
            )

            # [DW Save]
            dw_estimates_values = [p.item() for p in params_list_dw]
            dw_estimates_loss = dw_estimates_values + [loss]
            rmsre_dw = calculate_original_scale_metrics(dw_estimates_values, true_params_dict)

            a_date = '1222'
            grid_res = int(aggregated_data.shape[0] / 8)
            
            # Filename: irre
            input_filepath_dw = output_path / f"sim_irre_dw_{a_date}_{grid_res}.json"
            
            res_dw = alg_optimization(
                day=f"2024-07-01", cov_name=f"DW_{num_iter}", space_size=grid_res,
                lr=lr, params=dw_estimates_loss, time=0, rmsre=rmsre_dw
            )
            
            current_data = BaseLogger.load_list(input_filepath_dw)
            current_data.append(res_dw.__dict__)
            with input_filepath_dw.open('w', encoding='utf-8') as f:
                json.dump(current_data, f, separators=(",", ":"), indent=4)
            pd.DataFrame(current_data).to_csv(input_path / f"sim_irre_dW_v{int(v*100):03d}_{a_date}_{grid_res}.csv", index=False)


            # ------------------------------------------------------------------
            # STEP 5: Vecchia Optimization (Only runs if DW passed)
            # ------------------------------------------------------------------
            print(f"\n--- Vecchia Optimization ---")
            
            model_instance = kernels_reparam_space_time_gpu.fit_vecchia_lbfgs(
                smooth=v, input_map=mm_input_map, aggregated_data=aggregated_data,
                nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nheads
            )

            optimizer_vecc = model_instance.set_optimizer(
                params_list_vecc, lr=LBFGS_LR, max_iter=LBFGS_MAX_EVAL, 
                history_size=LBFGS_HISTORY_SIZE
            )

            start_time = time.time()
            out, steps_ran = model_instance.fit_vecc_lbfgs(
                params_list_vecc, optimizer_vecc, max_steps=LBFGS_MAX_STEPS, grad_tol=1e-7
            )
            epoch_time = time.time() - start_time
            
            rmsre_vecc = calculate_original_scale_metrics(out, true_params_dict)
            print(f"Vecchia RMSRE: {rmsre_vecc:.2f}%")

            # [Vecc Save]
            # Filename: irre
            input_filepath_vecc = output_path / f"sim_irre_vecc_{a_date}_{grid_res}.json"
            
            res_vecc = alg_optimization(
                day=f"2024-07-01", cov_name=f"Vecc_{num_iter}", space_size=grid_res,
                lr=lr, params=out, time=epoch_time, rmsre=rmsre_vecc
            )
            
            current_data_vecc = BaseLogger.load_list(input_filepath_vecc)
            current_data_vecc.append(res_vecc.__dict__)
            with input_filepath_vecc.open('w', encoding='utf-8') as f:
                json.dump(current_data_vecc, f, separators=(",", ":"), indent=4)
            pd.DataFrame(current_data_vecc).to_csv(input_path / f"sim_irre_vecc_{a_date}_v{int(v*100):03d}_LBFGS_{grid_res}.csv", index=False)

            dw_norm_list.append(rmsre_dw)
            vecc_norm_list.append(rmsre_vecc)
            print(f'Average DW Norm: {np.mean(dw_norm_list):.4f}, Vecc Norm: {np.mean(vecc_norm_list):.4f}')

        # ------------------------------------------------------------------
        # Exception Handling: Skip entire iteration if any step fails
        # ------------------------------------------------------------------
        except Exception as e:
            print(f"🔴 Iteration {num_iter+1} FAILED & SKIPPED due to error: {e}")
            continue 

if __name__ == "__main__":
    app()




