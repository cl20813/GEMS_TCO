# Standard libraries
import sys
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")
import os
import logging
import argparse 
import pandas as pd
import numpy as np
import pickle
import torch
import torch.optim as optim
import copy 
import time
import gc # Garbage collection

# Custom imports
from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
# Removed alg_optimization as requested
from GEMS_TCO import kernels_gpu_st_simulation_column as kernels_gpu_st_simulation_column

from typing import Optional, List, Tuple
from pathlib import Path
import typer
from GEMS_TCO import configuration as config
from datetime import datetime

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(1.0, help="learning rate (LBFGS usually needs higher)"), # Changed default to 1.0 for LBFGS
    mm_cond_number: int = typer.Option(8, help="Number of nearest neighbors"),
    keep_exact_loc: bool = typer.Option(True, help="whether to keep exact location data or not")
) -> None:

    # --- 1. SETUP & PATHS ---
    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define Output CSV
    




    today = datetime.now().strftime("%m%d%y")
    HEAD_CONFIGS = [300, 400, 500] 








    csv_filename = output_path / f"simulation_results_large_region_v{int(v*100)}_{today}.csv"
    print(f"Results will be saved to: {csv_filename}")

    # --- DEVICE SETUP ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32 if DEVICE.type == 'mps' else torch.float64
    print(f"Using Device: {DEVICE}")

    # --- TRUE PARAMETERS ---
    init_sigmasq   = 13.059
    init_range_lon = 0.195 
    init_range_lat = 0.154 
    init_advec_lat = 0.0218
    init_range_time = 1.0
    init_advec_lon = -0.1689
    init_nugget    = 0.247

    # Map parameters (Model Space)
    init_phi2 = 1.0 / init_range_lon
    init_phi1 = init_sigmasq * init_phi2
    init_phi3 = (init_range_lon / init_range_lat)**2
    init_phi4 = (init_range_lon / init_range_time)**2

    # Create Initial Parameters
    initial_vals = [np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), 
                    np.log(init_phi4), init_advec_lat, init_advec_lon, np.log(init_nugget)]

    # We keep a clean copy of initial parameters to reset every iteration
    params_list_template = [
        torch.tensor([val], requires_grad=True, dtype=DTYPE, device=DEVICE)
        for val in initial_vals
    ]

    # True Parameters Dictionary for Metric Calculation
    true_params_dict = {
        'sigmasq': 13.059, 'range_lat': 0.154, 'range_lon': 0.195,
        'range_time': 1.0, 'advec_lat': 0.0218, 'advec_lon': -0.1689, 'nugget': 0.247
    }
    
    # Mean Ozone
    OZONE_MEAN = 260.0

    # --- HELPER: COVARIANCE ---
    def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
        phi1, phi2, phi3, phi4 = torch.exp(params[0]), torch.exp(params[1]), torch.exp(params[2]), torch.exp(params[3])
        advec_lat, advec_lon = params[4], params[5]
        sigmasq = phi1 / phi2

        u_lat_eff = lags_x - advec_lat * lags_t
        u_lon_eff = lags_y - advec_lon * lags_t
        
        dist_sq = (u_lat_eff.pow(2) * phi3) + (u_lon_eff.pow(2)) + (lags_t.pow(2) * phi4)
        distance = torch.sqrt(dist_sq + 1e-8)
        
        return sigmasq * torch.exp(-distance * phi2)

    # --- HELPER: FFT SIMULATION ---
    def generate_exact_gems_field(lat_coords, lon_coords, t_steps, params):
        Nx, Ny, Nt = len(lat_coords), len(lon_coords), t_steps
        
        dlat = float(lat_coords[1] - lat_coords[0])
        dlon = float(lon_coords[1] - lon_coords[0])
        dt = 1.0 
        
        Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
        
        # Lags
        Lx_len = Px * dlat   
        lags_x = torch.arange(Px, device=DEVICE, dtype=DTYPE) * dlat
        lags_x[Px//2:] -= Lx_len 
        
        Ly_len = Py * dlon   
        lags_y = torch.arange(Py, device=DEVICE, dtype=DTYPE) * dlon
        lags_y[Py//2:] -= Ly_len

        Lt_len = Pt * dt     
        lags_t = torch.arange(Pt, device=DEVICE, dtype=DTYPE) * dt
        lags_t[Pt//2:] -= Lt_len

        L_x, L_y, L_t = torch.meshgrid(lags_x, lags_y, lags_t, indexing='ij')
        C_vals = get_model_covariance_on_grid(L_x, L_y, L_t, params)

        S = torch.fft.fftn(C_vals)
        S.real = torch.clamp(S.real, min=0)

        random_phase = torch.fft.fftn(torch.randn(Px, Py, Pt, device=DEVICE, dtype=DTYPE))
        weighted_freq = torch.sqrt(S.real) * random_phase
        field_sim = torch.fft.ifftn(weighted_freq).real
        
        return field_sim[:Nx, :Ny, :Nt]

    # --- HELPER: ORDERING ---
    def get_spatial_ordering(input_maps, mm_cond_number=10):
        key_list = list(input_maps.keys())
        data_for_coord = input_maps[key_list[0]]
        
        if isinstance(data_for_coord, torch.Tensor):
            data_for_coord = data_for_coord.cpu().numpy()

        coords1 = np.stack((data_for_coord[:, 0], data_for_coord[:, 1]), axis=-1)
        ord_mm = _orderings.maxmin_cpp(coords1)
        
        data_for_coord_reordered = data_for_coord[ord_mm]
        coords1_reordered = np.stack(
            (data_for_coord_reordered[:, 0], data_for_coord_reordered[:, 1]), axis=-1
        )
        nns_map = _orderings.find_nns_l2(locs=coords1_reordered, max_nn=mm_cond_number)
        return ord_mm, nns_map

    # --- HELPER: TRANSFORM & METRICS ---
    def get_physical_params(est_params):
        """Converts model parameters back to physical space"""
        est_t = torch.tensor(est_params, device='cpu', dtype=torch.float64).flatten()
        phi1, phi2, phi3, phi4 = torch.exp(est_t[0]), torch.exp(est_t[1]), torch.exp(est_t[2]), torch.exp(est_t[3])
        
        return {
            "sigmasq": (phi1 / phi2).item(),
            "range_lat": (1.0 / phi2 / torch.sqrt(phi3)).item(),
            "range_lon": (1.0 / phi2).item(),
            "range_time": (1.0 / phi2 / torch.sqrt(phi4)).item(),
            "advec_lat": est_t[4].item(),
            "advec_lon": est_t[5].item(),
            "nugget": torch.exp(est_t[6]).item()
        }

    def calculate_metrics(est_dict, true_dict):
        # Convert dicts to tensors for calc
        est_vec = torch.tensor([est_dict[k] for k in true_dict.keys()])
        true_vec = torch.tensor([true_dict[k] for k in true_dict.keys()])
        
        frob_norm = torch.linalg.norm(true_vec - est_vec).item()
        mape = torch.mean(torch.abs((true_vec - est_vec) / true_vec)).item() * 100 
        return frob_norm, mape

    # ==========================================
    # 🆕 GRID CONFIGURATION (Large Region)
    # ==========================================
    lats_sim = torch.arange(0, 10.0 + 0.001, 0.044, device=DEVICE, dtype=DTYPE)
    lons_sim = torch.arange(113.0, 130.0 + 0.001, 0.063, device=DEVICE, dtype=DTYPE)
    
    lats_flip = torch.flip(lats_sim, dims=[0])
    lons_flip = torch.flip(lons_sim, dims=[0])
    grid_lat, grid_lon = torch.meshgrid(lats_flip, lons_flip, indexing='ij')
    flat_lats = grid_lat.flatten()
    flat_lons = grid_lon.flatten()

    # ==========================================
    # 🔄 MAIN SIMULATION LOOP
    # ==========================================
    results = [] # Master list for logging
    num_iters = 100
    
    # Define Head Configurations to Loop Over
    

    for num_iter in range(num_iters):
        print(f"\n======================================")
        print(f"STARTING SIMULATION ITERATION {num_iter+1}/{num_iters}")
        print(f"======================================")

        # 1. Reset Parameters for Generation
        params_gen = copy.deepcopy(params_list_template)
        t_def = 8
        
        # 2. Generate Data
        print("-> Generating True Field (FFT)...")
        sim_field = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_gen)
        
        # 3. Add Noise & Format
        input_map = {}
        aggregated_list = [] 
        nugget_std = torch.sqrt(torch.exp(params_gen[6]))
        
        for t in range(t_def):
            field_t = sim_field[:, :, t] 
            field_t_flipped = torch.flip(field_t, dims=[0, 1]) 
            flat_vals = field_t_flipped.flatten()
            
            obs_vals = flat_vals + (torch.randn_like(flat_vals) * nugget_std) + OZONE_MEAN
            time_val = 21.0 + t
            flat_times = torch.full_like(flat_lats, time_val)
            
            clean_tensor = torch.stack([flat_lats, flat_lons, obs_vals, flat_times], dim=1).detach()
            
            key_str = f'2024_07_y24m07day01_hm{t:02d}:53'
            input_map[key_str] = clean_tensor
            aggregated_list.append(clean_tensor)

        aggregated_data = torch.cat(aggregated_list, dim=0)
        print(f"-> Data Generated. Shape: {aggregated_data.shape}")

        # 4. Ordering (MaxMin)
        ord_mm, nns_map = get_spatial_ordering(input_map, mm_cond_number=10)
        mm_input_map = {k: input_map[k][ord_mm] for k in input_map}
        
        # Move to GPU for fitting
        if isinstance(aggregated_data, torch.Tensor):
            aggregated_data = aggregated_data.to(DEVICE)

        # ==========================================
        # 🔄 HEADS LOOP (0, 100, 300)
        # ==========================================
        for n_head in HEAD_CONFIGS:
            gc.collect(); torch.cuda.empty_cache()
            print(f"\n   >>> Fitting Vecchia with {n_head} Heads...")
            
            # Reset Optimizer Params
            current_params = [p.detach().clone().to(DEVICE).requires_grad_(True) for p in params_list_template]
            
            # Instantiate Model
            model_instance = kernels_reparam_space_time.fit_vecchia_lbfgs(
                smooth = v,
                input_map = mm_input_map,
                aggregated_data = aggregated_data,
                nns_map = nns_map,
                mm_cond_number = mm_cond_number,
                nheads = n_head # Dynamic heads
            )

            # Setup L-BFGS
            optimizer_vecc = model_instance.set_optimizer(
                current_params,     
                lr=lr,            
                max_iter=100,        
                history_size=100 
            )

            # Fit
            start_time = time.time()
            out_params, steps = model_instance.fit_vecc_lbfgs(
                current_params,
                optimizer_vecc,
                max_steps=10, 
                grad_tol=1e-7
            )
            epoch_time = time.time() - start_time
            
            # Metrics
            est_phys_dict = get_physical_params(out_params)
            frob_norm, mape = calculate_metrics(est_phys_dict, true_params_dict)

            print(f"   [DONE] {n_head} Heads | Time: {epoch_time:.2f}s | MAPE: {mape:.2f}% | F-Norm: {frob_norm:.4f}")

            # -------------------------------------------------------
            # 💾 SAVE RESULTS (Append & Overwrite CSV)
            # -------------------------------------------------------
            row = {
                "Iteration": num_iter + 1,
                "Heads": n_head,
                "MAPE": mape,
                "FrobNorm": frob_norm,
                "Time": epoch_time,
                # Estimated Params
                "Est_SigmaSq":   est_phys_dict['sigmasq'],
                "Est_RangeLat":  est_phys_dict['range_lat'],
                "Est_RangeLon":  est_phys_dict['range_lon'],
                "Est_RangeTime": est_phys_dict['range_time'],
                "Est_AdvecLat":  est_phys_dict['advec_lat'],
                "Est_AdvecLon":  est_phys_dict['advec_lon'],
                "Est_Nugget":    est_phys_dict['nugget'],
            }
            results.append(row)
            
            # Save immediately
            pd.DataFrame(results).round(6).to_csv(csv_filename, index=False)
            print(f"   ✅ Saved to {csv_filename}")

if __name__ == "__main__":
    app()


