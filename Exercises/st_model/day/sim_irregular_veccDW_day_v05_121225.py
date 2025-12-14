# Standard libraries
import sys
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")
import os
import logging
import argparse # Argument parsing

# Data manipulation and analysis
import pandas as pd
import numpy as np
import pickle
import torch
import torch.optim as optim
import copy                    # clone tensor
import time
from sklearn.neighbors import BallTree
# Custom imports


from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization, alg_opt_Encoder
from GEMS_TCO import kernels_gpu_st_simulation_column as kernels_gpu_st_simulation_column


from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2, exact_location_filter
from GEMS_TCO import debiased_whittle
from torch.nn import Parameter
import torch
import torch.fft


app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()

def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.1, help="learning rate"),
    step: int = typer.Option(80, help="Number of iterations in optimization"),

    gamma_par: float = typer.Option(0.5, help="decreasing factor for learning rate"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Number of nearest neighbors in Vecchia approx."),
    
    mm_cond_number: int = typer.Option(8, help="Number of nearest neighbors in Vecchia approx."),
    params: List[str] = typer.Option(['20', '8.25', '5.25', '.2', '.2', '.05', '5'], help="Initial parameters"),
    ## mm-cond-number should be called in command line not mm_cond_number
    ## negative number can be a problem when parsing with typer
    epochs: int = typer.Option(120, help="Number of iterations in optimization"),
    nheads: int = typer.Option(300, help="Number of iterations in optimization"),
    keep_exact_loc: bool = typer.Option(True, help="whether to keep exact location data or not")
) -> None:

    output_path = input_path = Path(config.amarel_estimates_day_path)
    # --- simulate data
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")


    DTYPE= torch.float32 if DEVICE.type == 'mps' else torch.float64

    # TRUE PARAMETERS
    init_sigmasq   = 13.059
    init_range_lon = 0.195 
    init_range_lat = 0.154 
    init_advec_lat = 0.0218
    init_range_time = 1.0
    init_advec_lon = -0.1689
    init_nugget    = 0.247

    # Map parameters
    init_phi2 = 1.0 / init_range_lon
    init_phi1 = init_sigmasq * init_phi2
    init_phi3 = (init_range_lon / init_range_lat)**2
    init_phi4 = (init_range_lon / init_range_time)**2

    # Create Initial Parameters
    initial_vals = [np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), 
                    np.log(init_phi4), init_advec_lat, init_advec_lon, np.log(init_nugget)]

    params_list = [
        # Changed: uses dynamic DTYPE
        torch.tensor([val], requires_grad=True, dtype=DTYPE, device=DEVICE)
        for val in initial_vals
    ]

    params_list2 = copy.deepcopy(params_list)
    params_list_gen_data = copy.deepcopy(params_list)
    # Mean Ozone
    OZONE_MEAN = 260.0

    # --- 2. EXACT COVARIANCE ---
    def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
        phi1, phi2, phi3, phi4 = torch.exp(params[0]), torch.exp(params[1]), torch.exp(params[2]), torch.exp(params[3])
        advec_lat, advec_lon = params[4], params[5]
        sigmasq = phi1 / phi2

        u_lat_eff = lags_x - advec_lat * lags_t
        u_lon_eff = lags_y - advec_lon * lags_t
        
        dist_sq = (u_lat_eff.pow(2) * phi3) + (u_lon_eff.pow(2)) + (lags_t.pow(2) * phi4)
        distance = torch.sqrt(dist_sq + 1e-8)
        
        return sigmasq * torch.exp(-distance * phi2)

    # --- 3. FFT SIMULATION ---
    def generate_exact_gems_field(lat_coords, lon_coords, t_steps, params):
        Nx = len(lat_coords)
        Ny = len(lon_coords)
        Nt = t_steps
        
        print(f"Exact Grid Size: {Nx} (Lat) x {Ny} (Lon) x {Nt} (Time) = {Nx*Ny*Nt} points")
        
        dlat = float(lat_coords[1] - lat_coords[0])
        dlon = float(lon_coords[1] - lon_coords[0])
        dt = 1.0 
        
        Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
        
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

    # --- 4. REGULAR GRID FUNCTIONS (FIXED WITH ROUNDING) ---

    def make_target_grid(lat_start, lat_end, lat_step, lon_start, lon_end, lon_step, device, dtype):
        """
        Constructs a grid explicitly from start to end.
        CRITICAL: Includes rounding to 4 decimal places to prevent "Tensor size does not match" errors.
        """
        # 1. Generate Latitudes (Descending from 5.0)
        # We use a small epsilon to ensure the 'end' is included if it's a multiple
        lats = torch.arange(lat_start, lat_end - 0.0001, lat_step, device=device, dtype=dtype)
        lats = torch.round(lats * 10000) / 10000  # <--- FIX: Round to 4 decimals
        
        # 2. Generate Longitudes (Descending from 133.0)
        lons = torch.arange(lon_start, lon_end - 0.0001, lon_step, device=device, dtype=dtype)
        lons = torch.round(lons * 10000) / 10000  # <--- FIX: Round to 4 decimals

        print(f"Grid Generation debug: Lat Range {lats[0]:.4f}-{lats[-1]:.4f}, Lon Range {lons[0]:.4f}-{lons[-1]:.4f}")
        print(f"Unique Lats: {len(lats)}, Unique Lons: {len(lons)}")

        # 3. Meshgrid (indexing='ij' -> Lat is rows, Lon is cols)
        grid_lat, grid_lon = torch.meshgrid(lats, lons, indexing='ij')

        # 4. Flatten
        flat_lats = grid_lat.flatten()
        flat_lons = grid_lon.flatten()

        # 5. Stack
        center_points = torch.stack([flat_lats, flat_lons], dim=1)
        
        # Return grid AND dimensions (Nx, Ny) for verification
        return center_points, len(lats), len(lons)

    def coarse_by_center_tensor(input_map_tensors: dict, target_grid_tensor: torch.Tensor):
        coarse_map = {}
        
        # BallTree requires CPU Numpy
        query_points_np = target_grid_tensor.cpu().numpy()
        query_points_rad = np.radians(query_points_np)
        
        for key, val_tensor in input_map_tensors.items():
            # Source locations (Perturbed)
            source_locs_np = val_tensor[:, :2].cpu().numpy()
            source_locs_rad = np.radians(source_locs_np)
            
            # NN Search
            tree = BallTree(source_locs_rad, metric='haversine')
            dist, ind = tree.query(query_points_rad, k=1)
            nearest_indices = ind.flatten()
            
            # Map values back to tensor
            indices_tensor = torch.tensor(nearest_indices, device=val_tensor.device, dtype=torch.long)
            gathered_vals = val_tensor[indices_tensor, 2]
            gathered_times = val_tensor[indices_tensor, 3]
            
            # Construct Regular Tensor
            new_tensor = torch.stack([
                target_grid_tensor[:, 0], # Regular Lat
                target_grid_tensor[:, 1], # Regular Lon
                gathered_vals,            # Mapped Value
                gathered_times            # Mapped Time
            ], dim=1)
            
            coarse_map[key] = new_tensor

        return coarse_map

    def get_spatial_ordering(
            
            input_maps: torch.Tensor,
            mm_cond_number: int = 10
        ) -> Tuple[np.ndarray, np.ndarray]:
            
            key_list = list(input_maps.keys())
            data_for_coord = input_maps[key_list[0]]
            
            # --- FIX START ---
            # Check if input is Tensor, if so convert to Numpy for KDTree processing
            if isinstance(data_for_coord, torch.Tensor):
                data_for_coord = data_for_coord.cpu().numpy()
            # --- FIX END ---

            x1 = data_for_coord[:, 0]
            y1 = data_for_coord[:, 1]
            
            # Now this works because x1, y1 are numpy arrays
            coords1 = np.stack((x1, y1), axis=-1)

            # Calculate MaxMin ordering
            ord_mm = _orderings.maxmin_cpp(coords1)
            
            # Reorder coordinates to find nearest neighbors
            data_for_coord_reordered = data_for_coord[ord_mm]
            coords1_reordered = np.stack(
                (data_for_coord_reordered[:, 0], data_for_coord_reordered[:, 1]), 
                axis=-1
            )
            
            # Calculate nearest neighbors map
            nns_map = _orderings.find_nns_l2(locs=coords1_reordered, max_nn=mm_cond_number)
            return ord_mm, nns_map
    
    dw_norm_list = []
    vecc_norm_list = []
    vecc_col_norm_list = []

    num_iters = 10
    for num_iter in range(num_iters):
        print(f"Iteration {num_iter+1}/{num_iters}")

        params_list_vecc = [
            p.detach().clone().to(DEVICE).requires_grad_(True) 
            for p in params_list2
        ]

        params_list_dw = [
                    p.detach().clone().to(DEVICE).requires_grad_(True) 
                    for p in params_list2
                ]


        perturbation_scale = 0.001  # Adjust this value as needed
        LOC_ERR_STD = 0.024

        lats_sim = torch.arange(0, 5.0 + perturbation_scale, 0.044, device=DEVICE, dtype=DTYPE)
        lons_sim = torch.arange(123.0, 133.0 + perturbation_scale, 0.063, device=DEVICE, dtype=DTYPE)
        t_def = 8
        
        print("1. Generating True Field...")
        sim_field = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_list_gen_data)
        
        print("2. Formatting Output...")
        input_map = {}
        aggregated_list = [] 
        
        nugget_std = torch.sqrt(torch.exp(params_list_vecc[6]))
        
        # Flip to Descending Order
        lats_flip = torch.flip(lats_sim, dims=[0])
        lons_flip = torch.flip(lons_sim, dims=[0])
        
        grid_lat, grid_lon = torch.meshgrid(lats_flip, lons_flip, indexing='ij')
        flat_lats = grid_lat.flatten()
        flat_lons = grid_lon.flatten()
        
        for t in range(t_def):
            # Flip field to match coordinates
            field_t = sim_field[:, :, t] 
            field_t_flipped = torch.flip(field_t, dims=[0, 1]) 
            flat_vals = field_t_flipped.flatten()
            # Add Noise + Mean
            obs_vals = flat_vals + (torch.randn_like(flat_vals) * nugget_std) + OZONE_MEAN

            # Add Location Perturbation
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

        print(f"\nDone.")
        print(f"Aggregated Tensor Shape: {aggregated_data.shape}")
        print(f"Device: {aggregated_data.device}")
        print(f"Dtype: {aggregated_data.dtype}")
        
        torch.set_printoptions(precision=4, sci_mode=True)
        print("Sample Output (Lat Desc, Lon Desc):")
        print(aggregated_data[:6])
        
        print(f"\nGradient Check: {aggregated_data.requires_grad} (Should be False)")


        # Step sizes
        step_lat = 0.044
        step_lon = 0.063

        # Generate Target Grid: TOP-DOWN to ensure 5.0 and 133.0 are included
        # 5.0 -> 0.0 (Descending) | 133.0 -> 123.0 (Descending)
        target_grid, Nx_reg, Ny_reg = make_target_grid(
            lat_start=5.0, lat_end=0.0, lat_step=-step_lat,
            lon_start=133.0, lon_end=123.0, lon_step=-step_lon,
            device=DEVICE, dtype=DTYPE
        )

        print(f"Target Regular Grid Shape: {target_grid.shape}")

        # Map Perturbed Data to Regular Grid
        coarse_map = coarse_by_center_tensor(input_map, target_grid)

        # Aggregate Clean Data
        coarse_aggregated_list = list(coarse_map.values())
        coarse_aggregated_data = torch.cat(coarse_aggregated_list, dim=0)

        # --- CRITICAL: OVERWRITE VARIABLES FOR DOWNSTREAM TASKS ---
        input_map = coarse_map
        aggregated_data = coarse_aggregated_data

        print(f"Target Regular Grid Shape: {target_grid.shape}")

        print("\n--- Regularization Complete ---")
        print(f"Final Data Shape: {aggregated_data.shape}")

        # --- 7. VERIFICATION ---
        # Verify integrity for Whittle (Must be perfect rect)
        u_lat = torch.unique(aggregated_data[:, 0])
        u_lon = torch.unique(aggregated_data[:, 1])

        print(f"\nGrid Integrity Check:")
        print(f"Unique Latitudes: {len(u_lat)}")
        print(f"Unique Longitudes: {len(u_lon)}")
        print(f"Expected Total Points per Day: {len(u_lat) * len(u_lon)}")
        print(f"Actual Points per Day: {aggregated_data.shape[0] // t_def}")

        if (len(u_lat) * len(u_lon)) == (aggregated_data.shape[0] // t_def):
            print("SUCCESS: Grid is perfect and ready for Whittle.")
        else:
            print("WARNING: Grid mismatch detected! Whittle will fail.")

        ord_mm, nns_map = get_spatial_ordering(input_map, mm_cond_number=10)
        mm_input_map = {}
        for key in input_map:
            mm_input_map[key] = input_map[key][ord_mm]  # Extract only Lat and Lon columns


    # --- 1. Global Configuration & Constants ---
        dwl = debiased_whittle.debiased_whittle_likelihood()
        TAPERING_FUNC = dwl.cgn_hamming 
        
        print(f"Using device: {DEVICE}")

        # Global L-BFGS Settings
        LBFGS_LR = 1.0
        LBFGS_MAX_STEPS = 10      # 10 to 20  
        LBFGS_HISTORY_SIZE = 100   
        LBFGS_MAX_EVAL = 100       # line search from 50 to 80
        DWL_MAX_STEPS = 20         
        LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3
        #DELTA_LAT, DELTA_LON = 0.044, 0.063 
        

        # Assuming data access is correct
        daily_aggregated_tensors_dw = [aggregated_data]
        daily_hourly_maps_dw = [input_map]
        daily_aggregated_tensors_vecc = [aggregated_data]
        daily_hourly_maps_vecc = [mm_input_map]

        daily_hourly_map_dw = input_map
        daily_aggregated_tensor_dw = aggregated_data

        daily_hourly_map_vecc = mm_input_map
        daily_aggregated_tensor_vecc = aggregated_data

        if isinstance(daily_aggregated_tensor_vecc, torch.Tensor):
            daily_aggregated_tensor_vecc = daily_aggregated_tensor_vecc.to(DEVICE)


# -------------------------------------------------------
        # STEP C: Phase 1 - Debiased Whittle Optimization
        # -------------------------------------------------------
        print(f"\n--- Debiased Whittle {num_iter} ---")
        
        # Helper list for preprocess (can use raw values or params_list)
        # We use the raw floats for the preprocessor init


        day_idx=0
        # Initialize Preprocessor with the wrapper lists (0-index because list has length 1)
        db = debiased_whittle.debiased_whittle_preprocess(
            daily_aggregated_tensors_dw, 
            daily_hourly_maps_dw, 
            day_idx= day_idx, # It is index 0 in our wrapper list # this automatically lead to day_idx
            params_list= params_list_dw, 
            lat_range=[0,5], 
            lon_range=[123.0, 133.0]
        )

        cur_df = db.generate_spatially_filtered_days(0, 5, 123, 133)
        
        unique_times = torch.unique(cur_df[:, TIME_COL])
        time_slices_list = [cur_df[cur_df[:, TIME_COL] == t_val] for t_val in unique_times]

        # --- 1. Pre-compute J-vector, Taper Grid, and Taper Autocorrelation ---
        print("Pre-computing J-vector (Hamming taper)...")
        
        # --- 💥 REVISED: Renamed 'p' to 'p_time' 💥 ---
        J_vec, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered( 
            time_slices_list,
            tapering_func=TAPERING_FUNC, 
            lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL,
            device=DEVICE
        )

        if J_vec is None or J_vec.numel() == 0 or n1 == 0 or n2 == 0 or p_time == 0:
            print(f"Error: J-vector generation failed for Day {day_idx+1}.")
            exit()
        

        I_sample = dwl.calculate_sample_periodogram_vectorized(J_vec)
        taper_autocorr_grid = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

        # Set up Optimizer for Whittle
        optimizer_dw = torch.optim.LBFGS(
            params_list, lr=1.0, max_iter=20, history_size=100, 
            line_search_fn="strong_wolfe", tolerance_grad=1e-5  # 1e-5 to 1e-7
        )

        print(f"Running Whittle L-BFGS on {DEVICE}...")
        # --- END REVISION ---

        # Run Whittle Optimization      
        nat_str, phi_str, raw_str, loss, steps = dwl.run_lbfgs_tapered(
            params_list=params_list,
            optimizer=optimizer_dw,
            I_sample=I_sample,
            n1=n1, n2=n2, p_time=p_time,
            taper_autocorr_grid=taper_autocorr_grid, 
            max_steps=DWL_MAX_STEPS,
            device=DEVICE
        )

# -------------------------------------------------------
# -------------------------------------------------------
        # STEP D: Handoff - Create 'new_params_list'
        # -------------------------------------------------------
        
        # Scale loss for reporting (optional, based on your logic)
        loss_scaled = loss * n1 * n2 * 8  
        
        # 1. Create fresh tensors for Vecchia
        # We DETACH from the Whittle graph so Vecchia starts fresh

        dw_estimates_values = [p.item() for p in params_list]
        dw_estimates_loss = dw_estimates_values + [loss]


        # 2. Apply lower bound to nugget (index -1)
        #with torch.no_grad():
        #    new_params_list[-1].clamp_min_(-2.0)


        frob_norm_dw = sum((p_true.item() - p_est)**2 
                        for p_true, p_est in zip(params_list2, dw_estimates_values))

        # Optional: Take sqrt if you want the actual Euclidean/Frobenius distance
        frob_norm_dw = frob_norm_dw ** 0.5

        # (Your JSON/CSV saving code remains here...)
        input_filepath = output_path / f"sim_irre_dw_1212{ ( daily_aggregated_tensors_dw[0].shape[0]/8 ) }.json"
        res = alg_optimization( f"2024-07-{day_idx+1}", f"DW_{num_iter}", ( daily_aggregated_tensors_dw[0].shape[0]/8 ) , lr,  step , dw_estimates_loss, 0 , frob_norm_dw)
        loaded_data = res.load(input_filepath)
        loaded_data.append( res.toJSON() )
        res.save(input_filepath,loaded_data)
        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'frob_norm'] # 0 for epoch
        csv_filepath = input_path/f"sim_irre_dW_v{int(v*100):03d}_121225_{(daily_aggregated_tensors_vecc[0].shape[0]/8 )}.csv"
        res.tocsv( loaded_data, fieldnames,csv_filepath )



        # 2 - Vecchia L-BFGS Optimization
        
        # --- 💥 Instantiate the GPU Batched Class ---
        # NOTE: Ensure fit_vecchia_lbfgs is the NEW class we defined
        model_instance = kernels_reparam_space_time.fit_vecchia_lbfgs(
                smooth = v,
                input_map = daily_hourly_map_vecc,
                aggregated_data = daily_aggregated_tensor_vecc,
                nns_map = nns_map,
                mm_cond_number = mm_cond_number,
                nheads = nheads
            )

        new_params_list = [
            p.detach().clone().to(DEVICE).requires_grad_(True)
            for p in params_list2
        ]

        # --- 💥 Set L-BFGS Optimizer ---
        optimizer_vecc = model_instance.set_optimizer(
                    new_params_list,     
                    lr=LBFGS_LR,            
                    max_iter=LBFGS_MAX_EVAL,        
                    history_size=LBFGS_HISTORY_SIZE 
                )

        print(f"\n--- Vecchia max min Optimization ( {num_iter+1}) ---")
        start_time = time.time()
        
        # --- 💥 Call the Batched Fit Method ---
        # REMOVED: model_instance.matern_cov_aniso_STABLE_log_reparam
        out, steps_ran = model_instance.fit_vecc_lbfgs(
                new_params_list,
                optimizer_vecc,
                # covariance_function argument is GONE
                max_steps=LBFGS_MAX_STEPS, 
                grad_tol=1e-7
            )

        end_time = time.time()
        epoch_time = end_time - start_time
        
        B = out[:-1]

        frob_norm = sum((p_true.item() - p_est)**2 
                        for p_true, p_est in zip(params_list2, B))

        frob_norm = frob_norm ** 0.5
        print(f"Vecchia Optimization finished in {epoch_time:.2f}s. Results: {out}")

        # (Your Result Saving Logic...)
        input_filepath = output_path / f"sim_irre_vecc_1212_{ ( daily_aggregated_tensors_vecc[0].shape[0]/8 ) }.json"
        
        res = alg_optimization( f"2024-07-{day_idx+1}", f"Vecc_{num_iter}", ( daily_aggregated_tensors_vecc[0].shape[0]/8 ) , lr,  step , out, epoch_time, frob_norm )
        loaded_data = res.load(input_filepath)
        loaded_data.append( res.toJSON() )
        res.save(input_filepath,loaded_data)
        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'frob_norm'] # 0 for epoch

        csv_filepath = input_path/f"sim_irre_vecc_1212_v{int(v*100):03d}_LBFGS_{(daily_aggregated_tensors_vecc[0].shape[0]/8 )}.csv"
        res.tocsv( loaded_data, fieldnames,csv_filepath )


        ''' 
        # 3 - Vecchia L-BFGS, but column conditioning set
        print(f"\n--- Vecchia column conditioning Optimization ( {num_iter+1}) ---")

        # --- CONFIGURATION ---
        v = 0.5              # Smoothness
        mm_cond_number = 14    # Neighbors
        nheads = 113*3           # 0 = Pure Vecchia
        lr = 1.0             # LBFGS learning rate
        LBFGS_MAX_STEPS = 10
        LBFGS_HISTORY_SIZE = 100
        LBFGS_LR = 1.0
        LBFGS_MAX_EVAL = 100    

        new_params_list_col = [
            p.detach().clone().to(DEVICE).requires_grad_(True)
            for p in params_list2
        ]

        model_instance_col = kernels_gpu_st_simulation_column.fit_vecchia_lbfgs(
            smooth=v,
            input_map=input_map,
            aggregated_data=aggregated_data,
            nns_map=nns_map,
            mm_cond_number=mm_cond_number,
            nheads=nheads
        )
        
        optimizer_vecc_col = model_instance_col.set_optimizer(
                    new_params_list_col,     
                    lr=LBFGS_LR,            
                    max_iter=LBFGS_MAX_EVAL,        
                    history_size=LBFGS_HISTORY_SIZE 
                )

        start_time = time.time()

        out_col, steps_ran = model_instance_col.fit_vecc_lbfgs(
                new_params_list_col,
                optimizer_vecc_col,
                # covariance_function argument is GONE
                max_steps=LBFGS_MAX_STEPS, 
                grad_tol=1e-7
            )

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"\nOptimization finished in {epoch_time:.2f}s.")

        C = out_col[:-1]

        frob_norm_col = sum((p_true.item() - p_est)**2 
                        for p_true, p_est in zip(params_list2, C))

        frob_norm_col = frob_norm_col ** 0.5
        print(f"Vecchia Optimization finished in {epoch_time:.2f}s. Results: {out_col}")

        # (Your Result Saving Logic...)
        input_filepath = output_path / f"sim_vecc_col_1212_{ ( daily_aggregated_tensors_vecc[0].shape[0]/8 ) }.json"
        res = alg_optimization( f"2024-07-{day_idx+1}", f"Vecc_{num_iter}", ( daily_aggregated_tensors_vecc[0].shape[0]/8 ) , lr,  step , out_col, epoch_time, frob_norm_col )
        loaded_data = res.load(input_filepath)
        loaded_data.append( res.toJSON() )
        res.save(input_filepath,loaded_data)
        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'frob_norm'] # 0 for epoch

        csv_filepath = input_path/f"sim_vecc_col_1212_v{int(v*100):03d}_LBFGS_{(daily_aggregated_tensors_vecc[0].shape[0]/8 )}.csv"
        res.tocsv( loaded_data, fieldnames,csv_filepath )

        '''

        
        dw_norm_list.append( frob_norm_dw )
        vecc_norm_list.append( frob_norm )
        #vecc_col_norm_list.append( frob_norm_col )
        print(f'average dw norm: {np.mean( dw_norm_list )}, vecc norm: {np.mean( vecc_norm_list )}')

    #print(f'average dw norm: {np.mean( dw_norm_list )}, vecc norm: {np.mean( vecc_norm_list )}, vecc col norm: {np.mean( vecc_col_norm_list )}')

if __name__ == "__main__":
    app()



