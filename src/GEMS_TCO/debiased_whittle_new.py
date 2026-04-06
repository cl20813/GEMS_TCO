# Configuration
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"

import sys
import cmath

sys.path.append(gems_tco_path)

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F


class debiased_whittle_preprocess:
    def __init__(self, daily_aggregated_tensors, daily_hourly_maps, day_idx, params_list, lat_range, lon_range):
        self.day_idx = day_idx
        self.daily_hourly_map = daily_hourly_maps[day_idx]

    def subset_tensor(self,df_tensor: torch.Tensor, lat_s: float, lat_e: float, lon_s: float,lon_e: float) -> torch.Tensor:
        """Subsets a tensor to a specific lat/lon range."""
        #lat_mask = (df_tensor[:, 0] >= -5) & (df_tensor[:, 0] <= 6.3)
        #lon_mask = (df_tensor[:, 1] >= 118) & (df_tensor[:, 1] <= 134.2)
        lat_mask = (df_tensor[:, 0] >= lat_s) & (df_tensor[:, 0] <= lat_e)
        lon_mask = (df_tensor[:, 1] >= lon_s) & (df_tensor[:, 1] <= lon_e)

        df_sub = df_tensor[lat_mask & lon_mask].clone()
        return df_sub

    def apply_first_difference_2d_tensor(self, df_tensor: torch.Tensor, kernel_type: str = 'new') -> torch.Tensor:
        """
        Applies a 2D difference filter using convolution.
        kernel_type:
          'old' : [-2,1; 1,0]  Z = (X(i+1,j)-X(i,j)) + (X(i,j+1)-X(i,j))
          'new' : [-1,1; 1,-1] Z = Delta_lat(Delta_lon X)  (separable)
        """
        if df_tensor.size(0) == 0:
            return torch.empty(0, 4, dtype=torch.float64)

        if df_tensor.dtype != torch.float64:
            df_tensor = df_tensor.to(torch.float64)

        # 1. Get grid dimensions and validate
        unique_lats = torch.unique(df_tensor[:, 0])
        unique_lons = torch.unique(df_tensor[:, 1])
        lat_count, lon_count = unique_lats.size(0), unique_lons.size(0)

        if df_tensor.size(0) != lat_count * lon_count:
            raise ValueError("Tensor size does not match grid dimensions. Must be a complete grid.")
        if lat_count < 2 or lon_count < 2:
            return torch.empty(0, 4)

        # 2. Reshape data and define the correct kernel
        ozone_data = df_tensor[:, 2].reshape(1, 1, lat_count, lon_count)

        if kernel_type == 'old':
            diff_kernel = torch.tensor([[[[-2., 1.],
                                        [ 1., 0.]]]], dtype=torch.float64)
        else:  # 'new'
            # Z = -X(i,j) + X(i+1,j) + X(i,j+1) - X(i+1,j+1) = -Delta_lat(Delta_lon X)
            # Sign-flipped vs true tensor product [[1,-1],[-1,1]], but |H|^2 is identical.
            diff_kernel = torch.tensor([[[[-1., 1.],
                                        [ 1., -1.]]]], dtype=torch.float64)

        diff_kernel = diff_kernel.to(df_tensor.device)
        
        # 3. Apply convolution (which acts as cross-correlation)
        filtered_grid = F.conv2d(ozone_data, diff_kernel, padding='valid').squeeze()

        # 4. Determine coordinates for the new, smaller grid
        # The new grid corresponds to the anchor points of the kernel
        new_lats = unique_lats[:-1]
        new_lons = unique_lons[:-1]

        # 5. Reconstruct the output tensor
        new_lat_grid, new_lon_grid = torch.meshgrid(new_lats, new_lons, indexing='ij')
        filtered_values = filtered_grid.flatten()
        time_value = df_tensor[0, 3].repeat(filtered_values.size(0))

        new_tensor = torch.stack([
            new_lat_grid.flatten(),
            new_lon_grid.flatten(),
            filtered_values,
            time_value
        ], dim=1)
        
        return new_tensor

    def generate_spatially_filtered_days(self, lat_s: float, lat_e: float, lon_s: float, lon_e: float,
                                          kernel_type: str = 'new'):
        tensors_to_aggregate = []

        for key, tensor in self.daily_hourly_map.items():
            subsetted = self.subset_tensor(tensor, lat_s, lat_e, lon_s, lon_e)
            if subsetted.size(0) > 0:
                try:
                    diff_applied = self.apply_first_difference_2d_tensor(subsetted, kernel_type=kernel_type)
                    if diff_applied.size(0) > 0:
                        tensors_to_aggregate.append(diff_applied)
                except ValueError as e:
                    print(f"Skipping data chunk on day {self.day_idx+1} due to error: {e}")

        if tensors_to_aggregate:
            subsetted_aggregated_day = torch.cat(tensors_to_aggregate, dim=0)
        return subsetted_aggregated_day

    def generate_raw_days(self, lat_s: float, lat_e: float, lon_s: float, lon_e: float):
        """No prewhitening — returns subsetted raw data concatenated across hours."""
        tensors_to_aggregate = []

        for key, tensor in self.daily_hourly_map.items():
            subsetted = self.subset_tensor(tensor, lat_s, lat_e, lon_s, lon_e)
            if subsetted.size(0) > 0:
                tensors_to_aggregate.append(subsetted)

        if tensors_to_aggregate:
            return torch.cat(tensors_to_aggregate, dim=0)
        return torch.empty(0, 4, dtype=torch.float64)
    


class debiased_whittle_likelihood: # (full_vecc_dw_likelihoods):
    
    # NOTE: The __init__ was empty. If it needs to call super(), 
    # it should be added back. I'm keeping it as you provided.
    def __init__(self):
        pass
    
    # =========================================================================
    # 1. Tapering & Data Functions
    # =========================================================================
    ''' 
    @staticmethod
    def cgn_hamming(u, n1, n2):
        """Computes a 2D Hamming window."""
        u1, u2 = u
        device = u1.device if isinstance(u1, torch.Tensor) else (u2.device if isinstance(u2, torch.Tensor) else torch.device('cpu'))
        u1_tensor = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_tensor = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        n1_eff = float(n1) if n1 > 0 else 1.0
        n2_eff = float(n2) if n2 > 0 else 1.0
        hamming1 = 0.54 + 0.46 * torch.cos(2.0 * torch.pi * u1_tensor / n1_eff)
        hamming2 = 0.54 + 0.46 * torch.cos(2.0 * torch.pi * u2_tensor / n2_eff)
        return hamming1 * hamming2
    '''
    
    @staticmethod
    def cgn_hamming(u, n1, n2):
        u1, u2 = u
        device = u1.device if isinstance(u1, torch.Tensor) else (u2.device if isinstance(u2, torch.Tensor) else torch.device('cpu'))
        u1_tensor = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_tensor = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        n1_eff = float(n1) if n1 > 0 else 1.0
        n2_eff = float(n2) if n2 > 0 else 1.0

        # FIXED: Changed + to -
        # Edge: u=0 -> weight 0.08, this is the taper 
        # At quarter u=N/4,   0.54 - 0.46*cos(pi/2) = 0.54
        # At Center u=N,      0.54 - 0.46*cos(pi)   = 1
        # After this again goes back to 0.54 and 0.08 to other edge.


        hamming1 = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * u1_tensor / n1_eff)
        hamming2 = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * u2_tensor / n2_eff)
        return hamming1 * hamming2
    

    @staticmethod
    def calculate_taper_autocorrelation_fft(taper_grid, n1, n2, device):
        """
        Computes the normalized taper autocorrelation function c_gn(u) using FFT.
        """
        taper_grid = taper_grid.to(device) 
        H = torch.sum(taper_grid**2)
        if H < 1e-12:
            print("Warning: Sum of squared taper weights (H) is near zero.")
            return torch.zeros((2*n1-1, 2*n2-1), device=device, dtype=taper_grid.dtype)
        N1, N2 = 2 * n1 - 1, 2 * n2 - 1
        taper_fft = torch.fft.fft2(taper_grid, s=(N1, N2))
        power_spectrum = torch.abs(taper_fft)**2
        autocorr_unnormalized = torch.fft.ifft2(power_spectrum).real
        autocorr_shifted = torch.fft.fftshift(autocorr_unnormalized)
        c_gn_grid = autocorr_shifted / (H + 1e-12)
        return c_gn_grid 

    @staticmethod
    def generate_Jvector_tapered(tensor_list, tapering_func, lat_col, lon_col, val_col, device):
        """
        Generates J-vector for a single component using the specified taper,
        placing result on device.
        """
        # --- 💥 REVISED: p_time is the number of time points 💥 ---
        p_time = len(tensor_list)
        if p_time == 0: return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        valid_tensors = [t for t in tensor_list if t.numel() > 0 and t.shape[1] > max(lat_col, lon_col, val_col)]
        if not valid_tensors:
            print("Warning: No valid tensors found in tensor_list.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        try:
            all_lats_cpu = torch.cat([t[:, lat_col] for t in valid_tensors])
            all_lons_cpu = torch.cat([t[:, lon_col] for t in valid_tensors])
        except IndexError:
            print(f"Error: Invalid column index. Check tensor shapes.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        all_lats_cpu = all_lats_cpu[~torch.isnan(all_lats_cpu) & ~torch.isinf(all_lats_cpu)]
        all_lons_cpu = all_lons_cpu[~torch.isnan(all_lons_cpu) & ~torch.isinf(all_lons_cpu)]
        if all_lats_cpu.numel() == 0 or all_lons_cpu.numel() == 0:
            print("Warning: No valid coordinates after NaN/Inf filtering.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        unique_lats_cpu, unique_lons_cpu = torch.unique(all_lats_cpu), torch.unique(all_lons_cpu)
        n1, n2 = len(unique_lats_cpu), len(unique_lons_cpu)
        if n1 == 0 or n2 == 0:
            print("Warning: Grid dimensions are zero.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        lat_map = {lat.item(): i for i, lat in enumerate(unique_lats_cpu)}
        lon_map = {lon.item(): i for i, lon in enumerate(unique_lons_cpu)}

        u1_mesh_cpu, u2_mesh_cpu = torch.meshgrid(
            torch.arange(n1, dtype=torch.float64),
            torch.arange(n2, dtype=torch.float64),
            indexing='ij'
        )
        taper_grid = tapering_func((u1_mesh_cpu, u2_mesh_cpu), n1, n2).to(device) # Taper on device

        fft_results = []
        for tensor in tensor_list:
            data_grid = torch.zeros((n1, n2), dtype=torch.float64, device=device)
            if tensor.numel() > 0 and tensor.shape[1] > max(lat_col, lon_col, val_col):
                for row in tensor:
                    lat_item, lon_item = row[lat_col].item(), row[lon_col].item()
                    if not (np.isnan(lat_item) or np.isnan(lon_item)):
                        i = lat_map.get(lat_item)
                        j = lon_map.get(lon_item)
                        if i is not None and j is not None:
                            val = row[val_col]
                            val_num = val.item() if isinstance(val, torch.Tensor) else val
                            if not np.isnan(val_num) and not np.isinf(val_num):
                                data_grid[i, j] = val_num

            data_grid_tapered = data_grid * taper_grid

            if torch.isnan(data_grid_tapered).any() or torch.isinf(data_grid_tapered).any():
                print("Warning: NaN/Inf detected in data_grid_tapered before FFT. Replacing with zeros.")
                data_grid_tapered = torch.nan_to_num(data_grid_tapered, nan=0.0, posinf=0.0, neginf=0.0)

            fft_results.append(torch.fft.fft2(data_grid_tapered))

        if not fft_results:
            print("Warning: No FFT results generated.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), n1, n2, 0, taper_grid

        J_vector_tensor = torch.stack(fft_results, dim=2).to(device)

        H = torch.sum(taper_grid**2)
        if H < 1e-12:
            print("Warning: Normalization factor H is near zero.")
            norm_factor = torch.tensor(0.0, device=device, dtype=torch.float64)
        else:
            norm_factor = (torch.sqrt(1.0 / H) / (2.0 * cmath.pi)).to(device)

        result = J_vector_tensor * norm_factor
        if torch.isnan(result).any(): print("Warning: NaN in J_vector output.")
        return result, n1, n2, p_time, taper_grid # <-- Return p_time

    @staticmethod
    def calculate_sample_periodogram_vectorized(J_vector_tensor):
        """Calculates sample periodogram I_n = J J^H (pxp matrix for each spatial freq)."""
        if torch.isnan(J_vector_tensor).any() or torch.isinf(J_vector_tensor).any():
            print("Warning: NaN/Inf detected in J_vector_tensor input.")
            n1, n2, p = J_vector_tensor.shape
            return torch.full((n1, n2, p, p), float('nan'), dtype=torch.complex128, device=J_vector_tensor.device)

        J_col = J_vector_tensor.unsqueeze(-1)
        J_row_conj = J_vector_tensor.unsqueeze(-2).conj()
        result = J_col @ J_row_conj

        if torch.isnan(result).any(): print("Warning: NaN in periodogram matrix output.")
        return result

    # =========================================================================
    # 2. Covariance Functions (7-Parameter Version)
    # =========================================================================
    @staticmethod
    def cov_x_spatiotemporal_model_kernel(u1, u2, t, params):
        """
        Computes autocovariance of X using the 7-PARAMETER spatio-temporal kernel.
        u1, u2 are PHYSICAL lags (already scaled by deltas).
        t is the PHYSICAL time lag.
        """
        device = params.device
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float64)

        if torch.isnan(params).any() or torch.isinf(params).any():
            out_shape = torch.broadcast_shapes(u1_dev.shape, u2_dev.shape, t_dev.shape)
            return torch.full(out_shape, float('nan'), device=device, dtype=torch.float64)

        # --- A. Unpack and Recover Parameters ---
        phi1   = torch.exp(params[0])
        phi2   = torch.exp(params[1]) # range_lon_inv
        phi3   = torch.exp(params[2]) # (range_lon / range_lat)^2
        phi4   = torch.exp(params[3]) # beta^2
        advec_lat = params[4]
        advec_lon = params[5]
        nugget = torch.exp(params[6])

        epsilon = 1e-12
        sigmasq = phi1 / (phi2 + epsilon)  
        range_lon_inv = phi2
        range_lat_inv = torch.sqrt(phi3 + epsilon) * phi2
        beta_scaled_inv = torch.sqrt(phi4 + epsilon) * phi2 # This is beta * range_lon_inv

        # --- B. Calculate Anisotropic Advected Distance ---
        u1_adv = u1_dev - advec_lat * t_dev
        u2_adv = u2_dev - advec_lon * t_dev

        dist_sq = (u1_adv * range_lat_inv).pow(2) + \
                (u2_adv * range_lon_inv).pow(2) + \
                (t_dev * beta_scaled_inv).pow(2)
        
        distance = torch.sqrt(dist_sq + epsilon) 

        # --- C. Calculate Covariance (Matern 0.5 = Exponential) ---
        cov_smooth = sigmasq * torch.exp(-distance)

        # --- D. Add Nugget ---
        is_zero_lag = (torch.abs(u1_dev) < 1e-9) & (torch.abs(u2_dev) < 1e-9) & (torch.abs(t_dev) < 1e-9)
        final_cov = torch.where(is_zero_lag, cov_smooth + nugget, cov_smooth)

        if torch.isnan(final_cov).any(): print("Warning: NaN detected in cov_x_spatiotemporal_model_kernel output.")
        return final_cov

    @staticmethod
    def cov_spatial_difference(u1, u2, t, params, delta1, delta2):
        """
        [Optimized] Calculates Cov(Y, Y) fully vectorized.
        Input shapes:
          u1, u2 : (N1, N2, P, P) or (N1, N2, 1, 1)
          t      : (1, 1, P, P)

        Filter: {(0,0): -1, (1,0): 1, (0,1): 1, (1,1): -1}  (separable/product filter)
        4x4=16 cross terms, consolidated to 9 unique offsets:
          (0,0): 4,  (±delta1,0): -2 each,  (0,±delta2): -2 each,
          (±delta1,±delta2 all diag): 1 each
        """
        device = params.device

        # 1. Weights & Offsets for new filter {(0,0):-1, (1,0):1, (0,1):1, (1,1):-1}
        w_flat = torch.tensor([4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0], device=device, dtype=torch.float64)

        offsets_list = [
            [0.0, 0.0],              # weight  4
            [-delta1, 0.0],          # weight -2
            [delta1, 0.0],           # weight -2
            [0.0, -delta2],          # weight -2
            [0.0, delta2],           # weight -2
            [-delta1, -delta2],      # weight  1
            [delta1, delta2],        # weight  1
            [delta1, -delta2],       # weight  1
            [-delta1, delta2],       # weight  1
        ]
        offsets_tensor = torch.tensor(offsets_list, device=device, dtype=torch.float64) # (9, 2)
        n_terms = 9

        # 2. Expand Inputs for Broadcasting
        u1_exp = u1.unsqueeze(0) # (1, N1, N2, P, P)
        u2_exp = u2.unsqueeze(0) # (1, N1, N2, P, P)

        # Add offsets: (9, 1, 1, 1, 1) + (1, N1, N2, P, P) -> (9, N1, N2, P, P)
        off_x = offsets_tensor[:, 0].reshape(n_terms, 1, 1, 1, 1)
        off_y = offsets_tensor[:, 1].reshape(n_terms, 1, 1, 1, 1)

        u1_shifted = u1_exp + off_x
        u2_shifted = u2_exp + off_y

        # 3. Compute Kernel for all 9 terms at once
        term_cov = debiased_whittle_likelihood.cov_x_spatiotemporal_model_kernel(
            u1_shifted, u2_shifted, t, params
        ) # Result: (9, N1, N2, P, P)

        # 4. Weighted Sum
        w_tensor = w_flat.reshape(n_terms, 1, 1, 1, 1)
        cov_final = torch.sum(term_cov * w_tensor, dim=0) # (N1, N2, P, P)

        return cov_final

    @staticmethod
    def cn_bar_tapered(u1, u2, t, params, n1, n2, taper_autocorr_grid, delta1, delta2):
        """
        [Optimized] Computes c_Y(u) * c_gn(u) for batched inputs.
        u1, u2 : Grid indices (N1, N2, 1, 1)
        t      : Physical time lag (1, 1, P, P)
        """
        # 1. Physical Lags로 변환
        lag_u1 = u1 * delta1
        lag_u2 = u2 * delta2
        
        # 2. Covariance Calculation (Vectorized)
        cov_X_value = debiased_whittle_likelihood.cov_spatial_difference(lag_u1, lag_u2, t, params, delta1, delta2)
        
        # 3. Taper Value Lookup
        # u1, u2 are indices. We need to broadcast taper_autocorr_grid (2*N1-1, 2*N2-1) to (N1, N2, P, P)
        u1_idx = u1.long().squeeze(-1).squeeze(-1) # (N1, N2)
        u2_idx = u2.long().squeeze(-1).squeeze(-1) # (N1, N2)
        
        idx1 = torch.clamp(n1 - 1 + u1_idx, 0, 2 * n1 - 2)
        idx2 = torch.clamp(n2 - 1 + u2_idx, 0, 2 * n2 - 2)
        
        taper_val = taper_autocorr_grid[idx1, idx2] # (N1, N2)
        
        # Broadcast taper_val to (N1, N2, P, P)
        taper_val = taper_val.unsqueeze(-1).unsqueeze(-1)
        
        return cov_X_value * taper_val

    @staticmethod
    def expected_periodogram_fft_tapered(params, n1, n2, p_time, taper_autocorr_grid, delta1, delta2):
        """
        [Optimized] Fully vectorized expected periodogram calculation.
        Removes O(P^2) loops and O(9) spatial loops.
        """
        device = params.device if isinstance(params, torch.Tensor) else params[0].device
        if isinstance(params, list):
            params_tensor = torch.cat([p.to(device) for p in params])
        else:
            params_tensor = params.to(device)

        # 1. Spatial Grid Setup (N1, N2, 1, 1)
        u1_lags = torch.arange(n1, dtype=torch.float64, device=device)
        u2_lags = torch.arange(n2, dtype=torch.float64, device=device)
        u1_mesh, u2_mesh = torch.meshgrid(u1_lags, u2_lags, indexing='ij')
        
        u1_b = u1_mesh.unsqueeze(-1).unsqueeze(-1) # (N1, N2, 1, 1)
        u2_b = u2_mesh.unsqueeze(-1).unsqueeze(-1) # (N1, N2, 1, 1)

        # 2. Time Grid Setup (1, 1, P, P) -> The "No Loop" Magic
        t_vec = torch.arange(p_time, dtype=torch.float64, device=device)
        t_q, t_r = torch.meshgrid(t_vec, t_vec, indexing='ij')
        t_diff = (t_q - t_r).unsqueeze(0).unsqueeze(0) # (1, 1, P, P)
        
        # 3. Calculate Aliasing Terms (All at once)
        # Each term returns (N1, N2, P, P)
        term1 = debiased_whittle_likelihood.cn_bar_tapered(u1_b, u2_b, t_diff, 
                                                           params_tensor, n1, n2, taper_autocorr_grid, delta1, delta2)
        
        term2 = debiased_whittle_likelihood.cn_bar_tapered(u1_b - n1, u2_b, t_diff, 
                                                           params_tensor, n1, n2, taper_autocorr_grid, delta1, delta2)
        
        term3 = debiased_whittle_likelihood.cn_bar_tapered(u1_b, u2_b - n2, t_diff, 
                                                           params_tensor, n1, n2, taper_autocorr_grid, delta1, delta2)
        
        term4 = debiased_whittle_likelihood.cn_bar_tapered(u1_b - n1, u2_b - n2, t_diff,
                                                           params_tensor, n1, n2, taper_autocorr_grid, delta1, delta2)
        
        tilde_cn_tensor = term1 + term2 + term3 + term4 # (N1, N2, P, P)
        
        # 4. FFT (Spatial dims 0, 1 only)
        # Input is complex? No, covariance is real. Output is complex.
        # But for Whittle, we usually take Real part if symmetric, 
        # but let's stick to standard FFT logic.
        
        # Convert to complex for FFT (optional in PyTorch but safe)
        tilde_cn_tensor_c = tilde_cn_tensor.to(torch.complex128)
        
        fft_result = torch.fft.fft2(tilde_cn_tensor_c, dim=(0, 1))
        
        # Take Real part (Power Spectrum is real)
        fft_result_real = fft_result.real 
        
        normalization_factor = 1.0 / (4.0 * cmath.pi**2)
        result = fft_result_real * normalization_factor

        return result

    # =========================================================================
    # 4. Likelihood Calculation (Tapered)
    # =========================================================================
    
    @staticmethod
    def whittle_likelihood_loss_tapered(params, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2):
        """
        Whittle Likelihood Loss (AVERAGED) using data tapering.

        Separable filter [-1,1;1,-1] has |H(ω₁,ω₂)|² = 4sin²(ω₁/2)·4sin²(ω₂/2),
        which is zero on the entire ω₁=0 row AND ω₂=0 column.
        These frequencies are excluded via boolean mask BEFORE any matrix operations
        to avoid near-singular solves that propagate NaN across the full batch.
        Valid interior frequencies: (n1-1)×(n2-1)  (skip row 0 and col 0).
        """
        device = I_sample.device
        params_tensor = params.to(device)

        if torch.isnan(params_tensor).any() or torch.isinf(params_tensor).any():
            print("Warning: NaN/Inf detected in input parameters to likelihood.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        I_expected = debiased_whittle_likelihood.expected_periodogram_fft_tapered(
            params_tensor, n1, n2, p_time, taper_autocorr_grid,
            delta1, delta2
        )

        if torch.isnan(I_expected).any() or torch.isinf(I_expected).any():
            print("Warning: NaN/Inf returned from expected_periodogram calculation.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        # ── Axis exclusion: mask BEFORE any matrix op ─────────────────────────
        # I_expected and I_sample: (n1, n2, p, p). Skip ω₁=0 (row 0) and ω₂=0 (col 0).
        # Valid slice: rows 1..n1-1, cols 1..n2-1  → shape (n1-1, n2-1, p, p)
        if n1 > 1 and n2 > 1:
            I_exp_valid  = I_expected[1:, 1:]    # (n1-1, n2-1, p, p)
            I_samp_valid = I_sample[1:, 1:]
            num_terms    = (n1 - 1) * (n2 - 1)
        else:
            I_exp_valid  = I_expected
            I_samp_valid = I_sample
            num_terms    = n1 * n2

        if num_terms == 0:
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        eye_matrix   = torch.eye(p_time, dtype=torch.complex128, device=device)
        diag_vals    = torch.abs(I_exp_valid.diagonal(dim1=-2, dim2=-1))
        mean_diag_abs = diag_vals.mean().item() if diag_vals.numel() > 0 and not torch.isnan(diag_vals).all() else 1.0
        diag_load    = max(mean_diag_abs * 1e-8, 1e-9)
        I_exp_stable = I_exp_valid + eye_matrix * diag_load

        sign, logabsdet = torch.linalg.slogdet(I_exp_stable)
        if torch.any(sign.real <= 1e-9):
            log_det_term = torch.where(sign.real > 1e-9, logabsdet,
                                       torch.tensor(1e10, device=device, dtype=torch.float64))
        else:
            log_det_term = logabsdet

        if torch.isnan(I_samp_valid).any() or torch.isinf(I_samp_valid).any():
            print("Warning: NaN/Inf detected in I_sample (valid frequencies).")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        try:
            solved_term = torch.linalg.solve(I_exp_stable, I_samp_valid)
            trace_term  = torch.einsum('...ii->...', solved_term).real
        except torch.linalg.LinAlgError as e:
            print(f"Warning: LinAlgError during solve: {e}.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        if torch.isnan(trace_term).any() or torch.isinf(trace_term).any():
            print("Warning: NaN/Inf in trace_term. Returning NaN loss.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        likelihood_terms = log_det_term + trace_term  # (n1-1, n2-1)

        if torch.isnan(likelihood_terms).any():
            print("Warning: NaN in likelihood_terms. Returning NaN loss.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        avg_loss = likelihood_terms.sum() / num_terms

        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            print("Warning: NaN/Inf in final loss. Returning Inf penalty.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        return avg_loss


    # =========================================================================
    # 5. Training Loop & Helpers (💥 NEWLY ADDED 💥)
    # =========================================================================
    
    @staticmethod
    def get_printable_params_7param(p_list):
        """Helper to convert 7-param log list to natural scale for printing."""
        valid_tensors = [p for p in p_list if isinstance(p, torch.Tensor)]
        if not valid_tensors: return "Invalid params_list"
        p_cat = torch.cat([p.detach().clone().cpu() for p in valid_tensors])
        
        if len(p_cat) != 7:
            return f"Expected 7 params, got {len(p_cat)}."

        log_params = p_cat
        if torch.isnan(log_params).any() or torch.isinf(log_params).any():
            return "[NaN/Inf in log_params]"
            
        try:
            phi1 = torch.exp(log_params[0])
            phi2 = torch.exp(log_params[1]) # range_lon_inv
            phi3 = torch.exp(log_params[2]) # (range_lon / range_lat)^2
            phi4 = torch.exp(log_params[3]) # beta^2
            advec_lat = log_params[4]       # Not in log scale
            advec_lon = log_params[5]       # Not in log scale
            nugget = torch.exp(log_params[6])

            epsilon = 1e-12
            sigmasq = phi1 / (phi2 + epsilon)
            range_lon = 1.0 / (phi2 + epsilon)
            range_lat = 1.0 / (torch.sqrt(phi3 + epsilon) * phi2 + epsilon)
            range_time = range_lon / torch.sqrt(phi4 + epsilon)
            
            return (f"sigmasq: {sigmasq.item():.4f}, range_lat: {range_lat.item():.4f}, "
                    f"range_lon: {range_lon.item():.4f}, range_time: {range_time.item():.4f}, "
                    f"advec_lat: {advec_lat.item():.4f}, advec_lon: {advec_lon.item():.4f}, "
                    f"nugget: {nugget.item():.4f}")
        except Exception as e:
            return f"[Error in param conversion: {e}]"

    @staticmethod
    def get_phi_params_7param(log_params_list):
        """Helper to print 7-param reparameterized (phi-scale) params."""
        try:
            p_cat = torch.cat([p.detach().clone().cpu() for p in log_params_list])
            phi1 = torch.exp(p_cat[0])
            phi2 = torch.exp(p_cat[1])
            phi3 = torch.exp(p_cat[2])
            phi4 = torch.exp(p_cat[3])
            advec_lat = p_cat[4]
            advec_lon = p_cat[5]
            nugget = torch.exp(p_cat[6]) 
            
            return (f"phi1: {phi1.item():.4f}, phi2: {phi2.item():.4f}, phi3: {phi3.item():.4f}, "
                    f"phi4: {phi4.item():.4f}, advec_lat: {advec_lat.item():.4f}, advec_lon: {advec_lon.item():.4f}, "
                    f"nugget: {nugget.item():.4f}")
        except Exception:
            return "[Error in reparam conversion]"

    @staticmethod
    def get_raw_log_params_7param(log_params_list):
        """Helper to print the raw 7 params being optimized."""
        try:
            p_cat = torch.cat([p.detach().clone().cpu() for p in log_params_list])
            return (f"log_phi1: {p_cat[0].item():.4f}, log_phi2: {p_cat[1].item():.4f}, "
                    f"log_phi3: {p_cat[2].item():.4f}, log_phi4: {p_cat[3].item():.4f}, "
                    f"advec_lat: {p_cat[4].item():.4f}, advec_lon: {p_cat[5].item():.4f}, "
                    f"log_nugget: {p_cat[6].item():.4f}")
        except Exception:
            return "[Error in raw param conversion]"
    
    @staticmethod
    def run_lbfgs_tapered(params_list, optimizer, I_sample, n1, n2, p_time, taper_autocorr_grid, max_steps=50, device='cpu',grad_tol=1e-5):
        """Training loop using L-BFGS optimizer with improved convergence checks."""
        
        #params_list = [p.to(device) for p in params_list]
        # You should assume the params_list passed to the function is already on the correct device
        
        best_params_state = [p.detach().clone() for p in params_list]
        steps_completed = 0
        DELTA_LAT, DELTA_LON = 0.044, 0.063 
        
        loss_tol = 1e-12 
        
        best_loss = float('inf')
        prev_loss_item = float('inf') 

        I_sample_dev = I_sample.to(device)
        taper_autocorr_grid_dev = taper_autocorr_grid.to(device) 


        

        # (debiased_whittle.py 내부의 run_lbfgs_tapered 메소드 안)

        def closure():
            optimizer.zero_grad()
            params_tensor = torch.cat(params_list) 
            
            loss = debiased_whittle_likelihood.whittle_likelihood_loss_tapered(
                params_tensor, I_sample_dev, n1, n2, p_time, taper_autocorr_grid_dev, DELTA_LAT, DELTA_LON
            )
            
            # 1. Loss 자체가 NaN인 경우
            if not torch.isfinite(loss):
                print("⚠️ Loss is NaN/Inf inside closure.")
                # 여기서 에러를 던져야 메인 루프의 except로 바로 점프합니다.
                raise RuntimeError("Numerical Instability: Loss is NaN/Inf") 
            
            loss.backward()
    
            # 2. 기울기(Gradient)가 NaN인 경우
            for param in params_list:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    print(f"⚠️ Warning: NaN/Inf gradient detected.")
                    # 마찬가지로 즉시 에러 발생
                    raise RuntimeError("Numerical Instability: Gradient is NaN/Inf")
            
            return loss

        # --- End of closure ---

        for i in range(max_steps):
            steps_completed = i + 1
            
            loss = optimizer.step(closure)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {i+1}/{max_steps}: Loss is NaN/Inf. Stopping optimization.")
                break
                
            current_loss_item = loss.item()
            
            if current_loss_item < best_loss:
                params_valid = not any(torch.isnan(p.data).any() or torch.isinf(p.data).any() for p in params_list)
                if params_valid:
                    best_loss = current_loss_item
                    best_params_state = [p.detach().clone() for p in params_list]
            
            max_abs_grad = 0.0
            with torch.no_grad():
                for p in params_list:
                    if p.grad is not None:
                        max_abs_grad = max(max_abs_grad, p.grad.abs().item())

            print(f'--- Step {i+1}/{max_steps} ---')
            print(f' Loss: {current_loss_item:.6f} | Max Grad: {max_abs_grad:.6e}')
            print(f'  Params (Raw Log): {debiased_whittle_likelihood.get_raw_log_params_7param(params_list)}')
            
            loss_change = abs(current_loss_item - prev_loss_item)
            
            if i > 2: # Warmup period
                if max_abs_grad < grad_tol:
                    print(f"\n--- Converged on gradient norm (max|grad| < {grad_tol}) at step {i+1} ---")
                    break
                if loss_change < loss_tol:
                    print(f"\n--- Converged on loss change (change < {loss_tol}) at step {i+1} ---")
                    break
            
            prev_loss_item = current_loss_item

        print("\n--- Training Complete ---")
        if best_params_state is None:
            print("Training failed to find a valid model state.")
            return None, None, None, None, steps_completed

        final_natural_params_str = debiased_whittle_likelihood.get_printable_params_7param(best_params_state)
        final_phi_params_str = debiased_whittle_likelihood.get_phi_params_7param(best_params_state)
        final_raw_params_str = debiased_whittle_likelihood.get_raw_log_params_7param(best_params_state)
        final_loss_rounded = round(best_loss, 3) if best_loss != float('inf') else float('inf')

        print(f'\nFINAL BEST STATE ACHIEVED (during training):')
        print(f'Best Loss: {final_loss_rounded}')
        
        return final_natural_params_str, final_phi_params_str, final_raw_params_str, final_loss_rounded, steps_completed