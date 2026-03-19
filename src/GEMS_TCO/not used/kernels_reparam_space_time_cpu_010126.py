# Standard libraries
import sys
import time
import numpy as np
import torch
import warnings
from typing import Dict, Any, Callable, List, Tuple
from scipy.special import gamma

# Add your custom path
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)
sys.path.append("/cache/home/jl2815/tco")

# --- BASE CLASS ---
class SpatioTemporalModel:
    def __init__(self, smooth:float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, nns_map:Dict[str, Any], mm_cond_number: int):
        self.device = aggregated_data.device
        self.smooth = smooth
        self.smooth_tensor = torch.tensor(self.smooth, dtype=torch.float64, device=self.device)
        gamma_val = torch.tensor(gamma(self.smooth), dtype=torch.float64, device=self.device)
        self.matern_const = ( (2**(1-self.smooth)) / gamma_val )

        self.input_map = input_map
        self.aggregated_data = aggregated_data[:,:4]
        self.key_list = list(input_map.keys())
        self.size_per_hour = len(input_map[self.key_list[0]])
        self.mm_cond_number = mm_cond_number

        # Process NNS Map
        nns_map = list(nns_map) 
        for i in range(len(nns_map)):  
            tmp = np.delete(nns_map[i][:self.mm_cond_number], np.where(nns_map[i][:self.mm_cond_number] == -1))
            if tmp.size > 0:
                nns_map[i] = tmp
            else:
                nns_map[i] = []
        self.nns_map = nns_map

    def _convert_raw_params_to_interpretable(self, raw_params_list: List[float]) -> Dict[str, float]:
        """모든 하위 클래스에서 공유하는 파라미터 변환 함수"""
        try:
            phi1, phi2, phi3, phi4 = np.exp(raw_params_list[0:4])
            advec_lat, advec_lon = raw_params_list[4], raw_params_list[5]
            nugget = np.exp(raw_params_list[6])
            
            range_lon = 1.0 / phi2
            sigmasq = phi1 / phi2 
            range_lat = range_lon / np.sqrt(phi3)
            range_time = 1/(np.sqrt(phi4) * phi2) 

            return {
                "sigma_sq": sigmasq,
                "range_lon": range_lon,
                "range_lat": range_lat,
                "range_time": range_time,
                "advec_lat": advec_lat,
                "advec_lon": advec_lon,
                "nugget": nugget
            }
        except Exception as e:
            print(f"\nWarning: Could not convert raw params. Error: {e}")
            return {}
    
    # --- SINGLE MATRIX KERNEL ---
    def precompute_coords_aniso_STABLE(self, dist_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        phi3, phi4, advec_lat, advec_lon = dist_params
        
        # x columns: [lat, lon, val, time]
        u_lat_adv = x[:, 0] - advec_lat * x[:, 3]
        u_lon_adv = x[:, 1] - advec_lon * x[:, 3]
        u_t = x[:, 3]
        
        v_lat_adv = y[:, 0] - advec_lat * y[:, 3]
        v_lon_adv = y[:, 1] - advec_lon * y[:, 3]
        v_t = y[:, 3]

        delta_lat = u_lat_adv.unsqueeze(1) - v_lat_adv.unsqueeze(0)
        delta_lon = u_lon_adv.unsqueeze(1) - v_lon_adv.unsqueeze(0)
        delta_t   = u_t.unsqueeze(1)       - v_t.unsqueeze(0)

        dist_sq = (delta_lat.pow(2) * phi3) + delta_lon.pow(2) + (delta_t.pow(2) * phi4)
        return torch.sqrt(dist_sq + 1e-8)

    def matern_cov_aniso_STABLE_log_reparam(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        phi1, phi2, phi3, phi4 = np.exp(params[0:4].detach().cpu().numpy()) if not isinstance(params, torch.Tensor) else torch.exp(params[0:4])
        # Note: 위 라인은 tensor/numpy 호환성을 위해 안전장치를 둠. 기본적으로는 tensor 연산.
        phi1, phi2, phi3, phi4 = torch.exp(params[0]), torch.exp(params[1]), torch.exp(params[2]), torch.exp(params[3])
        nugget = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        
        sigmasq = phi1 / phi2
        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance = self.precompute_coords_aniso_STABLE(dist_params, x, y)
        
        cov = sigmasq * torch.exp(-distance * phi2)
        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8) 
        return cov

    def full_likelihood_avg(self, params: torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, covariance_function: Callable) -> torch.Tensor:
        """
        [수정됨] Vecchia Heads 계산 시에도 6 Features를 사용하도록 통일
        """
        input_data = input_data.to(torch.float64)
        y = y.to(torch.float64)
        N = input_data.shape[0]
        if N == 0: return torch.tensor(0.0, device=self.device, dtype=torch.float64)
        
        # 1. Covariance Calculation
        cov_matrix = covariance_function(params=params, y=input_data, x=input_data)
        
        try:
            jitter = torch.eye(cov_matrix.shape[0], device=self.device, dtype=torch.float64) * 1e-6
            L = torch.linalg.cholesky(cov_matrix + jitter)
        except torch.linalg.LinAlgError:
            return torch.tensor(torch.inf, device=params.device, dtype=params.dtype)
        
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        
        # 2. Design Matrix Construction (6 Features: Int, Lat, Lon, Time, LxT, LonxT)
        lat  = input_data[:, 0]
        lon  = input_data[:, 1]
        time_val = input_data[:, 3]
        ones = torch.ones(N, device=self.device, dtype=torch.float64)
        
        locs = torch.stack([
            ones, lat, lon, time_val, lat*time_val, lon*time_val
        ], dim=1) # Shape (N, 6)
        
        if y.dim() == 1: y_col = y.unsqueeze(-1)
        else: y_col = y
        
        # 3. Forward Solve (Whitening)
        combined_rhs = torch.cat((locs, y_col), dim=1)
        Z_combined = torch.linalg.solve_triangular(L, combined_rhs, upper=False)
        
        Z_X = Z_combined[:, :6] # 6 features
        Z_y = Z_combined[:, 6:]

        # 4. GLS Beta
        tmp1 = torch.matmul(Z_X.T, Z_X)
        tmp2 = torch.matmul(Z_X.T, Z_y)
        
        try:
            jitter_beta = torch.eye(tmp1.shape[0], device=self.device, dtype=torch.float64) * 1e-8
            beta = torch.linalg.solve(tmp1 + jitter_beta, tmp2)
        except torch.linalg.LinAlgError:
            return torch.tensor(torch.inf, device=locs.device, dtype=locs.dtype)

        # 5. Residuals & NLL
        Z_residual = Z_y - torch.matmul(Z_X, beta)
        quad_form = torch.matmul(Z_residual.T, Z_residual)

        neg_log_lik_sum = 0.5 * (log_det + quad_form.squeeze())
        return neg_log_lik_sum / N


# --- BATCHED GPU/CPU CLASS ---
class VecchiaBatched(SpatioTemporalModel):
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        
        self.device = torch.device('cpu') # CPU Target
        self.nheads = nheads
        self.max_neighbors = mm_cond_number 
        self.is_precomputed = False
        
        self.X_batch = None
        self.Y_batch = None
        self.Locs_batch = None
        self.Heads_data = None 

    def precompute_conditioning_sets(self):
        print("Pre-computing for CPU (NumPy Native)...", end=" ")
        
        key_list = list(self.input_map.keys())
        cut_line = self.nheads
        
        # 1. Heads Data
        heads_list = []
        for key in key_list:
            day_data = self.input_map[key]
            if isinstance(day_data, torch.Tensor): day_data = day_data.numpy()
            heads_list.append(day_data[:cut_line])
        
        self.Heads_data = torch.from_numpy(np.concatenate(heads_list, axis=0)).to(dtype=torch.float64)

        # 2. Tails Data
        tasks = []
        for time_idx, key in enumerate(key_list):
            day_data = self.input_map[key]
            indices = range(cut_line, len(day_data))
            for idx in indices:
                tasks.append((time_idx, idx))

        total_vecchia_points = len(tasks)
        max_storage_dim = (self.max_neighbors + 1) * 3
        
        X_batch_np = np.full((total_vecchia_points, max_storage_dim, 3), 1e6, dtype=np.float64)
        Y_batch_np = np.zeros((total_vecchia_points, max_storage_dim, 1), dtype=np.float64)
        
        # Locs_batch needs to hold 6 features
        Locs_batch_np = np.zeros((total_vecchia_points, max_storage_dim, 6), dtype=np.float64)

        for i, (time_idx, index) in enumerate(tasks):
            current_np = self.input_map[key_list[time_idx]]
            if isinstance(current_np, torch.Tensor): current_np = current_np.numpy()
            
            current_row = current_np[index].reshape(1, -1) 
            mm_neighbors = self.nns_map[index]
            past = list(mm_neighbors)
            
            data_list = []
            if past: data_list.append(current_np[past])
            if time_idx > 0: 
                prev = self.input_map[key_list[time_idx - 1]]
                if isinstance(prev, torch.Tensor): prev = prev.numpy()
                data_list.append(prev[past + [index], :])
            if time_idx > 1: 
                prev2 = self.input_map[key_list[time_idx - 2]]
                if isinstance(prev2, torch.Tensor): prev2 = prev2.numpy()
                data_list.append(prev2[past + [index], :])

            if data_list:
                neighbors_block = np.vstack(data_list)
            else:
                neighbors_block = np.empty((0, current_row.shape[1]))

            combined_data = np.concatenate([neighbors_block, current_row], axis=0)
            actual_len = combined_data.shape[0]
            start_slot = max_storage_dim - actual_len
            
            X_batch_np[i, start_slot:, 0] = combined_data[:, 0]
            X_batch_np[i, start_slot:, 1] = combined_data[:, 1]
            X_batch_np[i, start_slot:, 2] = combined_data[:, 3]
            Y_batch_np[i, start_slot:, 0] = combined_data[:, 2]
            
            # --- Feature Engineering (6 Features) ---
            lat = combined_data[:, 0]
            lon = combined_data[:, 1]
            t   = combined_data[:, 3]
            
            Locs_batch_np[i, start_slot:, 0] = 1.0 
            Locs_batch_np[i, start_slot:, 1] = lat
            Locs_batch_np[i, start_slot:, 2] = lon
            Locs_batch_np[i, start_slot:, 3] = t
            Locs_batch_np[i, start_slot:, 4] = lat * t
            Locs_batch_np[i, start_slot:, 5] = lon * t

        self.X_batch = torch.from_numpy(X_batch_np)
        self.Y_batch = torch.from_numpy(Y_batch_np)
        self.Locs_batch = torch.from_numpy(Locs_batch_np)
        
        self.is_precomputed = True
        print(f"Done. CPU Tensors Ready. Size: {self.X_batch.shape}")

    def batched_manual_dist(self, dist_params, x_batch):
        phi3, phi4, advec_lat, advec_lon = dist_params
        x_lat, x_lon, x_time = x_batch[:, :, 0], x_batch[:, :, 1], x_batch[:, :, 2]
        
        u_lat = x_lat - advec_lat * x_time
        u_lon = x_lon - advec_lon * x_time
        
        d_lat = u_lat.unsqueeze(2) - u_lat.unsqueeze(1)
        d_lon = u_lon.unsqueeze(2) - u_lon.unsqueeze(1)
        d_t   = x_time.unsqueeze(2) - x_time.unsqueeze(1)

        dist_sq = (d_lat.pow(2) * phi3) + (d_lon.pow(2)) + (d_t.pow(2) * phi4)
        return torch.sqrt(dist_sq + 1e-8)

    def matern_cov_batched(self, params, x_batch):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        
        cov = (phi1 / phi2) * torch.exp(-self.batched_manual_dist(dist_params, x_batch) * phi2)
        
        B, N, _ = x_batch.shape
        identity = torch.eye(N, dtype=torch.float64).unsqueeze(0).expand(B, N, N)
        return cov + identity * (nugget + 1e-6)

    def vecchia_batched_likelihood(self, params):
        if not self.is_precomputed: raise RuntimeError("Run precompute first!")
        
        def adapter_cov_func(params, x, y):
            return self.matern_cov_aniso_STABLE_log_reparam(params, x, y)

        head_nll_sum = 0.0
        if self.Heads_data.shape[0] > 0:
            # Note: Now uses the updated 6-feature full_likelihood_avg
            head_nll_avg = self.full_likelihood_avg(params, self.Heads_data, self.Heads_data[:, 2], adapter_cov_func)
            head_nll_sum = head_nll_avg * self.Heads_data.shape[0]

        # Tails (Batched)
        cov_batch = self.matern_cov_batched(params, self.X_batch)
        
        try:
            L_batch = torch.linalg.cholesky(cov_batch)
        except torch.linalg.LinAlgError:
            return torch.tensor(float('inf'), dtype=torch.float64)
            
        Z_locs = torch.linalg.solve_triangular(L_batch, self.Locs_batch, upper=False)
        Z_y    = torch.linalg.solve_triangular(L_batch, self.Y_batch, upper=False)
        
        Xt_Cinv_X = torch.bmm(Z_locs.transpose(1, 2), Z_locs)
        Xt_Cinv_y = torch.bmm(Z_locs.transpose(1, 2), Z_y)
        
        jitter = torch.eye(6, dtype=torch.float64).unsqueeze(0) * 1e-8 # 6 features
        beta = torch.linalg.solve(Xt_Cinv_X + jitter, Xt_Cinv_y)
        
        z_residuals = Z_y - torch.bmm(Z_locs, beta)
        z_target = z_residuals[:, -1, :] 
        sigma_cond = L_batch[:, -1, -1]
        
        log_det = 2 * torch.log(sigma_cond)
        quad_form = z_target.squeeze() ** 2
        
        vecchia_nll_sum = 0.5 * (log_det + quad_form).sum()

        total_nll = head_nll_sum + vecchia_nll_sum
        total_count = self.Heads_data.shape[0] + self.X_batch.shape[0]
        
        return total_nll / total_count


# --- FITTING CLASS ---
class fit_vecchia_lbfgs(VecchiaBatched): 
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)

    def set_optimizer(self, param_groups, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100):
        optimizer = torch.optim.LBFGS(
            param_groups, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size
        )
        return optimizer

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor], optimizer: torch.optim.LBFGS, max_steps: int = 50, grad_tol: float = 1e-7):
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print("--- Starting Batched L-BFGS Optimization (CPU) ---")

        def closure():
            optimizer.zero_grad()
            params = torch.stack(params_list)
            loss = self.vecchia_batched_likelihood(params)
            loss.backward()
            return loss

        for i in range(max_steps):
            loss = optimizer.step(closure)
            
            max_abs_grad = 0.0
            grad_values = []
            with torch.no_grad():
                for p in params_list:
                    if p.grad is not None:
                        grad_values.append(abs(p.grad.item())) 
                if grad_values: max_abs_grad = max(grad_values)

                print(f'--- Step {i+1}/{max_steps} / Loss: {loss.item():.6f} ---')
            
            if max_abs_grad < grad_tol:
                print(f"\nConverged at step {i+1}")
                break

        final_params_tensor = torch.stack(params_list).detach()
        final_raw_params_list = [p.item() for p in final_params_tensor]
        final_loss = loss.item()
        
        interpretable = self._convert_raw_params_to_interpretable(final_raw_params_list)
        print("Final Interpretable Params:", interpretable)
        
        return final_raw_params_list + [final_loss], i

# --- FULL LIKELIHOOD CLASS ---
class FullLikelihoodCPU(SpatioTemporalModel):
    def __init__(self, smooth: float, input_map: Dict[str, Any], aggregated_data: torch.Tensor):
        super().__init__(smooth, input_map, aggregated_data, nns_map=[], mm_cond_number=0)
        self.device = torch.device('cpu')
        
        # 1. Integrate Data
        all_pts = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in input_map.values()]
        self.full_data = torch.cat(all_pts, dim=0).to(self.device, dtype=torch.float64)
        
        # 2. Design Matrix (6 Features)
        N = self.full_data.shape[0]
        lat, lon, time_val = self.full_data[:, 0], self.full_data[:, 1], self.full_data[:, 3]
        self.X_design = torch.stack([
            torch.ones(N, dtype=torch.float64), lat, lon, time_val, lat*time_val, lon*time_val
        ], dim=1)
        self.y = self.full_data[:, 2].unsqueeze(-1)
        
        print(f"✅ Full Likelihood initialized with {N} points.")

    def compute_nll(self, params: torch.Tensor) -> torch.Tensor:
        cov = self.matern_cov_aniso_STABLE_log_reparam(params, self.full_data, self.full_data)
        
        try:
            L = torch.linalg.cholesky(cov + torch.eye(cov.shape[0]) * 1e-7)
        except torch.linalg.LinAlgError:
            return torch.tensor(float('inf'))

        Z_X = torch.linalg.solve_triangular(L, self.X_design, upper=False)
        Z_y = torch.linalg.solve_triangular(L, self.y, upper=False)
        
        XT_Sinv_X = Z_X.T @ Z_X
        XT_Sinv_y = Z_X.T @ Z_y
        beta = torch.linalg.solve(XT_Sinv_X + torch.eye(6)*1e-8, XT_Sinv_y)
        
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        res = Z_y - Z_X @ beta
        quad_form = res.T @ res
        
        nll = 0.5 * (log_det + quad_form.squeeze())
        return nll / self.full_data.shape[0]