from scipy.special import gamma
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import time


# --- SpatioTemporalModel Class ---
class SpatioTemporalModel:
    def __init__(self, smooth: float, input_map: Dict[str, Any], nns_map: Dict[str, Any], mm_cond_number: int):
        self.smooth = smooth
        self.mm_cond_number = mm_cond_number
        self.input_map = input_map
        self.key_list = list(input_map.keys())

        gamma_val = torch.tensor(gamma(self.smooth), dtype=torch.float64)
        self.matern_const = (2 ** (1 - self.smooth)) / gamma_val

        # Process NNS Map: remove -1 padding
        nns_map = list(nns_map)
        for i in range(len(nns_map)):
            tmp = np.delete(nns_map[i], np.where(nns_map[i] == -1))
            nns_map[i] = tmp if tmp.size > 0 else np.array([], dtype=np.int64)
        self.nns_map = nns_map

    def precompute_coords_aniso_STABLE(self, dist_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates anisotropic space-time distance for Head points (Exact GP)."""
        phi3, phi4, advec_lat, advec_lon = dist_params

        u_vec = torch.stack([x[:, 0] - advec_lat * x[:, 3],
                             x[:, 1] - advec_lon * x[:, 3],
                             x[:, 3]], dim=1)
        v_vec = torch.stack([y[:, 0] - advec_lat * y[:, 3],
                             y[:, 1] - advec_lon * y[:, 3],
                             y[:, 3]], dim=1)

        one = torch.ones(1, device=x.device, dtype=phi3.dtype)
        weights = torch.stack([phi3.view(1), one, phi4.view(1)]).view(-1)

        u_sq = (u_vec.pow(2) * weights).sum(dim=1, keepdim=True)
        v_sq = (v_vec.pow(2) * weights).sum(dim=1, keepdim=True)
        uv   = (u_vec * weights) @ v_vec.T

        return torch.sqrt((u_sq - 2 * uv + v_sq.T).clamp(min=1e-8))

    def matern_cov_aniso_STABLE_log_reparam(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget    = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        sigmasq   = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance    = self.precompute_coords_aniso_STABLE(dist_params, x, y)
        cov         = sigmasq * torch.exp(-distance * phi2)

        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8)
        return cov


# --- Batched Vecchia Class ---
class VecchiaBatched(SpatioTemporalModel):
    def __init__(self, smooth: float, input_map: Dict[str, Any],
                 nns_map: Dict[str, Any], mm_cond_number: int, nheads: int,
                 limit_A: int = 8, limit_B: int = 8, limit_C: int = 8, daily_stride: int = 8):
        super().__init__(smooth, input_map, nns_map, mm_cond_number)

        first_val = next(iter(input_map.values()))
        self.device = first_val.device if isinstance(first_val, torch.Tensor) else torch.device('cpu')
        self.nheads       = nheads
        self.limit_A      = limit_A
        self.limit_B      = limit_B
        self.limit_C      = limit_C
        self.daily_stride = daily_stride

        # Feature 개수: 9개 [Intercept, Lat_centered, D1~D7]
        self.n_features   = 9
        self.is_precomputed = False
        self.lat_mean_val   = 0.0

        self.X_A = self.Y_A = self.Locs_A = None
        self.X_AB = self.Y_AB = self.Locs_AB = None
        self.X_ABC = self.Y_ABC = self.Locs_ABC = None
        self.Heads_data = None

    def batched_manual_dist(self, dist_params, x_batch):
        phi3, phi4, advec_lat, advec_lon = dist_params
        x_lat  = x_batch[:, :, 0] - advec_lat * x_batch[:, :, 2]
        x_lon  = x_batch[:, :, 1] - advec_lon * x_batch[:, :, 2]
        x_time = x_batch[:, :, 2]

        d_lat = x_lat.unsqueeze(2)  - x_lat.unsqueeze(1)
        d_lon = x_lon.unsqueeze(2)  - x_lon.unsqueeze(1)
        d_t   = x_time.unsqueeze(2) - x_time.unsqueeze(1)

        return torch.sqrt((d_lat.pow(2) * phi3) + d_lon.pow(2) + (d_t.pow(2) * phi4) + 1e-8)

    def matern_cov_batched(self, params, x_batch):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget      = torch.exp(params[6])
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])

        cov = (phi1 / phi2) * torch.exp(-self.batched_manual_dist(dist_params, x_batch) * phi2)
        B, N, _ = x_batch.shape
        eye = torch.eye(N, device=self.device, dtype=torch.float64).unsqueeze(0).expand(B, N, N)
        return cov + eye * (nugget + 1e-6)

    def precompute_conditioning_sets(self):
        limit_A, limit_B, limit_C = self.limit_A, self.limit_B, self.limit_C
        daily_stride = self.daily_stride

        max_dim_A   = limit_A
        max_dim_AB  = limit_A + (limit_B + 1)
        max_dim_ABC = limit_A + (limit_B + 1) + (limit_C + 1)

        n_stored = next((len(m) for m in self.nns_map if len(m) > 0), 0)
        print(f"🚀 Pre-computing 3-group Vecchia "
              f"[A={max_dim_A}, AB={max_dim_AB}, ABC={max_dim_ABC}, stored={n_stored}]...", end=" ")

        # 데이터 통합
        all_data_list = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                         for d in self.input_map.values()]
        Real_Data  = torch.cat(all_data_list, dim=0).to(self.device, dtype=torch.float32)
        n_real, num_cols = Real_Data.shape

        # NaN 마스크
        is_nan_real = torch.isnan(Real_Data[:, 2])
        valid_lats  = Real_Data[~is_nan_real, 0]
        self.lat_mean_val = valid_lats.mean().item() if valid_lats.numel() > 0 else Real_Data[:, 0].mean().item()
        print(f"[Mean Lat: {self.lat_mean_val:.4f}]", end=" ")
        is_nan_mask_np = is_nan_real.cpu().numpy()

        # 슬롯별 고유 dummy 블록 (NaN 지역 안전망)
        n_dummies  = max_dim_ABC
        dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float32)
        for k in range(n_dummies):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        Full_Data      = torch.cat([Real_Data, dummy_block], dim=0)
        dummy_start    = n_real
        is_nan_mask_np = np.append(is_nan_mask_np, np.zeros(n_dummies, dtype=bool))

        # 배치 인덱스 구성
        key_list      = list(self.input_map.keys())
        day_lengths   = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        n_time_steps  = len(key_list)
        use_set_C     = daily_stride < n_time_steps

        heads_indices  = []
        batch_list_A   = []
        batch_list_AB  = []
        batch_list_ABC = []

        def add_valid_neighbors(indices_to_check, current_indices, cap):
            count = 0
            for idx in indices_to_check:
                if count >= cap:
                    break
                if not is_nan_mask_np[idx]:
                    current_indices.append(idx)
                    count += 1

        for time_idx, key in enumerate(key_list):
            day_len = day_lengths[time_idx]
            offset  = cumulative_len[time_idx]

            # Heads
            for local_idx in range(min(day_len, self.nheads)):
                idx = offset + local_idx
                if not is_nan_mask_np[idx]:
                    heads_indices.append(idx)
            if self.nheads >= day_len:
                continue

            for local_idx in range(self.nheads, day_len):
                target_idx = offset + local_idx
                if is_nan_mask_np[target_idx]:
                    continue

                current_indices = []

                # Set A: 현재 시점 t 공간 이웃
                add_valid_neighbors(
                    (offset + self.nns_map[local_idx]).tolist(), current_indices, cap=limit_A)

                has_B = time_idx > 0
                has_C = time_idx >= daily_stride

                # Set B: t-1, independent filtering
                if has_B:
                    prev_off = cumulative_len[time_idx - 1]
                    prev_len = day_lengths[time_idx - 1]
                    if local_idx < prev_len:
                        add_valid_neighbors([prev_off + local_idx], current_indices, cap=1)
                    nbs = self.nns_map[local_idx]
                    add_valid_neighbors(
                        (prev_off + nbs[nbs < prev_len]).tolist(), current_indices, cap=limit_B)

                # Set C: t-daily_stride, independent filtering
                if has_C:
                    pd_idx = time_idx - daily_stride
                    pd_off = cumulative_len[pd_idx]
                    pd_len = day_lengths[pd_idx]
                    if local_idx < pd_len:
                        add_valid_neighbors([pd_off + local_idx], current_indices, cap=1)
                    nbs = self.nns_map[local_idx]
                    add_valid_neighbors(
                        (pd_off + nbs[nbs < pd_len]).tolist(), current_indices, cap=limit_C)

                # 그룹 결정 및 dummy 패딩
                if has_C:
                    max_d, target_list = max_dim_ABC, batch_list_ABC
                elif has_B:
                    max_d, target_list = max_dim_AB, batch_list_AB
                else:
                    max_d, target_list = max_dim_A, batch_list_A

                n_valid = len(current_indices)
                if n_valid < max_d:
                    row = [dummy_start + k for k in range(max_d - n_valid)] + current_indices
                else:
                    row = current_indices[-max_d:]
                target_list.append(row)

        # 그룹별 GPU 텐서 빌드
        heads_tensor    = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
        self.Heads_data = (Full_Data[heads_tensor].contiguous().to(torch.float64)
                           if len(heads_indices) > 0
                           else torch.empty(0, num_cols, device=self.device, dtype=torch.float64))

        def build_tensors(idx_list, max_d):
            if not idx_list:
                return None, None, None, None, None
            T    = torch.tensor(idx_list, device=self.device, dtype=torch.long)
            G    = Full_Data[T]
            X    = G[..., [0, 1, 3]].contiguous().to(torch.float64)
            Y    = G[..., 2].unsqueeze(-1).contiguous().to(torch.float64)
            ones = torch.ones_like(G[..., 0]).unsqueeze(-1)
            lat  = (G[..., 0] - self.lat_mean_val).unsqueeze(-1)
            dums = G[..., 4:11]
            Locs = torch.cat([ones, lat, dums], dim=-1).contiguous().to(torch.float64)
            is_dummy = (T >= dummy_start).unsqueeze(-1)
            Locs = Locs.masked_fill(is_dummy, 0.0)
            Y    = Y.masked_fill(is_dummy, 0.0)
            return X, Y, Locs, T, is_dummy

        self.X_A,   self.Y_A,   self.Locs_A,   self._T_A,   self._is_dummy_A   = build_tensors(batch_list_A,   max_dim_A)
        self.X_AB,  self.Y_AB,  self.Locs_AB,  self._T_AB,  self._is_dummy_AB  = build_tensors(batch_list_AB,  max_dim_AB)
        self.X_ABC, self.Y_ABC, self.Locs_ABC, self._T_ABC, self._is_dummy_ABC = build_tensors(batch_list_ABC, max_dim_ABC)

        # Store metadata for fast Y-only refresh (refresh_y_from_input_map)
        self._heads_tensor_stored = heads_tensor if len(heads_indices) > 0 else None
        self._dummy_start_stored  = dummy_start
        self._n_real_stored       = n_real
        self._n_dummies_stored    = n_dummies

        self.n_tails = len(batch_list_A) + len(batch_list_AB) + len(batch_list_ABC)
        self.is_precomputed = True
        print(f"[Set C: {use_set_C}] ✅ Done. "
              f"(Heads: {len(heads_indices)}, "
              f"Tails A/AB/ABC: {len(batch_list_A)}/{len(batch_list_AB)}/{len(batch_list_ABC)})")

    def refresh_y_from_input_map(self):
        """Update only Y tensors using current input_map values.

        Call this after updating self.input_map when the spatial structure
        (locations, conditioning sets) is unchanged but observation values differ
        — e.g. in GIM bootstrap loops. Much faster than full re-precompute.
        """
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first!")
        all_data = torch.cat(
            [torch.from_numpy(d) if isinstance(d, np.ndarray) else d
             for d in self.input_map.values()], dim=0
        ).to(self.device, dtype=torch.float32)
        # Append zeros for dummy rows
        y_full = torch.cat([
            all_data[:, 2],
            torch.zeros(self._n_dummies_stored, device=self.device, dtype=torch.float32)
        ])  # shape [n_real + n_dummies]
        # Refresh Heads
        if self._heads_tensor_stored is not None and self._heads_tensor_stored.numel() > 0:
            self.Heads_data[:, 2] = y_full[self._heads_tensor_stored].to(torch.float64)
        # Refresh tail Y tensors
        for T, is_dummy, attr in [
            (self._T_A,   self._is_dummy_A,   'Y_A'),
            (self._T_AB,  self._is_dummy_AB,  'Y_AB'),
            (self._T_ABC, self._is_dummy_ABC, 'Y_ABC'),
        ]:
            if T is not None:
                Y = y_full[T].unsqueeze(-1).to(torch.float64)
                Y = Y.masked_fill(is_dummy, 0.0)
                setattr(self, attr, Y.contiguous())

    def vecchia_batched_likelihood(self, params):
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first!")

        XT_Sinv_X = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y = torch.zeros((self.n_features, 1),               device=self.device, dtype=torch.float64)
        yT_Sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det   = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        # PART 1: Heads (Exact GP)
        if self.Heads_data.shape[0] > 0:
            ones  = torch.ones((self.Heads_data.shape[0], 1), device=self.device, dtype=torch.float64)
            lat   = (self.Heads_data[:, 0] - self.lat_mean_val).unsqueeze(-1)
            dums  = self.Heads_data[:, 4:11]
            X_h   = torch.cat([ones, lat, dums], dim=1)
            y_h   = self.Heads_data[:, 2].unsqueeze(-1)

            cov = self.matern_cov_aniso_STABLE_log_reparam(params, self.Heads_data, self.Heads_data)
            try:
                L = torch.linalg.cholesky(cov)
            except torch.linalg.LinAlgError:
                with torch.no_grad():
                    phi2      = torch.exp(params[1])
                    range_lon = 1.0 / phi2
                    range_lat = range_lon / torch.exp(params[2]).sqrt()
                    range_t   = range_lon / torch.exp(params[3]).sqrt()
                    nugget    = torch.exp(params[6])
                    sigmasq   = torch.exp(params[0]) / phi2
                    print(f"[Cholesky FAIL | Heads] "
                          f"sigmasq={sigmasq.item():.4f}  "
                          f"range_lon={range_lon.item():.4f}  range_lat={range_lat.item():.4f}  "
                          f"range_t={range_t.item():.4f}  nugget={nugget.item():.4e}")
                return torch.tensor(float('inf'), device=self.device)

            log_det   += 2 * torch.sum(torch.log(torch.diag(L)))
            Z_X = torch.linalg.solve_triangular(L, X_h, upper=False)
            Z_y = torch.linalg.solve_triangular(L, y_h, upper=False)
            XT_Sinv_X += Z_X.T @ Z_X
            XT_Sinv_y += Z_X.T @ Z_y
            yT_Sinv_y += (Z_y.T @ Z_y).squeeze()

        # PART 2: Tails (3-group sequential batching)
        chunk_size = 4096
        for X_b, Y_b, Locs_b in [(self.X_A,   self.Y_A,   self.Locs_A),
                                   (self.X_AB,  self.Y_AB,  self.Locs_AB),
                                   (self.X_ABC, self.Y_ABC, self.Locs_ABC)]:
            if X_b is None or X_b.shape[0] == 0:
                continue
            for start in range(0, X_b.shape[0], chunk_size):
                end = min(start + chunk_size, X_b.shape[0])
                cov_chunk = self.matern_cov_batched(params, X_b[start:end])
                try:
                    L_chunk = torch.linalg.cholesky(cov_chunk)
                except torch.linalg.LinAlgError:
                    with torch.no_grad():
                        phi2      = torch.exp(params[1])
                        range_lon = 1.0 / phi2
                        range_lat = range_lon / torch.exp(params[2]).sqrt()
                        range_t   = range_lon / torch.exp(params[3]).sqrt()
                        nugget    = torch.exp(params[6])
                        sigmasq   = torch.exp(params[0]) / phi2
                        print(f"[Cholesky FAIL | Tails] "
                              f"sigmasq={sigmasq.item():.4f}  "
                              f"range_lon={range_lon.item():.4f}  range_lat={range_lat.item():.4f}  "
                              f"range_t={range_t.item():.4f}  nugget={nugget.item():.4e}")
                    return torch.tensor(float('inf'), device=self.device)

                Z_locs = torch.linalg.solve_triangular(L_chunk, Locs_b[start:end], upper=False)
                Z_y    = torch.linalg.solve_triangular(L_chunk, Y_b[start:end],    upper=False)

                u_X = Z_locs[:, -1, :]
                u_y = Z_y[:, -1, :]
                log_det   += 2 * torch.sum(torch.log(L_chunk[:, -1, -1]))
                XT_Sinv_X += u_X.T @ u_X
                XT_Sinv_y += u_X.T @ u_y
                yT_Sinv_y += (u_y.T @ u_y).squeeze()

        # PART 3: Solve beta (GLS)
        jitter = torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-6
        try:
            beta = torch.linalg.solve(XT_Sinv_X + jitter, XT_Sinv_y)
        except torch.linalg.LinAlgError:
            with torch.no_grad():
                phi2      = torch.exp(params[1])
                range_lon = 1.0 / phi2
                range_lat = range_lon / torch.exp(params[2]).sqrt()
                range_t   = range_lon / torch.exp(params[3]).sqrt()
                nugget    = torch.exp(params[6])
                sigmasq   = torch.exp(params[0]) / phi2
                print(f"[Cholesky FAIL | GLS beta] "
                      f"sigmasq={sigmasq.item():.4f}  "
                      f"range_lon={range_lon.item():.4f}  range_lat={range_lat.item():.4f}  "
                      f"range_t={range_t.item():.4f}  nugget={nugget.item():.4e}")
            return torch.tensor(float('inf'), device=self.device)

        quad = yT_Sinv_y - 2 * (beta.T @ XT_Sinv_y) + (beta.T @ XT_Sinv_X @ beta)
        total_N = self.Heads_data.shape[0] + self.n_tails

        return 0.5 * (log_det + quad.squeeze()) / total_N

    def get_gls_beta(self, params):
        """Run the GLS forward pass and return the beta estimate.

        Performs the same Cholesky accumulation as vecchia_batched_likelihood but
        returns beta = (X' Σ⁻¹ X)⁻¹ X' Σ⁻¹ y instead of the scalar likelihood.
        Used to fix beta before computing per-unit observed-J contributions.
        """
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first!")

        XT_Sinv_X = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y = torch.zeros((self.n_features, 1),               device=self.device, dtype=torch.float64)

        if self.Heads_data.shape[0] > 0:
            ones = torch.ones((self.Heads_data.shape[0], 1), device=self.device, dtype=torch.float64)
            lat  = (self.Heads_data[:, 0] - self.lat_mean_val).unsqueeze(-1)
            dums = self.Heads_data[:, 4:11]
            X_h  = torch.cat([ones, lat, dums], dim=1)
            y_h  = self.Heads_data[:, 2].unsqueeze(-1)
            cov  = self.matern_cov_aniso_STABLE_log_reparam(params, self.Heads_data, self.Heads_data)
            L    = torch.linalg.cholesky(cov)
            Z_X  = torch.linalg.solve_triangular(L, X_h, upper=False)
            Z_y  = torch.linalg.solve_triangular(L, y_h, upper=False)
            XT_Sinv_X += Z_X.T @ Z_X
            XT_Sinv_y += Z_X.T @ Z_y

        chunk_size = 4096
        for X_b, Y_b, Locs_b in [(self.X_A,   self.Y_A,   self.Locs_A),
                                   (self.X_AB,  self.Y_AB,  self.Locs_AB),
                                   (self.X_ABC, self.Y_ABC, self.Locs_ABC)]:
            if X_b is None or X_b.shape[0] == 0:
                continue
            for start in range(0, X_b.shape[0], chunk_size):
                end = min(start + chunk_size, X_b.shape[0])
                cov_chunk = self.matern_cov_batched(params, X_b[start:end])
                L_chunk   = torch.linalg.cholesky(cov_chunk)
                Z_locs    = torch.linalg.solve_triangular(L_chunk, Locs_b[start:end], upper=False)
                Z_y       = torch.linalg.solve_triangular(L_chunk, Y_b[start:end],    upper=False)
                XT_Sinv_X += Z_locs[:, -1, :].T @ Z_locs[:, -1, :]
                XT_Sinv_y += Z_locs[:, -1, :].T @ Z_y[:, -1, :]

        jitter = torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-6
        return torch.linalg.solve(XT_Sinv_X + jitter, XT_Sinv_y)  # (n_features, 1)

    def vecchia_per_unit_nll_terms(self, params, beta):
        """Return per-unit NLL contributions with beta fixed.

        Each term: LL_i = log(σ_i) + 0.5 * r_i²
        where σ_i is the conditional standard deviation and r_i is the
        conditional normalized residual (both in the whitened Cholesky space).

        Summing over all units: Σ LL_i = 0.5 * (log_det + quad)
        so  nll = (1/N) Σ LL_i  matches vecchia_batched_likelihood.

        Returns: 1-D tensor of shape (total_N,). Differentiable w.r.t. params.
        """
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first!")

        terms = []

        # --- Heads (exact GP block) ---
        if self.Heads_data.shape[0] > 0:
            ones = torch.ones((self.Heads_data.shape[0], 1), device=self.device, dtype=torch.float64)
            lat  = (self.Heads_data[:, 0] - self.lat_mean_val).unsqueeze(-1)
            dums = self.Heads_data[:, 4:11]
            X_h  = torch.cat([ones, lat, dums], dim=1)
            y_h  = self.Heads_data[:, 2].unsqueeze(-1)
            cov  = self.matern_cov_aniso_STABLE_log_reparam(params, self.Heads_data, self.Heads_data)
            L    = torch.linalg.cholesky(cov)
            Z_X  = torch.linalg.solve_triangular(L, X_h, upper=False)
            Z_y  = torch.linalg.solve_triangular(L, y_h, upper=False)
            # residual in whitened space: L⁻¹(y - Xβ), one entry per head unit
            resid = (Z_y - Z_X @ beta)[:, 0]          # (N_heads,)
            log_L = torch.log(torch.diag(L))           # (N_heads,)
            terms.append(log_L + 0.5 * resid ** 2)

        # --- Tails (A / AB / ABC groups) ---
        chunk_size = 4096
        for X_b, Y_b, Locs_b in [(self.X_A,   self.Y_A,   self.Locs_A),
                                   (self.X_AB,  self.Y_AB,  self.Locs_AB),
                                   (self.X_ABC, self.Y_ABC, self.Locs_ABC)]:
            if X_b is None or X_b.shape[0] == 0:
                continue
            for start in range(0, X_b.shape[0], chunk_size):
                end       = min(start + chunk_size, X_b.shape[0])
                cov_chunk = self.matern_cov_batched(params, X_b[start:end])
                L_chunk   = torch.linalg.cholesky(cov_chunk)
                Z_locs    = torch.linalg.solve_triangular(L_chunk, Locs_b[start:end], upper=False)
                Z_y_chunk = torch.linalg.solve_triangular(L_chunk, Y_b[start:end],    upper=False)
                u_X       = Z_locs[:, -1, :]            # (B, n_feat)
                u_y       = Z_y_chunk[:, -1, 0]         # (B,)
                log_sigma = torch.log(L_chunk[:, -1, -1])  # (B,)
                resid     = u_y - (u_X @ beta)[:, 0]    # (B,)
                terms.append(log_sigma + 0.5 * resid ** 2)

        return torch.cat(terms)  # (total_N,)


# --- Fitting Class ---
class fit_vecchia_lbfgs(VecchiaBatched):

    def __init__(self, smooth: float, input_map: Dict[str, Any],
                 nns_map: Dict[str, Any], mm_cond_number: int, nheads: int,
                 limit_A: int = 8, limit_B: int = 8, limit_C: int = 8, daily_stride: int = 8):
        super().__init__(smooth, input_map, nns_map, mm_cond_number, nheads,
                         limit_A=limit_A, limit_B=limit_B, limit_C=limit_C, daily_stride=daily_stride)

    def set_optimizer(self, param_groups, lr=1.0, max_iter=20, max_eval=None,
                      tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10):
        return torch.optim.LBFGS(
            param_groups, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=history_size, line_search_fn="strong_wolfe"
        )

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        phi1, phi2, phi3, phi4 = np.exp(raw[0]), np.exp(raw[1]), np.exp(raw[2]), np.exp(raw[3])
        return {
            "sigma_sq":   phi1 / phi2,
            "range_lon":  1.0 / phi2,
            "range_lat":  1.0 / (phi2 * np.sqrt(phi3)),
            "range_time": 1.0 / (phi2 * np.sqrt(phi4)),
            "advec_lat":  raw[4],
            "advec_lon":  raw[5],
            "nugget":     np.exp(raw[6]),
        }

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor], optimizer: torch.optim.LBFGS,
                       max_steps: int = 50, grad_tol: float = 1e-5):
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print("--- Starting Batched L-BFGS Optimization (GPU) ---")

        def closure():
            optimizer.zero_grad()
            loss = self.vecchia_batched_likelihood(torch.stack(params_list))
            loss.backward()
            return loss

        loss = None
        for i in range(max_steps):
            loss = optimizer.step(closure)

            with torch.no_grad():
                grads = [abs(p.grad.item()) for p in params_list if p.grad is not None]
                max_grad = max(grads) if grads else 0.0

                print(f'--- Step {i+1}/{max_steps} / Loss: {loss.item():.6f} ---')
                for j, p in enumerate(params_list):
                    g = p.grad.item() if p.grad is not None else 'N/A'
                    print(f'  Param {j}: Value={p.item():.4f}, Grad={g}')
                print(f'  Max Abs Grad: {max_grad:.6e}')
                print("-" * 30)

            if max_grad < grad_tol:
                print(f"\nConverged at step {i+1}")
                break

        raw = [p.item() for p in params_list]
        final_loss = loss.item() if loss is not None else float('inf')
        print("Final Interpretable Params:", self._convert_params(raw))

        return raw + [final_loss], i
