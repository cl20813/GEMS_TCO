"""
kernels_vecchia_hybrid.py

Hybrid Vecchia kernel supporting Matérn smooth=0.5 and smooth=1.5.

Conditioning structure per target point:
  Set A (t):   spatial NN of target, limit_A points
  Set B (t-1): same-location anchor (1) + local spatial NN (limit_B_local)
               + fresh NN around shifted upstream center (lag1_fresh_count)
  Set C (t-2): same-location anchor (1) + local spatial NN (limit_C_local)
               + fresh NN around 2× shifted center (lag2_fresh_count)

Kernel formulas (d = phi2 * anisotropic_distance):
  smooth=0.5: C = sigmasq * exp(-d)
  smooth=1.5: C = sigmasq * (1 + d) * exp(-d)
"""

import numpy as np
import torch
from sklearn.neighbors import BallTree

from GEMS_TCO import kernels_vecchia


class HybridVecchiaFit(kernels_vecchia.fit_vecchia_lbfgs):
    """Hybrid Vecchia model with fresh shifted-center lag neighbors.

    Extends fit_vecchia_lbfgs with:
    - Matérn 0.5 or 1.5 covariance selection via `smooth`
    - Hybrid conditioning: local spatial NN + upstream-shifted fresh NN at each lag
    """

    def __init__(
        self,
        smooth: float,
        input_map: dict,
        nns_map,
        mm_cond_number: int,
        nheads: int,
        limit_A: int = 20,
        limit_B_local: int = 16,
        limit_C_local: int = 12,
        daily_stride: int = 2,
        spatial_coords=None,
        lag1_lon_offset: float = 0.063,
        lag1_fresh_count: int = 2,
        lag2_fresh_count: int = 2,
    ):
        if smooth not in (0.5, 1.5):
            raise ValueError(f"smooth must be 0.5 or 1.5, got {smooth}")
        super().__init__(
            smooth, input_map, nns_map, mm_cond_number, nheads,
            limit_A=limit_A,
            limit_B=limit_B_local,
            limit_C=limit_C_local,
            daily_stride=daily_stride,
        )
        self.spatial_coords = spatial_coords
        self.lag1_lon_offset = float(abs(lag1_lon_offset))
        self.lag1_fresh_count = int(lag1_fresh_count)
        self.lag2_fresh_count = int(lag2_fresh_count)

    # ------------------------------------------------------------------
    # Covariance overrides: smooth-aware versions
    # ------------------------------------------------------------------

    def matern_cov_batched(self, params, x_batch):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        d = self.batched_manual_dist(dist_params, x_batch)
        scaled_d = d * phi2

        if self.smooth == 0.5:
            cov = (phi1 / phi2) * torch.exp(-scaled_d)
        else:  # 1.5
            cov = (phi1 / phi2) * (1.0 + scaled_d) * torch.exp(-scaled_d)

        B, N, _ = x_batch.shape
        eye = torch.eye(N, device=self.device, dtype=torch.float64).unsqueeze(0).expand(B, N, N)
        return cov + eye * (nugget + 1e-6)

    def matern_cov_aniso_STABLE_log_reparam(self, params, x, y):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        sigmasq = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance = self.precompute_coords_aniso_STABLE(dist_params, x, y)
        scaled_d = distance * phi2

        if self.smooth == 0.5:
            cov = sigmasq * torch.exp(-scaled_d)
        else:  # 1.5
            cov = sigmasq * (1.0 + scaled_d) * torch.exp(-scaled_d)

        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8)
        return cov

    # ------------------------------------------------------------------
    # Shift-center lookup helpers
    # ------------------------------------------------------------------

    def _spatial_coords_np(self, n_points: int) -> np.ndarray:
        if self.spatial_coords is not None:
            coords = np.asarray(self.spatial_coords[:n_points], dtype=np.float64)
        else:
            all_data = [
                torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                for d in self.input_map.values()
            ]
            coords = all_data[0][:n_points, :2].cpu().numpy().astype(np.float64)
        coords = coords.copy()
        nan_mask = np.isnan(coords).any(axis=1)
        coords[nan_mask] = np.array([0.0, 1000.0])
        return coords

    def _build_shift_lookup(self, n_points: int, multiplier: float) -> np.ndarray:
        coords = self._spatial_coords_np(n_points)
        tree = BallTree(np.radians(coords), metric="haversine")
        lats = coords[:, 0]
        lons = coords[:, 1]
        valid = ~np.isnan(coords).any(axis=1)
        lon_min = float(np.nanmin(lons[valid]))
        lon_max = float(np.nanmax(lons[valid]))
        base_ids = np.arange(n_points, dtype=np.int64)
        target_lons = lons + multiplier * self.lag1_lon_offset
        outside = (~valid) | (target_lons < lon_min) | (target_lons > lon_max)
        query = np.column_stack([np.radians(lats), np.radians(target_lons)])
        _, idx = tree.query(query, k=1)
        lookup = idx.flatten().astype(np.int64)
        lookup[outside] = base_ids[outside]
        return lookup

    # ------------------------------------------------------------------
    # Hybrid conditioning-set construction
    # ------------------------------------------------------------------

    def precompute_conditioning_sets(self):
        limit_A    = int(self.limit_A)
        lag1_local = int(self.limit_B)        # local NN count at lag 1
        lag2_local = int(self.limit_C)        # local NN count at lag 2
        lag1_fresh = int(self.lag1_fresh_count)
        lag2_fresh = int(self.lag2_fresh_count)
        daily_stride = int(self.daily_stride)

        max_dim_A   = limit_A
        max_dim_AB  = limit_A + 1 + lag1_local + lag1_fresh
        max_dim_ABC = max_dim_AB + 1 + lag2_local + lag2_fresh

        n_stored = next((len(m) for m in self.nns_map if len(m) > 0), 0)
        print(
            f"Pre-computing HybridVecchia (smooth={self.smooth}) "
            f"[A={max_dim_A}, AB={max_dim_AB}, ABC={max_dim_ABC}, "
            f"B=local{lag1_local}+fresh{lag1_fresh}, "
            f"C=local{lag2_local}+fresh{lag2_fresh}, "
            f"lag1_offset={self.lag1_lon_offset:.4f}, stored={n_stored}]...",
            end=" ",
        )

        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        Real_Data = torch.cat(all_data_list, dim=0).to(self.device, dtype=torch.float32)
        n_real, num_cols = Real_Data.shape

        is_nan_real = torch.isnan(Real_Data[:, 2])
        valid_lats = Real_Data[~is_nan_real, 0]
        self.lat_mean_val = (
            valid_lats.mean().item() if valid_lats.numel() > 0
            else Real_Data[:, 0].mean().item()
        )
        print(f"[Mean Lat: {self.lat_mean_val:.4f}]", end=" ")
        is_nan_mask_np = is_nan_real.cpu().numpy()

        n_dummies = max_dim_ABC
        dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float32)
        for k in range(n_dummies):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        Full_Data = torch.cat([Real_Data, dummy_block], dim=0)
        dummy_start = n_real
        is_nan_mask_np = np.append(is_nan_mask_np, np.zeros(n_dummies, dtype=bool))

        key_list       = list(self.input_map.keys())
        day_lengths    = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        n_time_steps   = len(key_list)
        use_set_C      = daily_stride < n_time_steps

        n_pts_per_day = day_lengths[0]
        lag1_center   = self._build_shift_lookup(n_pts_per_day, multiplier=1.0)
        lag2_center   = self._build_shift_lookup(n_pts_per_day, multiplier=2.0)

        heads_indices  = []
        batch_list_A   = []
        batch_list_AB  = []
        batch_list_ABC = []

        def add_valid_neighbors(indices_to_check, current_indices, cap):
            count = 0
            for idx in indices_to_check:
                if count >= cap:
                    break
                idx = int(idx)
                if idx not in current_indices and not is_nan_mask_np[idx]:
                    current_indices.append(idx)
                    count += 1

        for time_idx, key in enumerate(key_list):
            day_len = day_lengths[time_idx]
            offset  = cumulative_len[time_idx]

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
                nbs_current = (
                    self.nns_map[local_idx]
                    if local_idx < len(self.nns_map)
                    else np.array([], dtype=np.int64)
                )
                add_valid_neighbors((offset + nbs_current).tolist(), current_indices, cap=limit_A)

                has_B = time_idx > 0
                has_C = time_idx >= daily_stride

                if has_B:
                    prev_off = cumulative_len[time_idx - 1]
                    prev_len = day_lengths[time_idx - 1]

                    if local_idx < prev_len:
                        add_valid_neighbors([prev_off + local_idx], current_indices, cap=1)

                    local_candidates = [
                        prev_off + int(v)
                        for v in nbs_current
                        if int(v) < prev_len and int(v) != local_idx
                    ]
                    add_valid_neighbors(local_candidates, current_indices, cap=lag1_local)

                    center_B = int(lag1_center[local_idx]) if local_idx < len(lag1_center) else local_idx
                    if center_B >= prev_len:
                        center_B = local_idx
                    nbs_B = (
                        self.nns_map[center_B]
                        if center_B < len(self.nns_map)
                        else np.array([], dtype=np.int64)
                    )
                    fresh_candidates_B = [prev_off + center_B] + [
                        prev_off + int(v)
                        for v in nbs_B
                        if int(v) < prev_len and int(v) != local_idx
                    ]
                    add_valid_neighbors(fresh_candidates_B, current_indices, cap=lag1_fresh)

                if has_C:
                    pd_idx = time_idx - daily_stride
                    pd_off = cumulative_len[pd_idx]
                    pd_len = day_lengths[pd_idx]

                    if local_idx < pd_len:
                        add_valid_neighbors([pd_off + local_idx], current_indices, cap=1)

                    local_candidates = [
                        pd_off + int(v)
                        for v in nbs_current
                        if int(v) < pd_len and int(v) != local_idx
                    ]
                    add_valid_neighbors(local_candidates, current_indices, cap=lag2_local)

                    center_C = int(lag2_center[local_idx]) if local_idx < len(lag2_center) else local_idx
                    if center_C >= pd_len:
                        center_C = local_idx
                    nbs_C = (
                        self.nns_map[center_C]
                        if center_C < len(self.nns_map)
                        else np.array([], dtype=np.int64)
                    )
                    fresh_candidates_C = [pd_off + center_C] + [
                        pd_off + int(v)
                        for v in nbs_C
                        if int(v) < pd_len and int(v) != local_idx
                    ]
                    add_valid_neighbors(fresh_candidates_C, current_indices, cap=lag2_fresh)

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

        heads_tensor = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
        self.Heads_data = (
            Full_Data[heads_tensor].contiguous().to(torch.float64)
            if len(heads_indices) > 0
            else torch.empty((0, num_cols), device=self.device, dtype=torch.float64)
        )

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

        self._heads_tensor_stored = heads_tensor if len(heads_indices) > 0 else None
        self._dummy_start_stored  = dummy_start
        self._n_real_stored       = n_real
        self._n_dummies_stored    = n_dummies
        self.n_tails = len(batch_list_A) + len(batch_list_AB) + len(batch_list_ABC)

        print(
            f"[SetC={use_set_C}] Done. "
            f"(Heads={len(heads_indices)}, "
            f"Tails A/AB/ABC={len(batch_list_A)}/{len(batch_list_AB)}/{len(batch_list_ABC)})"
        )
        self.is_precomputed = True
        return self
