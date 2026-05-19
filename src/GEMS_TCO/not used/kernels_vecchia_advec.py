"""
kernels_vecchia_advec.py

Advection-aware Vecchia approximation.

Conditioning set structure (vs standard VecchiaBatched):
  t              : limit_A spatial neighbors  (unchanged)
  t-1  (Set B)   : same loc + 1 advec center + limit_B neighbors around that advec center
  t-C  (Set C)   : same loc + 1 advec center + limit_C neighbors around that advec center

Advection offsets (fixed, motivated by avg advec_lon ~= -0.1, lon_res = 0.063):
  Set B advec offset: +advec_lon_offset          (default 0.063*2 = 0.126 deg east)
  Set C advec offset: +2 * advec_lon_offset      (0.252 deg east)

Usage:
  model = fit_vecchia_lbfgs_advec(
      smooth=0.5, input_map=..., nns_map=..., mm_cond_number=100, nheads=0,
      limit_A=8, limit_B=8, limit_C=8, daily_stride=8,
      spatial_coords=ordered_grid_coords_np,   # [N_grid, 2]  (lat, lon)
      advec_lon_offset=0.126,
  )
  model.precompute_conditioning_sets()
"""
import numpy as np
import torch
from typing import Dict, Any, List, Optional

from GEMS_TCO.kernels_vecchia import VecchiaBatched


# ── Advec-aware precompute ────────────────────────────────────────────────────

class VecchiaBatchedAdvec(VecchiaBatched):
    """Vecchia approximation with one advection-adjusted conditioning point per temporal lag."""

    def __init__(self, smooth: float, input_map: Dict[str, Any],
                 nns_map: Dict[str, Any], mm_cond_number: int, nheads: int,
                 limit_A: int = 8, limit_B: int = 8, limit_C: int = 8,
                 daily_stride: int = 8,
                 spatial_coords: Optional[np.ndarray] = None,
                 advec_lon_offset: Optional[float] = None,
                 advec_lon_step: Optional[float] = None):
        super().__init__(smooth, input_map, nns_map, mm_cond_number, nheads,
                         limit_A=limit_A, limit_B=limit_B, limit_C=limit_C,
                         daily_stride=daily_stride)
        # ordered grid coordinates [N_grid, 2] (lat, lon) — same ordering as input_map rows
        self.spatial_coords = spatial_coords
        # Backward-compatible alias: advec_lon_step was the old name, but this
        # is an offset distance, not a grid step count.
        if advec_lon_offset is None:
            advec_lon_offset = 0.126 if advec_lon_step is None else advec_lon_step
        self.advec_lon_offset = advec_lon_offset
        self.advec_lon_step = advec_lon_offset

    # ── Advec neighbor lookup ─────────────────────────────────────────────────

    def _build_advec_lookup(self, n_points: int):
        """Batch-query nearest neighbors shifted east by 1x and 2x advec_lon_offset."""
        from sklearn.neighbors import BallTree

        if self.spatial_coords is not None:
            coords_np = np.asarray(self.spatial_coords[:n_points], dtype=np.float64)
        else:
            all_data = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                        for d in self.input_map.values()]
            coords_raw = all_data[0][:n_points, :2].cpu().numpy().astype(np.float64)
            nan_mask = np.isnan(coords_raw).any(axis=1)
            coords_np = coords_raw.copy()
            coords_np[nan_mask] = np.array([0.0, 1000.0])  # out-of-range dummy

        tree = BallTree(np.radians(coords_np), metric='haversine')

        lats = coords_np[:, 0]
        lons = coords_np[:, 1]

        # Set B: shift 1× eastward (upstream at t-1 given advec_lon < 0)
        q_B = np.column_stack([np.radians(lats), np.radians(lons + self.advec_lon_offset)])
        _, idx_B = tree.query(q_B, k=1)
        advec_nb_B = idx_B.flatten().astype(np.int64)

        # Set C: shift 2× eastward
        q_C = np.column_stack([np.radians(lats), np.radians(lons + 2.0 * self.advec_lon_offset)])
        _, idx_C = tree.query(q_C, k=1)
        advec_nb_C = idx_C.flatten().astype(np.int64)

        return advec_nb_B, advec_nb_C

    # ── Core override ─────────────────────────────────────────────────────────

    def precompute_conditioning_sets(self):
        limit_A, limit_B, limit_C = self.limit_A, self.limit_B, self.limit_C
        daily_stride = self.daily_stride

        # Each temporal lag gets +1 slot for the advec neighbor
        max_dim_A   = limit_A
        max_dim_AB  = limit_A + (limit_B + 1) + 1
        max_dim_ABC = limit_A + (limit_B + 1) + 1 + (limit_C + 1) + 1

        n_stored = next((len(m) for m in self.nns_map if len(m) > 0), 0)
        print(f"🚀 Pre-computing Advec Vecchia "
              f"[A={max_dim_A}, AB={max_dim_AB}, ABC={max_dim_ABC}, "
              f"advec_offset={self.advec_lon_offset:.4f}, stored={n_stored}]...", end=" ")

        # ── Stack all data ────────────────────────────────────────────────────
        all_data_list = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                         for d in self.input_map.values()]
        Real_Data  = torch.cat(all_data_list, dim=0).to(self.device, dtype=torch.float32)
        n_real, num_cols = Real_Data.shape

        is_nan_real = torch.isnan(Real_Data[:, 2])
        valid_lats  = Real_Data[~is_nan_real, 0]
        self.lat_mean_val = (valid_lats.mean().item() if valid_lats.numel() > 0
                             else Real_Data[:, 0].mean().item())
        print(f"[Mean Lat: {self.lat_mean_val:.4f}]", end=" ")
        is_nan_mask_np = is_nan_real.cpu().numpy()

        # ── Dummy block ───────────────────────────────────────────────────────
        n_dummies   = max_dim_ABC
        dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float32)
        for k in range(n_dummies):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        Full_Data      = torch.cat([Real_Data, dummy_block], dim=0)
        dummy_start    = n_real
        is_nan_mask_np = np.append(is_nan_mask_np, np.zeros(n_dummies, dtype=bool))

        # ── Index structures ──────────────────────────────────────────────────
        key_list       = list(self.input_map.keys())
        day_lengths    = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        n_time_steps   = len(key_list)
        use_set_C      = daily_stride < n_time_steps

        # ── Precompute advec neighbors (batch, O(N log N)) ────────────────────
        n_pts_per_day = day_lengths[0]
        advec_nb_B, advec_nb_C = self._build_advec_lookup(n_pts_per_day)

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

                # Set A: spatial neighbors at t
                add_valid_neighbors(
                    (offset + self.nns_map[local_idx]).tolist(), current_indices, cap=limit_A)

                has_B = time_idx > 0
                has_C = time_idx >= daily_stride

                # Set B: t-1  (same loc + upstream-centered neighbors + upstream center)
                if has_B:
                    prev_off = cumulative_len[time_idx - 1]
                    prev_len = day_lengths[time_idx - 1]

                    if local_idx < prev_len:
                        add_valid_neighbors([prev_off + local_idx], current_indices, cap=1)

                    adv_B = int(advec_nb_B[local_idx] if local_idx < len(advec_nb_B) else local_idx)
                    center_B = adv_B if adv_B < prev_len and adv_B != local_idx else local_idx
                    nbs = (self.nns_map[center_B]
                           if center_B < len(self.nns_map)
                           else np.array([], dtype=np.int64))
                    nbs_B_candidates = (
                        prev_off + nbs[(nbs < prev_len) & (nbs != local_idx) & (nbs != center_B)]
                    ).tolist()
                    add_valid_neighbors(
                        nbs_B_candidates, current_indices, cap=limit_B)

                    # Advec center: nearest grid point shifted 1x eastward at t-1
                    before_adv = len(current_indices)
                    if center_B != local_idx:
                        add_valid_neighbors([prev_off + center_B], current_indices, cap=1)
                    if len(current_indices) == before_adv:
                        add_valid_neighbors(nbs_B_candidates, current_indices, cap=1)

                # Set C: t-daily_stride  (same loc + upstream-centered neighbors + upstream center)
                if has_C:
                    pd_idx = time_idx - daily_stride
                    pd_off = cumulative_len[pd_idx]
                    pd_len = day_lengths[pd_idx]

                    if local_idx < pd_len:
                        add_valid_neighbors([pd_off + local_idx], current_indices, cap=1)

                    adv_C = int(advec_nb_C[local_idx] if local_idx < len(advec_nb_C) else local_idx)
                    center_C = adv_C if adv_C < pd_len and adv_C != local_idx else local_idx
                    nbs = (self.nns_map[center_C]
                           if center_C < len(self.nns_map)
                           else np.array([], dtype=np.int64))
                    nbs_C_candidates = (
                        pd_off + nbs[(nbs < pd_len) & (nbs != local_idx) & (nbs != center_C)]
                    ).tolist()
                    add_valid_neighbors(
                        nbs_C_candidates, current_indices, cap=limit_C)

                    # Advec center: nearest grid point shifted 2x eastward at t-daily_stride
                    before_adv = len(current_indices)
                    if center_C != local_idx:
                        add_valid_neighbors([pd_off + center_C], current_indices, cap=1)
                    if len(current_indices) == before_adv:
                        add_valid_neighbors(nbs_C_candidates, current_indices, cap=1)

                # Group + dummy padding
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

        # ── GPU tensors ───────────────────────────────────────────────────────
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

        self._heads_tensor_stored = heads_tensor if len(heads_indices) > 0 else None
        self._dummy_start_stored  = dummy_start
        self._n_real_stored       = n_real
        self._n_dummies_stored    = n_dummies

        self.n_tails = len(batch_list_A) + len(batch_list_AB) + len(batch_list_ABC)
        self.is_precomputed = True
        print(f"[Set C: {use_set_C}] ✅ Done. "
              f"(Heads: {len(heads_indices)}, "
              f"Tails A/AB/ABC: {len(batch_list_A)}/{len(batch_list_AB)}/{len(batch_list_ABC)})")


# ── Fitting class (mirrors fit_vecchia_lbfgs from kernels_vecchia.py) ─────────

class fit_vecchia_lbfgs_advec(VecchiaBatchedAdvec):

    def __init__(self, smooth: float, input_map: Dict[str, Any],
                 nns_map: Dict[str, Any], mm_cond_number: int, nheads: int,
                 limit_A: int = 8, limit_B: int = 8, limit_C: int = 8,
                 daily_stride: int = 8,
                 spatial_coords: Optional[np.ndarray] = None,
                 advec_lon_offset: Optional[float] = None,
                 advec_lon_step: Optional[float] = None):
        super().__init__(smooth, input_map, nns_map, mm_cond_number, nheads,
                         limit_A=limit_A, limit_B=limit_B, limit_C=limit_C,
                         daily_stride=daily_stride,
                         spatial_coords=spatial_coords,
                         advec_lon_offset=advec_lon_offset,
                         advec_lon_step=advec_lon_step)

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

        print("--- Starting Advec L-BFGS Optimization ---")

        def closure():
            optimizer.zero_grad()
            loss = self.vecchia_batched_likelihood(torch.stack(params_list))
            loss.backward()
            return loss

        loss = None
        for i in range(max_steps):
            loss = optimizer.step(closure)

            with torch.no_grad():
                grads    = [abs(p.grad.item()) for p in params_list if p.grad is not None]
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

        raw        = [p.item() for p in params_list]
        final_loss = loss.item() if loss is not None else float('inf')
        print("Final Interpretable Params:", self._convert_params(raw))

        return raw + [final_loss], i
