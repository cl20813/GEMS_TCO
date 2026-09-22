"""
kernels_vecchia_advec_band.py

Fixed-budget advection-band Vecchia conditioning.

Target comparison:
  Base:
    t   : 10 spatial neighbors
    t-1 : same loc + 6 local neighbors
    t-2 : same loc + 4 local neighbors

  AdvecBand:
    t   : 10 spatial neighbors
    t-1 : same loc + 4 local neighbors + upstream(+2 cells) + upstream(+3 cells)
    t-2 : same loc + 4 local neighbors

The two models have the same maximum conditioning budget.  If an upstream
point is invalid, duplicated, or falls on the same location after nearest-grid
matching, the vacant slot is filled with the next available local t-1 neighbor
instead of a dummy row.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from GEMS_TCO.kernels_vecchia import fit_vecchia_lbfgs


class fit_vecchia_lbfgs_advec_band(fit_vecchia_lbfgs):
    """Vecchia LBFGS model with a two-point upstream band at t-1 only."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        nns_map: Dict[str, Any],
        mm_cond_number: int,
        nheads: int,
        limit_A: int = 10,
        limit_B: int = 4,
        limit_C: int = 4,
        daily_stride: int = 2,
        spatial_coords: Optional[np.ndarray] = None,
        lon_resolution: float = 0.063,
        advec_cell_offsets: Sequence[int] = (2, 3),
    ):
        super().__init__(
            smooth,
            input_map,
            nns_map,
            mm_cond_number,
            nheads,
            limit_A=limit_A,
            limit_B=limit_B,
            limit_C=limit_C,
            daily_stride=daily_stride,
        )
        self.spatial_coords = spatial_coords
        self.lon_resolution = float(lon_resolution)
        self.advec_cell_offsets = tuple(int(v) for v in advec_cell_offsets)

    def _spatial_coords_np(self, n_points: int) -> np.ndarray:
        if self.spatial_coords is not None:
            coords_np = np.asarray(self.spatial_coords[:n_points], dtype=np.float64)
        else:
            all_data = [
                torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                for d in self.input_map.values()
            ]
            coords_np = all_data[0][:n_points, :2].cpu().numpy().astype(np.float64)

        coords_np = coords_np.copy()
        nan_mask = np.isnan(coords_np).any(axis=1)
        coords_np[nan_mask] = np.array([0.0, 1000.0])
        return coords_np

    def _build_advec_band_lookup(self, n_points: int) -> np.ndarray:
        """Return [n_points, n_offsets] nearest east-shifted local indices."""
        from sklearn.neighbors import BallTree

        coords_np = self._spatial_coords_np(n_points)
        tree = BallTree(np.radians(coords_np), metric="haversine")
        lats = coords_np[:, 0]
        lons = coords_np[:, 1]

        lookups = []
        for cell_offset in self.advec_cell_offsets:
            q = np.column_stack(
                [
                    np.radians(lats),
                    np.radians(lons + cell_offset * self.lon_resolution),
                ]
            )
            _, idx = tree.query(q, k=1)
            lookups.append(idx.flatten().astype(np.int64))

        if not lookups:
            return np.empty((n_points, 0), dtype=np.int64)
        return np.column_stack(lookups)

    @staticmethod
    def _valid_local_ids(values: Iterable[int], upper: int) -> List[int]:
        return [int(v) for v in values if int(v) < upper]

    def precompute_conditioning_sets(self):
        limit_A, limit_B, limit_C = self.limit_A, self.limit_B, self.limit_C
        daily_stride = self.daily_stride
        n_advec = len(self.advec_cell_offsets)

        max_dim_A = limit_A
        max_dim_AB = limit_A + (limit_B + 1) + n_advec
        max_dim_ABC = limit_A + (limit_B + 1) + n_advec + (limit_C + 1)

        n_stored = next((len(m) for m in self.nns_map if len(m) > 0), 0)
        print(
            "Pre-computing AdvecBand Vecchia "
            f"[A={max_dim_A}, AB={max_dim_AB}, ABC={max_dim_ABC}, "
            f"B_local={limit_B}, C_local={limit_C}, "
            f"advec_cells={self.advec_cell_offsets}, stored={n_stored}]...",
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
            valid_lats.mean().item()
            if valid_lats.numel() > 0
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

        key_list = list(self.input_map.keys())
        day_lengths = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        n_time_steps = len(key_list)
        use_set_C = daily_stride < n_time_steps

        n_pts_per_day = day_lengths[0]
        advec_band = self._build_advec_band_lookup(n_pts_per_day)

        heads_indices = []
        batch_list_A = []
        batch_list_AB = []
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
            offset = cumulative_len[time_idx]

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

                add_valid_neighbors(
                    (offset + self.nns_map[local_idx]).tolist(),
                    current_indices,
                    cap=limit_A,
                )

                has_B = time_idx > 0
                has_C = time_idx >= daily_stride

                if has_B:
                    prev_off = cumulative_len[time_idx - 1]
                    prev_len = day_lengths[time_idx - 1]

                    if local_idx < prev_len:
                        add_valid_neighbors([prev_off + local_idx], current_indices, cap=1)

                    adv_ids = []
                    if local_idx < advec_band.shape[0]:
                        adv_ids = self._valid_local_ids(advec_band[local_idx], prev_len)

                    adv_exclude = {v for v in adv_ids if v != local_idx}
                    nbs = self.nns_map[local_idx]
                    local_candidates = [
                        prev_off + int(v)
                        for v in nbs
                        if int(v) < prev_len and int(v) not in adv_exclude
                    ]
                    add_valid_neighbors(local_candidates, current_indices, cap=limit_B)

                    for adv_id in adv_ids:
                        before = len(current_indices)
                        if adv_id != local_idx:
                            add_valid_neighbors([prev_off + adv_id], current_indices, cap=1)
                        if len(current_indices) == before:
                            add_valid_neighbors(local_candidates, current_indices, cap=1)

                if has_C:
                    pd_idx = time_idx - daily_stride
                    pd_off = cumulative_len[pd_idx]
                    pd_len = day_lengths[pd_idx]

                    if local_idx < pd_len:
                        add_valid_neighbors([pd_off + local_idx], current_indices, cap=1)
                    nbs = self.nns_map[local_idx]
                    add_valid_neighbors(
                        (pd_off + nbs[nbs < pd_len]).tolist(),
                        current_indices,
                        cap=limit_C,
                    )

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
            else torch.empty(0, num_cols, device=self.device, dtype=torch.float64)
        )

        def build_tensors(idx_list, max_d):
            if not idx_list:
                return None, None, None, None, None
            T = torch.tensor(idx_list, device=self.device, dtype=torch.long)
            G = Full_Data[T]
            X = G[..., [0, 1, 3]].contiguous().to(torch.float64)
            Y = G[..., 2].unsqueeze(-1).contiguous().to(torch.float64)
            ones = torch.ones_like(G[..., 0]).unsqueeze(-1)
            lat = (G[..., 0] - self.lat_mean_val).unsqueeze(-1)
            dums = G[..., 4:11]
            Locs = torch.cat([ones, lat, dums], dim=-1).contiguous().to(torch.float64)
            is_dummy = (T >= dummy_start).unsqueeze(-1)
            Locs = Locs.masked_fill(is_dummy, 0.0)
            Y = Y.masked_fill(is_dummy, 0.0)
            return X, Y, Locs, T, is_dummy

        self.X_A, self.Y_A, self.Locs_A, self._T_A, self._is_dummy_A = build_tensors(
            batch_list_A, max_dim_A
        )
        self.X_AB, self.Y_AB, self.Locs_AB, self._T_AB, self._is_dummy_AB = build_tensors(
            batch_list_AB, max_dim_AB
        )
        self.X_ABC, self.Y_ABC, self.Locs_ABC, self._T_ABC, self._is_dummy_ABC = build_tensors(
            batch_list_ABC, max_dim_ABC
        )

        self._heads_tensor_stored = heads_tensor if len(heads_indices) > 0 else None
        self._dummy_start_stored = dummy_start
        self._n_real_stored = n_real
        self._n_dummies_stored = n_dummies

        self.n_tails = len(batch_list_A) + len(batch_list_AB) + len(batch_list_ABC)
        self.is_precomputed = True
        print(
            f"[Set C: {use_set_C}] Done. "
            f"(Heads: {len(heads_indices)}, "
            f"Tails A/AB/ABC: {len(batch_list_A)}/{len(batch_list_AB)}/{len(batch_list_ABC)})"
        )
