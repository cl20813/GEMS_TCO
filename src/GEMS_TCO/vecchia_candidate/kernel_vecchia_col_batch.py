"""
kernel_vecchia_col_batch.py

Batched reverse-L Vecchia kernel.

This version is intentionally not a template-reuse kernel.  It keeps the V3
reverse-L conditioning geometry, but evaluates the likelihood in a target-wise
batched Vecchia style so the experiment isolates whether the reverse-L
conditioning logic improves RMSRE.

Default conditioning:
  per_lag_conditioning_count = 14
  lag_count = 2
  nominal m = 14 * (2 + 1) = 42

The reverse-L scan is defined on the regular grid.  If
use_data_coords_for_offsets=True, covariance distances are computed from the
actual input-map coordinates, e.g. real Source_Latitude/Source_Longitude.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.special import gamma


class ReverseLColumnVecchiaFitBatch:
    """Reverse-L conditioning geometry with batched Vecchia likelihood."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        mm_cond_number: int = 300,
        nheads: int = 0,
        grid_coords: Optional[np.ndarray] = None,
        head_right_cols: int = 0,
        above_count: int = 3,
        right_col_count: int = 3,
        per_lag_conditioning_count: int = 14,
        right_neighbor_count: Optional[int] = None,
        lag_count: int = 2,
        include_lag_self: bool = False,
        lat_round_decimals: int = 6,
        lon_round_decimals: int = 6,
        target_chunk_size: int = 4096,
        use_data_coords_for_offsets: bool = False,
        **_,
    ):
        if smooth not in (0.5, 1.5):
            raise ValueError(f"smooth must be 0.5 or 1.5, got {smooth}")
        if right_neighbor_count is not None and int(right_neighbor_count) != int(per_lag_conditioning_count):
            raise ValueError(
                "ReverseLColumnVecchiaFitBatch uses per_lag_conditioning_count as "
                "the total spatial conditioning cap per lag; pass only one of "
                "per_lag_conditioning_count or a matching right_neighbor_count."
            )

        self.smooth = float(smooth)
        self.input_map = input_map
        self.mm_cond_number = int(mm_cond_number)
        self.nheads = int(nheads)  # API compatibility; head_right_cols controls this kernel.
        self.grid_coords = grid_coords
        self.head_right_cols = int(head_right_cols)
        self.above_count = int(above_count)
        self.right_col_count = int(right_col_count)
        self.per_lag_conditioning_count = int(per_lag_conditioning_count)
        self.right_neighbor_count = int(per_lag_conditioning_count)
        self.lag_count = int(lag_count)
        self.include_lag_self = bool(include_lag_self)
        self.lat_round_decimals = int(lat_round_decimals)
        self.lon_round_decimals = int(lon_round_decimals)
        self.target_chunk_size = int(target_chunk_size)
        self.use_data_coords_for_offsets = bool(use_data_coords_for_offsets)

        first_val = next(iter(input_map.values()))
        self.device = first_val.device if isinstance(first_val, torch.Tensor) else torch.device("cpu")
        self.n_features = 9
        self.lat_mean_val = 0.0
        self.is_precomputed = False

        self.Full_Data = None
        self.Heads_data = None
        self.Batched_Groups = []
        self.Grouped_Batches = []  # kept empty for old diagnostics that expect the attribute
        self.n_tails = 0
        self.n_templates = np.nan

        self._n_real = 0
        self._n_grid = 0
        self._n_time = 0
        self._n_lat = 0
        self._n_lon = 0
        self._local_to_row = None
        self._local_to_col = None
        self._row_col_to_local = {}
        self._stencil_cache = {}

        gamma_val = torch.tensor(gamma(self.smooth), dtype=torch.float64)
        self.matern_const = (2 ** (1 - self.smooth)) / gamma_val

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _regular_coords_np(self, n_points: int) -> np.ndarray:
        if self.grid_coords is not None:
            coords = np.asarray(self.grid_coords[:n_points], dtype=np.float64)
        else:
            first = next(iter(self.input_map.values()))
            if isinstance(first, torch.Tensor):
                coords = first[:n_points, :2].detach().cpu().numpy().astype(np.float64)
            else:
                coords = np.asarray(first[:n_points, :2], dtype=np.float64)
        if coords.shape[1] != 2:
            raise ValueError("grid_coords must have columns [lat, lon]")
        return coords

    def _build_grid_maps(self, coords: np.ndarray):
        lat_key = np.round(coords[:, 0], self.lat_round_decimals)
        lon_key = np.round(coords[:, 1], self.lon_round_decimals)
        lats = np.sort(np.unique(lat_key))
        lons = np.sort(np.unique(lon_key))
        lat_to_row = {v: i for i, v in enumerate(lats)}
        lon_to_col = {v: i for i, v in enumerate(lons)}

        local_to_row = np.empty(coords.shape[0], dtype=np.int64)
        local_to_col = np.empty(coords.shape[0], dtype=np.int64)
        row_col_to_local = {}
        for i, (la, lo) in enumerate(zip(lat_key, lon_key)):
            r = lat_to_row[la]
            c = lon_to_col[lo]
            local_to_row[i] = r
            local_to_col[i] = c
            row_col_to_local[(r, c)] = i
        return lats, lons, local_to_row, local_to_col, row_col_to_local

    def _get_local(self, row: int, col: int) -> Optional[int]:
        return self._row_col_to_local.get((int(row), int(col)))

    def _spatial_candidate_locals_uncapped(self, row: int, col: int) -> List[int]:
        key = (int(row), int(col))
        if key in self._stencil_cache:
            return self._stencil_cache[key]

        out: List[int] = []
        seen = set()

        for k in range(1, self.above_count + 1):
            nb = self._get_local(row + k, col)
            if nb is not None and nb not in seen:
                out.append(nb)
                seen.add(nb)

        right_candidates = []
        for dc in range(1, self.right_col_count + 1):
            c2 = col + dc
            if c2 >= self._n_lon:
                continue
            for r2 in range(row, -1, -1):
                nb = self._get_local(r2, c2)
                if nb is None or nb in seen:
                    continue
                down = row - r2
                right_candidates.append((down, dc, nb))

        right_candidates.sort(key=lambda x: (x[0], x[1]))
        for _, _, nb in right_candidates:
            if nb not in seen:
                out.append(nb)
                seen.add(nb)

        self._stencil_cache[key] = out
        return out

    # ------------------------------------------------------------------
    # Covariance/design helpers
    # ------------------------------------------------------------------

    def _raw_params(self, params: torch.Tensor):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        sigmasq = phi1 / phi2
        return phi1, phi2, phi3, phi4, params[4], params[5], nugget, sigmasq

    def _cov_from_deltas(self, d_lat, d_lon, d_t, params: torch.Tensor):
        _, phi2, phi3, phi4, advec_lat, advec_lon, _, sigmasq = self._raw_params(params)
        u_lat = d_lat - advec_lat * d_t
        u_lon = d_lon - advec_lon * d_t
        dist = torch.sqrt(d_lat.new_tensor(1e-8) + u_lat.pow(2) * phi3 + u_lon.pow(2) + d_t.pow(2) * phi4)
        scaled = dist * phi2
        if self.smooth == 0.5:
            return sigmasq * torch.exp(-scaled)
        return sigmasq * (1.0 + scaled) * torch.exp(-scaled)

    def _full_cov(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        cov = self._cov_from_deltas(diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], params)
        nugget = torch.exp(params[6])
        cov = cov.clone()
        cov.diagonal().add_(nugget + 1e-8)
        return cov

    def _cov_nn(self, neigh_coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = neigh_coords.unsqueeze(2) - neigh_coords.unsqueeze(1)
        cov = self._cov_from_deltas(diff[:, :, :, 0], diff[:, :, :, 1], diff[:, :, :, 2], params)
        nugget = torch.exp(params[6])
        m = neigh_coords.shape[1]
        eye = torch.eye(m, device=self.device, dtype=torch.float64).unsqueeze(0)
        return cov + eye * (nugget + 1e-6)

    def _cov_tn(self, target_coords: torch.Tensor, neigh_coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = target_coords.unsqueeze(1) - neigh_coords
        return self._cov_from_deltas(diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], params)

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        lat = (flat[:, 0:1] - self.lat_mean_val).to(torch.float64)
        dums = flat[:, 4:11].to(torch.float64)
        X = torch.cat([ones, lat, dums], dim=1)
        return X.reshape(*orig_shape, self.n_features)

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute_conditioning_sets(self):
        print(
            "Pre-computing Batched ReverseLColumn V3 "
            f"[heads_right={self.head_right_cols}, above={self.above_count}, "
            f"right_cols={self.right_col_count}, per_lag={self.per_lag_conditioning_count}, "
            f"lags={self.lag_count}, "
            f"coord_mode={'data' if self.use_data_coords_for_offsets else 'grid'}]...",
            end=" ",
        )
        t0 = time.time()

        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        all_data_list = [d.to(self.device, dtype=torch.float64) for d in all_data_list]
        real_data = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = real_data.shape
        self._n_real = n_real
        self._n_time = len(all_data_list)
        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if len(set(day_lengths)) != 1:
            raise ValueError(f"ReverseLColumnVecchiaFitBatch requires equal grid length per time, got {day_lengths}")
        self._n_grid = day_lengths[0]
        cumulative_len = np.cumsum([0] + day_lengths)

        coords_np = self._regular_coords_np(self._n_grid)
        lats, lons, local_to_row, local_to_col, row_col_to_local = self._build_grid_maps(coords_np)
        self._n_lat = len(lats)
        self._n_lon = len(lons)
        self._local_to_row = local_to_row
        self._local_to_col = local_to_col
        self._row_col_to_local = row_col_to_local
        self._stencil_cache = {}

        y = real_data[:, 2]
        valid_y_np = (~torch.isnan(y)).detach().cpu().numpy()
        valid_lats = real_data[~torch.isnan(y), 0]
        self.lat_mean_val = float(valid_lats.mean().item()) if valid_lats.numel() else float(real_data[:, 0].mean().item())

        time_values = []
        for d in all_data_list:
            good = ~torch.isnan(d[:, 3])
            time_values.append(float(d[good, 3].median().item()) if good.any() else 0.0)

        max_m = self.per_lag_conditioning_count * (self.lag_count + 1)
        dummy_block = torch.zeros((max_m, num_cols), device=self.device, dtype=torch.float64)
        for k in range(max_m):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        self.Full_Data = torch.cat([real_data, dummy_block], dim=0).contiguous()
        dummy_start = n_real
        valid_y_np = np.append(valid_y_np, np.ones(max_m, dtype=bool))

        head_locals = set()
        if self.head_right_cols > 0:
            for c in range(max(0, self._n_lon - self.head_right_cols), self._n_lon):
                for r in range(self._n_lat):
                    nb = self._get_local(r, c)
                    if nb is not None:
                        head_locals.add(nb)

        heads_indices: List[int] = []
        batch_lists: Dict[int, List[Tuple[int, List[int]]]] = {
            self.per_lag_conditioning_count: [],
            self.per_lag_conditioning_count * 2: [],
            self.per_lag_conditioning_count * 3: [],
        }
        m_sizes = []

        col_order = range(self._n_lon - 1, -1, -1)
        row_order = range(self._n_lat - 1, -1, -1)
        per_lag_cap = max(0, int(self.per_lag_conditioning_count))

        for time_idx in range(self._n_time):
            offset = int(cumulative_len[time_idx])
            active_lags = min(self.lag_count, time_idx) + 1
            max_d = per_lag_cap * active_lags

            for col in col_order:
                for row in row_order:
                    local_idx = self._get_local(row, col)
                    if local_idx is None:
                        continue
                    target_global = offset + local_idx
                    if not valid_y_np[target_global]:
                        continue

                    if local_idx in head_locals:
                        heads_indices.append(target_global)
                        continue

                    neigh_globals: List[int] = []
                    seen = set()
                    spatial_candidates = self._spatial_candidate_locals_uncapped(row, col)

                    for lag in range(active_lags):
                        neigh_time_idx = time_idx - lag
                        neigh_time_offset = int(cumulative_len[neigh_time_idx])
                        added_this_lag = 0

                        if lag > 0 and self.include_lag_self:
                            g = neigh_time_offset + local_idx
                            if g not in seen and valid_y_np[g]:
                                neigh_globals.append(g)
                                seen.add(g)
                                added_this_lag += 1

                        for nb_local in spatial_candidates:
                            if added_this_lag >= per_lag_cap:
                                break
                            g = neigh_time_offset + int(nb_local)
                            if g in seen or not valid_y_np[g]:
                                continue
                            neigh_globals.append(g)
                            seen.add(g)
                            added_this_lag += 1

                    m_sizes.append(len(neigh_globals))
                    if len(neigh_globals) < max_d:
                        padded = [dummy_start + k for k in range(max_d - len(neigh_globals))] + neigh_globals
                    else:
                        padded = neigh_globals[-max_d:]
                    batch_lists[max_d].append((target_global, padded))

        if heads_indices:
            h = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
            self.Heads_data = self.Full_Data[h].contiguous()
        else:
            self.Heads_data = torch.empty((0, num_cols), device=self.device, dtype=torch.float64)

        self.Batched_Groups = []
        for max_d in sorted(batch_lists):
            rows = batch_lists[max_d]
            if not rows:
                continue
            target_idx = torch.tensor([r[0] for r in rows], device=self.device, dtype=torch.long)
            neigh_idx = torch.tensor([r[1] for r in rows], device=self.device, dtype=torch.long)
            is_dummy = neigh_idx >= dummy_start
            target_rows = self.Full_Data[target_idx].contiguous()
            neigh_rows = self.Full_Data[neigh_idx].contiguous()
            dummy_mask = is_dummy.unsqueeze(-1)
            coords_t = target_rows[:, [0, 1, 3]].contiguous()
            coords_n = neigh_rows[:, :, [0, 1, 3]].contiguous()
            X_t = self._design_from_rows(target_rows).contiguous()
            y_t = target_rows[:, 2:3].contiguous()
            X_n = self._design_from_rows(neigh_rows).masked_fill(dummy_mask, 0.0).contiguous()
            y_n = neigh_rows[:, :, 2].masked_fill(is_dummy, 0.0).contiguous()
            self.Batched_Groups.append({
                "max_m": int(max_d),
                "target_idx": target_idx,
                "neigh_idx": neigh_idx,
                "is_dummy": is_dummy,
                "coords_t": coords_t,
                "coords_n": coords_n,
                "X_t": X_t,
                "y_t": y_t,
                "X_n": X_n,
                "y_n": y_n,
            })

        self.n_tails = int(sum(g["target_idx"].shape[0] for g in self.Batched_Groups))
        self.is_precomputed = True
        self.Grouped_Batches = []
        if m_sizes:
            m_arr = np.asarray(m_sizes)
            m_msg = f"m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max()}"
        else:
            m_msg = "m empty"
        print(
            f"Done in {time.time() - t0:.1f}s. grid={self._n_lat}x{self._n_lon}, "
            f"heads={len(heads_indices)}, tails={self.n_tails}, "
            f"batches={[(g['max_m'], int(g['target_idx'].shape[0])) for g in self.Batched_Groups]}, {m_msg}"
        )
        return self

    # ------------------------------------------------------------------
    # Likelihood/optimization
    # ------------------------------------------------------------------

    def _check_precomputed(self):
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first")

    def _head_design_response(self):
        X_h = self._design_from_rows(self.Heads_data)
        y_h = self.Heads_data[:, 2:3]
        return X_h, y_h

    def _accumulate_gls_stats(self, params: torch.Tensor, include_y_quad: bool = True, catch_cholesky: bool = False):
        self._check_precomputed()
        XT_Sinv_X = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        if self.Heads_data.shape[0] > 0:
            coords = self.Heads_data[:, [0, 1, 3]].contiguous()
            X_h, y_h = self._head_design_response()
            try:
                K_h = self._full_cov(coords, params)
                L_h = torch.linalg.cholesky(K_h)
            except torch.linalg.LinAlgError:
                if catch_cholesky:
                    return None
                raise
            Z_X = torch.linalg.solve_triangular(L_h, X_h, upper=False)
            Z_y = torch.linalg.solve_triangular(L_h, y_h, upper=False)
            XT_Sinv_X += Z_X.T @ Z_X
            XT_Sinv_y += Z_X.T @ Z_y
            if include_y_quad:
                yT_Sinv_y += (Z_y.T @ Z_y).squeeze()
            log_det += 2.0 * torch.log(torch.diagonal(L_h)).sum()

        _, _, _, _, _, _, nugget, sigmasq = self._raw_params(params)
        total_sill = sigmasq + nugget

        for group in self.Batched_Groups:
            coords_t_all = group["coords_t"]
            coords_n_all = group["coords_n"]
            X_t_all = group["X_t"]
            y_t_all = group["y_t"]
            X_n_all = group["X_n"]
            y_n_all = group["y_n"]
            is_dummy = group["is_dummy"]
            B_total = int(coords_t_all.shape[0])
            m = int(group["max_m"])

            for start in range(0, B_total, self.target_chunk_size):
                end = min(start + self.target_chunk_size, B_total)
                dummy_mask = is_dummy[start:end].unsqueeze(-1)
                coords_t = coords_t_all[start:end]
                coords_n = coords_n_all[start:end]
                X_t = X_t_all[start:end]
                y_t = y_t_all[start:end]
                X_n = X_n_all[start:end]
                y_n = y_n_all[start:end]

                try:
                    K = self._cov_nn(coords_n, params)
                    k = self._cov_tn(coords_t, coords_n, params)
                    k = k.masked_fill(dummy_mask.squeeze(-1), 0.0)
                    L = torch.linalg.cholesky(K)
                    z = torch.linalg.solve_triangular(L, k.unsqueeze(-1), upper=False)
                    w = torch.linalg.solve_triangular(L.transpose(-1, -2), z, upper=True).squeeze(-1)
                except torch.linalg.LinAlgError:
                    if catch_cholesky:
                        return None
                    raise

                sigma_cond = total_sill - (z.squeeze(-1).pow(2)).sum(dim=1)
                if torch.any(sigma_cond <= 1e-10) or torch.isnan(sigma_cond).any():
                    if catch_cholesky:
                        return None
                    raise torch.linalg.LinAlgError("non-positive conditional variance")

                y_eff = y_t - (y_n * w).sum(dim=1, keepdim=True)
                X_eff = X_t - torch.einsum("bmf,bm->bf", X_n, w)
                inv_s = (1.0 / sigma_cond).unsqueeze(-1)

                XT_Sinv_X += X_eff.T @ (X_eff * inv_s)
                XT_Sinv_y += X_eff.T @ (y_eff * inv_s)
                if include_y_quad:
                    yT_Sinv_y += ((y_eff.squeeze(-1).pow(2) / sigma_cond).sum())
                log_det += torch.log(sigma_cond).sum()

        total_N = int(self.Heads_data.shape[0]) + int(self.n_tails)
        return XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, total_N

    def _gls_jitter(self):
        return torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-6

    def vecchia_batched_likelihood(self, params: torch.Tensor) -> torch.Tensor:
        stats = self._accumulate_gls_stats(params, include_y_quad=True, catch_cholesky=True)
        if stats is None:
            return torch.tensor(float("inf"), device=self.device, dtype=torch.float64)
        XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, total_N = stats
        try:
            beta = torch.linalg.solve(XT_Sinv_X + self._gls_jitter(), XT_Sinv_y)
        except torch.linalg.LinAlgError:
            return torch.tensor(float("inf"), device=self.device, dtype=torch.float64)
        quad = yT_Sinv_y - 2.0 * (beta.T @ XT_Sinv_y).squeeze() + (beta.T @ XT_Sinv_X @ beta).squeeze()
        return 0.5 * (log_det + quad) / total_N

    def get_gls_beta(self, params: torch.Tensor) -> torch.Tensor:
        XT_Sinv_X, XT_Sinv_y, _, _, _ = self._accumulate_gls_stats(params, include_y_quad=False, catch_cholesky=False)
        return torch.linalg.solve(XT_Sinv_X + self._gls_jitter(), XT_Sinv_y)

    def set_optimizer(
        self,
        param_groups,
        lr=1.0,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=10,
    ):
        return torch.optim.LBFGS(
            param_groups,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        phi1, phi2, phi3, phi4 = np.exp(raw[0]), np.exp(raw[1]), np.exp(raw[2]), np.exp(raw[3])
        return {
            "sigma_sq": phi1 / phi2,
            "range_lon": 1.0 / phi2,
            "range_lat": 1.0 / (phi2 * np.sqrt(phi3)),
            "range_time": 1.0 / (phi2 * np.sqrt(phi4)),
            "advec_lat": raw[4],
            "advec_lon": raw[5],
            "nugget": np.exp(raw[6]),
        }

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor], optimizer: torch.optim.LBFGS, max_steps: int = 50, grad_tol: float = 1e-5):
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print("--- Starting Batched Reverse-L Vecchia L-BFGS ---")

        def closure():
            optimizer.zero_grad()
            params = torch.stack([p.reshape(()) for p in params_list])
            loss = self.vecchia_batched_likelihood(params)
            loss.backward()
            return loss

        loss = None
        last_iter = 0
        for i in range(max_steps):
            last_iter = i
            loss = optimizer.step(closure)
            with torch.no_grad():
                grads = [abs(float(p.grad.detach().item())) for p in params_list if p.grad is not None]
                max_grad = max(grads) if grads else 0.0
                print(f"--- Step {i + 1}/{max_steps} / Loss: {float(loss.detach().item()):.4f} / Max Grad: {max_grad:.2e} ---")
                if max_grad < grad_tol:
                    print(f"Converged: max_grad {max_grad:.2e} < {grad_tol:.2e}")
                    break

        raw = [float(p.detach().cpu().item()) for p in params_list]
        final_loss = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float("nan")
        final_params = {k: round(float(v), 4) for k, v in self._convert_params(raw).items()}
        print("Final Interpretable Params:", final_params)
        return raw + [final_loss], last_iter


ReverseLColumnVecchiaFitV3 = ReverseLColumnVecchiaFitBatch

__all__ = ["ReverseLColumnVecchiaFitBatch", "ReverseLColumnVecchiaFitV3"]
