"""
kernels_space_050726.py

Pure-spatial Vecchia kernels for isolating same-time conditioning behavior.

Estimated covariance parameters:
  log(sigmasq), log(range_lat), log(range_lon), log(nugget)

The likelihood treats time slots as independent spatial replicates that share
the same covariance parameters.  There is no advection and no temporal range.
Regression coefficients are profiled by GLS, as in the spatio-temporal Vecchia
kernels used elsewhere in the project.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class _PureSpaceVecchiaBase:
    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        target_chunk_size: int = 4096,
    ):
        if smooth not in (0.5, 1.5):
            raise ValueError(f"smooth must be 0.5 or 1.5, got {smooth}")
        self.smooth = float(smooth)
        self.input_map = input_map
        first_val = next(iter(input_map.values()))
        self.device = first_val.device if isinstance(first_val, torch.Tensor) else torch.device("cpu")
        self.target_chunk_size = int(target_chunk_size)

        self.n_features = 9
        self.lat_mean_val = 0.0
        self.Full_Data = None
        self.Heads_data = None
        self.Batched_Groups = []
        self.Grouped_Batches = []
        self.n_tails = 0
        self.is_precomputed = False

    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_lat = torch.exp(params[1])
        range_lon = torch.exp(params[2])
        nugget = torch.exp(params[3])
        return sigmasq, range_lat, range_lon, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_lat, range_lon, _ = self._raw_params(params)
        dist = torch.sqrt(
            d_lat.new_tensor(1e-8)
            + (d_lat / range_lat).pow(2)
            + (d_lon / range_lon).pow(2)
        )
        if self.smooth == 0.5:
            return sigmasq * torch.exp(-dist)
        return sigmasq * (1.0 + dist) * torch.exp(-dist)

    def _cov_nn(self, neigh_coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = neigh_coords.unsqueeze(2) - neigh_coords.unsqueeze(1)
        cov = self._cov_from_deltas(diff[:, :, :, 0], diff[:, :, :, 1], params)
        _, _, _, nugget = self._raw_params(params)
        m = neigh_coords.shape[1]
        eye = torch.eye(m, device=self.device, dtype=torch.float64).unsqueeze(0)
        return cov + eye * (nugget + 1e-6)

    def _cov_tn(self, target_coords: torch.Tensor, neigh_coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = target_coords.unsqueeze(1) - neigh_coords
        return self._cov_from_deltas(diff[:, :, 0], diff[:, :, 1], params)

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        lat = (flat[:, 0:1] - self.lat_mean_val).to(torch.float64)
        dums = flat[:, 4:11].to(torch.float64)
        X = torch.cat([ones, lat, dums], dim=1)
        return X.reshape(*orig_shape, self.n_features)

    def _make_full_data(self, max_m: int):
        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        all_data_list = [d.to(self.device, dtype=torch.float64) for d in all_data_list]
        real_data = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = real_data.shape

        y = real_data[:, 2]
        coord_ok = ~torch.isnan(real_data[:, 0]) & ~torch.isnan(real_data[:, 1])
        obs_ok = (~torch.isnan(y)) & coord_ok
        valid_lats = real_data[obs_ok, 0]
        self.lat_mean_val = (
            float(valid_lats.mean().item()) if valid_lats.numel()
            else float(torch.nanmean(real_data[:, 0]).item())
        )

        max_m = max(0, int(max_m))
        if max_m > 0:
            dummy_block = torch.zeros((max_m, num_cols), device=self.device, dtype=torch.float64)
            for k in range(max_m):
                dummy_block[k, 0] = (k + 1) * 1e8
                dummy_block[k, 1] = (k + 1) * 1e8
                dummy_block[k, 3] = (k + 1) * 1e8
            full_data = torch.cat([real_data, dummy_block], dim=0).contiguous()
        else:
            full_data = real_data

        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if len(set(day_lengths)) != 1:
            raise ValueError(f"Pure-space kernels require equal grid length per time, got {day_lengths}")
        cumulative_len = np.cumsum([0] + day_lengths)
        valid_obs_np = obs_ok.detach().cpu().numpy()
        if max_m > 0:
            valid_obs_np = np.append(valid_obs_np, np.ones(max_m, dtype=bool))
        return all_data_list, full_data, n_real, num_cols, day_lengths, cumulative_len, valid_obs_np

    def _finalize_batches(
        self,
        full_data: torch.Tensor,
        target_neigh_pairs: List[Tuple[int, List[int]]],
        max_m: int,
        dummy_start: int,
        num_cols: int,
        heads_indices: Optional[List[int]] = None,
    ):
        heads_indices = heads_indices or []
        if heads_indices:
            h = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
            self.Heads_data = full_data[h].contiguous()
        else:
            self.Heads_data = torch.empty((0, num_cols), device=self.device, dtype=torch.float64)

        self.Batched_Groups = []
        if target_neigh_pairs:
            target_idx = torch.tensor([r[0] for r in target_neigh_pairs], device=self.device, dtype=torch.long)
            neigh_idx = torch.tensor([r[1] for r in target_neigh_pairs], device=self.device, dtype=torch.long)
            is_dummy = neigh_idx >= dummy_start if max_m > 0 else torch.zeros_like(neigh_idx, dtype=torch.bool)

            target_rows = full_data[target_idx].contiguous()
            neigh_rows = full_data[neigh_idx].contiguous() if max_m > 0 else torch.empty(
                (len(target_neigh_pairs), 0, num_cols), device=self.device, dtype=torch.float64
            )
            dummy_mask = is_dummy.unsqueeze(-1)
            self.Batched_Groups.append({
                "max_m": int(max_m),
                "target_idx": target_idx,
                "neigh_idx": neigh_idx,
                "is_dummy": is_dummy,
                "coords_t": target_rows[:, [0, 1]].contiguous(),
                "coords_n": neigh_rows[:, :, [0, 1]].contiguous() if max_m > 0 else neigh_rows[:, :, :2],
                "X_t": self._design_from_rows(target_rows).contiguous(),
                "y_t": target_rows[:, 2:3].contiguous(),
                "X_n": self._design_from_rows(neigh_rows).masked_fill(dummy_mask, 0.0).contiguous() if max_m > 0 else torch.empty(
                    (len(target_neigh_pairs), 0, self.n_features), device=self.device, dtype=torch.float64
                ),
                "y_n": neigh_rows[:, :, 2].masked_fill(is_dummy, 0.0).contiguous() if max_m > 0 else torch.empty(
                    (len(target_neigh_pairs), 0), device=self.device, dtype=torch.float64
                ),
            })

        self.Full_Data = full_data
        self.n_tails = int(sum(g["target_idx"].shape[0] for g in self.Batched_Groups))
        self.Grouped_Batches = []
        self.is_precomputed = True

    def _check_precomputed(self):
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first")

    def _accumulate_gls_stats(self, params: torch.Tensor, include_y_quad: bool = True, catch_cholesky: bool = False):
        self._check_precomputed()
        XT_Sinv_X = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        sigmasq, _, _, nugget = self._raw_params(params)
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
                coords_t = coords_t_all[start:end]
                X_t = X_t_all[start:end]
                y_t = y_t_all[start:end]

                if m > 0:
                    dummy_mask = is_dummy[start:end].unsqueeze(-1)
                    coords_n = coords_n_all[start:end]
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
                    y_eff = y_t - (y_n * w).sum(dim=1, keepdim=True)
                    X_eff = X_t - torch.einsum("bmf,bm->bf", X_n, w)
                else:
                    sigma_cond = total_sill.expand(coords_t.shape[0])
                    y_eff = y_t
                    X_eff = X_t

                if torch.any(sigma_cond <= 1e-10) or torch.isnan(sigma_cond).any():
                    if catch_cholesky:
                        return None
                    raise torch.linalg.LinAlgError("non-positive conditional variance")

                inv_s = (1.0 / sigma_cond).unsqueeze(-1)
                XT_Sinv_X += X_eff.T @ (X_eff * inv_s)
                XT_Sinv_y += X_eff.T @ (y_eff * inv_s)
                if include_y_quad:
                    yT_Sinv_y += (y_eff.squeeze(-1).pow(2) / sigma_cond).sum()
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

    def get_gls_beta(self, params_list: List[torch.Tensor]) -> torch.Tensor:
        params = torch.stack([p.reshape(()) for p in params_list])
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
        return {
            "sigmasq": float(np.exp(raw[0])),
            "range_lat": float(np.exp(raw[1])),
            "range_lon": float(np.exp(raw[2])),
            "nugget": float(np.exp(raw[3])),
        }

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor], optimizer: torch.optim.LBFGS, max_steps: int = 50, grad_tol: float = 1e-5):
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print("--- Starting Pure-Space Vecchia L-BFGS ---")

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
                print(f"--- Step {i + 1}/{max_steps} / Loss: {float(loss.detach().item()):.6f} / Max Grad: {max_grad:.2e} ---")
                if max_grad < grad_tol:
                    print(f"Converged: max_grad {max_grad:.2e} < {grad_tol:.2e}")
                    break

        raw = [float(p.detach().cpu().item()) for p in params_list]
        final_loss = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float("nan")
        print("Final Pure-Space Params:", self._convert_params(raw))
        return raw + [final_loss], last_iter


class HybridSpaceVecchiaFit(_PureSpaceVecchiaBase):
    """Same-time nearest-neighbor Vecchia using the existing max-min NN map."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        nns_map,
        limit_A: int = 20,
        target_chunk_size: int = 4096,
    ):
        super().__init__(smooth=smooth, input_map=input_map, target_chunk_size=target_chunk_size)
        self.nns_map = nns_map
        self.limit_A = int(limit_A)

    def precompute_conditioning_sets(self):
        print(f"Pre-computing HybridSpaceVecchia [A={self.limit_A}]...", end=" ")
        t0 = time.time()
        all_data_list, full_data, n_real, num_cols, day_lengths, cumulative_len, valid_obs_np = self._make_full_data(self.limit_A)
        dummy_start = n_real
        rows: List[Tuple[int, List[int]]] = []
        m_sizes = []

        for time_idx, day_len in enumerate(day_lengths):
            offset = int(cumulative_len[time_idx])
            for local_idx in range(day_len):
                target_global = offset + local_idx
                if not valid_obs_np[target_global]:
                    continue
                neigh = []
                seen = set()
                nbs_current = (
                    self.nns_map[local_idx]
                    if local_idx < len(self.nns_map)
                    else np.array([], dtype=np.int64)
                )
                for nb in nbs_current:
                    if len(neigh) >= self.limit_A:
                        break
                    nb = int(nb)
                    if nb < 0 or nb >= day_len:
                        continue
                    g = offset + nb
                    if g == target_global or g in seen or not valid_obs_np[g]:
                        continue
                    neigh.append(g)
                    seen.add(g)
                m_sizes.append(len(neigh))
                if len(neigh) < self.limit_A:
                    padded = [dummy_start + k for k in range(self.limit_A - len(neigh))] + neigh
                else:
                    padded = neigh[-self.limit_A:]
                rows.append((target_global, padded))

        self._finalize_batches(full_data, rows, self.limit_A, dummy_start, num_cols)
        m_arr = np.asarray(m_sizes) if m_sizes else np.asarray([0])
        print(
            f"Done in {time.time() - t0:.1f}s. tails={self.n_tails}, "
            f"m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max()}"
        )
        return self


class ColumnSpaceVecchiaFit(_PureSpaceVecchiaBase):
    """Same-time reverse-L column conditioning on a regular-grid scan."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        grid_coords: Optional[np.ndarray] = None,
        above_count: int = 2,
        right_col_count: int = 3,
        per_time_conditioning_count: int = 14,
        head_right_cols: int = 0,
        lat_round_decimals: int = 6,
        lon_round_decimals: int = 6,
        target_chunk_size: int = 4096,
    ):
        super().__init__(smooth=smooth, input_map=input_map, target_chunk_size=target_chunk_size)
        self.grid_coords = grid_coords
        self.above_count = int(above_count)
        self.right_col_count = int(right_col_count)
        self.per_time_conditioning_count = int(per_time_conditioning_count)
        self.head_right_cols = int(head_right_cols)
        self.lat_round_decimals = int(lat_round_decimals)
        self.lon_round_decimals = int(lon_round_decimals)
        self._n_lat = 0
        self._n_lon = 0
        self._row_col_to_local = {}
        self._stencil_cache = {}

    def _regular_coords_np(self, n_points: int) -> np.ndarray:
        if self.grid_coords is not None:
            coords = np.asarray(self.grid_coords[:n_points], dtype=np.float64)
        else:
            first = next(iter(self.input_map.values()))
            coords = first[:n_points, :2].detach().cpu().numpy() if isinstance(first, torch.Tensor) else np.asarray(first[:n_points, :2])
        return np.asarray(coords, dtype=np.float64)

    def _build_grid_maps(self, coords: np.ndarray):
        lat_key = np.round(coords[:, 0], self.lat_round_decimals)
        lon_key = np.round(coords[:, 1], self.lon_round_decimals)
        lats = np.sort(np.unique(lat_key))
        lons = np.sort(np.unique(lon_key))
        lat_to_row = {v: i for i, v in enumerate(lats)}
        lon_to_col = {v: i for i, v in enumerate(lons)}
        row_col_to_local = {}
        for i, (la, lo) in enumerate(zip(lat_key, lon_key)):
            row_col_to_local[(lat_to_row[la], lon_to_col[lo])] = i
        self._n_lat = len(lats)
        self._n_lon = len(lons)
        self._row_col_to_local = row_col_to_local
        self._stencil_cache = {}

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

    def precompute_conditioning_sets(self):
        print(
            f"Pre-computing ColumnSpaceVecchia [head_right={self.head_right_cols}, "
            f"above={self.above_count}, right_cols={self.right_col_count}, m={self.per_time_conditioning_count}]...",
            end=" ",
        )
        t0 = time.time()
        all_data_list, full_data, n_real, num_cols, day_lengths, cumulative_len, valid_obs_np = self._make_full_data(
            self.per_time_conditioning_count
        )
        n_grid = day_lengths[0]
        self._build_grid_maps(self._regular_coords_np(n_grid))
        dummy_start = n_real

        head_locals = set()
        if self.head_right_cols > 0:
            for c in range(max(0, self._n_lon - self.head_right_cols), self._n_lon):
                for r in range(self._n_lat):
                    nb = self._get_local(r, c)
                    if nb is not None:
                        head_locals.add(nb)

        rows: List[Tuple[int, List[int]]] = []
        heads: List[int] = []
        m_sizes = []
        col_order = range(self._n_lon - 1, -1, -1)
        row_order = range(self._n_lat - 1, -1, -1)

        for time_idx, day_len in enumerate(day_lengths):
            offset = int(cumulative_len[time_idx])
            for col in col_order:
                for row in row_order:
                    local_idx = self._get_local(row, col)
                    if local_idx is None:
                        continue
                    target_global = offset + local_idx
                    if not valid_obs_np[target_global]:
                        continue
                    if local_idx in head_locals:
                        heads.append(target_global)
                        continue

                    neigh = []
                    seen = set()
                    for nb_local in self._spatial_candidate_locals_uncapped(row, col):
                        if len(neigh) >= self.per_time_conditioning_count:
                            break
                        g = offset + int(nb_local)
                        if g in seen or not valid_obs_np[g]:
                            continue
                        neigh.append(g)
                        seen.add(g)
                    m_sizes.append(len(neigh))
                    if len(neigh) < self.per_time_conditioning_count:
                        padded = [dummy_start + k for k in range(self.per_time_conditioning_count - len(neigh))] + neigh
                    else:
                        padded = neigh[-self.per_time_conditioning_count:]
                    rows.append((target_global, padded))

        self._finalize_batches(full_data, rows, self.per_time_conditioning_count, dummy_start, num_cols, heads)
        m_arr = np.asarray(m_sizes) if m_sizes else np.asarray([0])
        print(
            f"Done in {time.time() - t0:.1f}s. grid={self._n_lat}x{self._n_lon}, "
            f"heads={len(heads)}, tails={self.n_tails}, "
            f"m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max()}"
        )
        return self


__all__ = ["HybridSpaceVecchiaFit", "ColumnSpaceVecchiaFit"]
