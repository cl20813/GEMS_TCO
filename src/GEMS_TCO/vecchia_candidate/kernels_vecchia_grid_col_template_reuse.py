
"""
kernels_vecchia_grid_col_template_reuse.py

Regular-grid reverse-L Vecchia kernel with template-level reuse.

This module intentionally uses regular grid coordinates, not recovered source
coordinates.  Conditioning sets are built from a deterministic right-to-left,
top-to-bottom scan order:

- the rightmost `head_right_cols` columns are treated as exact head points;
- each tail target conditions on an inverse-L stencil made of
  `above_count` points above the target in the same column plus
  `right_neighbor_count` points selected from the next `right_col_count`
  columns to the east;
- the same spatial stencil is repeated at lags t-1, ..., t-lag_count.

Targets with the same offset pattern are grouped.  For each parameter value the
small conditional system is Cholesky-factorized once per group/template, then
reused for all targets in that group.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from scipy.special import gamma


class ReverseLColumnVecchiaFit:
    """Regular-grid reverse-L Vecchia model with template grouped likelihood."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        mm_cond_number: int = 300,
        nheads: int = 0,
        grid_coords: Optional[np.ndarray] = None,
        head_right_cols: int = 3,
        above_count: int = 4,
        right_col_count: int = 3,
        right_neighbor_count: int = 80,
        lag_count: int = 2,
        include_lag_self: bool = False,
        lat_round_decimals: int = 6,
        lon_round_decimals: int = 6,
        target_chunk_size: int = 2048,
        head_mode: str = "all",
    ):
        if smooth not in (0.5, 1.5):
            raise ValueError(f"smooth must be 0.5 or 1.5, got {smooth}")
        if head_mode != "all":
            raise ValueError("Only head_mode='all' is implemented in this first version")
        self.smooth = float(smooth)
        self.input_map = input_map
        self.mm_cond_number = int(mm_cond_number)
        self.nheads = int(nheads)  # kept for API compatibility, not used.
        self.grid_coords = grid_coords
        self.head_right_cols = int(head_right_cols)
        self.above_count = int(above_count)
        self.right_col_count = int(right_col_count)
        self.right_neighbor_count = int(right_neighbor_count)
        self.lag_count = int(lag_count)
        self.include_lag_self = bool(include_lag_self)
        self.lat_round_decimals = int(lat_round_decimals)
        self.lon_round_decimals = int(lon_round_decimals)
        self.target_chunk_size = int(target_chunk_size)
        self.head_mode = head_mode

        first_val = next(iter(input_map.values()))
        if isinstance(first_val, torch.Tensor):
            self.device = first_val.device
        else:
            self.device = torch.device("cpu")

        self.n_features = 9  # intercept, centered lat, seven hour dummies
        self.lat_mean_val = 0.0
        self.is_precomputed = False

        self.Full_Data_Grid = None
        self.Heads_data = None
        self.Grouped_Batches = []
        self.n_tails = 0
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

    def _spatial_stencil_locals(self, row: int, col: int) -> List[int]:
        """Return deterministic spatial inverse-L local indices for a grid cell."""
        key = (int(row), int(col))
        if key in self._stencil_cache:
            return self._stencil_cache[key]

        out: List[int] = []
        seen = set()

        # Above means north / larger latitude. Rows are sorted by latitude ascending.
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
            for r2 in range(self._n_lat):
                nb = self._get_local(r2, c2)
                if nb is None or nb in seen:
                    continue
                # Prefer same-row vicinity, then nearer east column, then north side.
                right_candidates.append((abs(r2 - row), dc, 0 if r2 >= row else 1, r2, nb))
        right_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        for _, _, _, _, nb in right_candidates[: self.right_neighbor_count]:
            if nb not in seen:
                out.append(nb)
                seen.add(nb)

        self._stencil_cache[key] = out
        return out

    # ------------------------------------------------------------------
    # Covariance and design helpers
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

    def _stencil_cov_matrix(self, offsets: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = offsets.unsqueeze(1) - offsets.unsqueeze(0)
        cov = self._cov_from_deltas(diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], params)
        nugget = torch.exp(params[6])
        cov = cov.clone()
        cov.diagonal().add_(nugget + 1e-6)
        return cov

    def _cross_cov_vector(self, offsets: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self._cov_from_deltas(offsets[:, 0], offsets[:, 1], offsets[:, 2], params)

    def _full_cov(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        cov = self._cov_from_deltas(diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], params)
        nugget = torch.exp(params[6])
        cov = cov.clone()
        cov.diagonal().add_(nugget + 1e-8)
        return cov

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        lat = (flat[:, 0:1] - self.lat_mean_val).to(torch.float64)
        dums = flat[:, 4:11].to(torch.float64)
        X = torch.cat([ones, lat, dums], dim=1)
        return X.reshape(*orig_shape, self.n_features)

    # ------------------------------------------------------------------
    # Conditioning-set construction
    # ------------------------------------------------------------------

    def precompute_conditioning_sets(self):
        print(
            "Pre-computing ReverseLColumnVecchia "
            f"[heads_right={self.head_right_cols}, above={self.above_count}, "
            f"right_cols={self.right_col_count}, right_n={self.right_neighbor_count}, "
            f"lags={self.lag_count}]...",
            end=" ",
        )
        t0 = time.time()

        key_list = list(self.input_map.keys())
        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        all_data_list = [d.to(self.device, dtype=torch.float64) for d in all_data_list]
        self.Full_Data_Grid = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = self.Full_Data_Grid.shape
        self._n_real = n_real
        self._n_time = len(all_data_list)
        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if len(set(day_lengths)) != 1:
            raise ValueError(f"ReverseLColumnVecchia requires equal grid length per time, got {day_lengths}")
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

        y = self.Full_Data_Grid[:, 2]
        valid_y_np = (~torch.isnan(y)).detach().cpu().numpy()
        valid_lats = self.Full_Data_Grid[~torch.isnan(y), 0]
        self.lat_mean_val = float(valid_lats.mean().item()) if valid_lats.numel() else float(self.Full_Data_Grid[:, 0].mean().item())

        time_values = []
        for d in all_data_list:
            good = ~torch.isnan(d[:, 3])
            time_values.append(float(d[good, 3].median().item()) if good.any() else 0.0)

        head_locals = set()
        for c in range(max(0, self._n_lon - self.head_right_cols), self._n_lon):
            for r in range(self._n_lat):
                nb = self._get_local(r, c)
                if nb is not None:
                    head_locals.add(nb)

        heads_indices = []
        groups: Dict[Tuple[Tuple[float, float, float], ...], Dict[str, Any]] = {}
        m_sizes = []

        col_order = range(self._n_lon - 1, -1, -1)
        row_order = range(self._n_lat - 1, -1, -1)  # north to south

        def add_group(target_global: int, neigh_globals: List[int], offsets: List[Tuple[float, float, float]]):
            if len(neigh_globals) == 0:
                key = tuple()
                off_tensor = torch.empty((0, 3), device=self.device, dtype=torch.float64)
            else:
                key = tuple((round(a, 8), round(b, 8), round(c, 8)) for a, b, c in offsets)
                off_tensor = torch.tensor(offsets, device=self.device, dtype=torch.float64)
            if key not in groups:
                groups[key] = {"offsets": off_tensor, "batch_idx": [], "target_idx": []}
            groups[key]["batch_idx"].append(neigh_globals)
            groups[key]["target_idx"].append(target_global)
            m_sizes.append(len(neigh_globals))

        for time_idx in range(self._n_time):
            offset = int(cumulative_len[time_idx])
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

                    target_lat = float(coords_np[local_idx, 0])
                    target_lon = float(coords_np[local_idx, 1])
                    neigh_globals: List[int] = []
                    offsets_list: List[Tuple[float, float, float]] = []
                    seen = set()

                    spatial_locals = self._spatial_stencil_locals(row, col)
                    for lag in range(self.lag_count + 1):
                        neigh_time_idx = time_idx - lag
                        if neigh_time_idx < 0:
                            continue
                        neigh_time_offset = int(cumulative_len[neigh_time_idx])
                        dt = float(time_values[time_idx] - time_values[neigh_time_idx])

                        if lag > 0 and self.include_lag_self:
                            g = neigh_time_offset + local_idx
                            if g not in seen and valid_y_np[g]:
                                neigh_globals.append(g)
                                offsets_list.append((0.0, 0.0, dt))
                                seen.add(g)

                        for nb_local in spatial_locals:
                            g = neigh_time_offset + int(nb_local)
                            if g in seen or not valid_y_np[g]:
                                continue
                            nb_lat = float(coords_np[nb_local, 0])
                            nb_lon = float(coords_np[nb_local, 1])
                            neigh_globals.append(g)
                            offsets_list.append((target_lat - nb_lat, target_lon - nb_lon, dt))
                            seen.add(g)

                    add_group(target_global, neigh_globals, offsets_list)

        if heads_indices:
            h = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
            self.Heads_data = self.Full_Data_Grid[h].contiguous()
        else:
            self.Heads_data = torch.empty((0, num_cols), device=self.device, dtype=torch.float64)

        self.Grouped_Batches = []
        for val in groups.values():
            t_idx = torch.tensor(val["target_idx"], device=self.device, dtype=torch.long)
            if val["offsets"].shape[0] == 0:
                b_idx = torch.empty((len(val["target_idx"]), 0), device=self.device, dtype=torch.long)
            else:
                b_idx = torch.tensor(val["batch_idx"], device=self.device, dtype=torch.long)
            self.Grouped_Batches.append({"offsets": val["offsets"], "batch_idx": b_idx, "target_idx": t_idx})

        self.n_tails = int(sum(len(g["target_idx"]) for g in self.Grouped_Batches))
        self.is_precomputed = True
        if m_sizes:
            m_arr = np.asarray(m_sizes)
            m_msg = f"m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max()}"
        else:
            m_msg = "m empty"
        print(
            f"Done in {time.time() - t0:.1f}s. "
            f"grid={self._n_lat}x{self._n_lon}, heads={len(heads_indices)}, "
            f"tails={self.n_tails}, templates={len(self.Grouped_Batches)}, {m_msg}"
        )
        return self

    # ------------------------------------------------------------------
    # Likelihood and optimization
    # ------------------------------------------------------------------

    def _check_precomputed(self):
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first")

    def _accumulate_gls_stats(self, params: torch.Tensor, include_y_quad: bool = True, catch_cholesky: bool = False):
        self._check_precomputed()
        XT_Sinv_X = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        if self.Heads_data.shape[0] > 0:
            coords = self.Heads_data[:, [0, 1, 3]].contiguous()
            X_h = self._design_from_rows(self.Heads_data)
            y_h = self.Heads_data[:, 2:3]
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

        for group in self.Grouped_Batches:
            offsets = group["offsets"]
            target_idx = group["target_idx"]
            b_idx = group["batch_idx"]
            B_total = int(target_idx.shape[0])
            m = int(offsets.shape[0])

            if m == 0:
                inv_s = 1.0 / total_sill
                log_s = torch.log(total_sill)
                for start in range(0, B_total, self.target_chunk_size):
                    end = min(start + self.target_chunk_size, B_total)
                    rows_t = self.Full_Data_Grid[target_idx[start:end]]
                    X_t = self._design_from_rows(rows_t)
                    y_t = rows_t[:, 2:3]
                    XT_Sinv_X += (X_t.T @ X_t) * inv_s
                    XT_Sinv_y += (X_t.T @ y_t) * inv_s
                    if include_y_quad:
                        yT_Sinv_y += ((y_t.T @ y_t).squeeze() * inv_s).squeeze()
                    log_det += (end - start) * log_s
                continue

            try:
                K = self._stencil_cov_matrix(offsets, params)
                k = self._cross_cov_vector(offsets, params)
                L = torch.linalg.cholesky(K)
                z = torch.linalg.solve_triangular(L, k.unsqueeze(1), upper=False)
                w = torch.linalg.solve_triangular(L.T, z, upper=True).flatten()
            except torch.linalg.LinAlgError:
                if catch_cholesky:
                    return None
                raise

            sigma_cond = total_sill - torch.dot(z.flatten(), z.flatten())
            if torch.any(sigma_cond <= 1e-10) or torch.isnan(sigma_cond):
                if catch_cholesky:
                    return None
                raise torch.linalg.LinAlgError("non-positive conditional variance")
            inv_s = 1.0 / sigma_cond
            log_s = torch.log(sigma_cond)

            for start in range(0, B_total, self.target_chunk_size):
                end = min(start + self.target_chunk_size, B_total)
                rows_t = self.Full_Data_Grid[target_idx[start:end]]
                X_t = self._design_from_rows(rows_t)
                y_t = rows_t[:, 2:3]

                b_chunk = b_idx[start:end]
                flat_b = b_chunk.reshape(-1)
                rows_n = self.Full_Data_Grid[flat_b].reshape(end - start, m, -1)
                y_n = rows_n[:, :, 2]
                X_n = self._design_from_rows(rows_n)

                y_eff = y_t - (y_n @ w).unsqueeze(1)
                X_eff = X_t - torch.einsum("bmf,m->bf", X_n, w)

                XT_Sinv_X += (X_eff.T @ X_eff) * inv_s
                XT_Sinv_y += (X_eff.T @ y_eff) * inv_s
                if include_y_quad:
                    yT_Sinv_y += ((y_eff.T @ y_eff).squeeze() * inv_s).squeeze()
                log_det += (end - start) * log_s

        total_N = int(self.Heads_data.shape[0]) + int(self.n_tails)
        return XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, total_N

    def vecchia_structured_likelihood(self, params: torch.Tensor) -> torch.Tensor:
        stats = self._accumulate_gls_stats(params, include_y_quad=True, catch_cholesky=True)
        if stats is None:
            return torch.tensor(float("inf"), device=self.device, dtype=torch.float64)
        XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, total_N = stats
        jitter = torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-6
        try:
            beta = torch.linalg.solve(XT_Sinv_X + jitter, XT_Sinv_y)
        except torch.linalg.LinAlgError:
            return torch.tensor(float("inf"), device=self.device, dtype=torch.float64)
        quad = yT_Sinv_y - 2.0 * (beta.T @ XT_Sinv_y).squeeze() + (beta.T @ XT_Sinv_X @ beta).squeeze()
        return 0.5 * (log_det + quad) / total_N

    def get_gls_beta(self, params: torch.Tensor) -> torch.Tensor:
        XT_Sinv_X, XT_Sinv_y, _, _, _ = self._accumulate_gls_stats(params, include_y_quad=False, catch_cholesky=False)
        jitter = torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-6
        return torch.linalg.solve(XT_Sinv_X + jitter, XT_Sinv_y)

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
            "sigmasq": phi1 / phi2,
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

        print("--- Starting Reverse-L Column Vecchia L-BFGS ---")
        loss = None
        last_iter = 0

        def closure():
            optimizer.zero_grad()
            params = torch.stack([p.reshape(()) for p in params_list])
            val = self.vecchia_structured_likelihood(params)
            val.backward()
            return val

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

        out = [p.detach().cpu().clone() for p in params_list]
        out.append(loss.detach().cpu().clone() if isinstance(loss, torch.Tensor) else torch.tensor(float("nan")))
        return out, last_iter
