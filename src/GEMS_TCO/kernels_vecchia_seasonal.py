"""
kernels_vecchia_seasonal.py
===========================
Vecchia approximation for multi-month (Apr–Sep) spatiotemporal data.

Conditioning set — AR(1) + daily structure:
  target (s, t)
    ├─ Group 1  (t = 0)   : k_s spatial neighbors at t
    ├─ Group 2  (0 < t < 8): k_s@t  +  self@t-1  +  k_s@t-1
    └─ Group 3  (t ≥ 8)   : k_s@t  +  self@t-1  +  k_s@t-1
                                     +  self@t-8  +  k_s@t-8

t-2 … t-7 are intentionally skipped (AR(1) + daily, no intermediate lags).
Total conditioning: k_s + (1+k_s) + (1+k_s) = 3*k_s + 2  (= 20 for k_s=6)

Key difference from VecchiaBatched:
  - No "heads" (exact GP) — all observations use Vecchia.
    (Heads in VecchiaBatched require O(nheads*T)^2 memory; infeasible for T≥180*8.)
  - daily_stride is fixed at 8 (one GEMS day = 8 time steps).
  - Same covariance function and optimizer as existing code.
"""

from scipy.special import gamma
import numpy as np
import torch
from typing import Dict, Any, List


# ── Covariance (identical to SpatioTemporalModel) ─────────────────────────────

class _CovBase:
    def __init__(self, smooth: float):
        self.smooth = smooth
        gamma_val = torch.tensor(gamma(smooth), dtype=torch.float64)
        self.matern_const = (2 ** (1 - smooth)) / gamma_val

    def _aniso_dist(self, dist_params, x, y):
        phi3, phi4, adv_lat, adv_lon = dist_params
        def _shift(z):
            return torch.stack([z[:, 0] - adv_lat * z[:, 3],
                                z[:, 1] - adv_lon * z[:, 3],
                                z[:, 3]], dim=1)
        u, v = _shift(x), _shift(y)
        w = torch.stack([phi3.view(1),
                         torch.ones(1, device=x.device, dtype=phi3.dtype),
                         phi4.view(1)]).view(-1)
        u_sq = (u.pow(2) * w).sum(1, keepdim=True)
        v_sq = (v.pow(2) * w).sum(1, keepdim=True)
        uv   = (u * w) @ v.T
        return torch.sqrt((u_sq - 2*uv + v_sq.T).clamp(min=1e-8))

    def _batched_dist(self, dist_params, x_batch):
        phi3, phi4, adv_lat, adv_lon = dist_params
        lat  = x_batch[:, :, 0] - adv_lat  * x_batch[:, :, 2]
        lon  = x_batch[:, :, 1] - adv_lon  * x_batch[:, :, 2]
        t    = x_batch[:, :, 2]
        dl = lat.unsqueeze(2) - lat.unsqueeze(1)
        dlo = lon.unsqueeze(2) - lon.unsqueeze(1)
        dt  = t.unsqueeze(2)  - t.unsqueeze(1)
        return torch.sqrt(dl.pow(2)*phi3 + dlo.pow(2) + dt.pow(2)*phi4 + 1e-8)

    # Maximum advection speed (deg/hr).
    # Daily model: 0.02 deg/hr × 7h = 0.14 deg shift — fine.
    # Seasonal model: 0.02 × 744h = 14.9 deg — destroys distances.
    # Use tanh so optimizer can never exceed this bound (gradient always ≠ 0).
    ADV_MAX = 0.002   # 0.002 deg/hr × 744h = 1.5 deg max shift (≈ range_lat scale)

    def _effective_adv(self, params):
        """Bounded advection via tanh: adv ∈ (-ADV_MAX, ADV_MAX) always."""
        adv_lat = self.ADV_MAX * torch.tanh(params[4] / self.ADV_MAX)
        adv_lon = self.ADV_MAX * torch.tanh(params[5] / self.ADV_MAX)
        return adv_lat, adv_lon

    def cov_exact(self, params, x, y):
        """Full N×M covariance matrix (for small blocks)."""
        p       = torch.clamp(params, min=-15.0, max=15.0)
        phi1, phi2, phi3, phi4 = torch.exp(p[0:4])
        nugget  = torch.exp(p[6])
        adv_lat, adv_lon = self._effective_adv(params)   # tanh-bounded
        adv     = torch.stack([phi3, phi4, adv_lat, adv_lon])
        dist    = self._aniso_dist(adv, x, y)
        cov     = (phi1/phi2) * torch.exp(-dist * phi2)
        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8)
        return cov

    def cov_batched(self, params, x_batch):
        """Batched B×N×N covariance (for tail observations)."""
        p       = torch.clamp(params, min=-15.0, max=15.0)
        phi1, phi2, phi3, phi4 = torch.exp(p[0:4])
        nugget  = torch.exp(p[6])
        adv_lat, adv_lon = self._effective_adv(params)   # tanh-bounded
        adv     = torch.stack([phi3, phi4, adv_lat, adv_lon])
        dist    = self._batched_dist(adv, x_batch)
        cov     = (phi1/phi2) * torch.exp(-dist * phi2)
        B, N, _ = x_batch.shape
        eye = torch.eye(N, device=x_batch.device,
                        dtype=torch.float64).unsqueeze(0).expand(B, N, N)
        return cov + eye * (nugget + 1e-6)


# ── Main class ────────────────────────────────────────────────────────────────

class VecchiaAR1Daily(_CovBase):
    """
    Parameters
    ----------
    smooth     : Matérn smoothness ν
    input_map  : {time_key: Tensor[N, ≥11]}
                 Columns: [lat, lon, O3, Hours_elapsed, D1…D7]
                 Keys must be sorted chronologically.
                 Each time step must have the same number of rows (NaN for missing).
    nns_map    : spatial KNN indices (from orderings.find_nns_l2 on MaxMin-ordered locs)
    k_space    : number of spatial neighbors (default 6)
    """
    DAILY_STRIDE = 8          # one GEMS day = 8 time steps, fixed

    def __init__(self, smooth: float,
                 input_map: Dict[str, Any],
                 nns_map,
                 k_space: int = 6):
        super().__init__(smooth)
        self.input_map = input_map
        self.k_space   = k_space
        self.n_features = 9          # intercept + lat_centered + D1…D7

        first = next(iter(input_map.values()))
        self.device = first.device if isinstance(first, torch.Tensor) else torch.device('cpu')

        # Clean nns_map (remove -1 padding)
        nns = list(nns_map)
        for i in range(len(nns)):
            tmp = np.delete(nns[i], np.where(nns[i] == -1))
            nns[i] = tmp if tmp.size > 0 else np.array([], dtype=np.int64)
        self.nns_map = nns

        self.is_precomputed = False
        self.lat_mean_val   = 0.0

        # Filled by precompute_conditioning_sets
        self.G1_X = self.G1_Y = self.G1_Locs = None   # spatial only
        self.G2_X = self.G2_Y = self.G2_Locs = None   # + t-1
        self.G3_X = self.G3_Y = self.G3_Locs = None   # + t-1 + t-8
        self.n_obs = 0

    # ── Precompute ────────────────────────────────────────────────────────────

    def precompute_conditioning_sets(self):
        k = self.k_space
        S  = self.DAILY_STRIDE

        # Sizes: target + conditioning
        dim_G1 = 1 + k               # target + k_s@t
        dim_G2 = 1 + k + (1 + k)    # + self@t-1 + k_s@t-1  = 2k+2
        dim_G3 = 1 + k + (1+k)+(1+k)  # + self@t-8 + k_s@t-8 = 3k+3

        print(f"🚀 VecchiaAR1Daily  k_space={k}  "
              f"[G1={dim_G1-1}, G2={dim_G2-1}, G3={dim_G3-1} cond pts]", end=" ")

        key_list    = list(self.input_map.keys())
        all_tensors = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                       for d in self.input_map.values()]
        Real_Data   = torch.cat(all_tensors, dim=0).to(self.device, dtype=torch.float32)
        n_real, n_cols = Real_Data.shape

        is_nan_np = torch.isnan(Real_Data[:, 2]).cpu().numpy()
        valid_lats = Real_Data[~torch.isnan(Real_Data[:, 2]), 0]
        self.lat_mean_val = float(valid_lats.mean()) if valid_lats.numel() > 0 else 0.0

        # Dummy rows for padding
        n_dummy = max(dim_G1, dim_G2, dim_G3)
        dummy   = torch.zeros((n_dummy, n_cols), device=self.device, dtype=torch.float32)
        for k_ in range(n_dummy):
            dummy[k_, 0] = dummy[k_, 1] = dummy[k_, 3] = (k_+1) * 1e8
        Full_Data  = torch.cat([Real_Data, dummy], dim=0)
        dummy_start = n_real
        is_nan_np   = np.append(is_nan_np, np.zeros(n_dummy, dtype=bool))

        day_lens   = [len(d) for d in all_tensors]
        cum_len    = np.cumsum([0] + day_lens)
        n_T        = len(key_list)

        g1_list, g2_list, g3_list = [], [], []
        n_valid_total = 0

        def pick(indices, cap):
            """Return up to cap valid (non-NaN) indices."""
            out = []
            for idx in indices:
                if len(out) >= cap:
                    break
                if not is_nan_np[idx]:
                    out.append(idx)
            return out

        for t_idx in range(n_T):
            off = cum_len[t_idx]
            N_t = day_lens[t_idx]

            has_t1 = (t_idx >= 1)
            has_t8 = (t_idx >= S)

            off_t1 = cum_len[t_idx - 1]  if has_t1 else None
            N_t1   = day_lens[t_idx - 1] if has_t1 else 0
            off_t8 = cum_len[t_idx - S]  if has_t8 else None
            N_t8   = day_lens[t_idx - S] if has_t8 else 0

            for loc_idx in range(N_t):
                tgt = off + loc_idx
                if is_nan_np[tgt]:
                    continue
                n_valid_total += 1

                nbs = self.nns_map[loc_idx] if loc_idx < len(self.nns_map) else np.array([])

                # ── spatial neighbors at t ─────────────────────────────────
                sp_t = pick((off + nbs[nbs < N_t]).tolist(), k)

                # ── build conditioning list ────────────────────────────────
                cond = list(sp_t)   # k_s@t  (already ordered before target)

                if has_t1:
                    # self@t-1
                    if loc_idx < N_t1:
                        cond += pick([off_t1 + loc_idx], 1)
                    # k_s neighbors@t-1
                    cond += pick((off_t1 + nbs[nbs < N_t1]).tolist(), k)

                if has_t8:
                    # self@t-8
                    if loc_idx < N_t8:
                        cond += pick([off_t8 + loc_idx], 1)
                    # k_s neighbors@t-8
                    cond += pick((off_t8 + nbs[nbs < N_t8]).tolist(), k)

                # target goes LAST (Vecchia convention: last row is the target)
                row_indices = cond + [tgt]

                if   not has_t1: g1_list.append(row_indices)
                elif not has_t8: g2_list.append(row_indices)
                else:            g3_list.append(row_indices)

        self.n_obs = n_valid_total
        print(f"[T={n_T}, obs={n_valid_total}]", end=" ")

        self.G1_X, self.G1_Y, self.G1_Locs = self._build_tensors(
            g1_list, dim_G1, Full_Data, dummy_start, n_cols)
        self.G2_X, self.G2_Y, self.G2_Locs = self._build_tensors(
            g2_list, dim_G2, Full_Data, dummy_start, n_cols)
        self.G3_X, self.G3_Y, self.G3_Locs = self._build_tensors(
            g3_list, dim_G3, Full_Data, dummy_start, n_cols)

        self.is_precomputed = True
        print(f"✅ Done. (G1={len(g1_list)}, G2={len(g2_list)}, G3={len(g3_list)})")

    def _build_tensors(self, idx_list, max_dim, Full_Data, dummy_start, n_cols):
        if not idx_list:
            return None, None, None

        # Pad each row to max_dim (dummies at the front)
        padded = []
        for row in idx_list:
            n_pad = max_dim - len(row)
            pad   = [dummy_start + i for i in range(n_pad)]
            padded.append(pad + row)

        T         = torch.tensor(padded, device=self.device, dtype=torch.long)
        G         = Full_Data[T].to(torch.float64)                 # [B, max_dim, n_cols]
        X         = G[..., [0, 1, 3]].contiguous()                # [B, max_dim, 3]  lat,lon,time
        Y         = G[..., 2].unsqueeze(-1).contiguous()           # [B, max_dim, 1]
        ones      = torch.ones_like(G[..., 0]).unsqueeze(-1)
        lat_c     = (G[..., 0] - self.lat_mean_val).unsqueeze(-1)
        dums      = G[..., 4:11]                                   # D1…D7
        Locs      = torch.cat([ones, lat_c, dums], dim=-1).contiguous()  # [B, max_dim, 9]

        is_dummy  = (T >= dummy_start).unsqueeze(-1)
        Locs      = Locs.masked_fill(is_dummy, 0.0)
        Y         = Y.masked_fill(is_dummy, 0.0)
        return X, Y, Locs

    # ── Likelihood ────────────────────────────────────────────────────────────

    def vecchia_likelihood(self, params: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized negative log-likelihood (divided by n_obs).
        Compatible with autograd for L-BFGS.
        """
        if not self.is_precomputed:
            raise RuntimeError("Call precompute_conditioning_sets() first.")

        n_f = self.n_features
        XSX: torch.Tensor | None = None
        XSy: torch.Tensor | None = None
        ySy: torch.Tensor | None = None
        ld:  torch.Tensor | None = None

        # Differentiable penalty returned when cholesky fails.
        # Keeps computation graph alive so L-BFGS can update params.
        _penalty = torch.sum(params * 0) + 1e8

        chunk = 4096
        for X_b, Y_b, L_b in [(self.G1_X, self.G1_Y, self.G1_Locs),
                               (self.G2_X, self.G2_Y, self.G2_Locs),
                               (self.G3_X, self.G3_Y, self.G3_Locs)]:
            if X_b is None or X_b.shape[0] == 0:
                continue
            for s in range(0, X_b.shape[0], chunk):
                e = min(s + chunk, X_b.shape[0])
                cov = self.cov_batched(params, X_b[s:e])
                try:
                    L = torch.linalg.cholesky(cov)
                except torch.linalg.LinAlgError:
                    return _penalty

                Zl = torch.linalg.solve_triangular(L, L_b[s:e], upper=False)
                Zy = torch.linalg.solve_triangular(L, Y_b[s:e], upper=False)

                u_X = Zl[:, -1, :]
                u_y = Zy[:, -1, :]

                _ld  = 2 * torch.sum(torch.log(L[:, -1, -1]))
                _XSX = u_X.T @ u_X
                _XSy = u_X.T @ u_y
                _ySy = (u_y.T @ u_y).squeeze()

                ld  = _ld  if ld  is None else ld  + _ld
                XSX = _XSX if XSX is None else XSX + _XSX
                XSy = _XSy if XSy is None else XSy + _XSy
                ySy = _ySy if ySy is None else ySy + _ySy

        if ld is None:
            return _penalty

        jitter = torch.eye(n_f, device=self.device, dtype=torch.float64) * 1e-6
        try:
            beta = torch.linalg.solve(XSX + jitter, XSy)
        except torch.linalg.LinAlgError:
            return _penalty

        quad = ySy - 2*(beta.T @ XSy) + (beta.T @ XSX @ beta)
        return 0.5 * (ld + quad.squeeze()) / self.n_obs

    # ── Optimizer + fit ───────────────────────────────────────────────────────

    def fit(self, initial_vals: list,
            max_steps: int = 50, lr: float = 1.0,
            tolerance_grad: float = 1e-7, max_iter: int = 20):
        """
        L-BFGS fitting.  Always creates fresh params from initial_vals so
        repeated calls are not affected by previous runs.

        Parameters
        ----------
        initial_vals : list of 7 floats  [log_phi1 … log_phi4, adv_lat, adv_lon, log_nugget]
                       Note: adv_lat / adv_lon are the RAW (unconstrained) values passed to
                       tanh-bounding, so initial_vals[4:6] = 0.0 means effective adv = 0.
        """
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        # Always start from scratch — prevents stale-params issue.
        params_list = [
            torch.tensor([v], requires_grad=True, dtype=torch.float64, device=self.device)
            for v in initial_vals
        ]

        optimizer = torch.optim.LBFGS(
            params_list, lr=lr, max_iter=max_iter,
            tolerance_grad=tolerance_grad, tolerance_change=1e-9,
            history_size=100
        )

        def closure():
            optimizer.zero_grad()
            loss = self.vecchia_likelihood(torch.stack(params_list))
            loss.backward()
            return loss

        loss = None
        for step in range(max_steps):
            loss = optimizer.step(closure)
            with torch.no_grad():
                grads    = [abs(p.grad.item()) for p in params_list if p.grad is not None]
                max_grad = max(grads) if grads else 0.0
                print(f"Step {step+1:3d} | loss={loss.item():.6f} | max|grad|={max_grad:.2e}")
                if max_grad < tolerance_grad:
                    print(f"  Converged at step {step+1}.")
                    break

        raw = [p.item() for p in params_list]
        print("Final params (physical):", _to_physical(raw))
        return raw, loss.item() if loss is not None else float('inf')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_physical(raw, adv_max=0.002):
    """
    Convert raw (unconstrained) params to interpretable physical quantities.

    adv_max : must match _CovBase.ADV_MAX used during fitting (default 0.002 deg/hr).
    Advec reported in effective units after tanh-bounding.
    """
    import math
    def _sexp(x):
        try:
            return math.exp(float(x))
        except OverflowError:
            return float('inf')
    def _r(v):
        try: return round(v, 6)
        except (OverflowError, ValueError): return v
    phi1, phi2, phi3, phi4 = [_sexp(x) for x in raw[:4]]
    nugget = _sexp(raw[6])
    # Effective advec after tanh bounding (same as _effective_adv in the class)
    adv_lat_eff = adv_max * math.tanh(float(raw[4]) / adv_max)
    adv_lon_eff = adv_max * math.tanh(float(raw[5]) / adv_max)
    return {
        "sigma_sq":   _r(phi1/phi2 if phi2 else float('inf')),
        "range_lon":  _r(1/phi2    if phi2 else float('inf')),
        "range_lat":  _r(1/(phi2*phi3**0.5) if phi2 and phi3 else float('inf')),
        "range_time": _r(1/(phi2*phi4**0.5) if phi2 and phi4 else float('inf')),
        "advec_lat":  _r(adv_lat_eff),
        "advec_lon":  _r(adv_lon_eff),
        "nugget":     _r(nugget),
    }


def load_year_for_seasonal(year: int, data_dir,
                            mm_lookup: dict,
                            ord_mm,
                            months: list = None,
                            max_days: int = None,
                            stride_days: int = 1) -> dict:
    """
    Load Apr–Sep data for one year as an ordered {time_key: Tensor[N,11]} dict.

    Applies MaxMin ordering (ord_mm) to spatial rows.
    Centers O3 by monthly mean.
    Creates hour-of-day dummies (D1…D7) cycling 0→7 within each day.

    Parameters
    ----------
    year       : int
    data_dir   : Path to GEMS_DATA/Apr_to_Sep/
    mm_lookup  : {(year,month): float}
    ord_mm     : MaxMin spatial ordering indices (from get_spatial_ordering)
    max_days   : if set, use only first max_days days (for quick testing)
    stride_days: use every nth day (default 1 = all days)

    Returns
    -------
    data_map : {time_key: Tensor[N, 11]}
               Columns: lat, lon, O3_centered, Hours_elapsed, D1…D7
    """
    import pickle, torch, pandas as pd
    import torch.nn.functional as F
    from pathlib import Path

    data_dir = Path(data_dir)
    pkl_path = data_dir / f"tco_grid_apr_sep_{year}.pkl"
    idx_path = data_dir / "day_index_apr_sep_2022_2025.csv"

    with open(pkl_path, "rb") as f:
        merged = pickle.load(f)

    idx_df  = pd.read_csv(idx_path)
    day_idx = {
        row["date_str"]: [row[f"key_h{i}"] for i in range(8)
                          if pd.notna(row[f"key_h{i}"])]
        for _, row in idx_df.iterrows()
    }

    dates = sorted(d for d in day_idx if d.startswith(str(year)))

    # Optional month filter  e.g. months=[7] for July only
    if months is not None:
        dates = [d for d in dates if int(d[5:7]) in months]

    # Optional subsampling
    dates = dates[::stride_days]
    if max_days is not None:
        dates = dates[:max_days]

    data_map = {}
    dtype    = torch.float64

    for day_i, date_str in enumerate(dates):
        keys  = day_idx[date_str]
        month = int(date_str[5:7])
        mm    = mm_lookup.get((year, month), 0.0)

        for hour_i, key in enumerate(keys):
            if key not in merged:
                continue
            df = merged[key]

            # Hours_elapsed — normalize same as data_loader.py (subtract 477700)
            # Raw value ≈ 460,000 → adv_lat * t ≈ 9200 which destroys distances
            hrs = pd.to_numeric(df["Hours_elapsed"], errors="coerce")
            hrs = hrs.fillna(hrs.median())
            hrs = np.round(hrs.values - 477700).astype(np.float64)

            # O3 centered
            o3 = pd.to_numeric(df["ColumnAmountO3"], errors="coerce").values - mm

            n = len(df)
            base = torch.zeros((n, 4), dtype=dtype)
            # Use Source_Latitude/Longitude (actual irregular coords) for covariance,
            # exactly as load_working_data does with keep_ori=True.
            # MaxMin ordering (ord_mm) is computed on grid Latitude/Longitude externally
            # so row order is consistent across all time steps.
            if "Source_Latitude" in df.columns and "Source_Longitude" in df.columns:
                base[:, 0] = torch.tensor(df["Source_Latitude"].values,  dtype=dtype)
                base[:, 1] = torch.tensor(df["Source_Longitude"].values, dtype=dtype)
            else:
                base[:, 0] = torch.tensor(df["Latitude"].values,  dtype=dtype)
                base[:, 1] = torch.tensor(df["Longitude"].values, dtype=dtype)
            base[:, 2] = torch.tensor(o3,  dtype=dtype)
            base[:, 3] = torch.tensor(hrs, dtype=dtype)

            # Hour-of-day dummies (0-7 cycling)
            h_idx  = hour_i % 8
            dummies = F.one_hot(torch.tensor([h_idx]), num_classes=8
                                ).repeat(n, 1)[:, 1:].to(dtype)

            tensor = torch.cat([base, dummies], dim=1)   # [N, 11]

            # MaxMin spatial ordering
            if ord_mm is not None:
                tensor = tensor[ord_mm]

            data_map[key] = tensor

    print(f"Loaded {year}: {len(dates)} days × up to 8 hours = {len(data_map)} time steps")
    return data_map
