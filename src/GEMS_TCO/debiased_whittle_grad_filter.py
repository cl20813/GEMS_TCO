"""
Gradient-filter Debiased Whittle likelihood.

This module keeps the two first differences as a vector-valued spatial filter:

    D_lat X(i, j) = X(i + 1, j) - X(i, j)
    D_lon X(i, j) = X(i, j + 1) - X(i, j)

on the common anchor grid (n_lat - 1) x (n_lon - 1).  The Whittle likelihood
then treats the data as a 2 * T dimensional multivariate process, preserving
the cross-covariance between D_lat and D_lon instead of collapsing them into
the scalar filter D_lat + D_lon used by debiased_whittle_2110.
"""

import cmath
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
if gems_tco_path not in sys.path:
    sys.path.append(gems_tco_path)

from GEMS_TCO import debiased_whittle_2110 as _dw2110


LAT_COMPONENT = 0
LON_COMPONENT = 1


class debiased_whittle_preprocess(_dw2110.full_vecc_dw_likelihoods):
    """Preprocess regular-grid hourly maps into vector gradient components."""

    def __init__(self, daily_aggregated_tensors, daily_hourly_maps,
                 day_idx, params_list, lat_range, lon_range):
        super().__init__(daily_aggregated_tensors, daily_hourly_maps,
                         day_idx, params_list, lat_range, lon_range)

    def subset_tensor(self, df_tensor: torch.Tensor, lat_s: float, lat_e: float,
                      lon_s: float, lon_e: float) -> torch.Tensor:
        lat_mask = (df_tensor[:, 0] >= lat_s) & (df_tensor[:, 0] <= lat_e)
        lon_mask = (df_tensor[:, 1] >= lon_s) & (df_tensor[:, 1] <= lon_e)
        return df_tensor[lat_mask & lon_mask].clone()

    @staticmethod
    def _sort_complete_grid(df_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if df_tensor.dtype != torch.float64:
            df_tensor = df_tensor.to(torch.float64)
        unique_lats = torch.unique(df_tensor[:, 0])
        unique_lons = torch.unique(df_tensor[:, 1])
        n_lat, n_lon = unique_lats.numel(), unique_lons.numel()
        if df_tensor.size(0) != n_lat * n_lon:
            raise ValueError("Tensor size does not match complete grid dimensions.")
        sort_idx = (df_tensor[:, 0] * 1e6 + df_tensor[:, 1]).argsort()
        return df_tensor[sort_idx], unique_lats, unique_lons

    def apply_gradient_filter_2d_tensor(self, df_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (D_lat, D_lon) tensors on the same anchor grid.

        Each output has columns [lat, lon, filtered_value, time].
        Both components use anchors (lat[:-1], lon[:-1]) so their DFTs share the
        same spatial grid and can enter one multivariate periodogram.
        """
        if df_tensor.size(0) == 0:
            empty = torch.empty(0, 4, dtype=torch.float64, device=df_tensor.device)
            return empty, empty

        df_tensor, unique_lats, unique_lons = self._sort_complete_grid(df_tensor)
        n_lat, n_lon = unique_lats.numel(), unique_lons.numel()
        if n_lat < 2 or n_lon < 2:
            empty = torch.empty(0, 4, dtype=torch.float64, device=df_tensor.device)
            return empty, empty

        grid = df_tensor[:, 2].reshape(1, 1, n_lat, n_lon)
        kernel_lat = torch.tensor([[[[-1.0], [1.0]]]], dtype=torch.float64, device=df_tensor.device)
        kernel_lon = torch.tensor([[[[-1.0, 1.0]]]], dtype=torch.float64, device=df_tensor.device)

        d_lat_full = F.conv2d(grid, kernel_lat, padding="valid").squeeze(0).squeeze(0)
        d_lon_full = F.conv2d(grid, kernel_lon, padding="valid").squeeze(0).squeeze(0)

        d_lat = d_lat_full[:, :-1]
        d_lon = d_lon_full[:-1, :]

        anchor_lats = unique_lats[:-1]
        anchor_lons = unique_lons[:-1]
        lat_grid, lon_grid = torch.meshgrid(anchor_lats, anchor_lons, indexing="ij")
        time_value = df_tensor[0, 3].repeat(d_lat.numel())

        lat_tensor = torch.stack(
            [lat_grid.flatten(), lon_grid.flatten(), d_lat.flatten(), time_value],
            dim=1,
        )
        lon_tensor = torch.stack(
            [lat_grid.flatten(), lon_grid.flatten(), d_lon.flatten(), time_value],
            dim=1,
        )
        return lat_tensor, lon_tensor

    def generate_gradient_filtered_time_slices(self, lat_s: float, lat_e: float,
                                               lon_s: float, lon_e: float) -> List[torch.Tensor]:
        """
        Returns tensors in time-major component order:

            [D_lat(t0), D_lon(t0), D_lat(t1), D_lon(t1), ...].
        """
        slices: List[torch.Tensor] = []
        for _, tensor in self.daily_hourly_map.items():
            subsetted = self.subset_tensor(tensor, lat_s, lat_e, lon_s, lon_e)
            if subsetted.size(0) == 0:
                continue
            try:
                lat_diff, lon_diff = self.apply_gradient_filter_2d_tensor(subsetted)
            except ValueError as exc:
                print(f"[grad preprocess] skipping chunk on day {self.day_idx + 1}: {exc}")
                continue
            if lat_diff.size(0) > 0 and lon_diff.size(0) > 0:
                slices.extend([lat_diff, lon_diff])
        return slices

    def generate_spatially_filtered_days(self, lat_s: float, lat_e: float,
                                         lon_s: float, lon_e: float) -> torch.Tensor:
        """Compatibility helper returning all gradient rows with a component column."""
        rows = []
        for tensor in self.generate_gradient_filtered_time_slices(lat_s, lat_e, lon_s, lon_e):
            rows.append(tensor)
        if not rows:
            return torch.empty(0, 5, dtype=torch.float64)

        tagged = []
        for idx, tensor in enumerate(rows):
            component = torch.full((tensor.shape[0], 1), idx % 2,
                                   dtype=tensor.dtype, device=tensor.device)
            tagged.append(torch.cat([tensor, component], dim=1))
        return torch.cat(tagged, dim=0)


class debiased_whittle_likelihood:
    cgn_hamming = staticmethod(_dw2110.debiased_whittle_likelihood.cgn_hamming)
    calculate_taper_autocorrelation_fft = staticmethod(
        _dw2110.debiased_whittle_likelihood.calculate_taper_autocorrelation_fft)
    _fill_grid_from_tensor = staticmethod(_dw2110.debiased_whittle_likelihood._fill_grid_from_tensor)
    generate_Jvector_tapered = staticmethod(_dw2110.debiased_whittle_likelihood.generate_Jvector_tapered)
    generate_Jvector_tapered_mv = staticmethod(_dw2110.debiased_whittle_likelihood.generate_Jvector_tapered_mv)
    calculate_taper_autocorrelation_multivariate = staticmethod(
        _dw2110.debiased_whittle_likelihood.calculate_taper_autocorrelation_multivariate)
    calculate_sample_periodogram_vectorized = staticmethod(
        _dw2110.debiased_whittle_likelihood.calculate_sample_periodogram_vectorized)
    cov_x_spatiotemporal_model_kernel = staticmethod(
        _dw2110.debiased_whittle_likelihood.cov_x_spatiotemporal_model_kernel)
    get_printable_params_7param = staticmethod(
        _dw2110.debiased_whittle_likelihood.get_printable_params_7param)
    get_phi_params_7param = staticmethod(
        _dw2110.debiased_whittle_likelihood.get_phi_params_7param)
    get_raw_log_params_7param = staticmethod(
        _dw2110.debiased_whittle_likelihood.get_raw_log_params_7param)

    @staticmethod
    def component_metadata(p_total: int, n_components: int = 2,
                           device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if p_total % n_components != 0:
            raise ValueError(f"p_total={p_total} must be divisible by n_components={n_components}.")
        idx = torch.arange(p_total, dtype=torch.long, device=device)
        time_idx = idx // n_components
        component_idx = idx % n_components
        return time_idx.to(torch.float64), component_idx

    @staticmethod
    def _component_weights(component: int) -> Dict[Tuple[int, int], float]:
        if int(component) == LAT_COMPONENT:
            return {(0, 0): -1.0, (1, 0): 1.0}
        if int(component) == LON_COMPONENT:
            return {(0, 0): -1.0, (0, 1): 1.0}
        raise ValueError(f"Unknown gradient component: {component}")

    @staticmethod
    def cov_gradient_pair(u1, u2, t, params, delta1, delta2,
                          component_q: int, component_r: int):
        """Covariance between one gradient component at q and one at r."""
        device = params.device
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float64)

        out_shape = torch.broadcast_shapes(u1_dev.shape, u2_dev.shape, t_dev.shape)
        cov = torch.zeros(out_shape, device=device, dtype=torch.float64)
        weights_q = debiased_whittle_likelihood._component_weights(component_q)
        weights_r = debiased_whittle_likelihood._component_weights(component_r)

        for (a_lat, a_lon), w_a in weights_q.items():
            for (b_lat, b_lon), w_b in weights_r.items():
                lag_u1 = u1_dev + (a_lat - b_lat) * delta1
                lag_u2 = u2_dev + (a_lon - b_lon) * delta2
                cov = cov + w_a * w_b * debiased_whittle_likelihood.cov_x_spatiotemporal_model_kernel(
                    lag_u1, lag_u2, t_dev, params)
        return cov

    @staticmethod
    def cn_bar_tapered_gradient(u1, u2, t, params, n1, n2, taper_autocorr_grid,
                                delta1, delta2, q_idx, r_idx,
                                component_q: int, component_r: int):
        device = params.device
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float64)

        lag_u1 = u1_dev * delta1
        lag_u2 = u2_dev * delta2
        cov_value = debiased_whittle_likelihood.cov_gradient_pair(
            lag_u1, lag_u2, t_dev, params, delta1, delta2, component_q, component_r)

        idx1 = torch.clamp(n1 - 1 + u1_dev.long(), 0, 2 * n1 - 2)
        idx2 = torch.clamp(n2 - 1 + u2_dev.long(), 0, 2 * n2 - 2)
        if taper_autocorr_grid.ndim == 4:
            taper_value = taper_autocorr_grid[q_idx, r_idx, idx1, idx2]
        else:
            taper_value = taper_autocorr_grid[idx1, idx2]
        return cov_value * taper_value

    @staticmethod
    def expected_periodogram_fft_tapered(params, n1, n2, p_time, taper_autocorr_grid,
                                         delta1, delta2, n_components: int = 2):
        device = params.device if isinstance(params, torch.Tensor) else params[0].device
        params_tensor = torch.cat([p.to(device) for p in params]) if isinstance(params, list) else params.to(device)

        time_idx, component_idx = debiased_whittle_likelihood.component_metadata(
            p_time, n_components=n_components, device=device)
        u1_lags = torch.arange(n1, dtype=torch.float64, device=device)
        u2_lags = torch.arange(n2, dtype=torch.float64, device=device)
        u1_mesh, u2_mesh = torch.meshgrid(u1_lags, u2_lags, indexing="ij")

        rows = []
        for q in range(p_time):
            cols = []
            for r in range(p_time):
                t_diff = time_idx[q] - time_idx[r]
                cq = int(component_idx[q].item())
                cr = int(component_idx[r].item())
                term1 = debiased_whittle_likelihood.cn_bar_tapered_gradient(
                    u1_mesh, u2_mesh, t_diff, params_tensor, n1, n2, taper_autocorr_grid,
                    delta1, delta2, q, r, cq, cr)
                term2 = debiased_whittle_likelihood.cn_bar_tapered_gradient(
                    u1_mesh - n1, u2_mesh, t_diff, params_tensor, n1, n2, taper_autocorr_grid,
                    delta1, delta2, q, r, cq, cr)
                term3 = debiased_whittle_likelihood.cn_bar_tapered_gradient(
                    u1_mesh, u2_mesh - n2, t_diff, params_tensor, n1, n2, taper_autocorr_grid,
                    delta1, delta2, q, r, cq, cr)
                term4 = debiased_whittle_likelihood.cn_bar_tapered_gradient(
                    u1_mesh - n1, u2_mesh - n2, t_diff, params_tensor, n1, n2, taper_autocorr_grid,
                    delta1, delta2, q, r, cq, cr)
                cols.append((term1 + term2 + term3 + term4).to(torch.complex128))
            rows.append(torch.stack(cols, dim=-1))

        tilde_cn_tensor = torch.stack(rows, dim=-2)
        fft_result = torch.fft.fft2(tilde_cn_tensor, dim=(0, 1))
        result_raw = fft_result * (1.0 / (4.0 * cmath.pi ** 2))
        return (result_raw + result_raw.conj().transpose(-1, -2)) / 2.0

    @staticmethod
    def whittle_likelihood_loss_tapered(params, I_sample, n1, n2, p_time,
                                        taper_autocorr_grid, delta1, delta2,
                                        n_components: int = 2):
        device = I_sample.device
        params_tensor = params.to(device)
        if torch.isnan(params_tensor).any() or torch.isinf(params_tensor).any():
            return torch.tensor(float("nan"), device=device, dtype=torch.float64)

        I_expected = debiased_whittle_likelihood.expected_periodogram_fft_tapered(
            params_tensor, n1, n2, p_time, taper_autocorr_grid, delta1, delta2, n_components)
        if torch.isnan(I_expected).any() or torch.isinf(I_expected).any():
            return torch.tensor(float("nan"), device=device, dtype=torch.float64)

        eye = torch.eye(p_time, dtype=torch.complex128, device=device)
        diag_vals = torch.abs(I_expected.diagonal(dim1=-2, dim2=-1))
        mean_diag_abs = diag_vals.mean().item() if diag_vals.numel() else 1.0
        diag_load = max(mean_diag_abs * 1e-8, 1e-9)
        I_stable = I_expected + eye * diag_load

        sign, logabsdet = torch.linalg.slogdet(I_stable)
        log_det_term = torch.where(
            sign.real > 1e-9,
            logabsdet.real,
            torch.tensor(1e10, device=device, dtype=torch.float64),
        )

        try:
            solved = torch.linalg.solve(I_stable, I_sample)
        except torch.linalg.LinAlgError:
            return torch.tensor(float("inf"), device=device, dtype=torch.float64)
        trace_term = torch.einsum("...ii->...", solved).real
        terms = log_det_term + trace_term
        sum_loss = terms.sum() - terms[0, 0]
        denom = max(n1 * n2 - 1, 1)
        loss = sum_loss / denom
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float("inf"), device=device, dtype=torch.float64)
        return loss

    @staticmethod
    def run_lbfgs_tapered(params_list, optimizer, I_sample, n1, n2, p_time,
                          taper_autocorr_grid, max_steps=5, device="cpu",
                          grad_tol=1e-5, n_components: int = 2):
        best_params_state = [p.detach().clone() for p in params_list]
        best_loss = float("inf")
        prev_loss_item = float("inf")
        steps_completed = 0
        loss_tol = 1e-12
        delta_lat, delta_lon = 0.044, 0.063

        I_sample_dev = I_sample.to(device)
        taper_dev = taper_autocorr_grid.to(device)

        def closure():
            optimizer.zero_grad()
            params_tensor = torch.cat(params_list)
            loss = debiased_whittle_likelihood.whittle_likelihood_loss_tapered(
                params_tensor, I_sample_dev, n1, n2, p_time, taper_dev,
                delta_lat, delta_lon, n_components=n_components)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
            if any(p.grad is not None and
                   (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                   for p in params_list):
                optimizer.zero_grad()
            return loss

        for i in range(max_steps):
            steps_completed = i + 1
            loss = optimizer.step(closure)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {i + 1}/{max_steps}: loss NaN/Inf, stopping.")
                break

            cur_loss = loss.item()
            if cur_loss < best_loss and not any(
                torch.isnan(p.data).any() or torch.isinf(p.data).any()
                for p in params_list
            ):
                best_loss = cur_loss
                best_params_state = [p.detach().clone() for p in params_list]

            max_abs_grad = max(
                (p.grad.abs().max().item() for p in params_list if p.grad is not None),
                default=0.0,
            )
            print(f"--- Step {i + 1}/{max_steps} ---")
            print(f" Loss: {cur_loss:.6f} | Max Grad: {max_abs_grad:.6e}")
            print(f"  Params (Raw Log): {debiased_whittle_likelihood.get_raw_log_params_7param(params_list)}")

            if i > 2:
                if max_abs_grad < grad_tol:
                    print(f"Converged on gradient norm at step {i + 1}.")
                    break
                if abs(cur_loss - prev_loss_item) < loss_tol:
                    print(f"Converged on loss change at step {i + 1}.")
                    break
            prev_loss_item = cur_loss

        return (
            debiased_whittle_likelihood.get_printable_params_7param(best_params_state),
            debiased_whittle_likelihood.get_phi_params_7param(best_params_state),
            debiased_whittle_likelihood.get_raw_log_params_7param(best_params_state),
            round(best_loss, 3) if best_loss != float("inf") else float("inf"),
            steps_completed,
        )
