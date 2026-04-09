"""
debiased_whittle_nolf.py

Debiased Whittle with low-frequency exclusion (No-Low-Freq variant).

Modification from debiased_whittle_raw:
  - whittle_likelihood_loss_tapered accepts K1, K2 and excludes
    the box {(k1,k2) : k1 <= K1 AND k2 <= K2} from the Whittle sum.
  - run_lbfgs_tapered accepts and threads K1, K2 through.
  - All other methods (preprocessing, FFT, taper, expected periodogram)
    are inherited unchanged from debiased_whittle_raw.

Motivation:
  DW_raw overestimates range/sigmasq because large-scale spatial gradients
  (wavelength >> correlation range) contaminate low-frequency periodogram
  ordinates. Excluding these modes removes the bias without discarding
  information about short-range covariance structure.

  K1 = floor(n1 * alpha),  K2 = floor(n2 * alpha)
  Recommended alpha: 0.10 (minimal) or 0.20 (matches DW_mixed boundary).

  With K1=0, K2=0 (alpha→0): degenerates to DW_raw (only DC excluded).

Statistical justification:
  Equivalent to spectral REML (Patterson & Thompson 1971; Kitanidis 1983)
  for polynomial spatial trend. Fuentes (2002, JASA) applies this
  explicitly for spatial spectral estimation.
"""

import torch
import numpy as np

from GEMS_TCO.debiased_whittle_raw import (
    debiased_whittle_preprocess,          # noqa: F401  (re-exported for callers)
    debiased_whittle_likelihood as _RawDWL,
)


class debiased_whittle_likelihood(_RawDWL):
    """
    Drop-in replacement for debiased_whittle_raw.debiased_whittle_likelihood.
    Identical API except whittle_likelihood_loss_tapered and run_lbfgs_tapered
    accept optional K1, K2 keyword arguments.
    """

    # =========================================================================
    # Low-frequency Whittle loss
    # =========================================================================

    @staticmethod
    def whittle_likelihood_loss_tapered(
            params, I_sample, n1, n2, p_time,
            taper_autocorr_grid, delta1, delta2,
            K1: int = 0, K2: int = 0):
        """
        Whittle loss with low-frequency box exclusion.

        Excludes {(k1,k2) : k1 <= K1 AND k2 <= K2} from the sum.
        K1=0, K2=0  →  only (0,0) excluded  (same as DW_raw).
        K1=floor(n1*0.20), K2=floor(n2*0.20)  →  Option-B exclusion.

        Parameters
        ----------
        K1 : int  max k1 index to exclude (inclusive)
        K2 : int  max k2 index to exclude (inclusive)
        """
        device = I_sample.device
        params_tensor = params.to(device)

        if torch.isnan(params_tensor).any() or torch.isinf(params_tensor).any():
            print("Warning: NaN/Inf in input parameters.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        I_expected = debiased_whittle_likelihood.expected_periodogram_fft_tapered(
            params_tensor, n1, n2, p_time, taper_autocorr_grid, delta1, delta2)

        if torch.isnan(I_expected).any() or torch.isinf(I_expected).any():
            print("Warning: NaN/Inf in expected periodogram.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        eye_matrix  = torch.eye(p_time, dtype=torch.complex128, device=device)
        diag_vals   = torch.abs(I_expected.diagonal(dim1=-2, dim2=-1))
        mean_diag   = diag_vals.mean().item() if (diag_vals.numel() > 0 and
                       not torch.isnan(diag_vals).all()) else 1.0
        diag_load   = max(mean_diag * 1e-8, 1e-9)
        I_exp_stab  = I_expected + eye_matrix * diag_load

        sign, logabsdet = torch.linalg.slogdet(I_exp_stab)
        if torch.any(sign.real <= 1e-9):
            log_det_term = torch.where(
                sign.real > 1e-9, logabsdet.real,
                torch.tensor(1e10, device=device, dtype=torch.float64))
        else:
            log_det_term = logabsdet.real

        if torch.isnan(I_sample).any() or torch.isinf(I_sample).any():
            print("Warning: NaN/Inf in I_sample.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        try:
            solved_term = torch.linalg.solve(I_exp_stab, I_sample)
            trace_term  = torch.einsum('...ii->...', solved_term).real
        except torch.linalg.LinAlgError as e:
            print(f"Warning: LinAlgError: {e}.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        if torch.isnan(trace_term).any() or torch.isinf(trace_term).any():
            print("Warning: NaN/Inf in trace term.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        # ── Low-frequency exclusion mask ──────────────────────────────────────
        # Exclude {(k1,k2) : k1 <= K1 AND k2 <= K2}
        # likelihood_terms has shape (n1, n2)
        likelihood_terms = log_det_term + trace_term          # (n1, n2)

        k1_idx = torch.arange(n1, device=device)
        k2_idx = torch.arange(n2, device=device)
        k1_grid, k2_grid = torch.meshgrid(k1_idx, k2_idx, indexing='ij')
        exclude_mask = (k1_grid <= K1) & (k2_grid <= K2)     # True = skip
        include_mask = ~exclude_mask                          # True = use

        n_excluded = exclude_mask.sum().item()
        n_included = include_mask.sum().item()

        if n_included == 0:
            print("Warning: all frequencies excluded — check K1, K2.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        if torch.isnan(likelihood_terms).any():
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        sum_loss = likelihood_terms[include_mask].sum()
        avg_loss = sum_loss / n_included

        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            print(f"Warning: NaN/Inf in final loss (excluded {n_excluded} of "
                  f"{n1*n2} freq terms).")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        return avg_loss

    # =========================================================================
    # LBFGS loop — threads K1, K2 through closure
    # =========================================================================

    @staticmethod
    def run_lbfgs_tapered(
            params_list, optimizer, I_sample, n1, n2, p_time,
            taper_autocorr_grid, max_steps=5, device='cpu',
            grad_tol=1e-5, K1: int = 0, K2: int = 0):
        """
        L-BFGS loop.  K1, K2 are forwarded to whittle_likelihood_loss_tapered.
        """
        best_params_state = [p.detach().clone() for p in params_list]
        steps_completed   = 0
        DELTA_LAT, DELTA_LON = 0.044, 0.063

        loss_tol  = 1e-12
        best_loss = float('inf')
        prev_loss = float('inf')

        I_sample_dev       = I_sample.to(device)
        taper_autocorr_dev = taper_autocorr_grid.to(device)

        def closure():
            optimizer.zero_grad()
            params_tensor = torch.cat(params_list)
            loss = debiased_whittle_likelihood.whittle_likelihood_loss_tapered(
                params_tensor, I_sample_dev, n1, n2, p_time,
                taper_autocorr_dev, DELTA_LAT, DELTA_LON, K1, K2)
            if torch.isnan(loss) or torch.isinf(loss):
                return loss
            loss.backward()
            nan_grad = any(
                p.grad is not None and
                (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in params_list)
            if nan_grad:
                optimizer.zero_grad()
            return loss

        for i in range(max_steps):
            steps_completed = i + 1
            loss = optimizer.step(closure)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {i+1}/{max_steps}: Loss NaN/Inf. Stopping.")
                break

            cur = loss.item()
            if cur < best_loss:
                if not any(torch.isnan(p.data).any() or torch.isinf(p.data).any()
                           for p in params_list):
                    best_loss = cur
                    best_params_state = [p.detach().clone() for p in params_list]

            max_grad = 0.0
            with torch.no_grad():
                for p in params_list:
                    if p.grad is not None:
                        max_grad = max(max_grad, p.grad.abs().item())

            print(f'--- Step {i+1}/{max_steps} ---')
            print(f' Loss: {cur:.6f} | Max Grad: {max_grad:.6e}')
            print(f'  Params (Raw Log): '
                  f'{debiased_whittle_likelihood.get_raw_log_params_7param(params_list)}')

            if i > 2:
                if max_grad < grad_tol:
                    print(f"\n--- Converged (max|grad| < {grad_tol}) at step {i+1} ---")
                    break
                if abs(cur - prev_loss) < loss_tol:
                    print(f"\n--- Converged (loss_change < {loss_tol}) at step {i+1} ---")
                    break

            prev_loss = cur

        print("\n--- Training Complete ---")
        if best_params_state is None:
            return None, None, None, None, steps_completed

        final_loss = round(best_loss, 3) if best_loss != float('inf') else float('inf')
        return (debiased_whittle_likelihood.get_printable_params_7param(best_params_state),
                debiased_whittle_likelihood.get_phi_params_7param(best_params_state),
                debiased_whittle_likelihood.get_raw_log_params_7param(best_params_state),
                final_loss, steps_completed)
