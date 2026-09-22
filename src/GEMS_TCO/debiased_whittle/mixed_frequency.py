"""Mixed-frequency Debiased Whittle objective.

Low-index spatial frequencies use the identity-filtered periodogram, while
the remaining interior indices use the cross-differenced periodogram.  This
retains low-frequency advection information that cross differencing suppresses.

The cutoff is an index-space rule applied to two grids of different sizes.  It
is therefore a composite objective, not a proven one-to-one physical-frequency
partition.  Inference based on it requires separate calibration (for example,
simulation or a sandwich variance calculation).
"""

import cmath

import torch

from ._core import (
    BaseDebiasedWhittleLikelihood,
    BaseDebiasedWhittlePreprocessor,
    DebiasedWhittleFitResult,
    _optimize_parameters,
    _spectral_likelihood_terms,
)
from .filters import CROSS_DIFFERENCE, IDENTITY

# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — RAW  (no spatial filter, per-slice demean)
# ─────────────────────────────────────────────────────────────────────────────


class MixedFrequencyIdentityPreprocessor(BaseDebiasedWhittlePreprocessor):
    """Prepare the unfiltered, spatially demeaned objective component."""

    filter_spec = IDENTITY


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — DIFF  (2D separable filter [-1,1;1,-1])
# ─────────────────────────────────────────────────────────────────────────────


class MixedFrequencyCrossDifferencePreprocessor(BaseDebiasedWhittlePreprocessor):
    """Prepare the cross-differenced objective component."""

    filter_spec = CROSS_DIFFERENCE


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-frequency Debiased Whittle Likelihood
# ─────────────────────────────────────────────────────────────────────────────


class MixedFrequencyDebiasedWhittleLikelihood(BaseDebiasedWhittleLikelihood):
    """Composite DW likelihood using identity and cross-difference spectra."""

    # =========================================================================
    # 2b.  Covariance — RAW (identity filter)
    # =========================================================================

    @staticmethod
    def identity_covariance(u1, u2, t, params, delta1, delta2):
        """
        Cov_identity(u,v,τ) = C_X(u,v,τ).
        ``delta1`` and ``delta2`` are accepted for the shared callback API.
        """
        dev = params.device
        u1 = (
            u1.to(dev)
            if isinstance(u1, torch.Tensor)
            else torch.tensor(u1, device=dev, dtype=torch.float64)
        )
        u2 = (
            u2.to(dev)
            if isinstance(u2, torch.Tensor)
            else torch.tensor(u2, device=dev, dtype=torch.float64)
        )
        t = (
            t.to(dev)
            if isinstance(t, torch.Tensor)
            else torch.tensor(t, device=dev, dtype=torch.float64)
        )
        return MixedFrequencyDebiasedWhittleLikelihood.spatiotemporal_covariance(u1, u2, t, params)

    # =========================================================================
    # 2c.  Covariance — DIFF (2D separable filter)
    # =========================================================================

    @staticmethod
    def cross_difference_covariance(u1, u2, t, params, delta1, delta2):
        """
        Cov_Z(u,v,τ) = ΣΣ h_ab·h_cd·C_X(u+(a-c)δ1, v+(b-d)δ2, τ)
        h: {(0,0):-1, (1,0):+1, (0,1):+1, (1,1):-1}

        Expanded = 4C_X(u,v,τ)
                 - 2[C_X(u-δ1,v,τ)+C_X(u+δ1,v,τ)]
                 - 2[C_X(u,v-δ2,τ)+C_X(u,v+δ2,τ)]
                 + C_X(u-δ1,v-δ2,τ)+C_X(u+δ1,v-δ2,τ)
                 + C_X(u-δ1,v+δ2,τ)+C_X(u+δ1,v+δ2,τ)
        """
        weights = CROSS_DIFFERENCE.weights
        dev = params.device
        out = torch.broadcast_shapes(
            u1.shape if isinstance(u1, torch.Tensor) else (),
            u2.shape if isinstance(u2, torch.Tensor) else (),
            t.shape if isinstance(t, torch.Tensor) else (),
        )
        cov = torch.zeros(out, device=dev, dtype=torch.float64)
        u1d = (
            u1.to(dev)
            if isinstance(u1, torch.Tensor)
            else torch.tensor(u1, device=dev, dtype=torch.float64)
        )
        u2d = (
            u2.to(dev)
            if isinstance(u2, torch.Tensor)
            else torch.tensor(u2, device=dev, dtype=torch.float64)
        )
        td = (
            t.to(dev)
            if isinstance(t, torch.Tensor)
            else torch.tensor(t, device=dev, dtype=torch.float64)
        )
        for (a, b), wab in weights:
            for (c, d), wcd in weights:
                term = MixedFrequencyDebiasedWhittleLikelihood.spatiotemporal_covariance(
                    u1d + (a - c) * delta1, u2d + (b - d) * delta2, td, params
                )
                if torch.isnan(term).any():
                    return torch.full_like(cov, float("nan"))
                cov += wab * wcd * term
        return cov

    # =========================================================================
    # 3.  cn_bar helpers — one per covariance type
    # =========================================================================

    @staticmethod
    def _cn_bar(cov_fn, u1, u2, t, params, n1, n2, taper_auto, delta1, delta2, q=None, r=None):
        """c_Y(u)·c_gn(u) using the supplied covariance function."""
        dev = params.device
        u1d = (
            u1.to(dev)
            if isinstance(u1, torch.Tensor)
            else torch.tensor(u1, device=dev, dtype=torch.float64)
        )
        u2d = (
            u2.to(dev)
            if isinstance(u2, torch.Tensor)
            else torch.tensor(u2, device=dev, dtype=torch.float64)
        )
        td = (
            t.to(dev)
            if isinstance(t, torch.Tensor)
            else torch.tensor(t, device=dev, dtype=torch.float64)
        )
        cov_val = cov_fn(u1d * delta1, u2d * delta2, td, params, delta1, delta2)
        idx1 = n1 - 1 + u1d.long()
        idx2 = n2 - 1 + u2d.long()
        in_support = (idx1 >= 0) & (idx1 < 2 * n1 - 1) & (idx2 >= 0) & (idx2 < 2 * n2 - 1)
        safe_idx1 = torch.clamp(idx1, 0, 2 * n1 - 2)
        safe_idx2 = torch.clamp(idx2, 0, 2 * n2 - 2)
        if taper_auto.ndim == 4:
            if q is None or r is None:
                raise ValueError("Pair-specific taper autocorrelation requires q and r.")
            tv = taper_auto[q, r, safe_idx1, safe_idx2]
        elif taper_auto.ndim == 2:
            tv = taper_auto[safe_idx1, safe_idx2]
        else:
            raise ValueError("taper_auto must have two or four dimensions.")
        tv = torch.where(in_support, tv, torch.zeros_like(tv))
        if torch.isnan(cov_val).any() or torch.isnan(tv).any():
            return torch.full(
                torch.broadcast_shapes(cov_val.shape, tv.shape),
                float("nan"),
                device=dev,
                dtype=torch.float64,
            )
        return cov_val * tv

    # =========================================================================
    # 4.  Expected periodograms — one per covariance type
    # =========================================================================

    @staticmethod
    def _expected_periodogram(cov_fn, params, n1, n2, p_time, taper_auto, delta1, delta2):
        """
        E[I(ω)] = (1/(4π²)) FFT_2D[ Ã_n(u) ]   (aliasing sum, Lemma 2)
        Generic implementation controlled by the covariance callback.
        """
        if min(n1, n2, p_time) < 1:
            raise ValueError("n1, n2, and p_time must be positive.")
        if not isinstance(params, torch.Tensor) or params.ndim != 1 or params.numel() != 7:
            raise ValueError("params must be a one-dimensional tensor with seven elements.")
        valid_taper_shapes = {
            (2 * n1 - 1, 2 * n2 - 1),
            (p_time, p_time, 2 * n1 - 1, 2 * n2 - 1),
        }
        if tuple(taper_auto.shape) not in valid_taper_shapes:
            raise ValueError("taper_auto has an incompatible shape.")
        dev = params.device
        pt = params
        taper_auto = taper_auto.to(device=dev, dtype=torch.float64)

        u1m, u2m = torch.meshgrid(
            torch.arange(n1, dtype=torch.float64, device=dev),
            torch.arange(n2, dtype=torch.float64, device=dev),
            indexing="ij",
        )
        tl = torch.arange(p_time, dtype=torch.float64, device=dev)

        rows = []
        has_nan = False
        for q in range(p_time):
            cols = []
            for r in range(p_time):
                td = tl[q] - tl[r]
                _q = q if taper_auto.ndim == 4 else None
                _r = r if taper_auto.ndim == 4 else None

                def cb(du1, du2, _q=_q, _r=_r, td=td):
                    return MixedFrequencyDebiasedWhittleLikelihood._cn_bar(
                        cov_fn, du1, du2, td, pt, n1, n2, taper_auto, delta1, delta2, _q, _r
                    )

                grid = cb(u1m, u2m) + cb(u1m - n1, u2m) + cb(u1m, u2m - n2) + cb(u1m - n1, u2m - n2)
                if not torch.isfinite(grid).all():
                    has_nan = True
                    cols.append(torch.zeros(n1, n2, dtype=torch.complex128, device=dev))
                else:
                    cols.append(grid.to(torch.complex128))
            rows.append(torch.stack(cols, dim=-1))  # (n1, n2, p_time)
        cn = torch.stack(rows, dim=-2)  # (n1, n2, p_time, p_time)

        if has_nan:
            return torch.full(
                (n1, n2, p_time, p_time), float("nan"), dtype=torch.complex128, device=dev
            )
        result_raw = torch.fft.fft2(cn, dim=(0, 1)) * (1.0 / (4.0 * cmath.pi**2))
        return (result_raw + result_raw.conj().transpose(-1, -2)) / 2.0

    @staticmethod
    def identity_expected_periodogram(params, n1, n2, p_time, taper_auto, delta1, delta2):
        """Return the expected periodogram for the identity-filtered field."""
        return MixedFrequencyDebiasedWhittleLikelihood._expected_periodogram(
            MixedFrequencyDebiasedWhittleLikelihood.identity_covariance,
            params,
            n1,
            n2,
            p_time,
            taper_auto,
            delta1,
            delta2,
        )

    @staticmethod
    def cross_difference_expected_periodogram(
        params, n1d, n2d, p_time, taper_auto_diff, delta1, delta2
    ):
        """Return the expected periodogram for the cross-differenced field."""
        return MixedFrequencyDebiasedWhittleLikelihood._expected_periodogram(
            MixedFrequencyDebiasedWhittleLikelihood.cross_difference_covariance,
            params,
            n1d,
            n2d,
            p_time,
            taper_auto_diff,
            delta1,
            delta2,
        )

    @staticmethod
    def negative_log_likelihood(
        params,
        I_samp_raw,
        I_samp_diff,
        n1,
        n2,
        n1d,
        n2d,
        p_time,
        taper_auto_raw,
        taper_auto_diff,
        K1,
        K2,
        delta1=0.044,
        delta2=0.063,
    ):
        """
        L_mixed(θ) = Σ_{ω∈Ω_L} ℓ_raw(ω)  +  Σ_{ω∈Ω_H} ℓ_diff(ω)

        Ω_L = {k1≤K1, k2≤K2} \\ {(0,0)}    on raw  (n1 × n2)  grid
        Ω_H = {k1>K1 OR k2>K2}              on diff (n1d × n2d) grid

        The two index masks are disjoint within their respective grids.  The
        loss is averaged over the total number of selected terms.
        """
        if min(n1, n2, n1d, n2d, p_time) < 1:
            raise ValueError("All grid and time dimensions must be positive.")
        if not isinstance(params, torch.Tensor) or params.ndim != 1 or params.numel() != 7:
            raise ValueError("params must be a one-dimensional tensor with seven elements.")
        if I_samp_raw.shape != (n1, n2, p_time, p_time):
            raise ValueError("I_samp_raw has an incompatible shape.")
        if I_samp_diff.shape != (n1d, n2d, p_time, p_time):
            raise ValueError("I_samp_diff has an incompatible shape.")
        if not isinstance(K1, int) or not isinstance(K2, int):
            raise TypeError("K1 and K2 must be integer frequency cutoffs.")
        if not (0 <= K1 < min(n1, n1d)) or not (0 <= K2 < min(n2, n2d)):
            raise ValueError("K1 and K2 must lie inside both spatial frequency grids.")

        dev = I_samp_raw.device
        pt = params.to(dev)
        raw_sample = I_samp_raw.to(device=dev, dtype=torch.complex128)
        diff_sample = I_samp_diff.to(device=dev, dtype=torch.complex128)
        raw_taper = taper_auto_raw.to(device=dev, dtype=torch.float64)
        diff_taper = taper_auto_diff.to(device=dev, dtype=torch.float64)
        valid_raw_tapers = {
            (2 * n1 - 1, 2 * n2 - 1),
            (p_time, p_time, 2 * n1 - 1, 2 * n2 - 1),
        }
        valid_diff_tapers = {
            (2 * n1d - 1, 2 * n2d - 1),
            (p_time, p_time, 2 * n1d - 1, 2 * n2d - 1),
        }
        if tuple(raw_taper.shape) not in valid_raw_tapers:
            raise ValueError("taper_auto_raw has an incompatible shape.")
        if tuple(diff_taper.shape) not in valid_diff_tapers:
            raise ValueError("taper_auto_diff has an incompatible shape.")
        if not all(torch.isfinite(tensor).all() for tensor in (pt, raw_taper, diff_taper)):
            return torch.tensor(float("inf"), device=dev, dtype=torch.float64)

        # ── Expected periodograms ─────────────────────────────────────────────
        Ie_raw = MixedFrequencyDebiasedWhittleLikelihood.identity_expected_periodogram(
            pt, n1, n2, p_time, raw_taper, delta1, delta2
        )
        Ie_diff = MixedFrequencyDebiasedWhittleLikelihood.cross_difference_expected_periodogram(
            pt, n1d, n2d, p_time, diff_taper, delta1, delta2
        )

        # Invalid blocks are marked by the shared spectral solver and are
        # ignored only when the corresponding frequency is structurally excluded.

        # ── Per-frequency likelihood terms ────────────────────────────────────
        terms_raw = _spectral_likelihood_terms(Ie_raw, raw_sample)
        terms_diff = _spectral_likelihood_terms(Ie_diff, diff_sample)

        # ── Frequency masks ───────────────────────────────────────────────────
        # Low-freq: from raw grid
        #   (a) rectangle {k1≤K1, k2≤K2}
        #   (b) entire k1=0 row  — 2D diff filter H(0,ω2)=(e^0-1)(·)=0,
        #       so these frequencies have zero expected power under diff for ALL θ.
        #       Must be handled by raw, where f_raw(0,ω2) = f_X(0,ω2) > 0.
        #   (c) entire k2=0 col  — same reason: H(ω1,0)=(·)(e^0-1)=0.
        #   DC (0,0) excluded from likelihood.
        low_mask = torch.zeros(n1, n2, dtype=torch.bool, device=dev)
        low_mask[: K1 + 1, : K2 + 1] = True  # rectangle
        low_mask[0, :] = True  # k1=0 row  → raw (H_diff=0 there)
        low_mask[:, 0] = True  # k2=0 col  → raw (H_diff=0 there)
        low_mask[0, 0] = False  # DC always excluded

        # High-freq: from diff grid — interior only (k1>0 AND k2>0)
        #   |H(ω1,ω2)|² = 4sin²(ω1/2)·4sin²(ω2/2) > 0  for k1>0, k2>0.
        high_mask = torch.ones(n1d, n2d, dtype=torch.bool, device=dev)
        high_mask[: K1 + 1, : K2 + 1] = False  # exclude low-freq rectangle
        high_mask[0, :] = False  # k1=0 row: H_diff=0, zero expected power
        high_mask[:, 0] = False  # k2=0 col: H_diff=0, zero expected power

        # ── Sums ─────────────────────────────────────────────────────────────
        sum_low = terms_raw[low_mask].sum()
        sum_high = terms_diff[high_mask].sum()
        n_total = float(low_mask.sum() + high_mask.sum())

        if n_total < 1:
            return torch.tensor(float("inf"), device=dev, dtype=torch.float64)

        loss = (sum_low + sum_high) / n_total

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float("inf"), device=dev, dtype=torch.float64)
        return loss

    @classmethod
    def fit(
        cls,
        parameters,
        optimizer,
        identity_periodogram,
        cross_difference_periodogram,
        n1,
        n2,
        n1d,
        n2d,
        n_time,
        identity_taper_autocorrelation,
        cross_difference_taper_autocorrelation,
        K1,
        K2,
        delta1=0.044,
        delta2=0.063,
        max_steps=5,
        gradient_tolerance=1e-5,
        loss_tolerance=1e-12,
    ) -> DebiasedWhittleFitResult:
        """Fit the mixed-frequency covariance parameters."""
        parameters = tuple(parameters)
        if not parameters:
            raise ValueError("At least one parameter tensor is required.")
        device = parameters[0].device
        identity_periodogram = identity_periodogram.to(device)
        cross_difference_periodogram = cross_difference_periodogram.to(device)
        identity_taper_autocorrelation = identity_taper_autocorrelation.to(device)
        cross_difference_taper_autocorrelation = cross_difference_taper_autocorrelation.to(device)

        def objective(parameter_tensor):
            return cls.negative_log_likelihood(
                parameter_tensor,
                identity_periodogram,
                cross_difference_periodogram,
                n1,
                n2,
                n1d,
                n2d,
                n_time,
                identity_taper_autocorrelation,
                cross_difference_taper_autocorrelation,
                K1,
                K2,
                delta1,
                delta2,
            )

        return _optimize_parameters(
            parameters,
            optimizer,
            objective,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
            loss_tolerance=loss_tolerance,
        )


__all__ = [
    "MixedFrequencyIdentityPreprocessor",
    "MixedFrequencyCrossDifferencePreprocessor",
    "MixedFrequencyDebiasedWhittleLikelihood",
]
