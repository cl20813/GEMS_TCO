"""NumPy/SciPy construction of the Matérn correlation spline table.

This module deliberately has no PyTorch dependency.  The fitted Vecchia
classes and lightweight CPU diagnostics can therefore share exactly the same
spline coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gamma, kv


_MATERN_SPLINE_CACHE = {}


def _build_matern_spline_coeffs(
    smooth: float, n_points: int = 1200, r_max: float = 20.0
):
    """Return natural-cubic-spline coefficients for a Matérn correlation."""
    key = (round(float(smooth), 8), int(n_points), float(r_max))
    if key in _MATERN_SPLINE_CACHE:
        return _MATERN_SPLINE_CACHE[key]

    nu = float(smooth)
    if nu <= 0:
        raise ValueError(f"smooth must be positive, got {smooth}")

    r_arr = np.linspace(0.0, float(r_max), int(n_points), dtype=np.float64)
    f_arr = np.empty_like(r_arr)
    f_arr[0] = 1.0
    z = np.sqrt(2.0 * nu) * r_arr[1:]
    f_arr[1:] = (2.0 ** (1.0 - nu) / gamma(nu)) * (z**nu) * kv(nu, z)
    f_arr = np.nan_to_num(f_arr, nan=0.0, posinf=1.0, neginf=0.0)
    f_arr = np.clip(f_arr, 0.0, 1.0)

    cs = CubicSpline(r_arr, f_arr, bc_type="natural")
    coeffs = {
        "knots": r_arr,
        "a": cs.c[3].copy(),
        "b": cs.c[2].copy(),
        "c": cs.c[1].copy(),
        "d": cs.c[0].copy(),
        "r_max": float(r_max),
    }
    _MATERN_SPLINE_CACHE[key] = coeffs
    return coeffs
