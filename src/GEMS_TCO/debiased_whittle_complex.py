"""
debiased_whittle_complex.py  —  backward-compatibility alias

All DW modules now use complex (Hermitian-symmetric) expected periodogram.
This file re-exports debiased_whittle_raw so that existing imports still work.

Historical note:
  This module was created to hold the complex-Whittle variant that preserves
  Im(F(ω)) for advection sign recovery.  After the complex change was applied
  to debiased_whittle_raw (and all other DW modules), this file became
  redundant.  It is kept solely for backward compatibility with scripts that
  import `debiased_whittle_complex`.
"""

from GEMS_TCO.debiased_whittle_raw import (  # noqa: F401
    debiased_whittle_preprocess,
    debiased_whittle_likelihood,
)
