# Experimental diagnostics

This directory is for diagnostic methods that are still being developed or
validated.  It is deliberately outside `src/`, so these experiments are not
installed as part of the stable `GEMS_TCO` package.

Code belongs here while its statistical definition, numerical behavior, or
public API may still change. Promotion into `src/GEMS_TCO/` requires focused
tests, documented assumptions, and a stable interface.

`evaluate.py` is the former package-level collection of exploratory residual,
semivariogram, and plotting diagnostics. It remains available for research
reproduction, but it is intentionally not part of the installed API.
