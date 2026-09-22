# Torch Matérn diagnostic

`torch_matern_bessel.py` is a small-data validation bridge between the SciPy
direct-Bessel likelihood and Torch autograd. It is intentionally outside the
installable `GEMS_TCO` package: the implementation uses finite-difference
Bessel derivatives and is not a production inference engine.

Run it with the repository's `src` directory on `PYTHONPATH` so it can import
the canonical covariance implementation from `GEMS_TCO.spatial`.
