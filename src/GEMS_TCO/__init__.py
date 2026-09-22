"""Inference tools for GEMS total-column ozone models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("GEMS-TCO")
except PackageNotFoundError:  # Source checkout without an installed distribution.
    __version__ = "0.3.0"

__all__ = ["__version__"]
