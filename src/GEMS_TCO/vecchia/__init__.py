"""Stable package namespace for Vecchia implementations.

The package is being introduced incrementally.  Existing top-level modules
remain available so that research scripts continue to import and execute the
same implementation during the reorganization.
"""

from . import corridor_neighbors

__all__ = ["corridor_neighbors"]
