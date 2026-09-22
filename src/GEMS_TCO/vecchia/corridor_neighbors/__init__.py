"""Named corridor-neighbor configurations used by the Vecchia fits.

``local_lag432`` exposes the lighter 4/3/2 conditioning budget used for local
runs.  ``amarel_lag643`` exposes the larger 6/4/3 conditioning budget used for
Amarel runs.  These names describe the established run configurations; device
selection still follows the tensors passed to the unchanged implementation.
"""

from . import amarel_lag643, local_lag432

__all__ = ["local_lag432", "amarel_lag643"]
