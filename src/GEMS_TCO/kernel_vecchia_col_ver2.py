"""
kernel_vecchia_col_ver2.py

Second regular-grid column Vecchia experiment.

This variant keeps the template geometry much more rigid than
ReverseLColumnVecchiaFit:

- the rightmost `head_right_cols` columns are exact head points;
- for each tail target and each available lag, use at most
  `per_lag_conditioning_count` spatial conditioning points;
- same-column above points are added first, up to `above_count`;
- remaining slots are filled from the next `right_col_count` columns to the
  east, using rows equal to the target row and then rows below it.

For the high-resolution simulation requested on 2026-05-06, the intended
settings are above_count=3, right_col_count=3, per_lag_conditioning_count=50,
lag_count=2, so the nominal conditioning size is 150.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from GEMS_TCO.kernels_vecchia_column import ReverseLColumnVecchiaFit


class ReverseLColumnVecchiaFitV2(ReverseLColumnVecchiaFit):
    """Rigid reverse-L/downward-right column template for regular grids."""

    def __init__(
        self,
        *args,
        above_count: int = 3,
        right_col_count: int = 3,
        per_lag_conditioning_count: int = 50,
        right_neighbor_count: Optional[int] = None,
        **kwargs,
    ):
        if right_neighbor_count is not None and int(right_neighbor_count) != int(per_lag_conditioning_count):
            raise ValueError(
                "ReverseLColumnVecchiaFitV2 uses per_lag_conditioning_count as the "
                "total spatial conditioning cap per lag; pass only one of "
                "per_lag_conditioning_count or a matching right_neighbor_count."
            )
        self.per_lag_conditioning_count = int(per_lag_conditioning_count)
        super().__init__(
            *args,
            above_count=int(above_count),
            right_col_count=int(right_col_count),
            right_neighbor_count=int(per_lag_conditioning_count),
            **kwargs,
        )

    def _spatial_stencil_locals(self, row: int, col: int) -> List[int]:
        """Return same-column-above plus right-columns same-or-below stencil.

        Rows are sorted by latitude ascending, and the model scans north to
        south.  Therefore "above" means row+1, row+2, ... and "same or below"
        in right columns means row, row-1, row-2, ...
        """
        key = (int(row), int(col))
        if key in self._stencil_cache:
            return self._stencil_cache[key]

        out: List[int] = []
        seen = set()
        cap = max(0, int(self.per_lag_conditioning_count))

        for k in range(1, self.above_count + 1):
            if len(out) >= cap:
                break
            nb = self._get_local(row + k, col)
            if nb is not None and nb not in seen:
                out.append(nb)
                seen.add(nb)

        if len(out) < cap:
            right_candidates = []
            for dc in range(1, self.right_col_count + 1):
                c2 = col + dc
                if c2 >= self._n_lon:
                    continue
                for r2 in range(row, -1, -1):
                    nb = self._get_local(r2, c2)
                    if nb is None or nb in seen:
                        continue
                    down = row - r2
                    right_candidates.append((down, dc, nb))

            right_candidates.sort(key=lambda x: (x[0], x[1]))
            for _, _, nb in right_candidates:
                if len(out) >= cap:
                    break
                if nb not in seen:
                    out.append(nb)
                    seen.add(nb)

        self._stencil_cache[key] = out
        return out

    def expected_spatial_template_count_upper_bound(self) -> int:
        """Small diagnostic: exact count of distinct spatial offset patterns."""
        coords = self._regular_coords_np(self._n_grid)
        patterns = set()
        for col in range(max(0, self._n_lon - self.head_right_cols) - 1, -1, -1):
            for row in range(self._n_lat - 1, -1, -1):
                local = self._get_local(row, col)
                if local is None:
                    continue
                offsets = []
                for nb in self._spatial_stencil_locals(row, col):
                    offsets.append(
                        (
                            round(float(coords[local, 0] - coords[nb, 0]), 8),
                            round(float(coords[local, 1] - coords[nb, 1]), 8),
                        )
                    )
                patterns.add(tuple(offsets))
        return len(patterns)


__all__ = ["ReverseLColumnVecchiaFitV2"]
