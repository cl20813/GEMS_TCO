"""
kernels_st_lagcap_050726.py

Spatio-temporal Column V3 wrapper with lag-specific reverse-L conditioning caps.

This isolates whether real-data ST differences are driven by the amount of
conditioning information taken from t-1 and t-2.  The scan geometry is the same
as ReverseLColumnVecchiaFitBatch, but each lag can have its own cap, e.g.
    (14, 14, 14): baseline
    (14,  6, 14): keep only the nearest 6 reverse-L candidates at t-1
    (14,  0, 14): remove t-1 conditioning
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from GEMS_TCO.kernel_vecchia_col_batch import ReverseLColumnVecchiaFitBatch
from GEMS_TCO.kernels_st_trend_050726 import _STMeanDesignMixin


class ReverseLColumnLagCapVecchiaFitV3(ReverseLColumnVecchiaFitBatch):
    """Batched reverse-L Vecchia with separate conditioning caps by lag."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        mm_cond_number: int = 300,
        nheads: int = 0,
        grid_coords: Optional[np.ndarray] = None,
        head_right_cols: int = 0,
        above_count: int = 2,
        right_col_count: int = 3,
        per_lag_conditioning_counts: Sequence[int] = (14, 14, 14),
        include_lag_self: bool = False,
        lat_round_decimals: int = 6,
        lon_round_decimals: int = 6,
        target_chunk_size: int = 4096,
        use_data_coords_for_offsets: bool = True,
        **kwargs,
    ):
        counts = tuple(int(x) for x in per_lag_conditioning_counts)
        if len(counts) < 1:
            raise ValueError("per_lag_conditioning_counts must have at least one entry")
        if any(x < 0 for x in counts):
            raise ValueError("per_lag_conditioning_counts must be nonnegative")
        self.per_lag_conditioning_counts = counts

        super().__init__(
            smooth=smooth,
            input_map=input_map,
            mm_cond_number=mm_cond_number,
            nheads=nheads,
            grid_coords=grid_coords,
            head_right_cols=head_right_cols,
            above_count=above_count,
            right_col_count=right_col_count,
            per_lag_conditioning_count=max(counts),
            lag_count=len(counts) - 1,
            include_lag_self=include_lag_self,
            lat_round_decimals=lat_round_decimals,
            lon_round_decimals=lon_round_decimals,
            target_chunk_size=target_chunk_size,
            use_data_coords_for_offsets=use_data_coords_for_offsets,
            **kwargs,
        )

    def _active_lag_caps(self, time_idx: int) -> Tuple[int, ...]:
        max_lag = min(self.lag_count, int(time_idx))
        return tuple(self.per_lag_conditioning_counts[: max_lag + 1])

    def precompute_conditioning_sets(self):
        print(
            "Pre-computing Batched ReverseLColumn LagCap V3 "
            f"[heads_right={self.head_right_cols}, above={self.above_count}, "
            f"right_cols={self.right_col_count}, lag_caps={self.per_lag_conditioning_counts}, "
            f"coord_mode={'data' if self.use_data_coords_for_offsets else 'grid'}]...",
            end=" ",
        )
        t0 = time.time()

        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        all_data_list = [d.to(self.device, dtype=torch.float64) for d in all_data_list]
        real_data = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = real_data.shape
        self._n_real = n_real
        self._n_time = len(all_data_list)
        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if len(set(day_lengths)) != 1:
            raise ValueError(f"ReverseLColumnLagCapVecchiaFitV3 requires equal grid length per time, got {day_lengths}")
        self._n_grid = day_lengths[0]
        cumulative_len = np.cumsum([0] + day_lengths)

        coords_np = self._regular_coords_np(self._n_grid)
        lats, lons, local_to_row, local_to_col, row_col_to_local = self._build_grid_maps(coords_np)
        self._n_lat = len(lats)
        self._n_lon = len(lons)
        self._local_to_row = local_to_row
        self._local_to_col = local_to_col
        self._row_col_to_local = row_col_to_local
        self._stencil_cache = {}

        y = real_data[:, 2]
        valid_y_np = (~torch.isnan(y)).detach().cpu().numpy()
        valid_lats = real_data[~torch.isnan(y), 0]
        self.lat_mean_val = float(valid_lats.mean().item()) if valid_lats.numel() else float(real_data[:, 0].mean().item())

        max_m = int(sum(self.per_lag_conditioning_counts))
        dummy_block = torch.zeros((max_m, num_cols), device=self.device, dtype=torch.float64)
        for k in range(max_m):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        self.Full_Data = torch.cat([real_data, dummy_block], dim=0).contiguous()
        dummy_start = n_real
        valid_y_np = np.append(valid_y_np, np.ones(max_m, dtype=bool))

        head_locals = set()
        if self.head_right_cols > 0:
            for c in range(max(0, self._n_lon - self.head_right_cols), self._n_lon):
                for r in range(self._n_lat):
                    nb = self._get_local(r, c)
                    if nb is not None:
                        head_locals.add(nb)

        heads_indices: List[int] = []
        batch_lists: Dict[int, List[Tuple[int, List[int]]]] = {}
        m_sizes: List[int] = []

        col_order = range(self._n_lon - 1, -1, -1)
        row_order = range(self._n_lat - 1, -1, -1)

        for time_idx in range(self._n_time):
            offset = int(cumulative_len[time_idx])
            lag_caps = self._active_lag_caps(time_idx)
            max_d = int(sum(lag_caps))
            batch_lists.setdefault(max_d, [])

            for col in col_order:
                for row in row_order:
                    local_idx = self._get_local(row, col)
                    if local_idx is None:
                        continue
                    target_global = offset + local_idx
                    if not valid_y_np[target_global]:
                        continue

                    if local_idx in head_locals:
                        heads_indices.append(target_global)
                        continue

                    neigh_globals: List[int] = []
                    seen = set()
                    spatial_candidates = self._spatial_candidate_locals_uncapped(row, col)

                    for lag, per_lag_cap in enumerate(lag_caps):
                        per_lag_cap = int(per_lag_cap)
                        if per_lag_cap <= 0:
                            continue
                        neigh_time_idx = time_idx - lag
                        neigh_time_offset = int(cumulative_len[neigh_time_idx])
                        added_this_lag = 0

                        if lag > 0 and self.include_lag_self:
                            g = neigh_time_offset + local_idx
                            if g not in seen and valid_y_np[g]:
                                neigh_globals.append(g)
                                seen.add(g)
                                added_this_lag += 1

                        for nb_local in spatial_candidates:
                            if added_this_lag >= per_lag_cap:
                                break
                            g = neigh_time_offset + int(nb_local)
                            if g in seen or not valid_y_np[g]:
                                continue
                            neigh_globals.append(g)
                            seen.add(g)
                            added_this_lag += 1

                    m_sizes.append(len(neigh_globals))
                    if len(neigh_globals) < max_d:
                        padded = [dummy_start + k for k in range(max_d - len(neigh_globals))] + neigh_globals
                    else:
                        padded = neigh_globals[-max_d:]
                    batch_lists[max_d].append((target_global, padded))

        if heads_indices:
            h = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
            self.Heads_data = self.Full_Data[h].contiguous()
        else:
            self.Heads_data = torch.empty((0, num_cols), device=self.device, dtype=torch.float64)

        self.Batched_Groups = []
        for max_d in sorted(batch_lists):
            rows = batch_lists[max_d]
            if not rows:
                continue
            target_idx = torch.tensor([r[0] for r in rows], device=self.device, dtype=torch.long)
            neigh_idx = torch.tensor([r[1] for r in rows], device=self.device, dtype=torch.long)
            is_dummy = neigh_idx >= dummy_start
            target_rows = self.Full_Data[target_idx].contiguous()
            neigh_rows = self.Full_Data[neigh_idx].contiguous()
            dummy_mask = is_dummy.unsqueeze(-1)
            self.Batched_Groups.append({
                "max_m": int(max_d),
                "target_idx": target_idx,
                "neigh_idx": neigh_idx,
                "is_dummy": is_dummy,
                "coords_t": target_rows[:, [0, 1, 3]].contiguous(),
                "coords_n": neigh_rows[:, :, [0, 1, 3]].contiguous(),
                "X_t": self._design_from_rows(target_rows).contiguous(),
                "y_t": target_rows[:, 2:3].contiguous(),
                "X_n": self._design_from_rows(neigh_rows).masked_fill(dummy_mask, 0.0).contiguous(),
                "y_n": neigh_rows[:, :, 2].masked_fill(is_dummy, 0.0).contiguous(),
            })

        self.n_tails = int(sum(g["target_idx"].shape[0] for g in self.Batched_Groups))
        self.is_precomputed = True
        self.Grouped_Batches = []
        self.n_templates = np.nan

        if m_sizes:
            m_arr = np.asarray(m_sizes)
            m_msg = f"m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max()}"
        else:
            m_msg = "m empty"
        print(
            f"Done in {time.time() - t0:.1f}s. grid={self._n_lat}x{self._n_lon}, "
            f"heads={len(heads_indices)}, tails={self.n_tails}, "
            f"batches={[(g['max_m'], int(g['target_idx'].shape[0])) for g in self.Batched_Groups]}, {m_msg}"
        )
        return self


class ColumnSTLagCapTrendVecchiaFit(_STMeanDesignMixin, ReverseLColumnLagCapVecchiaFitV3):
    """Lag-cap Column ST kernel with selectable GLS mean design."""

    def __init__(self, *args, mean_design: str = "hour_spatial", **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mean_design(mean_design)

    def precompute_conditioning_sets(self):
        self._set_lon_mean_from_input_map()
        return super().precompute_conditioning_sets()


__all__ = ["ReverseLColumnLagCapVecchiaFitV3", "ColumnSTLagCapTrendVecchiaFit"]
