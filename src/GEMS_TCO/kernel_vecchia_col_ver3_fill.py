"""
kernel_vecchia_col_ver3_fill.py

Reverse-L Column V3 with missing-neighbor refill.

This keeps the same V3 nominal budget, m ~= 42:
  per_lag_conditioning_count=14 over t, t-1, t-2.

The difference from ReverseLColumnVecchiaFitV3 is in precompute:
for each target/lag, candidates are scanned beyond the nominal cap and missing
neighbors are skipped until the per-lag valid-neighbor cap is filled when
possible. This intentionally allows many more templates on real-data-like
missing patterns; the purpose is to test conditioning geometry/RMSRE, not cache
efficiency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from GEMS_TCO.kernel_vecchia_col_ver2 import ReverseLColumnVecchiaFitV2


class ReverseLColumnVecchiaFitV3FillMissing(ReverseLColumnVecchiaFitV2):
    """Column V3 that refills skipped missing neighbors from later candidates."""

    def __init__(self, *args, use_data_coords_for_offsets: bool = False, **kwargs):
        kwargs.setdefault("per_lag_conditioning_count", 14)
        super().__init__(*args, **kwargs)
        self.use_data_coords_for_offsets = bool(use_data_coords_for_offsets)

    def _spatial_candidate_locals_uncapped(self, row: int, col: int) -> List[int]:
        key = ("uncapped", int(row), int(col))
        if key in self._stencil_cache:
            return self._stencil_cache[key]

        out: List[int] = []
        seen = set()

        for k in range(1, self.above_count + 1):
            nb = self._get_local(row + k, col)
            if nb is not None and nb not in seen:
                out.append(nb)
                seen.add(nb)

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
            if nb not in seen:
                out.append(nb)
                seen.add(nb)

        self._stencil_cache[key] = out
        return out

    def precompute_conditioning_sets(self):
        print(
            "Pre-computing ReverseLColumnVecchia V3-fill "
            f"[heads_right={self.head_right_cols}, above={self.above_count}, "
            f"right_cols={self.right_col_count}, per_lag={self.per_lag_conditioning_count}, "
            f"lags={self.lag_count}, "
            f"coord_mode={'data' if self.use_data_coords_for_offsets else 'grid'}]...",
            end=" ",
        )
        t0 = time.time()

        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        all_data_list = [d.to(self.device, dtype=torch.float64) for d in all_data_list]
        self.Full_Data_Grid = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = self.Full_Data_Grid.shape
        self._n_real = n_real
        self._n_time = len(all_data_list)
        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if len(set(day_lengths)) != 1:
            raise ValueError(f"ReverseLColumnVecchia requires equal grid length per time, got {day_lengths}")
        self._n_grid = day_lengths[0]
        cumulative_len = np.cumsum([0] + day_lengths)

        coords_np = self._regular_coords_np(self._n_grid)
        data_coords_np = None
        if self.use_data_coords_for_offsets:
            data_coords_np = self.Full_Data_Grid[:, :2].detach().cpu().numpy().astype(np.float64)
        lats, lons, local_to_row, local_to_col, row_col_to_local = self._build_grid_maps(coords_np)
        self._n_lat = len(lats)
        self._n_lon = len(lons)
        self._local_to_row = local_to_row
        self._local_to_col = local_to_col
        self._row_col_to_local = row_col_to_local
        self._stencil_cache = {}

        y = self.Full_Data_Grid[:, 2]
        valid_y_np = (~torch.isnan(y)).detach().cpu().numpy()
        valid_lats = self.Full_Data_Grid[~torch.isnan(y), 0]
        self.lat_mean_val = (
            float(valid_lats.mean().item())
            if valid_lats.numel()
            else float(self.Full_Data_Grid[:, 0].mean().item())
        )

        time_values = []
        for d in all_data_list:
            good = ~torch.isnan(d[:, 3])
            time_values.append(float(d[good, 3].median().item()) if good.any() else 0.0)

        head_locals = set()
        for c in range(max(0, self._n_lon - self.head_right_cols), self._n_lon):
            for r in range(self._n_lat):
                nb = self._get_local(r, c)
                if nb is not None:
                    head_locals.add(nb)

        heads_indices = []
        groups: Dict[Tuple[Tuple[float, float, float], ...], Dict[str, Any]] = {}
        m_sizes = []
        per_lag_cap = max(0, int(self.per_lag_conditioning_count))

        col_order = range(self._n_lon - 1, -1, -1)
        row_order = range(self._n_lat - 1, -1, -1)

        def add_group(target_global: int, neigh_globals: List[int], offsets: List[Tuple[float, float, float]]):
            if len(neigh_globals) == 0:
                key = tuple()
                off_tensor = torch.empty((0, 3), device=self.device, dtype=torch.float64)
            else:
                key = tuple((round(a, 8), round(b, 8), round(c, 8)) for a, b, c in offsets)
                off_tensor = torch.tensor(offsets, device=self.device, dtype=torch.float64)
            if key not in groups:
                groups[key] = {"offsets": off_tensor, "batch_idx": [], "target_idx": []}
            groups[key]["batch_idx"].append(neigh_globals)
            groups[key]["target_idx"].append(target_global)
            m_sizes.append(len(neigh_globals))

        for time_idx in range(self._n_time):
            offset = int(cumulative_len[time_idx])
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

                    if self.use_data_coords_for_offsets:
                        target_lat = float(data_coords_np[target_global, 0])
                        target_lon = float(data_coords_np[target_global, 1])
                    else:
                        target_lat = float(coords_np[local_idx, 0])
                        target_lon = float(coords_np[local_idx, 1])
                    neigh_globals: List[int] = []
                    offsets_list: List[Tuple[float, float, float]] = []
                    seen = set()

                    spatial_candidates = self._spatial_candidate_locals_uncapped(row, col)
                    for lag in range(self.lag_count + 1):
                        neigh_time_idx = time_idx - lag
                        if neigh_time_idx < 0:
                            continue
                        neigh_time_offset = int(cumulative_len[neigh_time_idx])
                        dt = float(time_values[time_idx] - time_values[neigh_time_idx])
                        added_this_lag = 0

                        if lag > 0 and self.include_lag_self:
                            g = neigh_time_offset + local_idx
                            if g not in seen and valid_y_np[g]:
                                neigh_globals.append(g)
                                offsets_list.append((0.0, 0.0, dt))
                                seen.add(g)
                                added_this_lag += 1

                        for nb_local in spatial_candidates:
                            if added_this_lag >= per_lag_cap:
                                break
                            g = neigh_time_offset + int(nb_local)
                            if g in seen or not valid_y_np[g]:
                                continue
                            if self.use_data_coords_for_offsets:
                                nb_lat = float(data_coords_np[g, 0])
                                nb_lon = float(data_coords_np[g, 1])
                            else:
                                nb_lat = float(coords_np[nb_local, 0])
                                nb_lon = float(coords_np[nb_local, 1])
                            neigh_globals.append(g)
                            offsets_list.append((target_lat - nb_lat, target_lon - nb_lon, dt))
                            seen.add(g)
                            added_this_lag += 1

                    add_group(target_global, neigh_globals, offsets_list)

        if heads_indices:
            h = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
            self.Heads_data = self.Full_Data_Grid[h].contiguous()
        else:
            self.Heads_data = torch.empty((0, num_cols), device=self.device, dtype=torch.float64)

        self.Grouped_Batches = []
        for val in groups.values():
            t_idx = torch.tensor(val["target_idx"], device=self.device, dtype=torch.long)
            if val["offsets"].shape[0] == 0:
                b_idx = torch.empty((len(val["target_idx"]), 0), device=self.device, dtype=torch.long)
            else:
                b_idx = torch.tensor(val["batch_idx"], device=self.device, dtype=torch.long)
            self.Grouped_Batches.append({"offsets": val["offsets"], "batch_idx": b_idx, "target_idx": t_idx})

        self.n_tails = int(sum(len(g["target_idx"]) for g in self.Grouped_Batches))
        self.is_precomputed = True
        if m_sizes:
            m_arr = np.asarray(m_sizes)
            m_msg = f"m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max()}"
        else:
            m_msg = "m empty"
        print(
            f"Done in {time.time() - t0:.1f}s. "
            f"grid={self._n_lat}x{self._n_lon}, heads={len(heads_indices)}, "
            f"tails={self.n_tails}, templates={len(self.Grouped_Batches)}, {m_msg}"
        )
        return self


__all__ = ["ReverseLColumnVecchiaFitV3FillMissing"]
