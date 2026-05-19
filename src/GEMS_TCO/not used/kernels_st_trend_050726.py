"""
kernels_st_trend_050726.py

Spatio-temporal Vecchia wrappers with selectable GLS mean designs.

This file changes only the profiled GLS mean design.  The covariance parameter
mapping, conditioning sets, and Vecchia likelihood remain those of the parent
Hybrid/Column kernels.
"""

from __future__ import annotations

import numpy as np
import torch

from GEMS_TCO.kernel_vecchia_col_batch import ReverseLColumnVecchiaFitBatch
from GEMS_TCO.vecchia_candidate.kernels_vecchia_hybrid import HybridVecchiaFit


def _n_features_for_mean_design(mean_design: str) -> int:
    if mean_design == "base":
        return 9
    if mean_design == "latlon":
        return 10
    if mean_design == "hour_spatial":
        return 24
    raise ValueError(f"Unknown mean_design={mean_design!r}")


class _STMeanDesignMixin:
    def _init_mean_design(self, mean_design: str):
        if mean_design not in ("base", "latlon", "hour_spatial"):
            raise ValueError("mean_design must be one of: base, latlon, hour_spatial")
        self.mean_design = str(mean_design)
        self.n_features = _n_features_for_mean_design(self.mean_design)
        self.lon_mean_val = 0.0

    def _set_lon_mean_from_input_map(self):
        all_data = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        real_data = torch.cat(all_data, dim=0).to(self.device, dtype=torch.float64)
        y = real_data[:, 2]
        coord_ok = ~torch.isnan(real_data[:, 0]) & ~torch.isnan(real_data[:, 1])
        obs_ok = (~torch.isnan(y)) & coord_ok
        valid_lons = real_data[obs_ok, 1]
        self.lon_mean_val = (
            float(valid_lons.mean().item()) if valid_lons.numel()
            else float(torch.nanmean(real_data[:, 1]).item())
        )

    def _hour_dummies(self, flat: torch.Tensor) -> torch.Tensor:
        dums = flat[:, 4:11].to(torch.float64)
        if dums.shape[1] < 7:
            pad = torch.zeros((dums.shape[0], 7 - dums.shape[1]), device=self.device, dtype=torch.float64)
            dums = torch.cat([dums, pad], dim=1)
        return dums

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        lat = (flat[:, 0:1] - self.lat_mean_val).to(torch.float64)
        lon = (flat[:, 1:2] - self.lon_mean_val).to(torch.float64)
        dums = self._hour_dummies(flat)

        if self.mean_design == "base":
            X = torch.cat([ones, lat, dums], dim=1)
        elif self.mean_design == "latlon":
            X = torch.cat([ones, lat, lon, dums], dim=1)
        else:
            first_hour = (1.0 - dums.sum(dim=1, keepdim=True)).clamp(min=0.0, max=1.0)
            hour_onehot = torch.cat([first_hour, dums], dim=1)
            X = torch.cat([hour_onehot, hour_onehot * lat, hour_onehot * lon], dim=1)

        return X.reshape(*orig_shape, self.n_features)


class ColumnSTTrendVecchiaFit(_STMeanDesignMixin, ReverseLColumnVecchiaFitBatch):
    def __init__(self, *args, mean_design: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mean_design(mean_design)

    def precompute_conditioning_sets(self):
        self._set_lon_mean_from_input_map()
        return super().precompute_conditioning_sets()


class HybridSTTrendVecchiaFit(_STMeanDesignMixin, HybridVecchiaFit):
    def __init__(self, *args, mean_design: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mean_design(mean_design)

    def precompute_conditioning_sets(self):
        out = super().precompute_conditioning_sets()
        self._set_lon_mean_from_input_map()
        self._rebuild_tail_design_tensors()
        return out

    def _head_design_response(self):
        X_h = self._design_from_rows(self.Heads_data)
        y_h = self.Heads_data[:, 2].unsqueeze(-1)
        return X_h, y_h

    def _rebuild_tail_design_tensors(self):
        all_data = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        real_data = torch.cat(all_data, dim=0).to(self.device, dtype=torch.float64)
        n_real, num_cols = real_data.shape
        n_dummies = int(getattr(self, "_n_dummies_stored", 0))
        if n_dummies > 0:
            dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float64)
            full_data = torch.cat([real_data, dummy_block], dim=0)
        else:
            full_data = real_data

        for t_attr, mask_attr, locs_attr in [
            ("_T_A", "_is_dummy_A", "Locs_A"),
            ("_T_AB", "_is_dummy_AB", "Locs_AB"),
            ("_T_ABC", "_is_dummy_ABC", "Locs_ABC"),
        ]:
            T = getattr(self, t_attr, None)
            is_dummy = getattr(self, mask_attr, None)
            if T is None:
                setattr(self, locs_attr, None)
                continue
            G = full_data[T].to(torch.float64)
            Locs = self._design_from_rows(G).contiguous()
            if is_dummy is not None:
                Locs = Locs.masked_fill(is_dummy, 0.0)
            setattr(self, locs_attr, Locs)


__all__ = ["HybridSTTrendVecchiaFit", "ColumnSTTrendVecchiaFit"]
