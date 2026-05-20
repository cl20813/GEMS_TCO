"""
kernels_space_trend_050726.py

Pure-space Vecchia kernels with selectable GLS mean designs.

Mean designs:
  lat:
    intercept + centered latitude
  base:
    intercept + centered latitude + hour dummies
  latlon:
    intercept + centered latitude + centered longitude + hour dummies
  hour_spatial:
    hour-specific intercept + hour-specific centered latitude slope
    + hour-specific centered longitude slope

These wrappers reuse the same pure-space conditioning kernels as
kernels_space_050726.py and only change the GLS design matrix.
"""

from __future__ import annotations

import numpy as np
import torch

from GEMS_TCO.kernels_space_050726 import ColumnSpaceVecchiaFit, HybridSpaceVecchiaFit


def _n_features_for_mean_design(mean_design: str) -> int:
    if mean_design == "lat":
        return 2
    if mean_design == "base":
        return 9
    if mean_design == "latlon":
        return 10
    if mean_design == "hour_spatial":
        return 24
    raise ValueError(f"Unknown mean_design={mean_design!r}")


class _MeanDesignMixin:
    def _init_mean_design(self, mean_design: str):
        if mean_design not in ("lat", "base", "latlon", "hour_spatial"):
            raise ValueError("mean_design must be one of: lat, base, latlon, hour_spatial")
        self.mean_design = str(mean_design)
        self.n_features = _n_features_for_mean_design(self.mean_design)
        self.lon_mean_val = 0.0

    def _make_full_data(self, max_m: int):
        out = super()._make_full_data(max_m)
        _, full_data, n_real, _, _, _, _ = out
        real_data = full_data[:n_real]
        y = real_data[:, 2]
        coord_ok = ~torch.isnan(real_data[:, 0]) & ~torch.isnan(real_data[:, 1])
        obs_ok = (~torch.isnan(y)) & coord_ok
        valid_lons = real_data[obs_ok, 1]
        self.lon_mean_val = (
            float(valid_lons.mean().item()) if valid_lons.numel()
            else float(torch.nanmean(real_data[:, 1]).item())
        )
        return out

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

        if self.mean_design == "lat":
            X = torch.cat([ones, lat], dim=1)
        elif self.mean_design == "base":
            X = torch.cat([ones, lat, dums], dim=1)
        elif self.mean_design == "latlon":
            X = torch.cat([ones, lat, lon, dums], dim=1)
        else:
            first_hour = (1.0 - dums.sum(dim=1, keepdim=True)).clamp(min=0.0, max=1.0)
            hour_onehot = torch.cat([first_hour, dums], dim=1)
            X = torch.cat([hour_onehot, hour_onehot * lat, hour_onehot * lon], dim=1)

        return X.reshape(*orig_shape, self.n_features)


class HybridSpaceTrendVecchiaFit(_MeanDesignMixin, HybridSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mean_design(mean_design)


class ColumnSpaceTrendVecchiaFit(_MeanDesignMixin, ColumnSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mean_design(mean_design)


__all__ = ["HybridSpaceTrendVecchiaFit", "ColumnSpaceTrendVecchiaFit"]
