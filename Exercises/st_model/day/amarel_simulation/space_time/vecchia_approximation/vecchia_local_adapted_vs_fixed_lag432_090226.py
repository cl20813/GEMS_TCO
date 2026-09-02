#!/usr/bin/env python3
"""Local four-geometry Vecchia comparison with 4/3/2 block budgets.

This is a lightweight configuration layer over the common fitting/reporting
driver.  It preserves the calibrated M3+Q3 initializer, adapted corridor,
shifted-center, fixed-center, exact three-way union, conditional-eigen plots,
and fixed-zero nugget implementation while replacing every 6/4/3 graph by a
4/3/2 graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import vecchia_adapted_vs_fixed_lag643_090126 as core
from GEMS_TCO.vecchia_cluster import StrategyClusterVecchiaFit
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
)


BLOCK_SHAPE = (4, 4)
LAG_COUNTS = (4, 3, 2)


class CalibratedShiftedCenter4x4Lag432VecchiaFit(StrategyClusterVecchiaFit):
    """Lag-432 graph centered at calibrated one- and two-step shifts."""

    def __init__(
        self,
        smooth: float,
        input_map,
        grid_coords=None,
        reference_advec_lat: float = 0.0,
        reference_advec_lon: float = -0.126,
        daily_stride: int = 2,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
    ):
        self.reference_advec_lat = float(reference_advec_lat)
        self.reference_advec_lon = float(reference_advec_lon)
        self.past_offset_vector = np.asarray(
            [-self.reference_advec_lat, -self.reference_advec_lon],
            dtype=np.float64,
        )
        scalar_norm = max(
            float(np.hypot(self.reference_advec_lat, self.reference_advec_lon)),
            1e-12,
        )
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=BLOCK_SHAPE,
            strategy="offset_tapered",
            lag0_block_count=LAG_COUNTS[0],
            lag1_block_count=LAG_COUNTS[1],
            lag2_block_count=LAG_COUNTS[2],
            daily_stride=daily_stride,
            lag1_lon_offset=scalar_norm,
            lag2_lon_offset=2.0 * scalar_norm,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
        )
        self.temporal_basis = "calibrated_signed_2d_shifted_center"

    def _cluster_candidates_from_shifted_center(
        self,
        block_idx: int,
        multiplier: float,
        count: int,
    ) -> list[int]:
        if count <= 0 or self.cluster_centroids is None or self.cluster_all_tree is None:
            return []
        target = np.asarray(self.cluster_centroids[int(block_idx)], dtype=np.float64)
        center = target + float(multiplier) * self.past_offset_vector
        n_clusters = int(self.cluster_centroids.shape[0])
        k = min(n_clusters, max(1, int(count)))
        _, indices = self.cluster_all_tree.query(center, k=k)
        out: list[int] = []
        self._append_unique_int(out, np.asarray(indices, dtype=np.int64).reshape(-1))
        if len(out) < int(count) and k < n_clusters:
            _, indices = self.cluster_all_tree.query(center, k=n_clusters)
            self._append_unique_int(out, np.asarray(indices, dtype=np.int64).reshape(-1))
        return out[: int(count)]

    def _lag_candidates(self, block_idx: int, lag: int) -> list[int]:
        if int(lag) == 1:
            return self._cluster_candidates_from_shifted_center(
                block_idx, 1.0, LAG_COUNTS[1]
            )
        if int(lag) == 2:
            return self._cluster_candidates_from_shifted_center(
                block_idx, 2.0, LAG_COUNTS[2]
            )
        raise ValueError(f"Only lag 1 and lag 2 are supported, got {lag}")

    def cluster_summary(self) -> dict[str, Any]:
        out = super().cluster_summary()
        out.update(
            {
                "spec_name": "calibrated_shifted_center_4x4_lag432",
                "geometry": "shifted",
                "conditioning_mode": "calibrated_signed_2d_shifted_center",
                "reference_advec_lat": self.reference_advec_lat,
                "reference_advec_lon": self.reference_advec_lon,
                "past_step_lat": float(self.past_offset_vector[0]),
                "past_step_lon": float(self.past_offset_vector[1]),
            }
        )
        return out


class FixedCenterLag432VecchiaFit(StrategyClusterVecchiaFit):
    """Lag-432 graph with both past neighborhoods fixed at the target."""

    def __init__(
        self,
        smooth: float,
        input_map,
        grid_coords,
        daily_stride: int,
        target_chunk_size: int,
        min_target_points: int,
    ):
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=BLOCK_SHAPE,
            strategy="center_tapered",
            lag0_block_count=LAG_COUNTS[0],
            lag1_block_count=LAG_COUNTS[1],
            lag2_block_count=LAG_COUNTS[2],
            daily_stride=daily_stride,
            lag1_lon_offset=1e-12,
            lag2_lon_offset=2e-12,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
        )
        self.temporal_basis = "fixed_target_center"

    def cluster_summary(self) -> dict[str, Any]:
        out = super().cluster_summary()
        out.update(
            {
                "spec_name": "fixed_target_center_4x4_lag432",
                "geometry": "fixed",
                "conditioning_mode": "fixed_target_center",
                "reference_advec_lat": 0.0,
                "reference_advec_lon": 0.0,
                "past_step_lat": 0.0,
                "past_step_lon": 0.0,
            }
        )
        return out


class UnionLag432VecchiaFit(DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit):
    """Exact union of adapted, shifted, and fixed lag-432 block sets."""

    def __init__(
        self,
        smooth: float,
        input_map,
        grid_coords,
        reference_advec_lat: float,
        reference_advec_lon: float,
        daily_stride: int,
        target_chunk_size: int,
        min_target_points: int,
    ):
        self.component_lag_counts = {1: LAG_COUNTS[1], 2: LAG_COUNTS[2]}
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lat=reference_advec_lat,
            reference_advec_lon=reference_advec_lon,
            daily_stride=daily_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
        )
        self.lag1_block_count = 3 * LAG_COUNTS[1]
        self.lag2_block_count = 3 * LAG_COUNTS[2]
        self.lag1_max_blocks = self.lag1_block_count
        self.lag2_max_blocks = self.lag2_block_count
        self.lag1_local_blocks = self.lag1_max_blocks
        self.lag2_local_blocks = self.lag2_max_blocks
        self.all_neighbor_search = max(
            self.all_neighbor_search,
            self.lag1_max_blocks,
            self.lag2_max_blocks,
        ) + 8
        self.temporal_basis = "union_corridor_shifted_and_fixed"

    def _lag_candidates(self, block_idx: int, lag: int) -> list[int]:
        component_count = self.component_lag_counts[int(lag)]
        multipliers = (
            self.lag1_corridor_multipliers
            if int(lag) == 1
            else self.lag2_corridor_multipliers
        )
        adapted = self._cluster_candidates_from_vector_corridor(
            block_idx,
            multipliers,
            component_count,
        )
        shifted = CalibratedShiftedCenter4x4Lag432VecchiaFit._cluster_candidates_from_shifted_center(
            self,
            int(block_idx),
            1.0 if int(lag) == 1 else 2.0,
            component_count,
        )
        fixed = self._cluster_candidates_from_center(int(block_idx), component_count)
        out: list[int] = []
        self._append_unique_int(out, adapted)
        self._append_unique_int(out, shifted)
        self._append_unique_int(out, fixed)
        return out

    def cluster_summary(self) -> dict[str, Any]:
        out = super().cluster_summary()
        out.update(
            {
                "spec_name": "union_corridor_shifted_fixed_4x4_lag432",
                "geometry": "union",
                "conditioning_mode": "union_corridor_shifted_and_fixed",
                "component_lag1_block_count": LAG_COUNTS[1],
                "component_lag2_block_count": LAG_COUNTS[2],
                "union_component_count": 3,
            }
        )
        return out


class AdaptedFixedZeroNuggetLag432(
    core.FixedZeroNuggetMixin,
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
):
    pass


class ShiftedFixedZeroNuggetLag432(
    core.FixedZeroNuggetMixin,
    CalibratedShiftedCenter4x4Lag432VecchiaFit,
):
    pass


class FixedCenterFixedZeroNuggetLag432(
    core.FixedZeroNuggetMixin,
    FixedCenterLag432VecchiaFit,
):
    pass


class UnionFixedZeroNuggetLag432(
    core.FixedZeroNuggetMixin,
    UnionLag432VecchiaFit,
):
    pass


def configure_core() -> None:
    """Install lag-432 classes into the common execution/reporting engine."""
    core.LAG_COUNTS = LAG_COUNTS
    core.LAG_TAG = "lag432"
    core.AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit = (
        DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit
    )
    core.CalibratedShiftedCenter4x4Lag643VecchiaFit = (
        CalibratedShiftedCenter4x4Lag432VecchiaFit
    )
    core.FixedCenterLag643VecchiaFit = FixedCenterLag432VecchiaFit
    core.UnionLag643VecchiaFit = UnionLag432VecchiaFit
    core.AdaptedFixedZeroNuggetVecchiaFit = AdaptedFixedZeroNuggetLag432
    core.ShiftedFixedZeroNuggetVecchiaFit = ShiftedFixedZeroNuggetLag432
    core.FixedCenterFixedZeroNuggetVecchiaFit = FixedCenterFixedZeroNuggetLag432
    core.UnionFixedZeroNuggetVecchiaFit = UnionFixedZeroNuggetLag432


configure_core()


if __name__ == "__main__":
    core.main()
