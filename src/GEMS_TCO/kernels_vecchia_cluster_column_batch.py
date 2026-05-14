"""
kernels_vecchia_cluster_column_batch.py

Block-target, GPU-batched reverse-L column Vecchia for real GEMS TCO tests.

This module is intentionally small: it reuses the block conditional likelihood
from ``ClusterHybridVecchiaFit`` and changes only the block ordering and
conditioning geometry.

Design:
- target blocks are regular-grid cell blocks, default 3x3 cells;
- block order follows the GEMS/column scan: north row first, east-to-west,
  then one block row south;
- same-time conditioning uses reverse-L previous blocks: north/same-column
  blocks first, then east/right block columns;
- lagged conditioning can include the same block at previous time plus a capped
  number of the same reverse-L block candidates;
- covariance distances are still computed from the coordinates in ``input_map``.
  Pass ``grid_coords`` when ``input_map`` uses source coordinates so blocks are
  built on the regular grid while covariance uses real source locations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from GEMS_TCO.kernels_vecchia_cluster_hybrid import ClusterHybridVecchiaFit


class ClusterColumnVecchiaFitBatch(ClusterHybridVecchiaFit):
    """Cluster/block target Vecchia with reverse-L column block conditioning."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, torch.Tensor],
        grid_coords: Optional[np.ndarray] = None,
        block_shape: Tuple[int, int] = (3, 3),
        lag0_block_count: int = 6,
        lag1_same_block: bool = True,
        lag1_block_count: int = 2,
        lag2_same_block: bool = True,
        lag2_block_count: int = 1,
        daily_stride: int = 2,
        above_block_count: int = 2,
        right_block_count: int = 3,
        target_chunk_size: int = 64,
        min_target_points: int = 1,
    ):
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=block_shape,
            n_neighbor_blocks_t=lag0_block_count,
            lag1_same_block=lag1_same_block,
            lag1_local_blocks=lag1_block_count,
            lag1_shifted_blocks=0,
            lag2_same_block=lag2_same_block,
            lag2_local_blocks=lag2_block_count,
            lag2_shifted_blocks=0,
            daily_stride=daily_stride,
            lag1_lon_offset=0.0,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=None,
        )
        self.lag0_block_count = int(lag0_block_count)
        self.lag1_block_count = int(lag1_block_count)
        self.lag2_block_count = int(lag2_block_count)
        self.above_block_count = int(above_block_count)
        self.right_block_count = int(right_block_count)
        self.block_keys: List[Tuple[int, int]] = []
        self.block_key_to_idx: Dict[Tuple[int, int], int] = {}

    @staticmethod
    def _unique_inverse(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rounded = np.round(values.astype(np.float64), 10)
        unique = np.unique(rounded)
        lookup = {v: i for i, v in enumerate(unique)}
        inverse = np.array([lookup[v] for v in rounded], dtype=np.int64)
        return unique, inverse

    def _build_clusters(self, n_points: int):
        coords = self._grid_coords_np(n_points)
        _, row_idx = self._unique_inverse(coords[:, 0])
        _, col_idx = self._unique_inverse(coords[:, 1])

        br = self.block_shape[0]
        bc = self.block_shape[1]
        raw: Dict[Tuple[int, int], List[int]] = {}
        for local_idx, (r, c) in enumerate(zip(row_idx, col_idx)):
            key = (int(r // br), int(c // bc))
            raw.setdefault(key, []).append(int(local_idx))

        # Column/GEMS scan: north block rows first, east-to-west within a row.
        ordered_keys = sorted(raw, key=lambda x: (-x[0], -x[1]))
        self.block_keys = ordered_keys
        self.block_key_to_idx = {key: i for i, key in enumerate(ordered_keys)}
        self.cluster_points = [np.array(raw[k], dtype=np.int64) for k in ordered_keys]
        self.cluster_centroids = np.vstack([coords[p].mean(axis=0) for p in self.cluster_points])
        self.n_clusters = len(self.cluster_points)
        self.max_points_per_cluster = max(len(p) for p in self.cluster_points)

        # These are unused for column conditioning but kept for inherited summary compatibility.
        self.cluster_nns = np.full((self.n_clusters, 0), -1, dtype=np.int64)
        self.shift_lookup_lag1 = np.arange(self.n_clusters, dtype=np.int64)
        self.shift_lookup_lag2 = np.arange(self.n_clusters, dtype=np.int64)


    def _precompute_message(self) -> str:
        return (
            "Pre-computing ClusterColumnVecchiaFitBatch "
            f"(smooth={self.smooth}, block={self.block_shape}, "
            f"A=reverseL{self.lag0_block_count}, "
            f"B=same{int(self.lag1_same_block)}+reverseL{self.lag1_block_count}, "
            f"C=same{int(self.lag2_same_block)}+reverseL{self.lag2_block_count}, "
            f"stencil=above{self.above_block_count}+rightcols{self.right_block_count})..."
        )

    def _reverse_l_block_candidates(self, block_idx: int) -> List[int]:
        brow, bcol = self.block_keys[int(block_idx)]
        out: List[int] = []
        seen = set()

        def add(key: Tuple[int, int]):
            j = self.block_key_to_idx.get((int(key[0]), int(key[1])))
            if j is None or j == block_idx or j in seen:
                return
            # Only use previously scanned blocks.  Because the order is north/east first,
            # a previous block has a smaller ordered index.
            if j >= block_idx:
                return
            out.append(j)
            seen.add(j)

        # North/above blocks in the same block column.
        for k in range(1, self.above_block_count + 1):
            add((brow + k, bcol))

        # East/right block columns.  Same block row first, then progressively north.
        max_brow = max((r for r, _ in self.block_keys), default=brow)
        for up in range(0, max_brow - brow + 1):
            r2 = brow + up
            for dc in range(1, self.right_block_count + 1):
                add((r2, bcol + dc))

        return out

    @staticmethod
    def _append_unique_block_times(
        out: List[Tuple[int, int]],
        time_idx: int,
        block_candidates: Sequence[int],
        cap: int,
    ):
        added = 0
        seen = set(out)
        for block_idx in block_candidates:
            if added >= cap:
                break
            key = (int(time_idx), int(block_idx))
            if key in seen:
                continue
            out.append(key)
            seen.add(key)
            added += 1

    def _conditioning_blocks(self, block_idx: int, time_idx: int) -> Tuple[str, int, List[Tuple[int, int]]]:
        reverse_l_prev = self._reverse_l_block_candidates(int(block_idx))
        cond: List[Tuple[int, int]] = []

        self._append_unique_block_times(cond, time_idx, reverse_l_prev, self.lag0_block_count)
        label = "A"
        max_cond_blocks = self.lag0_block_count

        if time_idx > 0:
            label = "AB"
            prev_time = time_idx - 1
            if self.lag1_same_block:
                self._append_unique_block_times(cond, prev_time, [block_idx], 1)
            self._append_unique_block_times(cond, prev_time, reverse_l_prev, self.lag1_block_count)
            max_cond_blocks += int(self.lag1_same_block) + self.lag1_block_count

        if time_idx >= self.daily_stride:
            label = "ABC"
            prev2_time = time_idx - self.daily_stride
            if self.lag2_same_block:
                self._append_unique_block_times(cond, prev2_time, [block_idx], 1)
            self._append_unique_block_times(cond, prev2_time, reverse_l_prev, self.lag2_block_count)
            max_cond_blocks += int(self.lag2_same_block) + self.lag2_block_count

        return label, max_cond_blocks * self.max_points_per_cluster, cond

    def cluster_summary(self) -> Dict[str, float]:
        out = super().cluster_summary()
        out.update({
            "conditioning": "reverse_l_column_blocks",
            "lag0_block_count": self.lag0_block_count,
            "lag1_same_block": int(self.lag1_same_block),
            "lag1_block_count": self.lag1_block_count,
            "lag2_same_block": int(self.lag2_same_block),
            "lag2_block_count": self.lag2_block_count,
            "above_block_count": self.above_block_count,
            "right_block_count": self.right_block_count,
        })
        return out


__all__ = ["ClusterColumnVecchiaFitBatch"]
