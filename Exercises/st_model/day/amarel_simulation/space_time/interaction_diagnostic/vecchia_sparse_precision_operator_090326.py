#!/usr/bin/env python3
"""Sparse whitening/precision utilities for block-target Vecchia models.

For each model target block T and conditioning set C, this module constructs

    e_T = L_{T|C}^{-1} (r_T - K_TC K_CC^{-1} r_C),

where L_{T|C} is the Cholesky factor of the conditional covariance.  Stacking
these equations gives a square sparse whitening matrix B.  The fitted Vecchia
precision is then Omega = B.T @ B, but Omega never needs to be materialized:

    Omega @ v = B.T @ (B @ v).

The construction follows the exact block conditionals used by
ClusterHybridVecchiaFit._accumulate_gls_stats, including its covariance jitter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.sparse
import torch


@dataclass
class SparseVecchiaPrecision:
    whitener: scipy.sparse.csr_matrix
    valid_global_indices: np.ndarray
    global_to_compact: np.ndarray
    residual: np.ndarray
    design: np.ndarray
    response: np.ndarray
    metadata: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.whitener.shape[0])

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        return np.asarray(self.whitener.T @ (self.whitener @ x)).reshape(-1)

    def matmat(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float64)
        return np.asarray(self.whitener.T @ (self.whitener @ x))

    def covariance_solve(self, vector: np.ndarray) -> np.ndarray:
        """Apply Omega^{-1} with two sparse triangular solves.

        B is a permuted lower-triangular factor.  scipy.sparse.linalg.spsolve
        remains appropriate without explicitly recovering that permutation.
        This method is intended for checks, not for the Lanczos diagnostic.
        """
        from scipy.sparse.linalg import spsolve

        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        intermediate = spsolve(self.whitener.T.tocsr(), x)
        return np.asarray(spsolve(self.whitener, intermediate)).reshape(-1)


def compact_real_data(model: Any, beta: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Recreate the float32 precompute data path and compact valid rows."""
    parts = [
        (value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
        .to(device="cpu", dtype=torch.float32)
        for value in model.input_map.values()
    ]
    real = torch.cat(parts, dim=0).contiguous()
    valid_mask = torch.isfinite(real[:, 2]).detach().cpu().numpy()
    valid_global = np.flatnonzero(valid_mask).astype(np.int64)
    n_real = int(real.shape[0])
    global_to_compact = np.full(n_real, -1, dtype=np.int64)
    global_to_compact[valid_global] = np.arange(len(valid_global), dtype=np.int64)

    selected = real[torch.as_tensor(valid_global, dtype=torch.long)].to(torch.float64)
    ones = torch.ones((len(valid_global), 1), dtype=torch.float64)
    lat = selected[:, 0:1] - float(model.lat_mean_val)
    dummies = selected[:, 4:11]
    design = torch.cat([ones, lat, dummies], dim=1).detach().cpu().numpy()
    response = selected[:, 2].detach().cpu().numpy()
    beta_vector = np.asarray(beta, dtype=np.float64).reshape(-1)
    residual = response - design @ beta_vector
    return valid_global, global_to_compact, response, design, residual


def _block_whitening_coefficients(
    covariance: torch.Tensor,
    n_conditioning: int,
    n_target: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coefficients [-L_s^-1 W, L_s^-1] and conditional Cholesky."""
    c = int(n_conditioning)
    t = int(n_target)
    k_cc = covariance[:, :c, :c]
    k_ct = covariance[:, :c, c : c + t]
    k_tc = covariance[:, c : c + t, :c]
    k_tt = covariance[:, c : c + t, c : c + t]

    l_cc = torch.linalg.cholesky(k_cc)
    z = torch.linalg.solve_triangular(l_cc, k_ct, upper=False)
    kcc_inverse_kct = torch.linalg.solve_triangular(
        l_cc.transpose(-1, -2), z, upper=True
    )
    weights = kcc_inverse_kct.transpose(-1, -2)
    conditional = k_tt - torch.bmm(k_tc, kcc_inverse_kct)
    conditional = 0.5 * (conditional + conditional.transpose(-1, -2))
    l_conditional = torch.linalg.cholesky(conditional)

    identity = torch.eye(t, dtype=covariance.dtype, device=covariance.device)
    identity = identity.unsqueeze(0).expand(covariance.shape[0], t, t)
    inverse_l = torch.linalg.solve_triangular(l_conditional, identity, upper=False)
    conditioning_coefficients = -torch.bmm(inverse_l, weights)
    return conditioning_coefficients, inverse_l, l_conditional


def build_sparse_vecchia_precision(
    model: Any,
    params: torch.Tensor,
    beta: np.ndarray,
    chunk_size: int | None = None,
    coefficient_drop_tolerance: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> SparseVecchiaPrecision:
    """Construct B and compact residuals from a precomputed cluster model."""
    if not bool(getattr(model, "is_precomputed", False)):
        raise RuntimeError("Run model.precompute_conditioning_sets() first")
    if not hasattr(model, "_cluster_batches"):
        raise TypeError("Expected a cluster Vecchia model with _cluster_batches")
    if progress is None:
        progress = lambda _message: None
    started = time.perf_counter()
    valid_global, global_to_compact, response, design, residual = compact_real_data(
        model, beta
    )
    n_valid = int(len(valid_global))
    n_real = int(global_to_compact.size)
    requested_chunk = int(chunk_size or getattr(model, "target_chunk_size", 64))
    requested_chunk = max(1, requested_chunk)
    tolerance = float(coefficient_drop_tolerance)
    if tolerance < 0.0:
        raise ValueError("coefficient_drop_tolerance must be nonnegative")

    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    conditional_logdet_half = 0.0
    target_seen = np.zeros(n_valid, dtype=np.int16)
    n_block_rows = 0
    n_chunks = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(model._cluster_batches):
            c = int(batch.max_cond_points)
            t = int(batch.target_size)
            total_batch_rows = int(batch.X.shape[0])
            for start in range(0, total_batch_rows, requested_chunk):
                end = min(start + requested_chunk, total_batch_rows)
                covariance = model.matern_cov_batched(params, batch.X[start:end])
                conditioning, target, l_conditional = _block_whitening_coefficients(
                    covariance, c, t
                )
                coefficients = torch.cat([conditioning, target], dim=2)
                conditional_logdet_half += float(
                    torch.log(
                        torch.diagonal(l_conditional, dim1=-2, dim2=-1)
                    ).sum().detach().cpu().item()
                )

                global_indices = batch.T[start:end, : c + t].detach().cpu().numpy()
                compact_columns = np.full_like(global_indices, -1, dtype=np.int64)
                real_mask = global_indices < n_real
                compact_columns[real_mask] = global_to_compact[global_indices[real_mask]]
                target_global = global_indices[:, c : c + t]
                target_rows = global_to_compact[target_global]
                if np.any(target_rows < 0):
                    raise RuntimeError("A target row was not mapped to the valid compact index")
                np.add.at(target_seen, target_rows.reshape(-1), 1)

                coefficient_np = coefficients.detach().cpu().numpy()
                row_cube = np.broadcast_to(
                    target_rows[:, :, None], coefficient_np.shape
                )
                column_cube = np.broadcast_to(
                    compact_columns[:, None, :], coefficient_np.shape
                )
                keep = column_cube >= 0
                if tolerance > 0.0:
                    keep &= np.abs(coefficient_np) > tolerance
                else:
                    keep &= coefficient_np != 0.0
                row_chunks.append(np.asarray(row_cube[keep], dtype=np.int64))
                col_chunks.append(np.asarray(column_cube[keep], dtype=np.int64))
                data_chunks.append(np.asarray(coefficient_np[keep], dtype=np.float64))
                n_block_rows += end - start
                n_chunks += 1
            progress(
                f"batch {batch_index + 1}/{len(model._cluster_batches)}: "
                f"blocks={total_batch_rows}, c={c}, t={t}"
            )

    if np.any(target_seen != 1):
        values, counts = np.unique(target_seen, return_counts=True)
        raise RuntimeError(
            "Every valid observation must occur as one target; "
            f"coverage={dict(zip(values.tolist(), counts.tolist()))}"
        )
    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    values = np.concatenate(data_chunks)
    whitener = scipy.sparse.coo_matrix(
        (values, (rows, cols)), shape=(n_valid, n_valid), dtype=np.float64
    ).tocsr()
    whitener.sum_duplicates()
    whitener.sort_indices()
    nnz_by_row = np.diff(whitener.indptr)
    elapsed = time.perf_counter() - started
    metadata = {
        "n_valid": n_valid,
        "n_real_with_missing": n_real,
        "n_block_targets": int(n_block_rows),
        "n_build_chunks": int(n_chunks),
        "shape": list(whitener.shape),
        "nnz": int(whitener.nnz),
        "density": float(whitener.nnz / (n_valid * n_valid)),
        "nnz_per_row_mean": float(nnz_by_row.mean()),
        "nnz_per_row_median": float(np.median(nnz_by_row)),
        "nnz_per_row_max": int(nnz_by_row.max()),
        "csr_storage_bytes": int(
            whitener.data.nbytes + whitener.indices.nbytes + whitener.indptr.nbytes
        ),
        "coefficient_drop_tolerance": tolerance,
        "conditional_logdet_half": conditional_logdet_half,
        "build_s": elapsed,
    }
    return SparseVecchiaPrecision(
        whitener=whitener,
        valid_global_indices=valid_global,
        global_to_compact=global_to_compact,
        residual=residual,
        design=design,
        response=response,
        metadata=metadata,
    )


def native_gls_quadratic(
    model: Any,
    params: torch.Tensor,
    beta: np.ndarray,
) -> float:
    """Evaluate the exact quadratic accumulated by the native likelihood."""
    beta_tensor = torch.as_tensor(beta, dtype=params.dtype, device=params.device).reshape(-1, 1)
    with torch.no_grad():
        xt_sinv_x, xt_sinv_y, yt_sinv_y, _, _ = model._accumulate_gls_stats(
            params, include_y_quad=True, catch_cholesky=False
        )
        value = (
            yt_sinv_y
            - 2.0 * (beta_tensor.T @ xt_sinv_y).squeeze()
            + (beta_tensor.T @ xt_sinv_x @ beta_tensor).squeeze()
        )
    return float(value.detach().cpu().item())


def verify_precision_identity(
    precision: SparseVecchiaPrecision,
    native_quadratic: float,
) -> dict[str, float]:
    innovations = precision.whitener @ precision.residual
    sparse_quadratic = float(innovations @ innovations)
    relative_error = float(
        abs(sparse_quadratic - float(native_quadratic))
        / max(abs(float(native_quadratic)), 1e-15)
    )
    return {
        "native_quadratic": float(native_quadratic),
        "sparse_quadratic": sparse_quadratic,
        "relative_error": relative_error,
        "innovation_mean": float(np.mean(innovations)),
        "innovation_variance": float(np.mean(innovations**2)),
        "innovation_max_abs": float(np.max(np.abs(innovations))),
    }


__all__ = [
    "SparseVecchiaPrecision",
    "build_sparse_vecchia_precision",
    "native_gls_quadratic",
    "verify_precision_identity",
]
