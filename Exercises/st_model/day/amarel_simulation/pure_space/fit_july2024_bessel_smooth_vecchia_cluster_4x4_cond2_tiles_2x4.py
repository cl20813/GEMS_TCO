#!/usr/bin/env python3
"""July 2024 tile-wise anisotropic Matern fits with cluster Vecchia.

This is the Vecchia counterpart to the full-likelihood 2x4 tile experiment.
Each hour is split into 2x4 tiles; inside each tile, targets are 4x4 grid-cell
clusters and each cluster conditions on two previous max-min cluster blocks.

The covariance is anisotropic pure-space Matern with estimated smoothness and
the same phi reparameterization used in the space-time code.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, solve_triangular

try:
    from GEMS_TCO.kernels_space_iso_cluster_052426 import ClusterSpaceIsoTrendVecchiaFit
    from GEMS_TCO.matern_bessel_anisotropic import (
        covariance_from_deltas,
        fit_vecchia_matern_from_batches,
        natural_from_raw,
        vecchia_batches_to_numpy,
    )
except ImportError:
    _candidates = [
        Path(__file__).parents[5] / "src",
        Path("/home/jl2815/tco"),
    ]
    for _p in _candidates:
        if (_p / "GEMS_TCO").is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
            break
    from GEMS_TCO.kernels_space_iso_cluster_052426 import ClusterSpaceIsoTrendVecchiaFit
    from GEMS_TCO.matern_bessel_anisotropic import (
        covariance_from_deltas,
        fit_vecchia_matern_from_batches,
        natural_from_raw,
        vecchia_batches_to_numpy,
    )

from fit_july2024_bessel_smooth_full_likelihood_tiles_2x4 import (
    assign_tiles,
    choose_hour,
    deterministic_subset,
    make_manifest,
    monthly_output_dir,
    output_root,
    parse_float_pair,
    plot_monthly_tile_parameter_maps,
    prepare_hour_data,
    read_hour_table,
    read_manifest,
    resolve_manifest,
    round_numeric,
    tile_geometry,
)


METHOD = "vecc_cluster_4x4_cond2"


def parse_block_shape(text: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in str(text).lower().replace("x", ",").split(",") if x.strip()]
    if len(vals) != 2 or vals[0] <= 0 or vals[1] <= 0:
        raise argparse.ArgumentTypeError("block shape must look like 4x4")
    return vals[0], vals[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="July 2024 2x4 tile Bessel-smooth 4x4-cond2 Vecchia fits.")
    p.add_argument("--mode", choices=["manifest", "fit", "summarize", "all"], required=True)
    p.add_argument("--input", default=os.environ.get("DATA_PATH"))
    p.add_argument("--output-dir", default=os.environ.get(
        "OUTDIR",
        "/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/vecchia_cluster_4x4_cond2",
    ))
    p.add_argument("--monthly-output-dir", default=os.environ.get("MONTHLY_OUTDIR", ""))
    p.add_argument("--manifest", default=None)
    p.add_argument("--month", default=os.environ.get("MONTH", "2024-07"))
    p.add_argument("--max-hours", type=int, default=int(os.environ.get("MAX_HOURS", "240")))
    p.add_argument("--expected-hours", type=int, default=int(os.environ.get("EXPECTED_HOURS", "240")))

    p.add_argument("--time-col", default=os.environ.get("TIME_COL", "auto"))
    p.add_argument("--x-col", default=os.environ.get("X_COL", "auto"))
    p.add_argument("--y-col", default=os.environ.get("Y_COL", "auto"))
    p.add_argument("--value-col", default=os.environ.get("VALUE_COL", "auto"))
    p.add_argument("--qa-col", default=os.environ.get("QA_COL", ""))
    p.add_argument("--qa-min", type=float, default=None)

    p.add_argument("--coords", choices=["raw", "lonlat"], default=os.environ.get("COORDS", "raw"))
    p.add_argument("--lat-range", type=parse_float_pair, default=parse_float_pair(os.environ.get("LAT_RANGE", "-3,2")))
    p.add_argument("--lon-range", type=parse_float_pair, default=parse_float_pair(os.environ.get("LON_RANGE", "121,131")))
    p.add_argument("--tile-y", type=int, default=int(os.environ.get("TILE_Y", "2")))
    p.add_argument("--tile-x", type=int, default=int(os.environ.get("TILE_X", "4")))
    p.add_argument("--min-tile-points", type=int, default=int(os.environ.get("MIN_TILE_POINTS", "200")))
    p.add_argument("--tile-max-points", type=int, default=int(os.environ.get("TILE_MAX_POINTS", "0")))
    p.add_argument("--tile-workers", type=int, default=int(os.environ.get("TILE_WORKERS", "1")))
    p.add_argument("--sample-seed", type=int, default=int(os.environ.get("SAMPLE_SEED", "202407")))

    p.add_argument("--cluster-block-shape", type=parse_block_shape, default=parse_block_shape(os.environ.get("CLUSTER_BLOCK_SHAPE", "4x4")))
    p.add_argument("--cluster-neighbor-blocks", type=int, default=int(os.environ.get("CLUSTER_NEIGHBOR_BLOCKS", "2")))
    p.add_argument("--target-chunk-size", type=int, default=int(os.environ.get("TARGET_CHUNK_SIZE", "128")))
    p.add_argument("--min-target-points", type=int, default=int(os.environ.get("MIN_TARGET_POINTS", "1")))

    p.add_argument("--nugget-mode", choices=["free", "fixed0"], default=os.environ.get("NUGGET_MODE", "free"))
    p.add_argument("--mean-design", choices=["lat", "latlon"], default=os.environ.get("MEAN_DESIGN", "lat"))
    p.add_argument("--range-lat-init", type=float, default=float(os.environ.get("RANGE_LAT_INIT", "0.35")))
    p.add_argument("--range-lon-init", type=float, default=float(os.environ.get("RANGE_LON_INIT", "0.35")))
    p.add_argument("--smooth-init", type=float, default=float(os.environ.get("SMOOTH_INIT", "0.5")))
    p.add_argument("--nugget-init", type=float, default=None)
    p.add_argument("--smooth-min", type=float, default=float(os.environ.get("SMOOTH_MIN", "0.05")))
    p.add_argument("--smooth-max", type=float, default=float(os.environ.get("SMOOTH_MAX", "2.5")))
    p.add_argument("--range-min", type=float, default=float(os.environ.get("RANGE_MIN", "0.03")))
    p.add_argument("--range-max", type=float, default=float(os.environ.get("RANGE_MAX", "5.0")))
    p.add_argument("--jitter", type=float, default=float(os.environ.get("JITTER", "1e-6")))
    p.add_argument("--n-restarts", type=int, default=int(os.environ.get("N_RESTARTS", "1")))
    p.add_argument("--maxiter", type=int, default=int(os.environ.get("MAXITER", "80")))
    p.add_argument("--maxfun", type=int, default=int(os.environ.get("MAXFUN", "0")))
    p.add_argument("--maxls", type=int, default=int(os.environ.get("MAXLS", "20")))
    p.add_argument("--maxcor", type=int, default=int(os.environ.get("MAXCOR", "20")))
    p.add_argument("--optimizer-method", default=os.environ.get("OPTIMIZER_METHOD", "L-BFGS-B"))
    p.add_argument(
        "--outlier-whitened-threshold",
        type=float,
        default=float(os.environ.get("OUTLIER_WHITENED_THRESHOLD", "10")),
        help="If >0, fit once, mark |Vecchia whitened residual| above this value as missing, then refit the tile.",
    )

    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--hour", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _append_message(existing: object, addition: str) -> str:
    msg = "" if existing is None or (isinstance(existing, float) and np.isnan(existing)) else str(existing)
    return addition if not msg else f"{msg}; {addition}"


def _fit_has_usable_qc_params(fit: dict) -> bool:
    raw = fit.get("raw_params")
    if raw is None:
        return False
    try:
        raw_arr = np.asarray(raw, dtype=float)
        loss = float(fit.get("loss", fit.get("nll", np.nan)))
    except (TypeError, ValueError):
        return False
    return raw_arr.size > 0 and bool(np.all(np.isfinite(raw_arr))) and bool(np.isfinite(loss))


def _build_vecchia_tile_model(y: np.ndarray, coords: np.ndarray, task: dict):
    data = np.zeros((len(y), 4), dtype=np.float64)
    data[:, 0] = coords[:, 0]
    data[:, 1] = coords[:, 1]
    data[:, 2] = y
    tensor = torch.from_numpy(data).to(dtype=torch.float64)
    model = ClusterSpaceIsoTrendVecchiaFit(
        smooth=0.5,
        input_map={"t0": tensor},
        grid_coords=np.asarray(coords, dtype=np.float64),
        block_shape=tuple(task["cluster_block_shape"]),
        n_neighbor_blocks=int(task["cluster_neighbor_blocks"]),
        target_chunk_size=int(task["target_chunk_size"]),
        min_target_points=int(task["min_target_points"]),
        mean_design=str(task["mean_design"]),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.precompute_conditioning_sets()
    return model, vecchia_batches_to_numpy(model)


def _fit_vecchia_tile_once(y: np.ndarray, coords: np.ndarray, task: dict):
    model, batches = _build_vecchia_tile_model(y, coords, task)
    fit = fit_vecchia_matern_from_batches(
        batches=batches,
        n_features=int(model.n_features),
        y_var=max(float(np.nanvar(y, ddof=1)), 1e-8),
        nugget_mode=str(task["nugget_mode"]),
        fixed_nugget=0.0,
        smooth_bounds=tuple(task["smooth_bounds"]),
        range_bounds=tuple(task["range_bounds"]),
        range_lat_init=float(task["range_lat_init"]),
        range_lon_init=float(task["range_lon_init"]),
        smooth_init=float(task["smooth_init"]),
        nugget_init=task["nugget_init"],
        jitter=float(task["jitter"]),
        n_restarts=int(task["n_restarts"]),
        maxiter=int(task["maxiter"]),
        maxfun=int(task["maxfun"]),
        maxls=int(task["maxls"]),
        maxcor=int(task["maxcor"]),
        method=str(task["optimizer_method"]),
    )
    return fit, model, batches


def vecchia_whitened_residuals_by_obs(
    model,
    batches: list[dict],
    fit: dict,
    task: dict,
    n_obs: int,
) -> np.ndarray:
    """Map target-row Vecchia whitened residuals back to the tile observation order."""
    raw = fit.get("raw_params")
    if raw is None:
        raise ValueError("fit record has no raw_params")
    params = natural_from_raw(raw, str(task["nugget_mode"]), 0.0, tuple(task["smooth_bounds"]))
    n_features = int(model.n_features)
    xt_sinv_x = np.zeros((n_features, n_features), dtype=np.float64)
    xt_sinv_y = np.zeros((n_features, 1), dtype=np.float64)

    for batch in batches:
        coords_all = batch["coords"]
        X_all = batch["X"]
        y_all = batch["y"]
        d_lat_all = batch.get("d_lat")
        d_lon_all = batch.get("d_lon")
        m = int(batch["max_cond_points"])
        t = int(batch["target_size"])
        target = slice(m, m + t)
        for i in range(coords_all.shape[0]):
            K = covariance_from_deltas(d_lat_all[i], d_lon_all[i], params, jitter=float(task["jitter"]))
            c, lower = cho_factor(K, lower=True, check_finite=False)
            z_X = solve_triangular(c, X_all[i], lower=lower, check_finite=False)
            z_y = solve_triangular(c, y_all[i], lower=lower, check_finite=False)
            u_X = z_X[target, :]
            u_y = z_y[target, :]
            xt_sinv_x += u_X.T @ u_X
            xt_sinv_y += u_X.T @ u_y

    beta = np.linalg.solve(xt_sinv_x + np.eye(n_features) * 1e-8, xt_sinv_y)
    out = np.full(int(n_obs), np.nan, dtype=np.float64)
    cluster_batches = getattr(model, "_cluster_batches", [])
    for batch_np, batch_obj in zip(batches, cluster_batches):
        coords_all = batch_np["coords"]
        X_all = batch_np["X"]
        y_all = batch_np["y"]
        d_lat_all = batch_np.get("d_lat")
        d_lon_all = batch_np.get("d_lon")
        rows_all = batch_obj.rows.detach().cpu().numpy()
        m = int(batch_np["max_cond_points"])
        t = int(batch_np["target_size"])
        target = slice(m, m + t)
        for i in range(coords_all.shape[0]):
            K = covariance_from_deltas(d_lat_all[i], d_lon_all[i], params, jitter=float(task["jitter"]))
            c, lower = cho_factor(K, lower=True, check_finite=False)
            z_X = solve_triangular(c, X_all[i], lower=lower, check_finite=False)
            z_y = solve_triangular(c, y_all[i], lower=lower, check_finite=False)
            resid = (z_y[target, :] - z_X[target, :] @ beta).reshape(-1)
            obs_idx = rows_all[i, target].reshape(-1)
            real = obs_idx < int(n_obs)
            out[obs_idx[real].astype(int)] = resid[real]
    return out


def fit_vecchia_tile_with_outlier_qc(y: np.ndarray, coords: np.ndarray, task: dict):
    threshold = float(task.get("outlier_whitened_threshold", 0.0))
    initial_fit, initial_model, initial_batches = _fit_vecchia_tile_once(y, coords, task)
    finite_initial = np.isfinite(y)
    qc = {
        "outlier_whitened_threshold": threshold,
        "n_qc_initial_fit": int(finite_initial.sum()),
        "n_qc_removed": 0,
        "n_qc_fit": int(finite_initial.sum()),
        "qc_max_abs_whitened": np.nan,
        "qc_refit": False,
        "qc_initial_success": bool(initial_fit.get("success", False)),
    }
    if threshold <= 0.0 or not _fit_has_usable_qc_params(initial_fit):
        return initial_fit, initial_model, qc

    try:
        w = vecchia_whitened_residuals_by_obs(initial_model, initial_batches, initial_fit, task, n_obs=len(y))
    except Exception as exc:
        initial_fit["message"] = _append_message(
            initial_fit.get("message", ""),
            f"outlier_qc_skipped_whitening_error {exc}",
        )
        return initial_fit, initial_model, qc
    abs_w = np.abs(w)
    bad = np.isfinite(abs_w) & (abs_w > threshold)
    qc["qc_max_abs_whitened"] = float(np.nanmax(abs_w)) if np.isfinite(abs_w).any() else np.nan
    qc["n_qc_removed"] = int(bad.sum())
    y_qc = y.copy()
    y_qc[bad] = np.nan
    qc["n_qc_fit"] = int(np.isfinite(y_qc).sum())
    if int(bad.sum()) == 0:
        return initial_fit, initial_model, qc
    min_refit_points = max(10, int(initial_model.n_features) + 2)
    if int(np.isfinite(y_qc).sum()) < min_refit_points:
        initial_fit["message"] = _append_message(
            initial_fit.get("message", ""),
            f"outlier_qc_skipped_too_few_after_removal {int(finite_initial.sum())}->{int(np.isfinite(y_qc).sum())}",
        )
        qc["n_qc_fit"] = int(finite_initial.sum())
        return initial_fit, initial_model, qc

    refit, refit_model, _ = _fit_vecchia_tile_once(y_qc, coords, task)
    refit["message"] = _append_message(
        refit.get("message", ""),
        f"whitened_outlier_qc |r|>{threshold:g} removed {int(bad.sum())}/{int(finite_initial.sum())}",
    )
    refit["pre_qc_loss"] = initial_fit.get("loss", np.nan)
    refit["pre_qc_nll"] = initial_fit.get("nll", np.nan)
    qc["qc_refit"] = True
    return refit, refit_model, qc


def _fit_tile_worker(task: dict) -> dict:
    base = dict(task["base"])
    y_all = np.asarray(task["y"], dtype=np.float64)
    coords_all = np.asarray(task["coords"], dtype=np.float64)
    n_tile = int(y_all.shape[0])
    if n_tile < int(task["min_tile_points"]):
        base.update({
            "n": n_tile,
            "n_fit": 0,
            "success": False,
            "message": "too_few_points",
            "loss": np.nan,
            "nll": np.nan,
            "fit_method": METHOD,
        })
        return base
    keep = deterministic_subset(n_tile, int(task["tile_max_points"]))
    y = y_all[keep]
    coords = coords_all[keep]
    try:
        fit, model, qc = fit_vecchia_tile_with_outlier_qc(y, coords, task)
        base.update(fit)
        base.update({
            "n": n_tile,
            "n_fit": int(qc.get("n_qc_fit", int(np.isfinite(y).sum()))),
            "n_initial_fit": int(np.isfinite(y).sum()),
            "fit_method": METHOD,
            "cluster_block_shape": f"{task['cluster_block_shape'][0]}x{task['cluster_block_shape'][1]}",
            "cluster_neighbor_blocks": int(task["cluster_neighbor_blocks"]),
            **qc,
        })
        base.update(model.cluster_summary())
        if int(task["tile_max_points"]) > 0 and n_tile > int(task["tile_max_points"]):
            base["message"] = f"{base.get('message', '')}; deterministic_thin {n_tile}->{len(y)}"
        return base
    except Exception as exc:
        base.update({
            "n": n_tile,
            "n_fit": int(len(y)),
            "success": False,
            "message": f"ERROR: {exc}",
            "loss": np.nan,
            "nll": np.nan,
            "fit_method": METHOD,
        })
        return base


def fit_tiles_for_hour(coords: np.ndarray, values: np.ndarray, tile_id: np.ndarray, tile_meta: dict, args: argparse.Namespace) -> pd.DataFrame:
    tasks = []
    tile_count = int(args.tile_y) * int(args.tile_x)
    for tid in range(tile_count):
        mask = tile_id == tid
        tasks.append({
            "base": tile_geometry(tid, tile_meta, int(args.tile_x)),
            "y": values[mask],
            "coords": coords[mask],
            "min_tile_points": int(args.min_tile_points),
            "tile_max_points": int(args.tile_max_points),
            "cluster_block_shape": tuple(args.cluster_block_shape),
            "cluster_neighbor_blocks": int(args.cluster_neighbor_blocks),
            "target_chunk_size": int(args.target_chunk_size),
            "min_target_points": int(args.min_target_points),
            "nugget_mode": str(args.nugget_mode),
            "mean_design": str(args.mean_design),
            "smooth_bounds": (float(args.smooth_min), float(args.smooth_max)),
            "range_bounds": (float(args.range_min), float(args.range_max)),
            "range_lat_init": float(args.range_lat_init),
            "range_lon_init": float(args.range_lon_init),
            "smooth_init": float(args.smooth_init),
            "nugget_init": args.nugget_init,
            "jitter": float(args.jitter),
            "n_restarts": int(args.n_restarts),
            "maxiter": int(args.maxiter),
            "maxfun": int(args.maxfun),
            "maxls": int(args.maxls),
            "maxcor": int(args.maxcor),
            "optimizer_method": str(args.optimizer_method),
            "outlier_whitened_threshold": float(args.outlier_whitened_threshold),
        })
    workers = max(1, int(args.tile_workers))
    if workers == 1:
        rows = [_fit_tile_worker(t) for t in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_fit_tile_worker, t) for t in tasks]
            for fut in as_completed(futures):
                rows.append(fut.result())
    return pd.DataFrame(rows).sort_values(["tile_y", "tile_x"]).reset_index(drop=True)


def fit_one_hour(args: argparse.Namespace) -> None:
    manifest = read_manifest(args)
    hour_index, hour, row = choose_hour(args, manifest)
    mode_dir = output_root(args)
    hourly_dir = mode_dir / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"h{hour_index:03d}_{hour.strftime('%Y%m%dT%H%MZ')}_{METHOD}_{args.nugget_mode}"
    out_csv = hourly_dir / f"{stem}_tiles.csv"
    out_json = hourly_dir / f"{stem}_meta.json"
    if out_csv.exists() and out_json.exists() and not args.overwrite:
        print(f"Skipping existing {out_csv}")
        return

    df_hour = read_hour_table(args, row)
    coords, values, _, meta = prepare_hour_data(df_hour, args)
    tile_id, tile_meta = assign_tiles(coords, int(args.tile_y), int(args.tile_x))
    tile_df = fit_tiles_for_hour(coords, values, tile_id, tile_meta, args)
    tile_df.insert(0, "hour_index", hour_index)
    tile_df.insert(1, "day_index", int(row["day_index"]))
    tile_df.insert(2, "hour_slot", int(row["hour_slot"]))
    tile_df.insert(3, "hour_utc", int(row["hour_utc"]))
    tile_df.insert(4, "hour", hour.strftime("%Y-%m-%dT%H:%M:%SZ"))
    tile_df.insert(5, "month", args.month)
    tile_df.insert(6, "method", METHOD)
    tile_df.insert(7, "nugget_mode", str(args.nugget_mode))
    round_numeric(tile_df, 6).to_csv(out_csv, index=False, float_format="%.6f")
    with out_json.open("w") as f:
        json.dump({
            "hour_index": hour_index,
            "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": METHOD,
            "nugget_mode": args.nugget_mode,
            "tile_shape": [int(args.tile_y), int(args.tile_x)],
            "cluster_block_shape": list(args.cluster_block_shape),
            "cluster_neighbor_blocks": int(args.cluster_neighbor_blocks),
            "tile_meta": tile_meta,
            "data_meta": meta,
            "args": vars(args),
        }, f, indent=2)
    print(f"Wrote {out_csv}")


def read_hourly_outputs(args: argparse.Namespace) -> pd.DataFrame:
    hourly_dir = output_root(args) / "hourly"
    files = sorted(hourly_dir.glob("*_tiles.csv"))
    if not files:
        raise SystemExit(f"No hourly tile files under {hourly_dir}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def summarize(args: argparse.Namespace) -> None:
    df = read_hourly_outputs(args)
    mode_dir = output_root(args)
    summary_dir = mode_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir = monthly_output_dir(args)
    monthly_dir.mkdir(parents=True, exist_ok=True)
    ym = str(args.month).replace("-", "")
    tag = f"{ym}_{METHOD}_{args.nugget_mode}"

    hourly_path = summary_dir / f"{tag}_hourly_tile_fits.csv"
    round_numeric(df, 6).to_csv(hourly_path, index=False, float_format="%.6f")

    d = df[df["success"].astype(str).str.lower().isin(["true", "1"])].copy()
    keys = ["method", "nugget_mode", "tile_y", "tile_x", "tile_id"]
    value_cols = [
        "sigmasq", "sigma", "range_lat", "range_lon", "smooth", "nugget",
        "phi1", "phi2", "phi3", "loss", "nll", "n", "n_fit",
        "n_initial_fit", "n_qc_removed", "qc_max_abs_whitened",
        "n_target_blocks", "n_target_points",
    ]
    agg = {f"{c}_mean": (c, "mean") for c in value_cols if c in d.columns}
    agg.update({f"{c}_median": (c, "median") for c in ["sigmasq", "range_lat", "range_lon", "smooth", "nugget"] if c in d.columns})
    agg.update({f"{c}_sd": (c, "std") for c in ["sigmasq", "range_lat", "range_lon", "smooth", "nugget"] if c in d.columns})
    summary = d.groupby(keys, as_index=False).agg(
        n_hours=("hour_index", "nunique"),
        tile_center_lat=("tile_center_lat", "mean"),
        tile_center_lon=("tile_center_lon", "mean"),
        tile_lat_min=("tile_lat_min", "mean"),
        tile_lat_max=("tile_lat_max", "mean"),
        tile_lon_min=("tile_lon_min", "mean"),
        tile_lon_max=("tile_lon_max", "mean"),
        **agg,
    )
    summary_path = summary_dir / f"{tag}_tile_monthly_summary.csv"
    monthly_path = monthly_dir / f"{tag}_tile_monthly_summary.csv"
    round_numeric(summary, 6).to_csv(summary_path, index=False, float_format="%.6f")
    round_numeric(summary, 6).to_csv(monthly_path, index=False, float_format="%.6f")
    plot_paths = plot_monthly_tile_parameter_maps(summary, summary_dir, monthly_dir, tag)
    print(f"Wrote {hourly_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {monthly_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


def run_all(args: argparse.Namespace) -> None:
    make_manifest(args)
    manifest = read_manifest(args)
    for idx in range(len(manifest)):
        args.array_index = idx
        fit_one_hour(args)
    summarize(args)


def main() -> None:
    args = parse_args()
    if int(args.tile_y) != 2 or int(args.tile_x) != 4:
        print(f"WARNING: expected 2x4 tiles, got {args.tile_y}x{args.tile_x}")
    if tuple(args.cluster_block_shape) != (4, 4) or int(args.cluster_neighbor_blocks) != 2:
        print(
            "WARNING: requested Vecchia design is cluster-block 4x4 and "
            f"neighbor blocks 2, got {args.cluster_block_shape}, {args.cluster_neighbor_blocks}"
        )
    if args.mode == "manifest":
        make_manifest(args)
    elif args.mode == "fit":
        fit_one_hour(args)
    elif args.mode == "summarize":
        summarize(args)
    elif args.mode == "all":
        run_all(args)


if __name__ == "__main__":
    main()
