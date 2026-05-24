#!/usr/bin/env python3
"""
Offset-taper cluster Vecchia geometry sweep.

Created 2026-05-22.

This script is a second-stage follow-up to
``fit_sim_july_cluster_strategy_sweep_052226.py``.  The first sweep suggested
that target-center forcing is not useful and that ``offset_tapered`` is the
most promising cluster strategy.  Here we test that conclusion more carefully.

Important cluster-geometry clarification:
  The cluster target is a fixed non-overlapping grid block, not a sliding
  "each point plus its neighbors" stencil.  For 3x3 this is easy to picture as
  nine nearby grid cells.  For 4x4 there is no single center cell, so the
  answer can depend on how the 4-by-4 partition is anchored.  This sweep
  therefore includes longitude-shifted 4x4 block anchors and a 3x5
  longitude-wide block as alternatives.

Common temporal logic in every fit:
  t   : target-center previous max-min-ordered neighbor clusters.
  t-1 : clusters around target_lon + 0.126.
  t-2 : clusters around target_lon + 0.252, except explicit lag2_reuse_lag1
        tests where t-2 also uses target_lon + 0.126.

No target-center cluster is forced at t-1/t-2 in this sweep.

Specs tested by default:
  Geometry baseline with t/t-1/t-2 = 6/5/3:
    - 3x3_default
    - 4x4_default
    - 4x4_lon_right_o1
    - 4x4_lon_right_o2
    - 3x5_lon_wide

  Taper variants on 4x4_default and 4x4_lon_right_o1:
    - 6/4/3
    - 6/4/2
    - 6/3/2

  Lag-2 center variants:
    - selected 6/5/3 and 6/4/2 fits where t-2 reuses the 0.126 offset center
      instead of using the full 0.252 two-step offset.

Outputs:
  One compact row per fit in all_offset_taper_fits_summary.csv.  Truth is
  saved once in truth_params.json.  Running summaries are refreshed throughout.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
SRC = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from GEMS_TCO.vecchia_cluster import StrategyClusterVecchiaFit

from fit_sim_july_cluster_strategy_sweep_052226 import (
    DEVICE,
    DTYPE,
    P_LABELS,
    build_asset_bank,
    calculate_metrics,
    make_random_init,
    new_params,
    parse_range,
    parse_years,
    round_df,
    save_csv_rounded,
    set_seed,
    template_diagnostics,
    true_to_log_params,
)


ROUND_DECIMALS = 4
DELTA_LON_BASE = 0.063


ROW_COLUMNS = [
    "fit_id",
    "sim_id",
    "seed",
    "asset_year",
    "asset_day_idx",
    "asset_first_key",
    "data_kind",
    "spec_name",
    "block_design",
    "block_shape",
    "block_row_offset",
    "block_col_offset",
    "lag_pattern",
    "lag0_blocks",
    "lag1_blocks",
    "lag2_blocks",
    "lag2_center_mode",
    "lag1_lon_offset",
    "lag2_lon_offset",
    "n_cond_blocks_nominal",
    "n_cond_points_nominal",
    "n_valid",
    "n_grid",
    "valid_by_t",
    "loss",
    "converged",
    "fit_steps",
    "precompute_s",
    "fit_s",
    "total_s",
    "n_clusters",
    "max_points_per_cluster",
    "n_target_blocks",
    "n_target_points",
    "n_batches",
    "mean_m_by_template",
    "median_m_by_template",
    "max_m_by_template",
    "median_target_size_by_batch",
    "max_target_size_by_batch",
    "overall_rmsre",
    "spatial_rmsre",
    "advec_rmsre",
    "median_abs_error",
    "range_time_re",
    "nugget_re",
    *[f"est_{k}" for k in P_LABELS],
    *[f"abs_error_{k}" for k in P_LABELS],
    *[f"{k}_re" for k in P_LABELS],
    "error",
]


def append_row_csv(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})


def make_specs(include_lag2_reuse: bool = True) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(
        block_design: str,
        block_shape: tuple[int, int],
        lag_counts: tuple[int, int, int],
        block_row_offset: int = 0,
        block_col_offset: int = 0,
        lag2_center_mode: str = "offset252",
    ) -> None:
        lag0, lag1, lag2 = lag_counts
        spec_name = (
            f"{block_design}_lag{lag0}{lag1}{lag2}_{lag2_center_mode}"
        )
        specs.append(
            {
                "spec_name": spec_name,
                "block_design": block_design,
                "block_shape": block_shape,
                "block_row_offset": int(block_row_offset),
                "block_col_offset": int(block_col_offset),
                "lag_counts": lag_counts,
                "lag_pattern": f"{lag0}/{lag1}/{lag2}",
                "lag2_center_mode": lag2_center_mode,
            }
        )

    # Geometry baseline at the current best taper, 6/5/3.
    add("3x3_default", (3, 3), (6, 5, 3))
    add("4x4_default", (4, 4), (6, 5, 3))
    add("4x4_lon_right_o1", (4, 4), (6, 5, 3), block_col_offset=1)
    add("4x4_lon_right_o2", (4, 4), (6, 5, 3), block_col_offset=2)
    add("3x5_lon_wide", (3, 5), (6, 5, 3))

    # Taper strength variants for the most relevant 4x4 geometries.
    for design, col_offset in [("4x4_default", 0), ("4x4_lon_right_o1", 1)]:
        for lag_counts in [(6, 4, 3), (6, 4, 2), (6, 3, 2)]:
            add(design, (4, 4), lag_counts, block_col_offset=col_offset)

    if include_lag2_reuse:
        # Test whether t-2 should reuse the one-step offset center (0.126)
        # instead of moving to the two-step offset center (0.252).
        add("4x4_default", (4, 4), (6, 5, 3), lag2_center_mode="reuse_lag1_offset126")
        add("4x4_default", (4, 4), (6, 4, 2), lag2_center_mode="reuse_lag1_offset126")
        add("4x4_lon_right_o1", (4, 4), (6, 4, 2), block_col_offset=1, lag2_center_mode="reuse_lag1_offset126")

    # Preserve insertion order while removing accidental duplicates.
    seen = set()
    out = []
    for spec in specs:
        key = (
            spec["block_design"],
            spec["lag_counts"],
            spec["block_row_offset"],
            spec["block_col_offset"],
            spec["lag2_center_mode"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def fit_spec(
    source_map: dict[str, torch.Tensor],
    grid_coords_np: np.ndarray,
    initial_vals: list[float],
    spec: dict[str, Any],
    args: argparse.Namespace,
    truth: dict[str, float],
    smooth: float,
) -> dict[str, Any]:
    lag0, lag1, lag2 = spec["lag_counts"]
    lag2_offset = float(args.lag1_lon_offset) if spec["lag2_center_mode"] == "reuse_lag1_offset126" else float(args.lag2_lon_offset)
    params = new_params(initial_vals)
    model = StrategyClusterVecchiaFit(
        smooth=smooth,
        input_map=source_map,
        grid_coords=grid_coords_np,
        block_shape=spec["block_shape"],
        strategy="offset_tapered",
        lag0_block_count=lag0,
        lag1_block_count=lag1,
        lag2_block_count=lag2,
        daily_stride=args.daily_stride,
        lag1_lon_offset=args.lag1_lon_offset,
        lag2_lon_offset=lag2_offset,
        target_chunk_size=args.target_chunk_size,
        min_target_points=args.min_target_points,
        block_row_offset=spec["block_row_offset"],
        block_col_offset=spec["block_col_offset"],
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    diag = template_diagnostics(model)
    cluster_diag = model.cluster_summary()

    max_expected = int(spec["block_shape"][0]) * int(spec["block_shape"][1])
    if int(model.max_points_per_cluster) > max_expected:
        raise RuntimeError(
            f"Invalid cluster geometry: max_points_per_cluster={model.max_points_per_cluster}, "
            f"block_shape={spec['block_shape']}"
        )

    opt = model.set_optimizer(
        params,
        lr=1.0,
        max_iter=args.lbfgs_eval,
        max_eval=args.lbfgs_eval,
        history_size=args.lbfgs_hist,
    )
    t1 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=args.grad_tol)
    fit_s = time.time() - t1

    metrics, est = calculate_metrics(out, truth)
    loss = float(out[-1])
    fit_steps = int(fit_iter) + 1
    n_cond_blocks = int(lag0 + lag1 + lag2)
    row = {
        "spec_name": spec["spec_name"],
        "block_design": spec["block_design"],
        "block_shape": f"{spec['block_shape'][0]}x{spec['block_shape'][1]}",
        "block_row_offset": spec["block_row_offset"],
        "block_col_offset": spec["block_col_offset"],
        "lag_pattern": spec["lag_pattern"],
        "lag0_blocks": int(lag0),
        "lag1_blocks": int(lag1),
        "lag2_blocks": int(lag2),
        "lag2_center_mode": spec["lag2_center_mode"],
        "lag1_lon_offset": float(args.lag1_lon_offset),
        "lag2_lon_offset": float(lag2_offset),
        "n_cond_blocks_nominal": int(n_cond_blocks),
        "n_cond_points_nominal": int(n_cond_blocks * int(model.max_points_per_cluster)),
        "loss": loss,
        "converged": int(np.isfinite(loss) and fit_steps < int(args.lbfgs_steps)),
        "fit_steps": fit_steps,
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        **diag,
        **cluster_diag,
        **metrics,
        **{f"est_{k}": v for k, v in est.items()},
        "error": "",
    }
    del model, params, opt
    return row


def existing_completed(raw_csv: Path) -> set[tuple[int, str]]:
    if not raw_csv.exists():
        return set()
    df = pd.read_csv(raw_csv)
    if df.empty:
        return set()
    ok = df[df["error"].fillna("") == ""] if "error" in df.columns else df
    return {
        (int(r.sim_id), str(r.spec_name))
        for r in ok.itertuples(index=False)
        if pd.notna(r.sim_id) and pd.notna(r.spec_name)
    }


def make_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "error" not in df.columns:
        return pd.DataFrame()
    ok = df[df["error"].fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    metric_cols = [
        "loss",
        "overall_rmsre",
        "spatial_rmsre",
        "advec_rmsre",
        "median_abs_error",
        "range_time_re",
        "nugget_re",
        "precompute_s",
        "fit_s",
        "total_s",
        "n_target_points",
        "mean_m_by_template",
        "max_m_by_template",
    ]
    rows = []
    group_cols = ["spec_name", "block_design", "lag_pattern", "lag2_center_mode"]
    for keys, sub in ok.groupby(group_cols, dropna=False):
        row = {
            "spec_name": keys[0],
            "block_design": keys[1],
            "lag_pattern": keys[2],
            "lag2_center_mode": keys[3],
            "n": int(len(sub)),
            "n_cond_points_nominal_median": float(sub["n_cond_points_nominal"].median()),
            "converged_rate": float(sub["converged"].astype(float).mean()) if "converged" in sub else np.nan,
        }
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float) if col in sub else np.array([])
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_p90_p10"] = np.nan
                continue
            p10, p90 = np.percentile(vals, [10, 90])
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_p90_p10"] = float(p90 - p10)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["overall_rmsre_median", "median_abs_error_median"])


def make_param_summary(df: pd.DataFrame, truth: dict[str, float]) -> pd.DataFrame:
    if df.empty or "error" not in df.columns:
        return pd.DataFrame()
    ok = df[df["error"].fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["spec_name", "block_design", "lag_pattern", "lag2_center_mode"]
    for keys, sub in ok.groupby(group_cols, dropna=False):
        for p in P_LABELS:
            col = f"est_{p}"
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            tv = truth[p]
            abs_vals = np.abs(vals - tv)
            denom = abs(tv) if abs(tv) >= 0.01 else 1.0
            re_vals = abs_vals / denom
            p10_abs, p90_abs = np.percentile(abs_vals, [10, 90])
            p10_re, p90_re = np.percentile(re_vals, [10, 90])
            rows.append(
                {
                    "spec_name": keys[0],
                    "block_design": keys[1],
                    "lag_pattern": keys[2],
                    "lag2_center_mode": keys[3],
                    "parameter": p,
                    "true": tv,
                    "n": int(len(vals)),
                    "rmsre": float(np.sqrt(np.mean(re_vals**2))),
                    "median_abs_error": float(np.median(abs_vals)),
                    "p90_p10_abs_error": float(p90_abs - p10_abs),
                    "median_re": float(np.median(re_vals)),
                    "p90_p10_re": float(p90_re - p10_re),
                    "estimate_median": float(np.median(vals)),
                }
            )
    return pd.DataFrame(rows).sort_values(["parameter", "rmsre", "median_abs_error"])


def refresh_outputs(raw_csv: Path, model_csv: Path, param_csv: Path, error_csv: Path, txt_path: Path, truth: dict[str, float]) -> None:
    if not raw_csv.exists():
        return
    df = pd.read_csv(raw_csv)
    model_summary = make_model_summary(df)
    param_summary = make_param_summary(df, truth)
    errors = df[df["error"].fillna("") != ""] if "error" in df else pd.DataFrame()
    if not model_summary.empty:
        save_csv_rounded(model_summary, model_csv)
    if not param_summary.empty:
        save_csv_rounded(param_summary, param_csv)
    if not errors.empty:
        errors.to_csv(error_csv, index=False)

    ok_n = int((df["error"].fillna("") == "").sum()) if "error" in df else int(len(df))
    err_n = int((df["error"].fillna("") != "").sum()) if "error" in df else 0
    lines = [
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Rows: {len(df)} completed: {ok_n} errors: {err_n}",
        "",
        "Top offset-taper geometry summary:",
    ]
    if not model_summary.empty:
        cols = [
            "spec_name",
            "n",
            "overall_rmsre_median",
            "overall_rmsre_p90_p10",
            "median_abs_error_median",
            "median_abs_error_p90_p10",
            "total_s_median",
            "n_cond_points_nominal_median",
        ]
        cols = [c for c in cols if c in model_summary.columns]
        lines.append(round_df(model_summary[cols].head(25)).to_string(index=False))
    if not param_summary.empty:
        lines.extend(["", "Parameter summary preview:"])
        cols = ["spec_name", "parameter", "n", "rmsre", "median_abs_error", "p90_p10_abs_error", "median_re"]
        lines.append(round_df(param_summary[cols].head(40)).to_string(index=False))
    if not errors.empty:
        lines.extend(["", "Recent errors:"])
        lines.append(errors.tail(20).to_string(index=False))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]), flush=True)
    if not model_summary.empty:
        print(round_df(model_summary.head(15)).to_string(index=False), flush=True)


def choose_asset(asset_bank: list[dict[str, Any]], sim_id: int, args: argparse.Namespace) -> dict[str, Any]:
    if args.asset_sampling == "random":
        rng = np.random.default_rng(args.seed + 10_000 + sim_id)
        return asset_bank[int(rng.integers(0, len(asset_bank)))]
    return asset_bank[(sim_id - 1) % len(asset_bank)]


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-sims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument("--years", type=str, default="2022,2023,2024,2025")
    parser.add_argument("--day-idxs", type=str, default="all")
    parser.add_argument("--max-asset-days", type=int, default=0)
    parser.add_argument("--asset-sampling", type=str, default="cycle", choices=["cycle", "random"])
    parser.add_argument("--sim-data-root", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"))
    parser.add_argument("--data-kind", type=str, default="real_locations", choices=["gridded", "real_locations"])
    parser.add_argument("--lat-range", type=parse_range, default=parse_range("-3,2"))
    parser.add_argument("--lon-range", type=parse_range, default=parse_range("121,131"))
    parser.add_argument("--lag1-lon-offset", type=float, default=2.0 * DELTA_LON_BASE)
    parser.add_argument("--lag2-lon-offset", type=float, default=4.0 * DELTA_LON_BASE)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=15)
    parser.add_argument("--lbfgs-hist", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--init-noise", type=float, default=0.25)
    parser.add_argument("--summary-every", type=int, default=7)
    parser.add_argument("--no-center-response", dest="center_response", action="store_false")
    parser.set_defaults(center_response=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-lag2-reuse-tests", dest="include_lag2_reuse", action="store_false")
    parser.set_defaults(include_lag2_reuse=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jl2815/tco/exercise_output/estimates/day/offset_taper_geometry_sweep_052226"))


def main() -> None:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.require_cuda and DEVICE.type != "cuda":
        raise RuntimeError("--require-cuda was passed, but torch.cuda.is_available() is False")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / "all_offset_taper_fits_summary.csv"
    model_csv = args.out_dir / "running_model_summary.csv"
    param_csv = args.out_dir / "running_param_summary.csv"
    error_csv = args.out_dir / "running_errors.csv"
    txt_path = args.out_dir / "running_summary.txt"
    truth_json = args.out_dir / "truth_params.json"
    config_json = args.out_dir / "run_config.json"

    set_seed(args.seed)
    print("SRC:", SRC, flush=True)
    print("DEVICE:", DEVICE, flush=True)
    print("torch:", torch.__version__, flush=True)
    print("args:", vars(args), flush=True)

    asset_bank, truth = build_asset_bank(args)
    smooth = float(truth.get("smooth", 0.5))
    true_log = true_to_log_params(truth)
    specs = make_specs(include_lag2_reuse=args.include_lag2_reuse)
    truth_json.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    config_json.write_text(
        json.dumps(
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "src": SRC,
                "device": str(DEVICE),
                "n_assets_loaded": len(asset_bank),
                "n_specs": len(specs),
                "specs": specs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("truth:", {k: truth[k] for k in P_LABELS}, flush=True)
    print(f"Loaded {len(asset_bank)} day assets; smooth={smooth}; specs={len(specs)}", flush=True)

    done = existing_completed(raw_csv) if args.resume else set()
    fit_id = 0
    if raw_csv.exists():
        old = pd.read_csv(raw_csv)
        if not old.empty and "fit_id" in old:
            fit_ids = pd.to_numeric(old["fit_id"], errors="coerce").dropna()
            if not fit_ids.empty:
                fit_id = int(fit_ids.max())

    for sim_id in range(1, int(args.num_sims) + 1):
        print("\n" + "=" * 100, flush=True)
        print(f"Simulation {sim_id}/{args.num_sims}", flush=True)
        asset = choose_asset(asset_bank, sim_id, args)
        iter_seed = int(args.seed + sim_id)
        set_seed(iter_seed)
        rng = np.random.default_rng(iter_seed)
        initial_vals = make_random_init(rng, true_log, args.init_noise)
        source_map = {
            k: v.to(device=DEVICE, dtype=DTYPE, non_blocking=True).contiguous()
            for k, v in asset["source_map"].items()
        }
        grid_coords_np = np.asarray(asset["grid_coords_np"], dtype=np.float64)
        n_valid = sum(int(torch.isfinite(v[:, 2]).sum().item()) for v in source_map.values())
        print(
            f"asset=year{asset['year']} day_idx={asset['day_idx']} "
            f"n_grid={asset['n_grid']:,} n_valid={n_valid:,} valid={asset['valid_by_t']} "
            f"initial={[round(x, 4) for x in initial_vals]}",
            flush=True,
        )

        for spec in specs:
            key = (sim_id, spec["spec_name"])
            if args.resume and key in done:
                print(f"Skipping completed sim={sim_id} spec={spec['spec_name']}", flush=True)
                continue
            fit_id += 1
            print(f"\n--- fit_id={fit_id} spec={spec['spec_name']} ---", flush=True)
            row_base = {
                "fit_id": fit_id,
                "sim_id": sim_id,
                "seed": iter_seed,
                "asset_year": asset["year"],
                "asset_day_idx": asset["day_idx"],
                "asset_first_key": asset["day_keys"][0],
                "data_kind": args.data_kind,
                "n_valid": int(n_valid),
                "n_grid": int(asset["n_grid"]),
                "valid_by_t": json.dumps(asset["valid_by_t"], separators=(",", ":")),
            }
            try:
                row = fit_spec(source_map, grid_coords_np, initial_vals, spec, args, truth, smooth)
                row.update(row_base)
                compact = {
                    k: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
                    for k, v in row.items()
                    if k in [
                        "spec_name",
                        "loss",
                        "overall_rmsre",
                        "median_abs_error",
                        "spatial_rmsre",
                        "advec_rmsre",
                        "fit_steps",
                        "total_s",
                        "n_cond_points_nominal",
                    ]
                }
                print("RESULT:", compact, flush=True)
            except Exception as exc:
                row = {**row_base, **{k: spec.get(k, "") for k in ["spec_name", "block_design", "lag_pattern", "lag2_center_mode"]}, "error": repr(exc)}
                print("ERROR:", row, flush=True)

            append_row_csv(raw_csv, row, ROW_COLUMNS)
            if int(args.summary_every) > 0 and fit_id % int(args.summary_every) == 0:
                refresh_outputs(raw_csv, model_csv, param_csv, error_csv, txt_path, truth)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        del source_map
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    refresh_outputs(raw_csv, model_csv, param_csv, error_csv, txt_path, truth)
    print("Saved outputs under:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
