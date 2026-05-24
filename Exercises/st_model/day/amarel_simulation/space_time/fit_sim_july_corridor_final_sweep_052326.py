#!/usr/bin/env python3
"""
Final cluster Vecchia coverage sweep for the July ST simulation assets.

Created 2026-05-23.

This script does not generate new simulations.  It reads the real-location-like
July ST simulation assets created by:

  simulate_data/generate_july_st_circulant_real_locations_2022_2025.py

Question being tested:
  The previous cluster sweeps suggested that 4x4 offset-taper conditioning is
  useful, but a single shifted center can be too literal.  The more robust
  principle is:

      choose lagged conditioning blocks where s - v * delta_t is near zero

  In cluster form, that means covering a corridor of plausible advected
  locations rather than only snapping to one offset point.

The active candidate set is focused on eight models:

  1. corridor_width_4x4_lag643
     The updated corridor anchor rule.  For each lag, choose only enough corridor
     anchors to cover the plausible displacement interval given the 4x4
     longitude width, then fill remaining blocks around the corridor midpoint.

  2. half_step2_4x4_lag653
     The current best simulation baseline: t-1 offset = 0.5x |advec_lon| and
     t-2 offset = 1.0x |advec_lon|.

  3. exact_step2_4x4_lag653
     Simulation-only reference using the known one-step truth: t-1 offset =
     1.0x |advec_lon| and t-2 offset = 2.0x |advec_lon|.

  4--7. corridor_width lightweight variants
     Same updated corridor rule with smaller budgets: 6/4/2, 6/4/1, 5/4/2,
     and 4/3/2.

  8. corridor_wide_4x4_lag643
     A more conservative robust corridor: t-1 covers 0.2x--2.0x and t-2 covers
     0x--3.0x of the reference one-step displacement.

Sign convention:
  The simulation truth has negative advec_lon, but the existing hybrid Vecchia
  shifted-neighbor convention uses a positive longitude offset to look upstream
  in the lagged field.  Therefore this script uses abs(advec_lon) for the
  corridor and point-offset magnitudes.

Outputs:
  One compact row per fit in all_corridor_final_fits_summary.csv.  Truth and
  run config are saved once.  Running summaries include p90-p10, RMSRE, and
  median absolute error.
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
    round_df,
    save_csv_rounded,
    set_seed,
    template_diagnostics,
    true_to_log_params,
)
from fit_sim_july_offset_taper_geometry_sweep_052226 import (
    append_row_csv,
    choose_asset,
    existing_completed,
    make_model_summary,
    make_param_summary,
)


ROUND_DECIMALS = 4

ROW_COLUMNS = [
    "fit_id",
    "sim_id",
    "seed",
    "asset_year",
    "asset_day_idx",
    "asset_first_key",
    "data_kind",
    "spec_name",
    "conditioning_mode",
    "block_design",
    "block_shape",
    "block_row_offset",
    "block_col_offset",
    "lag_pattern",
    "lag2_center_mode",
    "lag0_blocks",
    "lag1_blocks",
    "lag2_blocks",
    "reference_advec_lon_abs",
    "lag1_lon_offset",
    "lag2_lon_offset",
    "lag1_lon_interval_lo",
    "lag1_lon_interval_hi",
    "lag2_lon_interval_lo",
    "lag2_lon_interval_hi",
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


def code_float(x: float) -> str:
    text = f"{float(x):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def make_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    v = float(abs(args.reference_advec_lon_abs))
    half = 0.5 * v
    full = 1.0 * v
    lag1_interval = (
        float(args.lag1_corridor_low_mult) * v,
        float(args.lag1_corridor_high_mult) * v,
    )
    lag2_interval = (
        float(args.lag2_corridor_low_mult) * v,
        float(args.lag2_corridor_high_mult) * v,
    )

    specs: list[dict[str, Any]] = []

    def add_corridor(
        lag_counts: tuple[int, int, int],
        anchor_mode: str,
        name_prefix: str = "corridor",
        lag1_interval_override: tuple[float, float] | None = None,
        lag2_interval_override: tuple[float, float] | None = None,
        lag2_center_mode: str = "corridor_0x_2x",
    ) -> None:
        lag0, lag1, lag2 = lag_counts
        use_lag1_interval = lag1_interval if lag1_interval_override is None else lag1_interval_override
        use_lag2_interval = lag2_interval if lag2_interval_override is None else lag2_interval_override
        if name_prefix == "corridor":
            prefix = "corridor" if anchor_mode == "budget" else f"corridor_{anchor_mode}"
        else:
            prefix = name_prefix
        specs.append(
            {
                "spec_name": (
                    f"{prefix}_4x4_lag{lag0}{lag1}{lag2}"
                    f"_v{code_float(v)}"
                    f"_t1_{code_float(use_lag1_interval[0])}_{code_float(use_lag1_interval[1])}"
                    f"_t2_{code_float(use_lag2_interval[0])}_{code_float(use_lag2_interval[1])}"
                ),
                "conditioning_mode": "corridor",
                "corridor_anchor_mode": anchor_mode,
                "block_design": "4x4_default",
                "block_shape": (4, 4),
                "block_row_offset": 0,
                "block_col_offset": 0,
                "lag_counts": lag_counts,
                "lag_pattern": f"{lag0}/{lag1}/{lag2}",
                "lag2_center_mode": lag2_center_mode,
                "lag1_lon_offset": full,
                "lag2_lon_offset": 2.0 * full,
                "lag1_lon_interval": use_lag1_interval,
                "lag2_lon_interval": use_lag2_interval,
            }
        )

    def add_point(name: str, lag_counts: tuple[int, int, int], lag1_offset: float, lag2_offset: float) -> None:
        lag0, lag1, lag2 = lag_counts
        lag2_center_mode = "reuse_lag1_half" if abs(lag2_offset - lag1_offset) < 1e-12 else "step2_full"
        specs.append(
            {
                "spec_name": (
                    f"{name}_4x4_lag{lag0}{lag1}{lag2}"
                    f"_t1_{code_float(lag1_offset)}_t2_{code_float(lag2_offset)}"
                ),
                "conditioning_mode": "point_offset",
                "corridor_anchor_mode": "none",
                "block_design": "4x4_default",
                "block_shape": (4, 4),
                "block_row_offset": 0,
                "block_col_offset": 0,
                "lag_counts": lag_counts,
                "lag_pattern": f"{lag0}/{lag1}/{lag2}",
                "lag2_center_mode": lag2_center_mode,
                "lag1_lon_offset": lag1_offset,
                "lag2_lon_offset": lag2_offset,
                "lag1_lon_interval": (0.0, 0.0),
                "lag2_lon_interval": (0.0, 0.0),
            }
        )

    # Updated width-anchor corridor rule.  The 6/4/3 row preserves its previous
    # spec name so an existing partially completed run can resume cleanly.
    add_corridor((6, 4, 3), anchor_mode="width")
    for lag_counts in [(6, 4, 2), (6, 4, 1), (5, 4, 2), (4, 3, 2)]:
        add_corridor(lag_counts, anchor_mode="width")

    # Best current point-offset baseline and simulation-only exact-truth
    # reference.  exact_step2 uses t-1=delta and t-2=2*delta.
    add_point("half_step2", (6, 5, 3), half, full)
    add_point("exact_step2", (6, 5, 3), full, 2.0 * full)

    # Robustness stress test for real-data uncertainty: broader one-step and
    # two-step corridors than the default [0.5, 1.5] / [0, 2] multipliers.
    add_corridor(
        (6, 4, 3),
        anchor_mode="width",
        name_prefix="corridor_width_wide",
        lag1_interval_override=(0.2 * v, 2.0 * v),
        lag2_interval_override=(0.0, 3.0 * v),
        lag2_center_mode="corridor_0x_3x",
    )
    return specs


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
    params = new_params(initial_vals)
    strategy = "offset_corridor_tapered" if spec["conditioning_mode"] == "corridor" else "offset_tapered"
    corridor_anchor_mode = spec.get("corridor_anchor_mode", "budget")
    if strategy != "offset_corridor_tapered":
        corridor_anchor_mode = "budget"
    model = StrategyClusterVecchiaFit(
        smooth=smooth,
        input_map=source_map,
        grid_coords=grid_coords_np,
        block_shape=spec["block_shape"],
        strategy=strategy,
        lag0_block_count=lag0,
        lag1_block_count=lag1,
        lag2_block_count=lag2,
        daily_stride=args.daily_stride,
        lag1_lon_offset=spec["lag1_lon_offset"],
        lag2_lon_offset=spec["lag2_lon_offset"],
        lag1_lon_interval=spec["lag1_lon_interval"],
        lag2_lon_interval=spec["lag2_lon_interval"],
        corridor_anchor_mode=corridor_anchor_mode,
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
        "conditioning_mode": spec["conditioning_mode"],
        "block_design": spec["block_design"],
        "block_shape": f"{spec['block_shape'][0]}x{spec['block_shape'][1]}",
        "block_row_offset": spec["block_row_offset"],
        "block_col_offset": spec["block_col_offset"],
        "lag_pattern": spec["lag_pattern"],
        "lag2_center_mode": spec["lag2_center_mode"],
        "lag0_blocks": int(lag0),
        "lag1_blocks": int(lag1),
        "lag2_blocks": int(lag2),
        "reference_advec_lon_abs": float(abs(args.reference_advec_lon_abs)),
        "lag1_lon_offset": float(spec["lag1_lon_offset"]),
        "lag2_lon_offset": float(spec["lag2_lon_offset"]),
        "lag1_lon_interval_lo": float(spec["lag1_lon_interval"][0]),
        "lag1_lon_interval_hi": float(spec["lag1_lon_interval"][1]),
        "lag2_lon_interval_lo": float(spec["lag2_lon_interval"][0]),
        "lag2_lon_interval_hi": float(spec["lag2_lon_interval"][1]),
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


def refresh_outputs(
    raw_csv: Path,
    model_csv: Path,
    param_csv: Path,
    error_csv: Path,
    txt_path: Path,
    truth: dict[str, float],
    active_spec_names: set[str] | None = None,
) -> None:
    if not raw_csv.exists():
        return
    df = pd.read_csv(raw_csv)
    if active_spec_names is not None and "spec_name" in df.columns:
        df = df[df["spec_name"].isin(active_spec_names)].copy()
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
        "Top corridor/half-offset final summary:",
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
        lines.append(round_df(param_summary[cols].head(50)).to_string(index=False))
    if not errors.empty:
        lines.extend(["", "Recent errors:"])
        lines.append(errors.tail(20).to_string(index=False))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]), flush=True)
    if not model_summary.empty:
        print(round_df(model_summary.head(15)).to_string(index=False), flush=True)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-sims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=523)
    parser.add_argument("--years", type=str, default="2022,2023,2024,2025")
    parser.add_argument("--day-idxs", type=str, default="all")
    parser.add_argument("--max-asset-days", type=int, default=0)
    parser.add_argument("--asset-sampling", type=str, default="cycle", choices=["cycle", "random"])
    parser.add_argument("--sim-data-root", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"))
    parser.add_argument("--data-kind", type=str, default="real_locations", choices=["gridded", "real_locations"])
    parser.add_argument("--lat-range", type=parse_range, default=parse_range("-3,2"))
    parser.add_argument("--lon-range", type=parse_range, default=parse_range("121,131"))
    parser.add_argument("--reference-advec-lon-abs", type=float, default=0.2)
    parser.add_argument("--lag1-corridor-low-mult", type=float, default=0.5)
    parser.add_argument("--lag1-corridor-high-mult", type=float, default=1.5)
    parser.add_argument("--lag2-corridor-low-mult", type=float, default=0.0)
    parser.add_argument("--lag2-corridor-high-mult", type=float, default=2.0)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=15)
    parser.add_argument("--lbfgs-hist", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--init-noise", type=float, default=0.25)
    parser.add_argument("--summary-every", type=int, default=5)
    parser.add_argument("--no-center-response", dest="center_response", action="store_false")
    parser.set_defaults(center_response=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jl2815/tco/exercise_output/estimates/day/corridor_final_sweep_052326"))


def main() -> None:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.require_cuda and DEVICE.type != "cuda":
        raise RuntimeError("--require-cuda was passed, but torch.cuda.is_available() is False")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / "all_corridor_final_fits_summary.csv"
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
    specs = make_specs(args)
    active_spec_names = {str(spec["spec_name"]) for spec in specs}
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
    for spec in specs:
        print("SPEC:", spec["spec_name"], flush=True)

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
                    if k
                    in [
                        "spec_name",
                        "conditioning_mode",
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
                row = {
                    **row_base,
                    "spec_name": spec.get("spec_name", ""),
                    "conditioning_mode": spec.get("conditioning_mode", ""),
                    "block_design": spec.get("block_design", ""),
                    "lag_pattern": spec.get("lag_pattern", ""),
                    "lag2_center_mode": spec.get("lag2_center_mode", ""),
                    "error": repr(exc),
                }
                print("ERROR:", row, flush=True)

            append_row_csv(raw_csv, row, ROW_COLUMNS)
            if int(args.summary_every) > 0 and fit_id % int(args.summary_every) == 0:
                refresh_outputs(raw_csv, model_csv, param_csv, error_csv, txt_path, truth, active_spec_names)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        del source_map
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    refresh_outputs(raw_csv, model_csv, param_csv, error_csv, txt_path, truth, active_spec_names)
    print("Saved outputs under:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
