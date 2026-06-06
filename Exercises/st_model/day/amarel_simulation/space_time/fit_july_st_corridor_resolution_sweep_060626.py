#!/usr/bin/env python3
"""
ST corridor Vecchia block-center max-min density sweep.

The experiment asks how fitted ST parameters move as spatial coverage/density
increases under a block-center max-min ordering:

    first 200 -> 400 -> 600 -> 1000 target blocks

The Vecchia target block is 4x4, so a full grid with about 18,000 spatial cells
has roughly 18,000 / 16 ~= 1,100 target blocks.  We order those block centers by
max-min, then keep all grid cells inside the first K blocks.  This keeps the
low-frequency/wide-coverage intuition of max-min ordering while making the
x-axis a stable categorical variable, which fixes the broken-looking monthly
plots from the earlier irregular prefix display.

Default real-data refinement:
  - 2022: fit smooth 0.2, 0.25, 0.3, 0.35
  - 2023/2024/2025: fit smooth 0.3, 0.35

Outputs are fit CSVs plus monthly-average parameter plots only.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

AMAREL_SRC = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
SRC = AMAREL_SRC if AMAREL_SRC.exists() else LOCAL_SRC
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    SPEC_NAME as VECCHIA_SPEC_NAME,
)
from GEMS_TCO import orderings

from fit_july2024_st_corridor_density_sweep_060426 import (
    DEFAULT_REAL_INIT_PHYSICAL,
    DTYPE,
    P_LABELS,
    backmap_params,
    clean_json_value,
    count_valid,
    load_real_assets,
    load_sim_assets,
    make_params_list,
    parse_day_idxs,
    parse_float_tokens,
    parse_int_tokens,
    parse_pair,
    resolve_device,
    save_rows,
    truth_metrics,
)
from GEMS_TCO.vecchia_st_spline import RealDataCorridorWidth4x4Lag643SplineFit


ROUND_DECIMALS = 6
ALL_FITS_CSV = "st_corridor_blockmaxmin_sweep_all_fits.csv"
SUMMARY_CSV = "st_corridor_blockmaxmin_monthly_summary.csv"
MISSING_PREFIX_CSV = "st_corridor_blockmaxmin_missing_prefixes.csv"


def default_output_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path("/home/jl2815/tco/exercise_output/summer/st_corridor_blockmaxmin_refine_smooth2022_060626")
    return Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_blockmaxmin_refine_smooth2022_060626")


def code_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_year_float_map(values: list[str] | str | None) -> dict[int, list[float]]:
    if not values:
        return {}
    raw = [values] if isinstance(values, str) else list(values)
    text = ";".join(str(v) for v in raw if str(v).strip())
    out: dict[int, list[float]] = {}
    for chunk in [c.strip() for c in text.split(";") if c.strip()]:
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(f"Expected YEAR:v1,v2 entry, got {chunk!r}")
        year_text, smooth_text = chunk.split(":", 1)
        year = int(year_text.strip())
        vals = [float(v.strip()) for v in smooth_text.split(",") if v.strip()]
        if not vals:
            raise argparse.ArgumentTypeError(f"No smooth values supplied for year {year}")
        out[year] = vals
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json_value(row), sort_keys=True) + "\n")


def grid_row_col_maps(grid_coords_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    coords = np.asarray(grid_coords_np, dtype=np.float64)
    lat_key = np.round(coords[:, 0], ROUND_DECIMALS)
    lon_key = np.round(coords[:, 1], ROUND_DECIMALS)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_to_row = {v: i for i, v in enumerate(lats)}
    lon_to_col = {v: i for i, v in enumerate(lons)}
    rows = np.asarray([lat_to_row[v] for v in lat_key], dtype=np.int64)
    cols = np.asarray([lon_to_col[v] for v in lon_key], dtype=np.int64)
    return rows, cols, int(len(lats)), int(len(lons))


def blockmaxmin_indices(grid_coords_np: np.ndarray, n_blocks_requested: int, block_shape: tuple[int, int] = BLOCK_SHAPE) -> tuple[np.ndarray, dict[str, Any]]:
    coords = np.asarray(grid_coords_np, dtype=np.float64)
    rows, cols, n_lat, n_lon = grid_row_col_maps(coords)
    by_block: dict[tuple[int, int], list[int]] = {}
    block_y = int(block_shape[0])
    block_x = int(block_shape[1])
    for idx, (row, col) in enumerate(zip(rows, cols)):
        key = (int(row) // block_y, int(col) // block_x)
        by_block.setdefault(key, []).append(int(idx))
    block_keys = sorted(by_block)
    centers = []
    for key in block_keys:
        idxs = np.asarray(by_block[key], dtype=np.int64)
        center = coords[idxs].mean(axis=0)
        centers.append(center)
    centers_np = np.asarray(centers, dtype=np.float64)
    if centers_np.size == 0:
        raise ValueError("No 4x4 blocks could be built from grid_coords_np.")
    # orderings.maxmin_cpp expects lon/lat ordering.
    lon_lat = np.column_stack([centers_np[:, 1], centers_np[:, 0]])
    order = np.asarray(orderings.maxmin_cpp(lon_lat), dtype=np.int64)
    if order.size and order.min() == 1 and order.max() == len(block_keys):
        order = order - 1
    if order.size != len(block_keys):
        raise RuntimeError(f"max-min block order length {order.size} != number of blocks {len(block_keys)}")
    n_use = min(int(n_blocks_requested), int(len(block_keys)))
    selected_blocks = [block_keys[int(i)] for i in order[:n_use]]
    selected = np.concatenate([np.asarray(by_block[key], dtype=np.int64) for key in selected_blocks])
    selected = np.asarray(sorted(set(int(i) for i in selected)), dtype=np.int64)
    meta = {
        "n_lat_grid": int(n_lat),
        "n_lon_grid": int(n_lon),
        "block_shape": f"{block_y}x{block_x}",
        "n_blocks_full": int(len(block_keys)),
        "n_blocks_requested": int(n_blocks_requested),
        "n_blocks_used": int(n_use),
        "n_grid_selected": int(selected.size),
    }
    return selected, meta


def asset_for_block_prefix(asset: dict[str, Any], n_blocks_requested: int) -> dict[str, Any]:
    grid = np.asarray(asset["grid_coords_np"], dtype=np.float64)
    idx, meta = blockmaxmin_indices(grid, int(n_blocks_requested), BLOCK_SHAPE)
    source_map = {k: v[idx].contiguous() for k, v in asset["source_map"].items()}
    n_valid, n_total, valid_by_t = count_valid(source_map)
    out = {
        **asset,
        "source_map": source_map,
        "grid_coords_np": grid[idx],
        "n_grid_full": int(grid.shape[0]),
        "n_block_prefix_requested": int(n_blocks_requested),
        "n_block_prefix_used": int(meta["n_blocks_used"]),
        "block_prefix_label": f"B{int(meta['n_blocks_used'])}",
        "n_grid_block_prefix": int(idx.size),
        "n_valid_block_prefix": int(n_valid),
        "n_total_block_prefix": int(n_total),
        "valid_by_t_block_prefix": valid_by_t,
        **meta,
    }
    return out


def fit_one_resolution(
    asset: dict[str, Any],
    dataset: str,
    true_smooth: float | None,
    fit_smooth: float,
    block_prefix: int,
    init_physical: dict[str, float],
    truth: dict[str, float] | None,
    reference_advec_lon_abs: float,
    device: torch.device,
    args: argparse.Namespace,
    fit_id: int,
) -> dict[str, Any]:
    sub = asset_for_block_prefix(asset, int(block_prefix))
    source_map = {
        k: v.to(device=device, dtype=DTYPE, non_blocking=True).contiguous()
        for k, v in sub["source_map"].items()
    }
    grid_coords_np = np.asarray(sub["grid_coords_np"], dtype=np.float64)
    n_valid, n_total, valid_by_t = count_valid(source_map)
    params_list = make_params_list(init_physical, dtype=DTYPE, device=device)

    model = RealDataCorridorWidth4x4Lag643SplineFit(
        smooth=float(fit_smooth),
        input_map=source_map,
        grid_coords=grid_coords_np,
        lag1_lon_offset=float(reference_advec_lon_abs),
        daily_stride=int(args.daily_stride),
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=int(args.min_target_points),
        spline_n_points=int(args.spline_n_points),
        spline_r_max=float(args.spline_r_max),
    )

    t0 = time.time()
    model.precompute_conditioning_sets()
    precompute_s = time.time() - t0

    optimizer = model.set_optimizer(
        params_list,
        lr=float(args.lbfgs_lr),
        max_iter=int(args.lbfgs_eval),
        max_eval=int(args.lbfgs_eval),
        history_size=int(args.lbfgs_history),
    )

    t1 = time.time()
    if args.suppress_fit_prints:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            out, steps_ran = model.fit_vecc_lbfgs(
                params_list, optimizer, max_steps=int(args.lbfgs_steps), grad_tol=float(args.grad_tol)
            )
    else:
        out, steps_ran = model.fit_vecc_lbfgs(
            params_list, optimizer, max_steps=int(args.lbfgs_steps), grad_tol=float(args.grad_tol)
        )
    fit_s = time.time() - t1
    est = backmap_params(out)
    loss = float(out[-1])
    nonfinite = [k for k in P_LABELS if not np.isfinite(float(est[k]))]
    if nonfinite or not np.isfinite(loss):
        raise RuntimeError(f"Non-finite fit result: loss={loss}, nonfinite_params={nonfinite}")
    cluster_summary = model.cluster_summary()

    row = {
        "fit_id": int(fit_id),
        "status": "ok",
        "error": "",
        "dataset": str(dataset),
        "data_kind": "real" if str(dataset) == "real" else "sim",
        "true_smooth": float(true_smooth) if true_smooth is not None else np.nan,
        "fit_smooth": float(fit_smooth),
        "smooth": float(fit_smooth),
        "year": int(sub["year"]),
        "month": int(sub["month"]),
        "day_idx": int(sub["day_idx"]),
        "day": str(sub["day"]),
        "block_prefix_requested": int(block_prefix),
        "block_prefix_used": int(sub["n_block_prefix_used"]),
        "block_prefix_label": str(sub["block_prefix_label"]),
        "block_prefix_order": {200: 0, 400: 1, 600: 2, 1000: 3}.get(int(block_prefix), int(block_prefix)),
        "n_grid_full": int(sub["n_grid_full"]),
        "n_blocks_full": int(sub["n_blocks_full"]),
        "n_grid_block_prefix": int(sub["n_grid_block_prefix"]),
        "n_rows_total": int(n_total),
        "n_valid_o3": int(n_valid),
        "valid_rate": float(n_valid / n_total) if n_total else np.nan,
        "valid_by_t": json.dumps(valid_by_t, separators=(",", ":")),
        "monthly_mean": float(sub["monthly_mean"]),
        "first_slot": sub["day_keys"][0] if sub.get("day_keys") else "",
        "last_slot": sub["day_keys"][-1] if sub.get("day_keys") else "",
        "spec_name": VECCHIA_SPEC_NAME,
        "smooth_kernel": "spline",
        "spline_n_points": int(args.spline_n_points),
        "spline_r_max": float(args.spline_r_max),
        "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
        "lag_pattern": f"{LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}",
        "reference_advec_lon_abs": float(reference_advec_lon_abs),
        "loss": loss,
        "steps_raw": int(steps_ran),
        "precompute_s": float(precompute_s),
        "fit_s": float(fit_s),
        "total_s": float(precompute_s + fit_s),
        **{f"est_{k}": float(est[k]) for k in P_LABELS},
        **truth_metrics(est, truth),
        **cluster_summary,
    }
    del model, params_list, optimizer, source_map
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def make_monthly_summary(ok: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "dataset",
        "true_smooth",
        "fit_smooth",
        "year",
        "month",
        "block_prefix_requested",
        "block_prefix_used",
        "block_prefix_label",
        "block_prefix_order",
    ]
    for param in P_LABELS:
        col = f"est_{param}"
        if col not in ok.columns:
            continue
        for keys, sub in ok.groupby(group_cols, dropna=False):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(
                {
                    "dataset": str(keys[0]),
                    "true_smooth": float(keys[1]) if pd.notna(keys[1]) else np.nan,
                    "fit_smooth": float(keys[2]),
                    "year": int(keys[3]),
                    "month": int(keys[4]),
                    "block_prefix_requested": int(keys[5]),
                    "block_prefix_used": int(keys[6]),
                    "block_prefix_label": str(keys[7]),
                    "block_prefix_order": int(keys[8]),
                    "parameter": param,
                    "n_days": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "p10": float(np.quantile(vals, 0.10)),
                    "p90": float(np.quantile(vals, 0.90)),
                }
            )
    return pd.DataFrame(rows)


def finite_estimate_mask(df: pd.DataFrame) -> pd.Series:
    est_cols = [f"est_{p}" for p in P_LABELS if f"est_{p}" in df.columns]
    if not est_cols:
        return pd.Series(False, index=df.index)
    vals = df[est_cols].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(vals.to_numpy(dtype=float)).all(axis=1), index=df.index)


def _style_for_fit_smooth(smooth: float):
    styles = {
        0.2: ("tab:red", "D", "-"),
        0.25: ("tab:purple", "v", "-"),
        0.3: ("tab:blue", "o", "-"),
        0.35: ("tab:green", "s", "-"),
        0.4: ("tab:olive", "P", "-"),
        0.5: ("tab:orange", "^", "-"),
    }
    return styles.get(round(float(smooth), 2), ("0.25", "o", "-"))


def common_complete_prefixes(summary: pd.DataFrame, dataset: str) -> list[int]:
    sub_dataset = summary[summary["dataset"] == dataset].copy()
    if sub_dataset.empty:
        return []
    base = sub_dataset[sub_dataset["parameter"] == P_LABELS[0]].copy()
    if base.empty:
        return []
    prefix_sets = []
    for _, sub in base.groupby(["year", "fit_smooth"], dropna=False):
        vals = pd.to_numeric(sub["block_prefix_requested"], errors="coerce").dropna().astype(int)
        prefix_sets.append(set(vals.tolist()))
    if not prefix_sets:
        return []
    common = set.intersection(*prefix_sets)
    if not common:
        return []
    order_map = (
        base[["block_prefix_requested", "block_prefix_order"]]
        .drop_duplicates()
        .sort_values(["block_prefix_order", "block_prefix_requested"])
    )
    ordered = [int(r.block_prefix_requested) for r in order_map.itertuples(index=False) if int(r.block_prefix_requested) in common]
    return ordered or sorted(common)


def plot_dataset_monthly(summary: pd.DataFrame, dataset: str, path: Path) -> None:
    sub_dataset = summary[summary["dataset"] == dataset].copy()
    if sub_dataset.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    axes_flat = axes.ravel()
    x_order = common_complete_prefixes(summary, dataset)
    if not x_order:
        vals = pd.to_numeric(sub_dataset["block_prefix_requested"], errors="coerce").dropna().astype(int)
        x_order = sorted(vals.unique().tolist())
    if not x_order:
        plt.close(fig)
        return
    x_labels = [str(x) for x in x_order]
    x_pos = np.arange(len(x_order))
    for ax, param in zip(axes_flat, P_LABELS):
        sub_param = sub_dataset[sub_dataset["parameter"] == param].copy()
        for keys, sub in sub_param.groupby(["year", "fit_smooth"], dropna=False):
            year, fit_smooth = int(keys[0]), float(keys[1])
            sub = sub.set_index("block_prefix_requested").reindex(x_order)
            y = sub["median"].to_numpy(dtype=float)
            p10 = sub["p10"].to_numpy(dtype=float)
            p90 = sub["p90"].to_numpy(dtype=float)
            color, marker, ls = _style_for_fit_smooth(fit_smooth)
            label = f"{year}, fit s={fit_smooth:g}"
            finite = np.isfinite(y)
            ax.plot(x_pos[finite], y[finite], marker=marker, linestyle=ls, linewidth=1.7, color=color, label=label)
            if finite.sum() >= 2:
                ax.fill_between(x_pos[finite], p10[finite], p90[finite], color=color, alpha=0.10, linewidth=0)
        ax.set_title(param)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(P_LABELS) :]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{dataset}: monthly parameter medians by max-min 4x4 block-prefix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_missing_prefix_report(df: pd.DataFrame, requested_prefixes: list[int]) -> pd.DataFrame:
    if df.empty or "block_prefix_requested" not in df.columns:
        return pd.DataFrame()
    requested = [int(x) for x in requested_prefixes]
    group_cols = ["dataset", "true_smooth", "fit_smooth", "year", "month"]
    group_cols = [c for c in group_cols if c in df.columns]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        status_ok = sub[sub["status"] == "ok"].copy() if "status" in sub.columns else sub.iloc[0:0].copy()
        ok = status_ok[finite_estimate_mask(status_ok)].copy() if not status_ok.empty else status_ok
        ok_prefixes = set(pd.to_numeric(ok["block_prefix_requested"], errors="coerce").dropna().astype(int).tolist())
        attempted_prefixes = set(pd.to_numeric(sub["block_prefix_requested"], errors="coerce").dropna().astype(int).tolist())
        prefix_numeric = pd.to_numeric(sub["block_prefix_requested"], errors="coerce")
        for prefix in requested:
            if prefix in ok_prefixes:
                continue
            sub_prefix = sub[prefix_numeric == prefix]
            latest_error = ""
            if "error" in sub_prefix.columns and sub_prefix["error"].notna().any():
                latest_error = str(sub_prefix["error"].dropna().iloc[-1])
            rows.append(
                {
                    **key_map,
                    "block_prefix_requested": int(prefix),
                    "attempted": bool(prefix in attempted_prefixes),
                    "attempt_rows": int(len(sub_prefix)),
                    "status_ok_rows": int((sub_prefix["status"] == "ok").sum()) if "status" in sub_prefix.columns else 0,
                    "finite_ok_rows": int(finite_estimate_mask(sub_prefix[sub_prefix["status"] == "ok"]).sum()) if "status" in sub_prefix.columns else 0,
                    "error_rows": int((sub_prefix["status"] == "error").sum()) if "status" in sub_prefix.columns else 0,
                    "latest_error": latest_error,
                }
            )
    return pd.DataFrame(rows)


def plot_all_monthly(summary: pd.DataFrame, out_dir: Path, monthly_out_dir: Path | None = None) -> None:
    if summary.empty:
        return
    plot_root = out_dir / "monthly_average_plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(summary["dataset"].dropna().unique()):
        plot_dataset_monthly(summary, str(dataset), plot_root / f"{dataset}_parameter_by_blockmaxmin.png")
        if monthly_out_dir is not None:
            plot_dataset_monthly(summary, str(dataset), monthly_out_dir / f"{dataset}_parameter_by_blockmaxmin.png")


def refresh_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    monthly_out_dir: Path | None = None,
    requested_prefixes: list[int] | None = None,
) -> None:
    if not rows:
        return
    df = save_rows(out_dir / ALL_FITS_CSV, rows)
    missing_count = 0
    if requested_prefixes is not None:
        missing = make_missing_prefix_report(df, requested_prefixes)
        missing_count = int(len(missing))
        save_rows(out_dir / MISSING_PREFIX_CSV, missing.to_dict(orient="records"))
    status_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else pd.DataFrame()
    ok = status_ok[finite_estimate_mask(status_ok)].copy() if not status_ok.empty else status_ok
    summary = make_monthly_summary(ok) if not ok.empty else pd.DataFrame()
    if not summary.empty:
        save_rows(out_dir / SUMMARY_CSV, summary)
        plot_all_monthly(summary, out_dir, monthly_out_dir)
    lines = [
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Rows: {len(df)}",
        f"Status ok: {int((df['status'] == 'ok').sum()) if 'status' in df.columns else 0}",
        f"Finite completed: {int(len(ok))}",
        f"Errors: {int((df['status'] == 'error').sum()) if 'status' in df.columns else 0}",
        f"Missing ok block-prefix combinations: {missing_count}",
        "",
        df.tail(12).to_string(index=False),
    ]
    (out_dir / "running_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ST corridor Vecchia max-min 4x4 block-prefix density sweep.")
    parser.add_argument("--datasets", nargs="+", default=["real"], choices=["real", "sim0p3", "sim0p5"])
    parser.add_argument("--real-fit-smooths", nargs="+", default=["0.3", "0.35"])
    parser.add_argument(
        "--real-fit-smooths-by-year",
        nargs="*",
        default=[
            "2022:0.2,0.25,0.3,0.35",
            "2023:0.3,0.35",
            "2024:0.3,0.35",
            "2025:0.3,0.35",
        ],
        help="Optional year-specific real-data smooth map, e.g. '2022:0.2,0.25;2023:0.3,0.35'.",
    )
    parser.add_argument("--sim-fit-smooths", nargs="+", default=["0.3", "0.5"])
    parser.add_argument("--block-prefixes", nargs="+", default=["200", "400", "600", "1000"])
    parser.add_argument("--days", default="0,15", help="'0,15' means July day_idx 0..14.")
    parser.add_argument("--real-years", nargs="+", default=["2022", "2023", "2024", "2025"])
    parser.add_argument("--sim-years", nargs="+", default=["2022", "2023", "2024", "2025"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--space", default="1,1")
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--sim0p3-root", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3"))
    parser.add_argument("--sim0p5-root", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5"))
    parser.add_argument("--sim0p5-fallback-root", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"))
    parser.add_argument("--sim-pickle-kind", default="real_locations", choices=["real_locations", "gridded"])
    parser.add_argument("--spline-n-points", type=int, default=4000)
    parser.add_argument("--spline-r-max", type=float, default=30.0)
    parser.add_argument("--real-reference-advec-lon-abs", type=float, default=REFERENCE_ADVEC_LON_ABS)
    parser.add_argument("--sim-reference-advec-lon-abs", type=float, default=0.2)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--center-response", dest="center_response", action="store_true", default=True)
    parser.add_argument("--no-center-response", dest="center_response", action="store_false")
    parser.add_argument("--sim-init", choices=["truth", "real_default"], default="truth")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-every", type=int, default=1)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--monthly-out-dir", type=Path, default=None)
    return parser


def load_existing(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / ALL_FITS_CSV
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int, int, float, int]]:
    keys = set()
    for row in rows:
        if str(row.get("status", "")) != "ok":
            continue
        keys.add(
            (
                str(row["dataset"]),
                int(row["year"]),
                int(row["day_idx"]),
                round(float(row["fit_smooth"]), 6),
                int(row["block_prefix_requested"]),
            )
        )
    return keys


def load_sim_case(args: argparse.Namespace, dataset: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    old_root = getattr(args, "sim_data_root", None)
    args.sim_data_root = Path(root)
    assets, truth = load_sim_assets(args)
    for asset in assets:
        asset["data_kind"] = dataset
    if old_root is not None:
        args.sim_data_root = old_root
    return assets, truth


def fit_smooths_for_asset(case: dict[str, Any], asset: dict[str, Any]) -> list[float]:
    by_year = case.get("fit_smooths_by_year") or {}
    year = int(asset["year"])
    if by_year and year in by_year:
        return [float(x) for x in by_year[year]]
    return [float(x) for x in case["fit_smooths"]]


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args)
    out_dir = args.out_dir or default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_out_dir = args.monthly_out_dir
    if monthly_out_dir is not None:
        monthly_out_dir.mkdir(parents=True, exist_ok=True)

    real_fit_smooths = parse_float_tokens(args.real_fit_smooths)
    real_fit_smooths_by_year = parse_year_float_map(args.real_fit_smooths_by_year)
    sim_fit_smooths = parse_float_tokens(args.sim_fit_smooths)
    block_prefixes = parse_int_tokens(args.block_prefixes)
    datasets = [str(d) for d in args.datasets]

    print("SRC:", SRC, flush=True)
    print("device:", device, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("datasets:", datasets, flush=True)
    print("real_fit_smooths:", real_fit_smooths, "sim_fit_smooths:", sim_fit_smooths, flush=True)
    print("real_fit_smooths_by_year:", real_fit_smooths_by_year, flush=True)
    print("block_prefixes:", block_prefixes, flush=True)
    print("days:", parse_day_idxs(args.days), "region:", parse_pair(args.lat_range), parse_pair(args.lon_range), flush=True)

    rows = load_existing(out_dir) if args.skip_existing else []
    done = completed_keys(rows) if args.skip_existing else set()
    fit_id = int(max([0] + [int(r.get("fit_id", 0)) for r in rows]))

    run_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "src": str(SRC),
        "device": str(device),
        "args": clean_json_value(vars(args)),
        "datasets": datasets,
        "real_fit_smooths": real_fit_smooths,
        "real_fit_smooths_by_year": real_fit_smooths_by_year,
        "sim_fit_smooths": sim_fit_smooths,
        "block_prefixes": block_prefixes,
        "density_definition": "first K max-min ordered 4x4 regular-grid target blocks; all cells inside selected blocks are kept",
    }
    (out_dir / "run_config.json").write_text(json.dumps(clean_json_value(run_config), indent=2, sort_keys=True), encoding="utf-8")

    cases: list[dict[str, Any]] = []
    if "real" in datasets:
        real_assets = load_real_assets(args)
        cases.append(
            {
                "dataset": "real",
                "assets": real_assets,
                "truth": None,
                "true_smooth": None,
                "fit_smooths": real_fit_smooths,
                "fit_smooths_by_year": real_fit_smooths_by_year,
                "init": DEFAULT_REAL_INIT_PHYSICAL,
                "reference_advec_lon_abs": float(args.real_reference_advec_lon_abs),
            }
        )
    if "sim0p3" in datasets:
        sim_assets, truth = load_sim_case(args, "sim0p3", args.sim0p3_root)
        init = {k: float(truth[k]) for k in P_LABELS} if args.sim_init == "truth" else DEFAULT_REAL_INIT_PHYSICAL
        cases.append(
            {
                "dataset": "sim0p3",
                "assets": sim_assets,
                "truth": truth,
                "true_smooth": 0.3,
                "fit_smooths": sim_fit_smooths,
                "init": init,
                "reference_advec_lon_abs": float(args.sim_reference_advec_lon_abs),
            }
        )
    if "sim0p5" in datasets:
        root = Path(args.sim0p5_root)
        if not root.exists() and Path(args.sim0p5_fallback_root).exists():
            root = Path(args.sim0p5_fallback_root)
        sim_assets, truth = load_sim_case(args, "sim0p5", root)
        init = {k: float(truth[k]) for k in P_LABELS} if args.sim_init == "truth" else DEFAULT_REAL_INIT_PHYSICAL
        cases.append(
            {
                "dataset": "sim0p5",
                "assets": sim_assets,
                "truth": truth,
                "true_smooth": 0.5,
                "fit_smooths": sim_fit_smooths,
                "init": init,
                "reference_advec_lon_abs": float(args.sim_reference_advec_lon_abs),
            }
        )

    for case in cases:
        dataset = case["dataset"]
        for asset in case["assets"]:
            asset_fit_smooths = fit_smooths_for_asset(case, asset)
            for block_prefix in block_prefixes:
                for fit_smooth in asset_fit_smooths:
                    key = (
                        str(dataset),
                        int(asset["year"]),
                        int(asset["day_idx"]),
                        round(float(fit_smooth), 6),
                        int(block_prefix),
                    )
                    if args.skip_existing and key in done:
                        print(f"Skipping existing ok fit: {key}", flush=True)
                        continue
                    fit_id += 1
                    base = {
                        "fit_id": int(fit_id),
                        "status": "error",
                        "dataset": str(dataset),
                        "year": int(asset["year"]),
                        "month": int(asset["month"]),
                        "day_idx": int(asset["day_idx"]),
                        "day": str(asset["day"]),
                        "fit_smooth": float(fit_smooth),
                        "smooth": float(fit_smooth),
                        "true_smooth": float(case["true_smooth"]) if case["true_smooth"] is not None else np.nan,
                        "block_prefix_requested": int(block_prefix),
                        "block_prefix_label": f"B{int(block_prefix)}",
                    }
                    print("\n" + "-" * 96, flush=True)
                    print(
                        f"fit_id={fit_id} dataset={dataset} year={asset['year']} day={asset['day']} "
                        f"true_s={base['true_smooth']} fit_s={fit_smooth} block_prefix={block_prefix}",
                        flush=True,
                    )
                    print("-" * 96, flush=True)
                    try:
                        row = fit_one_resolution(
                            asset=asset,
                            dataset=str(dataset),
                            true_smooth=case["true_smooth"],
                            fit_smooth=float(fit_smooth),
                            block_prefix=int(block_prefix),
                            init_physical=case["init"],
                            truth=case["truth"],
                            reference_advec_lon_abs=float(case["reference_advec_lon_abs"]),
                            device=device,
                            args=args,
                            fit_id=int(fit_id),
                        )
                        print(
                            pd.Series(
                                {
                                    k: row.get(k)
                                    for k in [
                                        "dataset",
                                        "day",
                                        "fit_smooth",
                                        "block_prefix_label",
                                        "n_grid_block_prefix",
                                        "loss",
                                        "fit_s",
                                        "est_sigmasq",
                                        "est_range_lat",
                                        "est_range_lon",
                                        "est_range_time",
                                        "est_advec_lat",
                                        "est_advec_lon",
                                        "est_nugget",
                                    ]
                                }
                            ).to_string(),
                            flush=True,
                        )
                    except Exception as exc:
                        row = {**base, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=10)}
                        print(f"ERROR: {row['error']}", flush=True)
                        traceback.print_exc()
                    rows.append(clean_json_value(row))
                    append_jsonl(out_dir / "st_corridor_blockmaxmin_sweep_all_fits.jsonl", row)
                    if int(args.summary_every) > 0 and fit_id % int(args.summary_every) == 0:
                        refresh_outputs(out_dir, rows, monthly_out_dir, requested_prefixes=block_prefixes)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    refresh_outputs(out_dir, rows, monthly_out_dir, requested_prefixes=block_prefixes)
    print("\nDone.", flush=True)
    print("fits:", out_dir / ALL_FITS_CSV, flush=True)
    print("summary:", out_dir / SUMMARY_CSV, flush=True)


if __name__ == "__main__":
    main()
