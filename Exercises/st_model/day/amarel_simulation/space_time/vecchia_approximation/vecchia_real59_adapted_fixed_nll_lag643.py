#!/usr/bin/env python3
"""Fit and compare adapted/fixed lag-6/4/3 Vecchia likelihoods.

The production window is July 1--30 in 2024 and 2025, excluding the
incomplete 2025-07-24 record, for 59 usable dates.  Each date is fitted twice:

* ``adapted`` uses the estimated advection corridor;
* ``fixed`` keeps past conditioning corridors centered on the target block.

This runner intentionally computes no conditional-eigen, dense-eigen, SLQ,
Lanczos, or Ritz diagnostic.  Its comparison target is the native Vecchia
negative log likelihood: each method's fitted objective on its own
conditioning graph, evaluated over all valid target observations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SUBMIT_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", str(HERE))).resolve()
AMAREL_ROOT = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
for candidate in (AMAREL_ROOT, SUBMIT_DIR, HERE, LOCAL_SRC):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import vecchia_adapted_fixed_lag643_core as core  # noqa: E402


METHODS = ("adapted", "fixed")
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
COLORS = {"adapted": "#1f77b4", "fixed": "#d62728"}
LABELS = {"adapted": "adapted corridor", "fixed": "fixed center"}
PARAMETERS = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)
RESULT_COLUMNS = (
    "dataset_id",
    "year",
    "date",
    "method",
    *PARAMETERS,
    "precompute_seconds",
    "fit_time_seconds",
    "outer_steps",
    "max_abs_gradient",
    "init_advec_lat",
    "init_advec_lon",
    "n_target_points",
    "native_nll_per_target",
    "native_nll_total",
    "gls_beta",
)
CHECKPOINT_NAME = "fit_checkpoint_native_nll.json"
FIT_CSV_NAME = "daily_fit_results.csv"
NLL_CSV_NAME = "daily_native_nll.csv"
SUMMARY_CSV_NAME = "native_nll_summary.csv"
PLOT_NAME = "daily_native_nll.png"
CONFIG_NAME = "run_config.json"

# fit_one_geometry normally offers an optional conditional-eigen diagnostic.
# This runner is likelihood-only, so disable it before any fit is constructed.
core.EIGEN_GEOMETRIES = ()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def date_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "dataset_id": f"real_{year}07{day:02d}",
            "data_kind": "real",
            "data_source": f"real_july_{year}",
            "year": year,
            "month": 7,
            "day": day,
            "date": f"{year}-07-{day:02d}",
        }
        for year in (2024, 2025)
        for day in range(1, 31)
        if not (year == 2025 and day == 24)
    ]
    if len(specs) != 59:
        raise RuntimeError(f"Internal date selection error: {len(specs)} dates")
    return specs


def normalize_record(record: dict[str, Any], source: Path) -> dict[str, Any]:
    """Normalize this runner's records and the earlier full-eigen checkpoint."""
    n_target_points = record.get("n_target_points", record.get("n_observations"))
    nll_per_target = record.get(
        "native_nll_per_target", record.get("native_nll_per_observation")
    )
    if n_target_points is None or nll_per_target is None:
        raise ValueError(
            f"{source}: record lacks target count or native Vecchia NLL"
        )
    n_target_points = int(n_target_points)
    nll_per_target = float(nll_per_target)
    total = record.get("native_nll_total", nll_per_target * n_target_points)
    normalized = {
        "dataset_id": str(record["dataset_id"]),
        "year": int(record["year"]),
        "date": str(record["date"]),
        "method": str(record["method"]),
        **{name: float(record[name]) for name in PARAMETERS},
        "precompute_seconds": float(record.get("precompute_seconds", 0.0)),
        "fit_time_seconds": float(record["fit_time_seconds"]),
        "outer_steps": int(record.get("outer_steps", 0)),
        "max_abs_gradient": record.get("max_abs_gradient"),
        "init_advec_lat": float(record["init_advec_lat"]),
        "init_advec_lon": float(record["init_advec_lon"]),
        "n_target_points": n_target_points,
        "native_nll_per_target": nll_per_target,
        "native_nll_total": float(total),
        "gls_beta": [float(value) for value in record.get("gls_beta", [])],
    }
    if normalized["method"] not in METHODS:
        raise ValueError(f"{source}: unexpected method {normalized['method']!r}")
    if n_target_points <= 0 or not math.isfinite(nll_per_target):
        raise ValueError(f"{source}: invalid likelihood record {normalized}")
    return normalized


def read_checkpoint(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list or a records object")
    normalized = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} is not an object")
        item = normalize_record(record, path)
        key = (item["dataset_id"], item["method"])
        if key in seen:
            raise ValueError(f"{path}: duplicate fit record {key}")
        seen.add(key)
        normalized.append(item)
    return normalized


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoint = args.output_root / CHECKPOINT_NAME
    if checkpoint.is_file():
        if args.import_checkpoint is not None:
            print(
                f"Ignoring --import-checkpoint because {checkpoint} already exists",
                flush=True,
            )
        return read_checkpoint(checkpoint)
    if args.import_checkpoint is None:
        return []
    if not args.import_checkpoint.is_file():
        raise FileNotFoundError(args.import_checkpoint)
    records = read_checkpoint(args.import_checkpoint)
    print(
        f"Imported {len(records)} fit records from {args.import_checkpoint}",
        flush=True,
    )
    return records


def ordered_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (str(row["date"]), METHOD_ORDER[str(row["method"])]),
    )


def results_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(ordered_records(records), columns=RESULT_COLUMNS)


def write_fit_csv(frame: pd.DataFrame, output_root: Path) -> None:
    columns = [
        "date",
        "method",
        *PARAMETERS,
        "precompute_seconds",
        "fit_time_seconds",
        "outer_steps",
        "max_abs_gradient",
        "init_advec_lat",
        "init_advec_lon",
    ]
    output = frame.loc[:, columns].copy()
    numeric = [column for column in columns if column not in {"date", "method"}]
    output.loc[:, numeric] = output.loc[:, numeric].apply(
        pd.to_numeric, errors="coerce"
    )
    output.to_csv(output_root / FIT_CSV_NAME, index=False, float_format="%.8g")


def add_paired_difference(frame: pd.DataFrame) -> pd.DataFrame:
    totals = frame.pivot(index="date", columns="method", values="native_nll_total")
    if set(METHODS).issubset(totals.columns):
        difference = totals["fixed"] - totals["adapted"]
    else:
        difference = pd.Series(dtype=float)
    output = frame.loc[
        :,
        [
            "date",
            "year",
            "method",
            "n_target_points",
            "native_nll_per_target",
            "native_nll_total",
        ],
    ].copy()
    output["fixed_minus_adapted_total_nll"] = output["date"].map(difference)
    return output


def write_nll_outputs(frame: pd.DataFrame, output_root: Path) -> None:
    nll = add_paired_difference(frame)
    nll.to_csv(
        output_root / NLL_CSV_NAME,
        index=False,
        float_format="%.10g",
    )
    if nll.empty:
        pd.DataFrame().to_csv(output_root / SUMMARY_CSV_NAME, index=False)
        return
    summary = (
        nll.groupby(["year", "method"], as_index=False)
        .agg(
            n_dates=("date", "nunique"),
            total_target_points=("n_target_points", "sum"),
            total_native_nll=("native_nll_total", "sum"),
            mean_daily_nll_per_target=("native_nll_per_target", "mean"),
        )
        .sort_values(["year", "method"])
    )
    summary["pooled_native_nll_per_target"] = (
        summary["total_native_nll"] / summary["total_target_points"]
    )
    summary.to_csv(
        output_root / SUMMARY_CSV_NAME,
        index=False,
        float_format="%.10g",
    )


def plot_native_nll(frame: pd.DataFrame, output_root: Path) -> None:
    if frame.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharey=True)
    for ax, year in zip(axes, (2024, 2025)):
        year_rows = frame[frame["year"] == year]
        for method in METHODS:
            rows = year_rows[year_rows["method"] == method].sort_values("date")
            ax.plot(
                rows["date"],
                rows["native_nll_per_target"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_title(f"Real {year} July: native Vecchia NLL per target")
        ax.set_ylabel("NLL / target")
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / PLOT_NAME, dpi=190, bbox_inches="tight")
    plt.close(fig)


def persist(records: list[dict[str, Any]], output_root: Path) -> None:
    compact = [
        {column: json_ready(record.get(column)) for column in RESULT_COLUMNS}
        for record in ordered_records(records)
    ]
    write_json(
        output_root / CHECKPOINT_NAME,
        {"schema_version": 1, "records": compact},
    )
    frame = results_frame(compact)
    write_fit_csv(frame, output_root)
    write_nll_outputs(frame, output_root)
    plot_native_nll(frame, output_root)


def build_fit_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        smooth=args.smooth,
        daily_stride=2,
        target_chunk_size=args.target_chunk_size,
        union_target_chunk_size=0,
        min_target_points=1,
        fixed_nugget=None,
        zero_nugget_fit_init=core.DEFAULT_REAL_INIT["nugget"],
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_steps=args.lbfgs_steps,
        lbfgs_eval=args.lbfgs_eval,
        lbfgs_history=args.lbfgs_history,
        grad_tol=args.grad_tol,
        suppress_fit_prints=args.suppress_fit_prints,
        diag_chunk_size=1,
        union_diag_chunk_size=1,
        resample_grid=1,
        empirical_max_lat_offset=20,
        empirical_max_lon_offset=20,
        empirical_min_pair_count=1000,
        empirical_smooth_bandwidth_deg=0.063,
        subgrid_max_condition_number=100.0,
        lat_range=args.lat_range,
        lon_range=args.lon_range,
        real_data_root=args.real_data_root,
        synthetic_data_root=Path("/unused"),
        hours_per_day=8,
        keep_exact_loc=True,
        truth_nugget=None,
        device=args.device,
        require_cuda=args.device == "cuda",
    )


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def fit_day(
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    completed = {
        str(record["method"])
        for record in records
        if str(record["dataset_id"]) == spec["dataset_id"]
    }
    if completed == set(METHODS):
        print("  Both likelihood fits already complete", flush=True)
        return

    fit_args = build_fit_args(args)
    device = core.resolve_device(fit_args)
    asset = core.load_real_asset(spec, fit_args)
    seed = core.m3_q3_seed(asset, fit_args)
    init = dict(core.DEFAULT_REAL_INIT)

    for method in METHODS:
        if method in completed:
            print(f"  Fit already complete: {method}", flush=True)
            continue
        clear_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"  Fitting {method}", flush=True)
        row, raw, _, _ = core.fit_one_geometry(
            method, asset, seed, init, device, fit_args
        )
        del raw
        n_target_points = int(row["n_target_points"])
        nll_per_target = float(row["final_native_nll"])
        record = {
            "dataset_id": spec["dataset_id"],
            "year": spec["year"],
            "date": spec["date"],
            "method": method,
            **{name: float(row[f"est_{name}"]) for name in PARAMETERS},
            "precompute_seconds": float(row["precompute_s"]),
            "fit_time_seconds": float(row["fit_s"]),
            "outer_steps": int(row["outer_steps"]),
            "max_abs_gradient": float(row["max_abs_gradient"]),
            "init_advec_lat": float(seed["seed_lat"]),
            "init_advec_lon": float(seed["seed_lon"]),
            "n_target_points": n_target_points,
            "native_nll_per_target": nll_per_target,
            "native_nll_total": nll_per_target * n_target_points,
            "gls_beta": [float(value) for value in row["gls_beta"]],
        }
        records.append(record)
        persist(records, args.output_root)
        print(
            f"    NLL/target={nll_per_target:.8f}; "
            f"total NLL={record['native_nll_total']:.4f}; "
            f"fit={record['fit_time_seconds']:.2f}s",
            flush=True,
        )
    del asset
    clear_device(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-data-root", type=Path, default=Path("/home/jl2815/tco/data")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/jl2815/tco/exercise_output/summer/"
            "vecchia_real59_adapted_fixed_nll_lag643"
        ),
    )
    parser.add_argument(
        "--import-checkpoint",
        type=Path,
        default=None,
        help=(
            "Seed an empty output root from this runner's checkpoint or the "
            "older fit_checkpoint_full_precision.json."
        ),
    )
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5, choices=(0.5,))
    parser.add_argument("--target-chunk-size", type=int, default=256)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=40)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--suppress-fit-prints", action="store_true")
    return parser


def validate_scope(records: list[dict[str, Any]], specs: list[dict[str, Any]]) -> None:
    selected = {str(spec["dataset_id"]) for spec in specs}
    unexpected = [
        (record["dataset_id"], record["method"])
        for record in records
        if str(record["dataset_id"]) not in selected
        or str(record["method"]) not in METHODS
    ]
    if unexpected:
        raise ValueError(f"Checkpoint contains records outside this run: {unexpected}")


def write_run_config(args: argparse.Namespace, n_imported: int) -> None:
    write_json(
        args.output_root / CONFIG_NAME,
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "methods": METHODS,
            "n_dates": 59,
            "excluded_date": "2025-07-24",
            "diagnostics": [],
            "comparison": "native Vecchia NLL on each method's own graph",
            "n_imported_records": n_imported,
            "arguments": vars(args),
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = date_specs()
    records = load_records(args)
    validate_scope(records, specs)
    write_run_config(args, len(records))
    if records:
        persist(records, args.output_root)

    for position, spec in enumerate(specs, start=1):
        print(f"\n[{position}/{len(specs)}] {spec['date']}", flush=True)
        fit_day(spec, records, args)

    persist(records, args.output_root)
    expected_rows = len(specs) * len(METHODS)
    if len(records) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} fit rows, found {len(records)}")
    print(
        f"Complete: {len(specs)} dates, {len(records)} likelihood fits in "
        f"{args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
