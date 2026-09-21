#!/usr/bin/env python3
"""Benchmark local CPU fitting chunks for adapted 4/3/2 and 6/4/3 Vecchia.

The default run performs exactly six independent fits on the 2024-07-03 real
GEMS TCO day:

    lag pattern 4/3/2 x target_chunk_size 64, 128, 256
    lag pattern 6/4/3 x target_chunk_size 64, 128, 256

Every fit starts from the same physical parameters and the same M3 masked-FFT
+ safeguarded-Q3 advection seed.  ``fixed`` corridor geometry is intentionally
not included.  The statistical nugget is estimated from an initial value of
0.247; it is not fixed at zero.

Timing fields separate conditioning-graph precomputation, LBFGS optimization,
the final objective evaluation, and total trial wall time.  Results are saved
after every trial so an interrupted run can resume safely.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
VECCHIA_APPROX = HERE.parent / "vecchia_approximation"
for path in (SRC, VECCHIA_APPROX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import vecchia_adapted_fixed_lag643_core as core  # noqa: E402
from GEMS_TCO.vecchia_realdata_adapted_corridor_width_4x4_lag643 import (  # noqa: E402
    AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit,
)
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (  # noqa: E402
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
)


DTYPE = torch.float64
MODEL_CLASSES = {
    "432": DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
    "643": AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit,
}
PARAMETER_NAMES = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def persist_results(records: list[dict[str, Any]], output_dir: Path) -> None:
    atomic_json(output_dir / "batch_benchmark_results.json", records)
    frame = pd.DataFrame(records)
    temporary = output_dir / "batch_benchmark_results.csv.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(output_dir / "batch_benchmark_results.csv")


def persist_summary(records: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(record for record in records if record.get("status") == "ok")
    if frame.empty:
        return
    frame = frame.sort_values(["lag_pattern", "target_chunk_size"]).copy()
    frame["fit_s_vs_batch64_pct"] = np.nan
    frame["fit_s_rank_within_lag"] = np.nan
    for lag_pattern, indexes in frame.groupby("lag_pattern", sort=False).groups.items():
        group = frame.loc[indexes]
        baseline = group.loc[group["target_chunk_size"].eq(64), "fit_s"]
        if not baseline.empty:
            frame.loc[indexes, "fit_s_vs_batch64_pct"] = (
                100.0 * (group["fit_s"] / float(baseline.iloc[0]) - 1.0)
            )
        frame.loc[indexes, "fit_s_rank_within_lag"] = group["fit_s"].rank(
            method="min"
        )
    columns = [
        "lag_pattern",
        "target_chunk_size",
        "precompute_s",
        "fit_s",
        "final_eval_s",
        "trial_wall_s",
        "fit_s_vs_batch64_pct",
        "fit_s_rank_within_lag",
        "optimizer_likelihood_calls",
        "outer_steps",
        "final_native_nll",
    ]
    summary = frame[columns]
    temporary = output_dir / "batch_benchmark_summary.csv.tmp"
    summary.to_csv(temporary, index=False)
    temporary.replace(output_dir / "batch_benchmark_summary.csv")


def mac_hardware_info() -> dict[str, str]:
    if platform.system() != "Darwin":
        return {}
    try:
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    wanted = {"Model Name", "Model Identifier", "Chip", "Total Number of Cores", "Memory"}
    info: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key in wanted:
            info[key.lower().replace(" ", "_")] = value.strip()
    return info


def parse_date(value: str) -> tuple[int, int, int]:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc
    return parsed.year, parsed.month, parsed.day


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2024-07-03")
    parser.add_argument(
        "--real-data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "batch_benchmark_results_090326",
    )
    parser.add_argument("--lag-patterns", nargs="+", choices=sorted(MODEL_CLASSES), default=["432", "643"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--force", action="store_true", help="Rerun completed lag/batch combinations.")
    return parser


def make_loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        real_data_root=args.real_data_root,
        lat_range="-3,2",
        lon_range="121,131",
        hours_per_day=8,
        keep_exact_loc=True,
        empirical_max_lat_offset=20,
        empirical_max_lon_offset=20,
        empirical_min_pair_count=1000,
        empirical_smooth_bandwidth_deg=0.063,
        subgrid_max_condition_number=100.0,
    )


def fit_one(
    lag_pattern: str,
    batch_size: int,
    asset: core.DayAsset,
    seed: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    trial_started = time.perf_counter()
    model_class = MODEL_CLASSES[lag_pattern]
    mapped = {
        key: tensor.to(device="cpu", dtype=DTYPE).contiguous()
        for key, tensor in asset.source_map.items()
    }
    model = model_class(
        smooth=float(args.smooth),
        input_map=mapped,
        grid_coords=asset.grid_coords,
        reference_advec_lat=float(seed["seed_lat"]),
        reference_advec_lon=float(seed["seed_lon"]),
        daily_stride=2,
        target_chunk_size=int(batch_size),
        min_target_points=1,
    )

    precompute_started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - precompute_started

    physical_init = {
        **core.DEFAULT_REAL_INIT,
        "advec_lat": float(seed["seed_lat"]),
        "advec_lon": float(seed["seed_lon"]),
    }
    raw_init = core.physical_to_raw(physical_init)
    params = [
        Parameter(torch.tensor(value, dtype=DTYPE, device="cpu"), requires_grad=True)
        for value in raw_init
    ]
    optimizer = model.set_optimizer(
        params,
        lr=float(args.lbfgs_lr),
        max_iter=int(args.lbfgs_eval),
        max_eval=int(args.lbfgs_eval),
        history_size=int(args.lbfgs_history),
    )

    original_likelihood = model.vecchia_batched_likelihood
    likelihood_calls = 0

    def counted_likelihood(raw_params: torch.Tensor) -> torch.Tensor:
        nonlocal likelihood_calls
        likelihood_calls += 1
        return original_likelihood(raw_params)

    model.vecchia_batched_likelihood = counted_likelihood
    fit_started = time.perf_counter()
    returned, step_index = model.fit_vecc_lbfgs(
        params,
        optimizer,
        max_steps=int(args.lbfgs_steps),
        grad_tol=float(args.grad_tol),
    )
    fit_s = time.perf_counter() - fit_started
    optimizer_likelihood_calls = likelihood_calls

    raw_final = [float(param.detach().item()) for param in params]
    final_eval_started = time.perf_counter()
    with torch.no_grad():
        final_nll = float(
            original_likelihood(torch.as_tensor(raw_final, dtype=DTYPE)).detach().item()
        )
    final_eval_s = time.perf_counter() - final_eval_started
    estimate = core.raw_to_physical(raw_final)
    gradients = [
        abs(float(param.grad.detach().item()))
        for param in params
        if param.grad is not None
    ]
    summary = model.cluster_summary()
    trial_wall_s = time.perf_counter() - trial_started

    record: dict[str, Any] = {
        "status": "ok",
        "date": asset.date,
        "lag_pattern": "/".join(lag_pattern),
        "geometry": "adapted",
        "target_chunk_size": int(batch_size),
        "device": "cpu",
        "dtype": str(DTYPE),
        "nugget_mode": "estimated",
        "init_nugget": float(core.DEFAULT_REAL_INIT["nugget"]),
        "init_advec_lat": float(seed["seed_lat"]),
        "init_advec_lon": float(seed["seed_lon"]),
        "precompute_s": float(precompute_s),
        "fit_s": float(fit_s),
        "final_eval_s": float(final_eval_s),
        "precompute_plus_fit_s": float(precompute_s + fit_s),
        "trial_wall_s": float(trial_wall_s),
        "optimizer_likelihood_calls": int(optimizer_likelihood_calls),
        "seconds_per_optimizer_likelihood_call": float(fit_s / max(optimizer_likelihood_calls, 1)),
        "outer_steps": int(step_index) + 1,
        "max_abs_gradient": max(gradients) if gradients else np.nan,
        "fit_returned_nll": float(returned[-1]),
        "final_native_nll": float(final_nll),
        "n_target_blocks": int(summary["n_target_blocks"]),
        "n_target_points": int(summary["n_target_points"]),
        "n_batches": int(summary["n_batches"]),
        "spec_name": str(summary["spec_name"]),
    }
    record.update({f"est_{name}": float(estimate[name]) for name in PARAMETER_NAMES})
    return record


def main() -> None:
    args = build_parser().parse_args()
    if any(size <= 0 for size in args.batch_sizes):
        raise ValueError("Every batch size must be positive")
    year, month, day = parse_date(args.date)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "batch_benchmark_results.json"
    records: list[dict[str, Any]] = []
    if results_path.is_file() and not args.force:
        loaded = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            records = loaded

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "date": args.date,
        "real_data_root": str(args.real_data_root.resolve()),
        "lag_patterns": list(args.lag_patterns),
        "batch_sizes": list(args.batch_sizes),
        "number_of_requested_fits": len(args.lag_patterns) * len(args.batch_sizes),
        "geometry": "adapted_only",
        "block_shape": [4, 4],
        "smooth": float(args.smooth),
        "dtype": str(DTYPE),
        "device": "cpu",
        "nugget_mode": "estimated",
        "base_initial_parameters_before_advection_seed": core.DEFAULT_REAL_INIT,
        "optimizer": {
            "name": "LBFGS",
            "lr": float(args.lbfgs_lr),
            "line_search_fn": "strong_wolfe",
            "max_iter": int(args.lbfgs_eval),
            "max_eval": int(args.lbfgs_eval),
            "history_size": int(args.lbfgs_history),
            "tolerance_grad": 1e-5,
            "tolerance_change": 1e-9,
            "outer_max_steps": int(args.lbfgs_steps),
            "outer_grad_tol": float(args.grad_tol),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
            **mac_hardware_info(),
        },
    }
    atomic_json(args.output_dir / "run_config.json", config)

    loader_args = make_loader_args(args)
    spec = {
        "dataset_id": f"real_{year}{month:02d}{day:02d}",
        "year": year,
        "month": month,
        "day": day,
        "date": args.date,
    }
    print(f"Loading {args.date} from {args.real_data_root}", flush=True)
    load_started = time.perf_counter()
    asset = core.load_real_asset(spec, loader_args)
    data_load_s = time.perf_counter() - load_started
    print(
        f"Loaded {asset.n_valid:,}/{asset.n_total:,} observations in {data_load_s:.3f}s",
        flush=True,
    )
    seed = core.m3_q3_seed(asset, loader_args)
    seed_record = {**seed, "data_load_s": data_load_s}
    atomic_json(args.output_dir / "initializer.json", seed_record)
    config["fit_initial_parameters"] = {
        **core.DEFAULT_REAL_INIT,
        "advec_lat": float(seed["seed_lat"]),
        "advec_lon": float(seed["seed_lon"]),
    }
    config["data_load_s"] = float(data_load_s)
    config["initializer_s"] = float(seed["initializer_s"])
    atomic_json(args.output_dir / "run_config.json", config)
    print(
        f"Adapted seed=({seed['seed_lat']:.9f}, {seed['seed_lon']:.9f}); "
        f"initializer={seed['initializer_s']:.3f}s",
        flush=True,
    )

    completed = {
        (str(record.get("lag_pattern", "")).replace("/", ""), int(record["target_chunk_size"]))
        for record in records
        if record.get("status") == "ok" and "target_chunk_size" in record
    }
    total = len(args.lag_patterns) * len(args.batch_sizes)
    trial_number = 0
    for lag_pattern in args.lag_patterns:
        for batch_size in args.batch_sizes:
            trial_number += 1
            key = (lag_pattern, int(batch_size))
            if key in completed and not args.force:
                print(f"[{trial_number}/{total}] skip completed lag={lag_pattern} batch={batch_size}", flush=True)
                continue
            gc.collect()
            print(f"[{trial_number}/{total}] fit lag={lag_pattern} batch={batch_size}", flush=True)
            try:
                record = fit_one(lag_pattern, batch_size, asset, seed, args)
                print(
                    f"  fit={record['fit_s']:.3f}s precompute={record['precompute_s']:.3f}s "
                    f"wall={record['trial_wall_s']:.3f}s NLL={record['final_native_nll']:.10f} "
                    f"calls={record['optimizer_likelihood_calls']}",
                    flush=True,
                )
            except Exception as exc:
                record = {
                    "status": "failed",
                    "date": asset.date,
                    "lag_pattern": "/".join(lag_pattern),
                    "geometry": "adapted",
                    "target_chunk_size": int(batch_size),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            if args.force:
                records = [
                    old
                    for old in records
                    if not (
                        str(old.get("lag_pattern", "")).replace("/", "") == lag_pattern
                        and int(old.get("target_chunk_size", -1)) == int(batch_size)
                    )
                ]
            records.append(record)
            persist_results(records, args.output_dir)
            gc.collect()

    ok = [record for record in records if record.get("status") == "ok"]
    persist_summary(records, args.output_dir)
    print(f"Completed {len(ok)}/{total} requested fits", flush=True)
    print(args.output_dir / "batch_benchmark_results.csv", flush=True)


if __name__ == "__main__":
    main()
