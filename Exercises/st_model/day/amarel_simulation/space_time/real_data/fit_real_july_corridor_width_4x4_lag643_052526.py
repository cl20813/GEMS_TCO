#!/usr/bin/env python3
"""
Fit July real-data ST Vecchia with the final corridor-width 4x4 lag-643 model.

Created 2026-05-25.

This is the Amarel production version of the corridor-width cluster Vecchia
logic selected from the simulation sweeps.

Model geometry:
  - 4x4 regular-grid target clusters.
  - lag pattern 6/4/3:
      t:   6 previous same-time clusters in max-min cluster order.
      t-1: 4 lagged clusters covering the longitude corridor.
      t-2: 3 lagged clusters covering the longitude corridor.
  - reference one-step |advec_lon| delta defaults to 0.126.
  - t-1 corridor is [0.5 delta, 1.5 delta].
  - t-2 corridor is [0, 2 delta].

Data geometry:
  - The fixed cluster grid is built from the regular Latitude/Longitude order.
  - The same fixed grid order is reused for all 8 hourly tensors in a day.
  - Covariances use Source_Latitude/Source_Longitude when --keep-exact-loc is on.

Outputs:
  - One row per year/day fit.
  - Running CSV, JSONL, and text summary are updated after each fit.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter


LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
AMAREL_SRC = Path("/home/jl2815/tco")
if AMAREL_SRC.exists():
    sys.path.insert(0, str(AMAREL_SRC))
elif LOCAL_SRC.exists():
    sys.path.insert(0, str(LOCAL_SRC))

from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    SPEC_NAME as VECCHIA_SPEC_NAME,
    build_model as build_corridor_width_643_model,
    model_spec as corridor_width_643_spec,
)


P_LABELS = [
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
]

DEFAULT_INIT_PHYSICAL = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0218,
    "advec_lon": -0.1689,
    "nugget": 0.247,
}


def parse_int_list(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def parse_pair(text: str, cast=float) -> list[Any]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got {text!r}")
    return [cast(parts[0]), cast(parts[1])]


def parse_days(text: str) -> list[int]:
    text = str(text).strip().lower()
    if text == "all":
        return list(range(31))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end))
    return [int(p) for p in parts]


def clean_json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [clean_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    return value


def physical_to_log_phi(params: dict[str, float]) -> list[float]:
    sigmasq = float(params["sigmasq"])
    range_lat = float(params["range_lat"])
    range_lon = float(params["range_lon"])
    range_time = float(params["range_time"])
    nugget = float(params["nugget"])

    phi2 = 1.0 / range_lon
    phi1 = sigmasq * phi2
    phi3 = (range_lon / range_lat) ** 2
    phi4 = (range_lon / range_time) ** 2
    return [
        np.log(phi1),
        np.log(phi2),
        np.log(phi3),
        np.log(phi4),
        float(params["advec_lat"]),
        float(params["advec_lon"]),
        np.log(nugget),
    ]


def backmap_params(out_params: list[float]) -> dict[str, float]:
    p = [float(x) for x in out_params[:7]]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
    range_lon = 1.0 / phi2
    return {
        "sigmasq": float(np.exp(p[0]) / phi2),
        "range_lat": float(range_lon / np.sqrt(phi3)),
        "range_lon": float(range_lon),
        "range_time": float(range_lon / np.sqrt(phi4)),
        "advec_lat": float(p[4]),
        "advec_lon": float(p[5]),
        "nugget": float(np.exp(p[6])),
    }


def make_params_list(init_physical: dict[str, float], dtype: torch.dtype, device: torch.device):
    return [
        Parameter(torch.tensor([val], dtype=dtype, device=device))
        for val in physical_to_log_phi(init_physical)
    ]


def assert_grid_order_consistent(df_map: dict[str, pd.DataFrame], keys: list[str], base_coords: np.ndarray) -> None:
    for key in keys:
        coords = df_map[key][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        if coords.shape != base_coords.shape or not np.allclose(coords, base_coords, equal_nan=True):
            raise RuntimeError(
                f"Regular grid coordinate order differs at {key}; "
                "cluster local-index mapping is not reusable."
            )


def count_valid(day_map: dict[str, torch.Tensor]) -> tuple[int, int]:
    n_valid = 0
    n_total = 0
    for tensor in day_map.values():
        n_total += int(tensor.shape[0])
        n_valid += int((~torch.isnan(tensor[:, 2])).sum().item())
    return n_valid, n_total


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json_value(row), sort_keys=True) + "\n")


def save_rows(csv_path: Path, rows: list[dict[str, Any]], decimals: int = 6) -> None:
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    df.to_csv(csv_path, index=False, float_format=f"%.{decimals}f")


def write_running_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    completed = int((df.get("status", pd.Series(dtype=str)) == "ok").sum()) if len(df) else 0
    errors = int((df.get("status", pd.Series(dtype=str)) == "error").sum()) if len(df) else 0

    lines = [
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Rows: {len(df)} completed: {completed} errors: {errors}",
        "",
    ]
    if completed:
        ok = df[df["status"] == "ok"].copy()
        show_cols = [
            "year",
            "day_idx",
            "day",
            "loss",
            "fit_s",
            "precompute_s",
            "est_sigmasq",
            "est_range_lat",
            "est_range_lon",
            "est_range_time",
            "est_advec_lat",
            "est_advec_lon",
            "est_nugget",
        ]
        show_cols = [c for c in show_cols if c in ok.columns]
        lines.append("Latest completed fits:")
        lines.append(ok[show_cols].tail(10).to_string(index=False))
        lines.append("")

        med_cols = [
            "est_sigmasq",
            "est_range_lat",
            "est_range_lon",
            "est_range_time",
            "est_advec_lat",
            "est_advec_lon",
            "est_nugget",
            "fit_s",
            "precompute_s",
        ]
        med_cols = [c for c in med_cols if c in ok.columns]
        if med_cols:
            med = ok.groupby("year", dropna=False)[med_cols].median(numeric_only=True).reset_index()
            lines.append("Median estimates/timing by year:")
            lines.append(med.round(6).to_string(index=False))
            lines.append("")

    if errors:
        err_cols = ["year", "day_idx", "day", "error"]
        err_cols = [c for c in err_cols if c in df.columns]
        lines.append("Recent errors:")
        lines.append(df[df["status"] == "error"][err_cols].tail(10).to_string(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_data_root() -> Path:
    amarel = Path(config.amarel_data_load_path)
    if amarel.exists():
        return amarel
    return Path(config.mac_data_load_path)


def default_output_root() -> Path:
    amarel_root = Path(config.amarel_estimates_day_path)
    if amarel_root.parent.exists() or Path("/home/jl2815").exists():
        return amarel_root / "real_july_corridor_width_4x4_lag643_052526"
    return (
        Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates")
        / "real_july_corridor_width_4x4_lag643_052526"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit July real data with corridor-width 4x4 lag-643 ST cluster Vecchia."
    )
    parser.add_argument("--years", nargs="+", default=["2022", "2023", "2024", "2025"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0,31", help="'all', '0,31', or comma-separated day indices.")
    parser.add_argument("--space", default="1,1", help="Lat/lon resolution, e.g. 1,1.")
    parser.add_argument("--lat-range", default="-3,2", help="Latitude range, e.g. --lat-range=-3,2.")
    parser.add_argument("--lon-range", default="121,131", help="Longitude range, e.g. 121,131.")
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--reference-advec-lon-abs", type=float, default=REFERENCE_ADVEC_LON_ABS)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--device", default=None, help="Optional explicit device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--round-decimals", type=int, default=6)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    years = parse_int_list(args.years)
    days = parse_days(args.days)
    lat_lon_resolution = [int(x) for x in parse_pair(args.space, int)]
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.cuda_fallback == "error":
        raise RuntimeError("CUDA is not available and --cuda-fallback=error was requested.")
    else:
        device = torch.device("cpu")

    dtype = torch.double
    data_root = args.data_root or default_data_root()
    output_root = args.output_root or default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "real_july_corridor_width_4x4_lag643_all_fits.csv"
    jsonl_path = output_root / "real_july_corridor_width_4x4_lag643_all_fits.jsonl"
    summary_path = output_root / "running_summary.txt"
    config_path = output_root / "run_config.json"

    model_spec = corridor_width_643_spec(args.reference_advec_lon_abs)
    run_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "years": years,
        "month": args.month,
        "days": days,
        "lat_lon_resolution": lat_lon_resolution,
        "lat_range": lat_range,
        "lon_range": lon_range,
        "smooth": args.smooth,
        "device": str(device),
        "dtype": str(dtype),
        "keep_exact_loc": bool(args.keep_exact_loc),
        "optimizer": {
            "lbfgs_lr": args.lbfgs_lr,
            "lbfgs_steps": args.lbfgs_steps,
            "lbfgs_eval": args.lbfgs_eval,
            "lbfgs_history": args.lbfgs_history,
            "grad_tol": args.grad_tol,
        },
        "init_physical": DEFAULT_INIT_PHYSICAL,
        "model_spec": model_spec,
    }
    config_path.write_text(json.dumps(clean_json_value(run_config), indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, int]] = set()
    if args.skip_existing and csv_path.exists():
        existing = pd.read_csv(csv_path)
        rows = existing.to_dict(orient="records")
        if "status" in existing.columns:
            ok_existing = existing[existing["status"] == "ok"]
            done_keys = {(str(row["year"]), int(row["day_idx"])) for _, row in ok_existing.iterrows()}

    print("device:", device)
    print("data_root:", data_root)
    print("output_root:", output_root)
    print("years:", years)
    print("days:", days)
    print("spec:", VECCHIA_SPEC_NAME)
    print("model_spec:", model_spec)

    data_loader = load_data_dynamic_processed(str(data_root))

    for year in years:
        print("\n" + "=" * 72)
        print(f"Loading real July data for {year}-{args.month:02d}")
        print("=" * 72)

        df_map, _, _, monthly_mean = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=lat_lon_resolution,
            mm_cond_number=1,
            years_=[str(year)],
            months_=[args.month],
            lat_range=lat_range,
            lon_range=lon_range,
            is_whittle=True,
        )
        key_idx = sorted(df_map)
        if not key_idx:
            raise RuntimeError(f"No data loaded for {year}-{args.month:02d} from {data_root}")

        base_grid_coords_np = df_map[key_idx[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        print("n hourly slots:", len(key_idx))
        print("monthly_mean:", monthly_mean)
        print("grid shape:", base_grid_coords_np.shape)

        for day_idx in days:
            if (str(year), int(day_idx)) in done_keys:
                print(f"Skipping existing ok fit: {year}-{args.month:02d}-{day_idx + 1:02d}")
                continue

            day = f"{year}-{args.month:02d}-{day_idx + 1:02d}"
            hour_indices = [day_idx * 8, (day_idx + 1) * 8]
            selected_keys = key_idx[hour_indices[0] : hour_indices[1]]

            print("\n" + "-" * 72)
            print(f"Fitting {day} day_idx={day_idx}, slots={hour_indices}, n_keys={len(selected_keys)}")
            print("-" * 72)

            row: dict[str, Any] = {
                "year": str(year),
                "month": int(args.month),
                "day_idx": int(day_idx),
                "day": day,
                "status": "error",
                "smooth": float(args.smooth),
                "spec_name": VECCHIA_SPEC_NAME,
                "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
                "lag_pattern": f"{LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}",
                "reference_advec_lon_abs": float(args.reference_advec_lon_abs),
                "monthly_mean": float(monthly_mean),
                "first_slot": selected_keys[0] if selected_keys else "",
                "last_slot": selected_keys[-1] if selected_keys else "",
            }

            try:
                if len(selected_keys) != 8:
                    raise RuntimeError(f"Expected 8 hourly slots for {day}, found {len(selected_keys)}")
                assert_grid_order_consistent(df_map, selected_keys, base_grid_coords_np)

                day_map_cpu, _ = data_loader.load_working_data(
                    df_map,
                    monthly_mean,
                    hour_indices,
                    ord_mm=None,
                    dtype=dtype,
                    keep_ori=args.keep_exact_loc,
                )
                day_map = {k: v.to(device) for k, v in day_map_cpu.items()}
                n_valid, n_total = count_valid(day_map)
                row.update(
                    {
                        "n_time_slots": len(day_map),
                        "n_rows_total": n_total,
                        "n_valid_o3": n_valid,
                        "valid_rate": float(n_valid / n_total) if n_total else np.nan,
                    }
                )

                params_list = make_params_list(DEFAULT_INIT_PHYSICAL, dtype=dtype, device=device)
                model = build_corridor_width_643_model(
                    smooth=args.smooth,
                    input_map=day_map,
                    grid_coords=base_grid_coords_np,
                    reference_advec_lon_abs=args.reference_advec_lon_abs,
                    daily_stride=args.daily_stride,
                    target_chunk_size=args.target_chunk_size,
                    min_target_points=args.min_target_points,
                )

                t0 = time.time()
                model.precompute_conditioning_sets()
                precompute_s = time.time() - t0

                optimizer = model.set_optimizer(
                    params_list,
                    lr=args.lbfgs_lr,
                    max_iter=args.lbfgs_eval,
                    max_eval=args.lbfgs_eval,
                    history_size=args.lbfgs_history,
                )

                t1 = time.time()
                if args.suppress_fit_prints:
                    import contextlib
                    import io

                    with contextlib.redirect_stdout(io.StringIO()):
                        out, steps_ran = model.fit_vecc_lbfgs(
                            params_list,
                            optimizer,
                            max_steps=args.lbfgs_steps,
                            grad_tol=args.grad_tol,
                        )
                else:
                    out, steps_ran = model.fit_vecc_lbfgs(
                        params_list,
                        optimizer,
                        max_steps=args.lbfgs_steps,
                        grad_tol=args.grad_tol,
                    )
                fit_s = time.time() - t1

                est = backmap_params(out)
                cluster_summary = model.cluster_summary()
                row.update(
                    {
                        "status": "ok",
                        "error": "",
                        "loss": float(out[-1]),
                        "steps_raw": int(steps_ran),
                        "precompute_s": float(precompute_s),
                        "fit_s": float(fit_s),
                        "total_s": float(precompute_s + fit_s),
                        **{f"est_{k}": float(est[k]) for k in P_LABELS},
                        **cluster_summary,
                    }
                )
                print(pd.Series(row).to_string())

                del model, params_list, optimizer, day_map, day_map_cpu
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
                print(f"ERROR fitting {day}: {row['error']}")
                traceback.print_exc()

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            rows.append(clean_json_value(row))
            append_jsonl(jsonl_path, row)
            save_rows(csv_path, rows, decimals=args.round_decimals)
            write_running_summary(summary_path, rows)

    print("\nDone.")
    print("csv:", csv_path)
    print("jsonl:", jsonl_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
