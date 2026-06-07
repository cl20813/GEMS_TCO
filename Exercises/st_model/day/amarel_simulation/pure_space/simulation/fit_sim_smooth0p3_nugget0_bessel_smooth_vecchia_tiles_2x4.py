#!/usr/bin/env python3
"""Pure-space Vecchia smooth estimation on smooth=0.3, nugget=0 simulation data.

This wrapper points the production 2x4-tile Bessel-Matern Vecchia fitter at the
reusable July ST simulation pickle generated with true smooth nu=0.3 and nugget
fixed to zero.  It is meant to test whether the pure-space smooth-free Vecchia
fit recovers the true local smoothness when smooth-nugget confounding is
removed from the data-generating process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PURE_SPACE_DIR = HERE.parent

DEFAULT_SIM_ROOT = Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0")
DEFAULT_OUT_ROOT = Path("/home/jl2815/tco/exercise_output/summer/sim_smooth0p3_nugget0_purespace_bessel_vecchia_2x4_fixed0_060726")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run pure-space 2x4-tile Vecchia smooth estimation on smooth=0.3, nugget=0 simulation data."
    )
    p.add_argument("--mode", choices=["manifest", "fit", "summarize", "run-hours", "all"], required=True)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    p.add_argument("--sim-kind", choices=["real_locations", "gridded"], default="real_locations")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--month", default=None)
    p.add_argument("--max-hours", type=int, default=80)
    p.add_argument("--expected-hours", type=int, default=80)
    p.add_argument("--hour-start", type=int, default=0)
    p.add_argument("--hour-end", type=int, default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--nugget-mode", choices=["free", "fixed0"], default="fixed0")
    p.add_argument("--lat-range", default="-3,2")
    p.add_argument("--lon-range", default="121,131")
    p.add_argument("--tile-y", type=int, default=2)
    p.add_argument("--tile-x", type=int, default=4)
    p.add_argument("--min-tile-points", type=int, default=200)
    p.add_argument("--tile-max-points", type=int, default=0)
    p.add_argument("--tile-workers", type=int, default=4)
    p.add_argument("--cluster-block-shape", default="4x4")
    p.add_argument("--cluster-neighbor-blocks", type=int, default=2)
    p.add_argument("--target-chunk-size", type=int, default=128)
    p.add_argument("--min-target-points", type=int, default=1)
    p.add_argument("--mean-design", choices=["lat", "latlon"], default="lat")
    p.add_argument("--range-lat-init", type=float, default=0.35)
    p.add_argument("--range-lon-init", type=float, default=0.35)
    p.add_argument("--smooth-init", type=float, default=0.5)
    p.add_argument("--smooth-min", type=float, default=0.05)
    p.add_argument("--smooth-max", type=float, default=2.5)
    p.add_argument("--range-min", type=float, default=0.03)
    p.add_argument("--range-max", type=float, default=5.0)
    p.add_argument("--jitter", type=float, default=1e-6)
    p.add_argument("--n-restarts", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--maxfun", type=int, default=0)
    p.add_argument("--maxls", type=int, default=20)
    p.add_argument("--maxcor", type=int, default=20)
    p.add_argument("--optimizer-method", default="L-BFGS-B")
    p.add_argument("--outlier-whitened-threshold", type=float, default=0.0)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    return p


def sim_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    year_dir = Path(args.sim_root) / f"{int(args.year)}_july_st_circulant"
    prefix = f"sim_july{int(args.year)}_st_circulant"
    data_path = year_dir / f"{prefix}_{args.sim_kind}.pkl"
    truth_path = year_dir / f"{prefix}_truth.json"
    if not data_path.exists():
        raise SystemExit(f"Simulation pickle not found: {data_path}")
    if not truth_path.exists():
        raise SystemExit(f"Simulation truth JSON not found: {truth_path}")
    return data_path, truth_path


def fit_paths(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = Path(args.out_root) / "vecchia_cluster_4x4_cond2_tiles_2x4"
    monthly_dir = Path(args.out_root) / "monthly_output" / "vecchia_cluster_4x4_cond2_tiles_2x4"
    return {
        "script": PURE_SPACE_DIR / "fit_july2024_bessel_smooth_vecchia_cluster_4x4_cond2_tiles_2x4.py",
        "out_dir": out_dir,
        "monthly_dir": monthly_dir,
        "manifest": out_dir / "manifest_hours.csv",
    }


def common_args(args: argparse.Namespace, data_path: Path, paths: dict[str, Path]) -> list[str]:
    month = args.month or f"{int(args.year)}-07"
    if args.sim_kind == "real_locations":
        x_col = "Source_Longitude"
        y_col = "Source_Latitude"
    else:
        x_col = "Longitude"
        y_col = "Latitude"
    return [
        "--input", str(data_path),
        "--output-dir", str(paths["out_dir"]),
        "--monthly-output-dir", str(paths["monthly_dir"]),
        "--manifest", str(paths["manifest"]),
        "--month", month,
        "--max-hours", str(int(args.max_hours)),
        "--expected-hours", str(int(args.expected_hours)),
        "--time-col", "hour",
        "--x-col", x_col,
        "--y-col", y_col,
        "--value-col", "ColumnAmountO3",
        "--coords", "raw",
        f"--lat-range={args.lat_range}",
        f"--lon-range={args.lon_range}",
        "--tile-y", str(int(args.tile_y)),
        "--tile-x", str(int(args.tile_x)),
    ]


def fit_args(args: argparse.Namespace, hour_idx: int) -> list[str]:
    return [
        "--array-index", str(int(hour_idx)),
        "--min-tile-points", str(int(args.min_tile_points)),
        "--tile-max-points", str(int(args.tile_max_points)),
        "--tile-workers", str(int(args.tile_workers)),
        "--cluster-block-shape", str(args.cluster_block_shape),
        "--cluster-neighbor-blocks", str(int(args.cluster_neighbor_blocks)),
        "--target-chunk-size", str(int(args.target_chunk_size)),
        "--min-target-points", str(int(args.min_target_points)),
        "--nugget-mode", str(args.nugget_mode),
        "--mean-design", str(args.mean_design),
        "--range-lat-init", str(float(args.range_lat_init)),
        "--range-lon-init", str(float(args.range_lon_init)),
        "--smooth-init", str(float(args.smooth_init)),
        "--smooth-min", str(float(args.smooth_min)),
        "--smooth-max", str(float(args.smooth_max)),
        "--range-min", str(float(args.range_min)),
        "--range-max", str(float(args.range_max)),
        "--jitter", str(float(args.jitter)),
        "--n-restarts", str(int(args.n_restarts)),
        "--maxiter", str(int(args.maxiter)),
        "--maxfun", str(int(args.maxfun)),
        "--maxls", str(int(args.maxls)),
        "--maxcor", str(int(args.maxcor)),
        "--optimizer-method", str(args.optimizer_method),
        "--outlier-whitened-threshold", str(float(args.outlier_whitened_threshold)),
    ]


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_base(base_mode: str, args: argparse.Namespace, data_path: Path, extra: list[str] | None = None) -> None:
    paths = fit_paths(args)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    paths["monthly_dir"].mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        str(paths["script"]),
        "--mode", base_mode,
        *common_args(args, data_path, paths),
    ]
    if extra:
        cmd.extend(extra)
    run_cmd(cmd, dry_run=bool(args.dry_run))


def write_run_metadata(args: argparse.Namespace, data_path: Path, truth_path: Path) -> None:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    meta = {
        "purpose": "pure-space Vecchia smooth estimation on smooth=0.3, nugget=0 simulation data",
        "simulation_data": str(data_path),
        "truth_json": str(truth_path),
        "truth": truth,
        "max_hours": int(args.max_hours),
        "nugget_mode": str(args.nugget_mode),
        "smooth_init": float(args.smooth_init),
        "outlier_whitened_threshold": float(args.outlier_whitened_threshold),
        "tile_grid": f"{int(args.tile_y)}x{int(args.tile_x)}",
        "cluster_block_shape": str(args.cluster_block_shape),
        "cluster_neighbor_blocks": int(args.cluster_neighbor_blocks),
    }
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    data_path, truth_path = sim_paths(args)
    write_run_metadata(args, data_path, truth_path)
    hour_end = args.hour_end if args.hour_end is not None else int(args.max_hours)

    if args.mode in {"manifest", "all"}:
        run_base("manifest", args, data_path)

    if args.mode == "fit":
        if args.array_index is None:
            raise SystemExit("--array-index is required for --mode fit")
        run_base("fit", args, data_path, fit_args(args, int(args.array_index)))

    if args.mode in {"run-hours", "all"}:
        for hour_idx in range(int(args.hour_start), int(hour_end)):
            run_base("fit", args, data_path, fit_args(args, int(hour_idx)))

    if args.mode in {"summarize", "all"}:
        run_base("summarize", args, data_path, ["--nugget-mode", str(args.nugget_mode)])


if __name__ == "__main__":
    main()
