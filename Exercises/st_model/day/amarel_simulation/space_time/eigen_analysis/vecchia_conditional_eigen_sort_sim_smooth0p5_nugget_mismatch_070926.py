#!/usr/bin/env python3
"""One-day smooth/nugget mismatch simulation for conditional eigen diagnostics.

This driver keeps the production conditional-eigen engine unchanged.  It
patches in four fixed-shape Matern variants and, only when needed, generates a
single smooth=0.5, nugget=0 July ST simulated day using the reusable circulant
generator.

Experiments:
  smoothness_mismatch:
    DGP smooth=0.5 nugget=0, fit smooth=0.5/0.3/1.0 with nugget fixed 0.
  nugget_mismatch:
    DGP smooth=0.5 nugget=0, fit smooth=0.5 with nugget fixed 0 and 2.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPACE_TIME_DIR = HERE.parent
for path in [HERE, SPACE_TIME_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import vecchia_conditional_eigen_sort_common_engine_061926 as sim_eig  # noqa: E402

from GEMS_TCO.vecchia_st_spline import (  # noqa: E402
    RealDataCorridorWidth4x4Lag643NoNuggetSplineFit,
)


RUN_STEM = "sim_july_st_s05_n0_vecchia_conditional_eigen_sort_smooth_nugget_mismatch_070926"
DTYPE = sim_eig.DTYPE
LOSS_DECIMALS = sim_eig.LOSS_DECIMALS
REFERENCE_ADVEC_LON_ABS = sim_eig.REFERENCE_ADVEC_LON_ABS

TRUE_INIT_PHYSICAL = {
    "sigmasq": 10.0,
    "range_lat": 0.2,
    "range_lon": 0.3,
    "range_time": 2.0,
    "advec_lat": 0.08,
    "advec_lon": -0.2,
    "nugget": 0.0,
}

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "matern_s05_n0_true": {
        "family": "matern",
        "smooth": 0.5,
        "fixed_nugget": 0.0,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.5 nugget0 true",
        "color": "#1f77b4",
    },
    "matern_s03_n0_rough": {
        "family": "matern",
        "smooth": 0.3,
        "fixed_nugget": 0.0,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.3 nugget0 rough",
        "color": "#d62728",
    },
    "matern_s10_n0_smooth": {
        "family": "matern",
        "smooth": 1.0,
        "fixed_nugget": 0.0,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=1.0 nugget0 smooth",
        "color": "#2ca02c",
    },
    "matern_s05_n2_over": {
        "family": "matern",
        "smooth": 0.5,
        "fixed_nugget": 2.0,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.5 nugget2 over",
        "color": "#ff7f0e",
    },
}

EXPERIMENT_VARIANTS = {
    "smoothness_mismatch": [
        "matern_s05_n0_true",
        "matern_s03_n0_rough",
        "matern_s10_n0_smooth",
    ],
    "nugget_mismatch": [
        "matern_s05_n0_true",
        "matern_s05_n2_over",
    ],
}


class RealDataCorridorWidth4x4Lag643FixedNuggetSplineFit(RealDataCorridorWidth4x4Lag643NoNuggetSplineFit):
    """Lag-643 Matern spline model with nugget fixed to any nonnegative value."""

    def __init__(self, *args, fixed_nugget: float = 0.0, **kwargs):
        self.fixed_nugget = float(fixed_nugget)
        if self.fixed_nugget < 0.0:
            raise ValueError(f"fixed_nugget must be nonnegative, got {fixed_nugget}")
        super().__init__(*args, **kwargs)

    def _nugget_from_params(self, params):
        return params.new_tensor(self.fixed_nugget)

    def _convert_params(self, raw):
        out = super()._convert_params(raw)
        out["nugget"] = float(self.fixed_nugget)
        return out


_ORIG_BUILD_MODEL = sim_eig.build_model
_ORIG_FIT_ONE_MODEL = sim_eig.fit_one_model


def build_model(spec: dict[str, Any], source_map: dict, grid_coords_np: np.ndarray, args: argparse.Namespace):
    family = str(spec.get("family", ""))
    if family == "matern":
        return RealDataCorridorWidth4x4Lag643FixedNuggetSplineFit(
            smooth=float(spec["smooth"]),
            fixed_nugget=float(spec.get("fixed_nugget", 0.0)),
            input_map=source_map,
            grid_coords=grid_coords_np,
            lag1_lon_offset=float(args.real_reference_advec_lon_abs),
            daily_stride=int(args.daily_stride),
            target_chunk_size=int(args.target_chunk_size),
            min_target_points=int(args.min_target_points),
            spline_n_points=int(args.spline_n_points),
            spline_r_max=float(args.spline_r_max),
        )
    return _ORIG_BUILD_MODEL(spec, source_map, grid_coords_np, args)


def fit_one_model(asset: sim_eig.DayAsset, model_variant: str, device, args: argparse.Namespace):
    row, model, beta = _ORIG_FIT_ONE_MODEL(asset, model_variant, device, args)
    fixed_nugget = float(MODEL_SPECS[model_variant].get("fixed_nugget", 0.0))
    row["fixed_nugget"] = fixed_nugget
    row["nugget_mode"] = f"fixed_{fixed_nugget:g}"
    row["est_nugget"] = fixed_nugget
    return row, model, beta


def install_engine_patches() -> None:
    sim_eig.RUN_STEM = RUN_STEM
    sim_eig.MODEL_SPECS = MODEL_SPECS
    sim_eig.TRUE_INIT_PHYSICAL = TRUE_INIT_PHYSICAL
    sim_eig.build_model = build_model
    sim_eig.fit_one_model = fit_one_model


def parse_day_idxs(text: str) -> list[int]:
    text = str(text).strip().lower()
    if text == "all":
        return list(range(31))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end))
    return [int(p) for p in parts]


def required_hours(days: str, hours_per_day: int) -> int:
    idxs = parse_day_idxs(days)
    if not idxs:
        raise ValueError(f"No day indices parsed from --days={days!r}")
    return (max(idxs) + 1) * int(hours_per_day)


def simulated_pickle_path(data_root: Path, year: int, sim_kind: str) -> Path:
    return Path(data_root) / f"{int(year)}_july_st_circulant" / f"sim_july{int(year)}_st_circulant_{sim_kind}.pkl"


def truth_path_for(data_root: Path, year: int) -> Path:
    return Path(data_root) / f"{int(year)}_july_st_circulant" / f"sim_july{int(year)}_st_circulant_truth.json"


def truth_matches(path: Path, min_hours: int) -> bool:
    if not path.exists():
        return False
    try:
        truth = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    checks = [
        abs(float(truth.get("smooth", np.nan)) - 0.5) <= 1e-12,
        abs(float(truth.get("nugget", np.nan)) - 0.0) <= 1e-12,
        abs(float(truth.get("sigmasq", np.nan)) - TRUE_INIT_PHYSICAL["sigmasq"]) <= 1e-8,
        abs(float(truth.get("range_lat", np.nan)) - TRUE_INIT_PHYSICAL["range_lat"]) <= 1e-8,
        abs(float(truth.get("range_lon", np.nan)) - TRUE_INIT_PHYSICAL["range_lon"]) <= 1e-8,
        abs(float(truth.get("range_time", np.nan)) - TRUE_INIT_PHYSICAL["range_time"]) <= 1e-8,
        int(truth.get("n_hours", 0)) >= int(min_hours),
    ]
    return all(checks)


def data_root_ready(data_root: Path, years: list[int], sim_kind: str, min_hours: int) -> bool:
    for year in years:
        if not simulated_pickle_path(data_root, year, sim_kind).exists():
            return False
        if not truth_matches(truth_path_for(data_root, year), min_hours):
            return False
    return True


def candidate_data_roots() -> list[Path]:
    if Path("/home/jl2815").exists():
        base = Path("/home/jl2815/tco/exercise_output/sim_data")
        return [
            base / "july_st_circulant_realpattern_smooth0p5_nugget0_oneday_070926",
            base / "july_st_circulant_realpattern_smooth0p5_nugget0",
            base / "july_st_circulant_realpattern_smooth0p5",
        ]
    base = Path("/Users/joonwonlee/Documents/GEMS_DATA/simulation")
    return [
        base / "july_st_circulant_realpattern_smooth0p5_nugget0_oneday_070926",
        base / "july_st_circulant_realpattern_smooth0p5_nugget0",
        base / "july_st_circulant_realpattern_smooth0p5",
    ]


def resolve_data_root(args: argparse.Namespace, years: list[int], min_hours: int) -> Path:
    if args.data_root is not None:
        return Path(args.data_root)
    for root in candidate_data_roots():
        if data_root_ready(root, years, str(args.sim_kind), min_hours):
            return root
    return candidate_data_roots()[0]


def default_generator_script() -> Path:
    if Path("/home/jl2815").exists():
        return Path("/home/jl2815/tco/simulate_data/generate_july_st_circulant_real_locations_2022_2025.py")
    return Path("/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data/generate_july_st_circulant_real_locations_2022_2025.py")


def default_generator_input_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path("/home/jl2815/tco/data")
    return Path("/Users/joonwonlee/Documents/GEMS_DATA")


def default_output_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path(f"/home/jl2815/tco/exercise_output/summer/{RUN_STEM}")
    return Path(f"/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/{RUN_STEM}")


def run_generator(args: argparse.Namespace, data_root: Path, years: list[int], max_hours: int) -> None:
    generator = Path(args.generator_script)
    if not generator.exists():
        raise FileNotFoundError(f"Generator script not found: {generator}")
    cmd = [
        sys.executable,
        str(generator),
        "--years",
        ",".join(str(y) for y in years),
        "--input-root",
        str(args.generator_input_root),
        "--output-dir",
        str(data_root),
        "--max-hours",
        str(max_hours),
        "--hours-per-day",
        str(args.hours_per_day),
        "--seed",
        str(args.seed),
        "--smooth",
        "0.5",
        "--sigmasq",
        str(TRUE_INIT_PHYSICAL["sigmasq"]),
        "--range-lat",
        str(TRUE_INIT_PHYSICAL["range_lat"]),
        "--range-lon",
        str(TRUE_INIT_PHYSICAL["range_lon"]),
        "--range-time",
        str(TRUE_INIT_PHYSICAL["range_time"]),
        "--advec-lat",
        str(TRUE_INIT_PHYSICAL["advec_lat"]),
        "--advec-lon",
        str(TRUE_INIT_PHYSICAL["advec_lon"]),
        "--nugget",
        "0.0",
        "--mean-intercept",
        str(args.mean_intercept),
        "--mean-lat-slope",
        str(args.mean_lat_slope),
        "--mean-lat-center",
        str(args.mean_lat_center),
        "--lat-range",
        str(args.lat_range),
        "--lon-range",
        str(args.lon_range),
        "--lat-factor-hr",
        str(args.lat_factor_hr),
        "--lon-factor-hr",
        str(args.lon_factor_hr),
        "--hr-pad",
        str(args.hr_pad),
    ]
    print("Generating one smooth=0.5 nugget=0 simulated asset:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_data(args: argparse.Namespace, data_root: Path, years: list[int], min_hours: int) -> None:
    if data_root_ready(data_root, years, str(args.sim_kind), min_hours):
        print(f"Reusing validated smooth=0.5 nugget=0 simulation data: {data_root}", flush=True)
        return
    if not args.generate_if_missing:
        raise FileNotFoundError(
            f"No validated smooth=0.5 nugget=0 {args.sim_kind} data found at {data_root}. "
            "Pass --generate-if-missing to create the one-day asset."
        )
    data_root.mkdir(parents=True, exist_ok=True)
    run_generator(args, data_root, years, min_hours)
    if not data_root_ready(data_root, years, str(args.sim_kind), min_hours):
        raise RuntimeError(f"Generator completed, but data validation still failed for {data_root}")


def common_cli_args(args: argparse.Namespace, data_root: Path, out_dir: Path, variants: list[str]) -> list[str]:
    cli = [
        "--data-root",
        str(data_root),
        "--sim-kind",
        str(args.sim_kind),
        "--years",
        *[str(y) for y in args.years],
        "--month",
        str(args.month),
        "--days",
        str(args.days),
        "--hours-per-day",
        str(args.hours_per_day),
        "--lat-range",
        str(args.lat_range),
        "--lon-range",
        str(args.lon_range),
        "--model-variants",
        *variants,
        "--real-reference-advec-lon-abs",
        str(args.real_reference_advec_lon_abs),
        "--daily-stride",
        str(args.daily_stride),
        "--target-chunk-size",
        str(args.target_chunk_size),
        "--diag-chunk-size",
        str(args.diag_chunk_size),
        "--min-target-points",
        str(args.min_target_points),
        "--spline-n-points",
        str(args.spline_n_points),
        "--spline-r-max",
        str(args.spline_r_max),
        "--lbfgs-lr",
        str(args.lbfgs_lr),
        "--lbfgs-steps",
        str(args.lbfgs_steps),
        "--lbfgs-eval",
        str(args.lbfgs_eval),
        "--lbfgs-history",
        str(args.lbfgs_history),
        "--grad-tol",
        str(args.grad_tol),
        "--cuda-fallback",
        str(args.cuda_fallback),
        "--brown-bridge-q",
        str(args.brown_bridge_q),
        "--resample-grid",
        str(args.resample_grid),
        "--out-dir",
        str(out_dir),
    ]
    if args.device:
        cli.extend(["--device", str(args.device)])
    if args.keep_exact_loc:
        cli.append("--keep-exact-loc")
    else:
        cli.append("--no-keep-exact-loc")
    if args.save_daily_curves:
        cli.append("--save-daily-curves")
    if args.suppress_fit_prints:
        cli.append("--suppress-fit-prints")
    return cli


def run_experiment(args: argparse.Namespace, data_root: Path, out_root: Path, experiment: str) -> None:
    variants = EXPERIMENT_VARIANTS[experiment]
    exp_stem = f"{RUN_STEM}_{experiment}"
    exp_out = out_root / experiment
    exp_out.mkdir(parents=True, exist_ok=True)
    sim_eig.RUN_STEM = exp_stem
    sim_eig.MODEL_SPECS = MODEL_SPECS
    print("\n" + "#" * 100, flush=True)
    print(f"Running {experiment}: variants={variants}", flush=True)
    print(f"Output: {exp_out}", flush=True)
    print("#" * 100, flush=True)

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(Path(sim_eig.__file__).resolve())] + common_cli_args(args, data_root, exp_out, variants)
        sim_eig.main()
    finally:
        sys.argv = old_argv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smooth=0.5 nugget=0 one-day ST Vecchia conditional eigen mismatch simulation.")
    parser.add_argument("--experiment", choices=["smoothness_mismatch", "nugget_mismatch", "both"], default="both")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--generator-script", type=Path, default=default_generator_script())
    parser.add_argument("--generator-input-root", type=Path, default=default_generator_input_root())
    parser.add_argument("--generate-if-missing", action="store_true")
    parser.add_argument("--out-root", type=Path, default=default_output_root())
    parser.add_argument("--years", nargs="+", default=["2023"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0", help="Default is one day: day_idx 0 only.")
    parser.add_argument("--hours-per-day", type=int, default=8)
    parser.add_argument("--sim-kind", choices=["gridded", "real_locations"], default="gridded")
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--real-reference-advec-lon-abs", type=float, default=REFERENCE_ADVEC_LON_ABS)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=32)
    parser.add_argument("--diag-chunk-size", type=int, default=64)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--spline-n-points", type=int, default=4000)
    parser.add_argument("--spline-r-max", type=float, default=30.0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=8)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--brown-bridge-q", type=float, default=sim_eig.BROWN_BRIDGE_Q95)
    parser.add_argument("--resample-grid", type=int, default=200)
    parser.add_argument("--save-daily-curves", action="store_true")
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--seed", type=int, default=20240701)
    parser.add_argument("--mean-intercept", type=float, default=260.0)
    parser.add_argument("--mean-lat-slope", type=float, default=1.0)
    parser.add_argument("--mean-lat-center", type=float, default=-0.5)
    parser.add_argument("--lat-factor-hr", type=int, default=100)
    parser.add_argument("--lon-factor-hr", type=int, default=10)
    parser.add_argument("--hr-pad", type=float, default=0.1)
    return parser


def main() -> None:
    install_engine_patches()
    args = build_arg_parser().parse_args()
    args.years = [int(y) for y in sim_eig.parse_tokens(args.years)]
    min_hours = required_hours(str(args.days), int(args.hours_per_day))
    data_root = resolve_data_root(args, args.years, min_hours)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ensure_data(args, data_root, args.years, min_hours)

    manifest = {
        "run_stem": RUN_STEM,
        "driver_script": str(Path(__file__).resolve()),
        "common_engine": str(Path(sim_eig.__file__).resolve()),
        "data_root": str(data_root),
        "out_root": str(out_root),
        "truth": TRUE_INIT_PHYSICAL | {"smooth": 0.5},
        "model_specs": sim_eig.clean_json_value(MODEL_SPECS),
        "experiments": EXPERIMENT_VARIANTS,
        "args": sim_eig.clean_json_value(vars(args)),
        "loss_label": f"Vecchia objective per target observation, printed to {LOSS_DECIMALS} decimals",
    }
    (out_root / "experiment_manifest.json").write_text(
        json.dumps(sim_eig.clean_json_value(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    experiments = list(EXPERIMENT_VARIANTS) if args.experiment == "both" else [str(args.experiment)]
    for experiment in experiments:
        run_experiment(args, data_root, out_root, experiment)

    print("\nAll requested mismatch experiments completed.", flush=True)


if __name__ == "__main__":
    main()
