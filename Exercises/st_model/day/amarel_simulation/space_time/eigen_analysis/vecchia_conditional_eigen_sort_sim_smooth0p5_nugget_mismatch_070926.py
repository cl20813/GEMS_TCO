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
import contextlib
import gc
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


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
    "matern_s05_n0_under": {
        "family": "matern",
        "smooth": 0.5,
        "fixed_nugget": 0.0,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.5 nugget0 under",
        "color": "#1f77b4",
    },
    "matern_s05_n1_true": {
        "family": "matern",
        "smooth": 0.5,
        "fixed_nugget": 1.0,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.5 nugget1 true",
        "color": "#2ca02c",
    },
    "matern_s03_n1_rough": {
        "family": "matern",
        "smooth": 0.3,
        "fixed_nugget": 1.0,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.3 nugget1 rough",
        "color": "#d62728",
    },
    "matern_s10_n1_smooth": {
        "family": "matern",
        "smooth": 1.0,
        "fixed_nugget": 1.0,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=1.0 nugget1 smooth",
        "color": "#9467bd",
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
    "nugget_mismatch_dgp1": [
        "matern_s05_n0_under",
        "matern_s05_n1_true",
        "matern_s05_n2_over",
    ],
    "smoothness_mismatch_dgp1": [
        "matern_s05_n1_true",
        "matern_s03_n1_rough",
        "matern_s10_n1_smooth",
    ],
}

DEFAULT_BOTH_EXPERIMENTS = ["smoothness_mismatch", "nugget_mismatch"]
DGP1_BOTH_EXPERIMENTS = ["smoothness_mismatch_dgp1", "nugget_mismatch_dgp1"]


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


def build_model(spec: dict[str, Any], source_map: dict, grid_coords_np: np.ndarray, args: argparse.Namespace):
    family = str(spec.get("family", ""))
    if family == "matern":
        fixed_nugget = float(spec.get("fixed_nugget", 0.0))
        if abs(fixed_nugget) <= 1e-15:
            return _ORIG_BUILD_MODEL(spec, source_map, grid_coords_np, args)
        return RealDataCorridorWidth4x4Lag643FixedNuggetSplineFit(
            smooth=float(spec["smooth"]),
            fixed_nugget=fixed_nugget,
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


def fit_one_model(asset: sim_eig.DayAsset, model_variant: str, device, args: argparse.Namespace, compute_beta: bool = True):
    spec = MODEL_SPECS[model_variant]
    fixed_nugget = float(spec.get("fixed_nugget", 0.0))
    source_map = {
        k: v.to(device=device, dtype=DTYPE, non_blocking=True).contiguous()
        for k, v in asset.source_map.items()
    }
    model = build_model(spec, source_map, asset.grid_coords_np, args)
    params_list = sim_eig.make_params_list(TRUE_INIT_PHYSICAL, dtype=DTYPE, device=device)

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
        with contextlib.redirect_stdout(io.StringIO()):
            out, steps_ran = model.fit_vecc_lbfgs(
                params_list,
                optimizer,
                max_steps=int(args.lbfgs_steps),
                grad_tol=float(args.grad_tol),
            )
    else:
        out, steps_ran = model.fit_vecc_lbfgs(
            params_list,
            optimizer,
            max_steps=int(args.lbfgs_steps),
            grad_tol=float(args.grad_tol),
        )
    fit_s = time.time() - t1

    raw_params = [float(x) for x in out[:6]]
    fit_loss_per_obs = float(out[-1])
    est = sim_eig.backmap_params(raw_params)
    est["nugget"] = fixed_nugget
    cluster_summary = model.cluster_summary()

    # Match the lighter 061826 fit script: after LBFGS, do not keep optimizer
    # state or autograd-connected Parameter objects while computing beta/eigen.
    for param in params_list:
        param.grad = None
    del optimizer, params_list
    gc.collect()
    if device_type(device) == "cuda":
        torch.cuda.empty_cache()

    beta = None
    if compute_beta:
        with torch.no_grad():
            params_tensor = torch.as_tensor(raw_params, device=device, dtype=DTYPE)
            beta = model.get_gls_beta(params_tensor).detach()

    row = {
        "status": "ok",
        "error": "",
        "year": int(asset.year),
        "month": int(asset.month),
        "day_idx": int(asset.day_idx),
        "day": str(asset.day_label),
        "model_variant": str(model_variant),
        "model_family": str(spec["family"]),
        "model_label": str(spec["label"]),
        "smooth": float(spec["smooth"]) if pd.notna(spec["smooth"]) else np.nan,
        "gc_alpha": float(spec["gc_alpha"]) if pd.notna(spec["gc_alpha"]) else np.nan,
        "gc_beta": float(spec["gc_beta"]) if pd.notna(spec["gc_beta"]) else np.nan,
        "nugget_mode": f"fixed_{fixed_nugget:g}",
        "fixed_nugget": fixed_nugget,
        "n_rows_per_time": int(asset.n_rows_per_time),
        "n_valid_o3": int(asset.n_valid),
        "n_total_rows": int(asset.n_total),
        "valid_rate": float(asset.n_valid / asset.n_total) if asset.n_total else np.nan,
        "monthly_mean": float(asset.monthly_mean),
        "first_key": asset.keys[0],
        "last_key": asset.keys[-1],
        "spec_name": sim_eig.VECCHIA_SPEC_NAME,
        "block_shape": f"{sim_eig.BLOCK_SHAPE[0]}x{sim_eig.BLOCK_SHAPE[1]}",
        "lag_pattern": f"{sim_eig.LAG_COUNTS[0]}/{sim_eig.LAG_COUNTS[1]}/{sim_eig.LAG_COUNTS[2]}",
        "reference_advec_lon_abs": float(args.real_reference_advec_lon_abs),
        "vecchia_loss_per_obs": fit_loss_per_obs,
        "steps_raw": int(steps_ran),
        "precompute_s": float(precompute_s),
        "fit_s": float(fit_s),
        "total_fit_s": float(precompute_s + fit_s),
        **{f"est_{k}": float(est[k]) for k in ["sigmasq", "range_lat", "range_lon", "range_time", "advec_lat", "advec_lon", "nugget"]},
        **cluster_summary,
    }
    row["est_nugget"] = fixed_nugget
    return row, model, beta


def install_engine_patches() -> None:
    sim_eig.RUN_STEM = RUN_STEM
    sim_eig.MODEL_SPECS = MODEL_SPECS
    sim_eig.TRUE_INIT_PHYSICAL = TRUE_INIT_PHYSICAL
    sim_eig.build_model = build_model
    sim_eig.fit_one_model = fit_one_model


def device_type(device) -> str:
    return str(getattr(device, "type", str(device).split(":")[0]))


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


def truth_matches(path: Path, min_hours: int, expected_nugget: float = 0.0) -> bool:
    if not path.exists():
        return False
    try:
        truth = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    checks = [
        abs(float(truth.get("smooth", np.nan)) - 0.5) <= 1e-12,
        abs(float(truth.get("nugget", np.nan)) - float(expected_nugget)) <= 1e-12,
        abs(float(truth.get("sigmasq", np.nan)) - TRUE_INIT_PHYSICAL["sigmasq"]) <= 1e-8,
        abs(float(truth.get("range_lat", np.nan)) - TRUE_INIT_PHYSICAL["range_lat"]) <= 1e-8,
        abs(float(truth.get("range_lon", np.nan)) - TRUE_INIT_PHYSICAL["range_lon"]) <= 1e-8,
        abs(float(truth.get("range_time", np.nan)) - TRUE_INIT_PHYSICAL["range_time"]) <= 1e-8,
        int(truth.get("n_hours", 0)) >= int(min_hours),
    ]
    return all(checks)


def data_root_ready(data_root: Path, years: list[int], sim_kind: str, min_hours: int, expected_nugget: float = 0.0) -> bool:
    for year in years:
        if not simulated_pickle_path(data_root, year, sim_kind).exists():
            return False
        if not truth_matches(truth_path_for(data_root, year), min_hours, expected_nugget=expected_nugget):
            return False
    return True


def candidate_data_roots() -> list[Path]:
    if Path("/home/jl2815").exists():
        base = Path("/home/jl2815/tco/exercise_output/sim_data")
        return [
            base / "july_st_circulant_realpattern_smooth0p5_nugget0_oneday_070926",
            base / "july_st_circulant_realpattern_smooth0p5_nugget0",
            base / "july_st_circulant_realpattern_smooth0p5",
            base / "july_st_circulant_realpattern",
        ]
    base = Path("/Users/joonwonlee/Documents/GEMS_DATA/simulation")
    return [
        base / "july_st_circulant_realpattern_smooth0p5_nugget0_oneday_070926",
        base / "july_st_circulant_realpattern_smooth0p5_nugget0",
        base / "july_st_circulant_realpattern_smooth0p5",
        base / "july_st_circulant_realpattern",
    ]


def resolve_data_root(args: argparse.Namespace, years: list[int], min_hours: int) -> Path:
    if args.data_root is not None:
        return Path(args.data_root)
    for root in candidate_data_roots():
        if data_root_ready(root, years, str(args.sim_kind), min_hours, expected_nugget=float(args.truth_nugget)):
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
        f"--advec-lon={TRUE_INIT_PHYSICAL['advec_lon']}",
        "--nugget",
        str(float(args.truth_nugget)),
        "--mean-intercept",
        str(args.mean_intercept),
        "--mean-lat-slope",
        str(args.mean_lat_slope),
        f"--mean-lat-center={args.mean_lat_center}",
        f"--lat-range={args.lat_range}",
        f"--lon-range={args.lon_range}",
        "--lat-factor-hr",
        str(args.lat_factor_hr),
        "--lon-factor-hr",
        str(args.lon_factor_hr),
        "--hr-pad",
        str(args.hr_pad),
    ]
    print(f"Generating one smooth=0.5 nugget={float(args.truth_nugget):g} simulated asset:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_data(args: argparse.Namespace, data_root: Path, years: list[int], min_hours: int) -> None:
    expected_nugget = float(args.truth_nugget)
    if data_root_ready(data_root, years, str(args.sim_kind), min_hours, expected_nugget=expected_nugget):
        print(f"Reusing validated smooth=0.5 nugget={expected_nugget:g} simulation data: {data_root}", flush=True)
        return
    if not args.generate_if_missing:
        raise FileNotFoundError(
            f"No validated smooth=0.5 nugget={expected_nugget:g} {args.sim_kind} data found at {data_root}. "
            "Pass --generate-if-missing to create the asset."
        )
    data_root.mkdir(parents=True, exist_ok=True)
    run_generator(args, data_root, years, min_hours)
    if not data_root_ready(data_root, years, str(args.sim_kind), min_hours, expected_nugget=expected_nugget):
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
        f"--lat-range={args.lat_range}",
        f"--lon-range={args.lon_range}",
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


def run_experiment(
    args: argparse.Namespace,
    data_root: Path,
    out_root: Path,
    experiment: str,
    variants: list[str] | None = None,
    run_stem_suffix: str | None = None,
) -> None:
    variants = list(variants or EXPERIMENT_VARIANTS[experiment])
    exp_stem = f"{RUN_STEM}_{experiment}"
    if run_stem_suffix:
        exp_stem = f"{exp_stem}_{run_stem_suffix}"
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


def wrapper_worker_cli(args: argparse.Namespace, data_root: Path, out_root: Path, experiment: str, model_variant: str) -> list[str]:
    return wrapper_stage_worker_cli(args, data_root, out_root, experiment, model_variant, "full")


def wrapper_stage_worker_cli(
    args: argparse.Namespace,
    data_root: Path,
    out_root: Path,
    experiment: str,
    model_variant: str,
    worker_stage: str,
) -> list[str]:
    cli = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-experiment",
        str(experiment),
        "--worker-model-variant",
        str(model_variant),
        "--worker-stage",
        str(worker_stage),
        "--data-root",
        str(data_root),
        "--out-root",
        str(out_root),
        "--years",
        *[str(y) for y in args.years],
        "--month",
        str(args.month),
        "--days",
        str(args.days),
        "--hours-per-day",
        str(args.hours_per_day),
        "--truth-nugget",
        str(args.truth_nugget),
        "--sim-kind",
        str(args.sim_kind),
        f"--lat-range={args.lat_range}",
        f"--lon-range={args.lon_range}",
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
        "--save-daily-curves",
    ]
    if args.device:
        cli.extend(["--device", str(args.device)])
    if args.keep_exact_loc:
        cli.append("--keep-exact-loc")
    else:
        cli.append("--no-keep-exact-loc")
    if args.suppress_fit_prints:
        cli.append("--suppress-fit-prints")
    return cli


def combined_summary_path(out_root: Path, experiment: str) -> Path:
    return out_root / experiment / f"{RUN_STEM}_{experiment}_summary.csv"


def worker_summary_path(out_root: Path, experiment: str, model_variant: str) -> Path:
    return out_root / experiment / f"{RUN_STEM}_{experiment}_{model_variant}_summary.csv"


def fit_artifact_path(out_root: Path, experiment: str, model_variant: str) -> Path:
    return out_root / experiment / "fit_artifacts" / f"{RUN_STEM}_{experiment}_{model_variant}_fit.csv"


def write_rows(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(sim_eig.ROUND_DECIMALS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=f"%.{sim_eig.ROUND_DECIMALS}f")
    return df


def fitted_params_from_row(row: pd.Series | dict[str, Any]) -> list[float]:
    return sim_eig.physical_to_log_phi(
        {
            "sigmasq": float(row["est_sigmasq"]),
            "range_lat": float(row["est_range_lat"]),
            "range_lon": float(row["est_range_lon"]),
            "range_time": float(row["est_range_time"]),
            "advec_lat": float(row["est_advec_lat"]),
            "advec_lon": float(row["est_advec_lon"]),
            "nugget": float(row.get("est_nugget", 0.0)),
        }
    )


def run_fit_only_worker(
    args: argparse.Namespace,
    data_root: Path,
    out_root: Path,
    experiment: str,
    model_variant: str,
) -> None:
    args.data_root = Path(data_root)
    exp_out = out_root / experiment
    exp_out.mkdir(parents=True, exist_ok=True)
    device = sim_eig.resolve_device(args)
    print(f"FIT-ONLY worker: {experiment}/{model_variant} device={device}", flush=True)
    rows: list[dict[str, Any]] = []
    for asset in sim_eig.load_day_assets(args):
        model = None
        beta = None
        try:
            row, model, beta = fit_one_model(asset, model_variant, device, args, compute_beta=False)
            row["worker_stage"] = "fit"
            print(
                pd.Series(
                    {
                        "day": row["day"],
                        "model": row["model_label"],
                        "loss/obs": f"{row['vecchia_loss_per_obs']:.{LOSS_DECIMALS}f}",
                        "fit_s": f"{row['fit_s']:.1f}",
                    }
                ).to_string(),
                flush=True,
            )
        except Exception as exc:
            row = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": sim_eig.traceback.format_exc(limit=12),
                "year": int(asset.year),
                "month": int(asset.month),
                "day_idx": int(asset.day_idx),
                "day": str(asset.day_label),
                "model_variant": str(model_variant),
                "worker_stage": "fit",
            }
            print(f"FIT ERROR for {model_variant}: {row['error']}", flush=True)
            sim_eig.traceback.print_exc()
        finally:
            if beta is not None:
                del beta
            if model is not None:
                del model
            gc.collect()
            if device_type(device) == "cuda":
                torch.cuda.empty_cache()
        rows.append(sim_eig.clean_json_value(row))
    path = fit_artifact_path(out_root, experiment, model_variant)
    write_rows(path, rows)
    print(f"FIT-ONLY artifact: {path}", flush=True)


def run_diag_only_worker(
    args: argparse.Namespace,
    data_root: Path,
    out_root: Path,
    experiment: str,
    model_variant: str,
) -> None:
    args.data_root = Path(data_root)
    exp_out = out_root / experiment
    exp_out.mkdir(parents=True, exist_ok=True)
    fit_path = fit_artifact_path(out_root, experiment, model_variant)
    if not fit_path.exists():
        raise FileNotFoundError(f"Missing fit artifact for diagnostic worker: {fit_path}")

    fit_df = pd.read_csv(fit_path)
    device = sim_eig.resolve_device(args)
    print(f"DIAG-ONLY worker: {experiment}/{model_variant} device={device}", flush=True)
    rows: list[dict[str, Any]] = []
    for asset in sim_eig.load_day_assets(args):
        model = None
        beta = None
        curve = None
        match = fit_df[
            (pd.to_numeric(fit_df["year"], errors="coerce") == int(asset.year))
            & (pd.to_numeric(fit_df["day_idx"], errors="coerce") == int(asset.day_idx))
            & (fit_df["model_variant"].astype(str) == str(model_variant))
        ]
        if match.empty:
            row = {
                "status": "error",
                "error": f"No fit row found in {fit_path}",
                "year": int(asset.year),
                "month": int(asset.month),
                "day_idx": int(asset.day_idx),
                "day": str(asset.day_label),
                "model_variant": str(model_variant),
                "worker_stage": "diag",
            }
            rows.append(row)
            continue

        fit_row = match.iloc[0].to_dict()
        if str(fit_row.get("status", "")) != "ok":
            fit_row["worker_stage"] = "diag"
            rows.append(sim_eig.clean_json_value(fit_row))
            continue

        try:
            spec = MODEL_SPECS[model_variant]
            source_map = {
                k: v.to(device=device, dtype=DTYPE, non_blocking=True).contiguous()
                for k, v in asset.source_map.items()
            }
            model = build_model(spec, source_map, asset.grid_coords_np, args)
            model.precompute_conditioning_sets()
            with torch.no_grad():
                params = torch.as_tensor(fitted_params_from_row(fit_row), device=device, dtype=DTYPE)
                beta = model.get_gls_beta(params).detach()
                t_diag = time.time()
                curve, diag_summary = sim_eig.conditional_eigen_curve(model, params, beta, args)
                diag_s = time.time() - t_diag
            row = {**fit_row, **diag_summary, "diag_s": float(diag_s), "worker_stage": "diag", "status": "ok", "error": ""}
            print(
                pd.Series(
                    {
                        "model": row["model_label"],
                        "loss/obs": f"{row['vecchia_loss_per_obs']:.{LOSS_DECIMALS}f}",
                        "conditional_loss/score": f"{row['conditional_loss_per_score']:.{LOSS_DECIMALS}f}",
                        "mean_y2": f"{row['mean_y2']:.5f}",
                        "D": f"{row['max_abs_bridge_scaled']:.5f}",
                        "n_scores": row["n_conditional_scores"],
                        "diag_s": f"{diag_s:.1f}",
                    }
                ).to_string(),
                flush=True,
            )
            curve_path = (
                exp_out
                / "daily_curves"
                / f"year_{asset.year}"
                / f"sim_{asset.year}_day{asset.day_idx + 1:02d}_{model_variant}_conditional_eig_curve.csv"
            )
            curve_path.parent.mkdir(parents=True, exist_ok=True)
            curve.round(sim_eig.ROUND_DECIMALS).to_csv(curve_path, index=False, float_format=f"%.{sim_eig.ROUND_DECIMALS}f")
        except Exception as exc:
            row = {
                **fit_row,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": sim_eig.traceback.format_exc(limit=12),
                "year": int(asset.year),
                "month": int(asset.month),
                "day_idx": int(asset.day_idx),
                "day": str(asset.day_label),
                "model_variant": str(model_variant),
                "worker_stage": "diag",
            }
            print(f"DIAG ERROR for {model_variant}: {row['error']}", flush=True)
            sim_eig.traceback.print_exc()
        finally:
            if curve is not None:
                del curve
            if beta is not None:
                del beta
            if model is not None:
                del model
            gc.collect()
            if device_type(device) == "cuda":
                torch.cuda.empty_cache()
        rows.append(sim_eig.clean_json_value(row))

    path = worker_summary_path(out_root, experiment, model_variant)
    write_rows(path, rows)
    print(f"DIAG-ONLY summary: {path}", flush=True)


def aggregate_isolated_experiment(args: argparse.Namespace, out_root: Path, experiment: str, variants: list[str]) -> None:
    exp_out = out_root / experiment
    summary_frames: list[pd.DataFrame] = []
    for model_variant in variants:
        path = worker_summary_path(out_root, experiment, model_variant)
        if not path.exists():
            raise FileNotFoundError(f"Missing worker summary for {model_variant}: {path}")
        summary_frames.append(pd.read_csv(path))

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not summary.empty:
        numeric_cols = summary.select_dtypes(include=[np.number]).columns
        summary[numeric_cols] = summary[numeric_cols].round(sim_eig.ROUND_DECIMALS)
    combined_path = combined_summary_path(out_root, experiment)
    summary.to_csv(combined_path, index=False, float_format=f"%.{sim_eig.ROUND_DECIMALS}f")

    if summary.empty or "status" not in summary.columns:
        return
    ok_summary = summary[summary["status"].astype(str) == "ok"].copy()
    if ok_summary.empty:
        return

    avg_rows: list[pd.DataFrame] = []
    summary_rows = sim_eig.clean_json_value(ok_summary.to_dict(orient="records"))
    for (year, day_idx), day_summary in ok_summary.groupby(["year", "day_idx"], sort=True):
        year_int = int(year)
        day_idx_int = int(day_idx)
        day_label = str(day_summary["day"].iloc[0]) if "day" in day_summary.columns else f"{year_int}-07-day{day_idx_int + 1:02d}"
        curves: dict[str, pd.DataFrame] = {}
        for _, row in day_summary.iterrows():
            model_variant = str(row["model_variant"])
            curve_path = (
                exp_out
                / "daily_curves"
                / f"year_{year_int}"
                / f"sim_{year_int}_day{day_idx_int + 1:02d}_{model_variant}_conditional_eig_curve.csv"
            )
            if not curve_path.exists():
                print(f"WARNING: missing daily curve for combined plot: {curve_path}", flush=True)
                continue
            curve = pd.read_csv(curve_path)
            curves[model_variant] = curve
            avg_rows.append(
                sim_eig.resample_curve(curve, int(args.resample_grid)).assign(
                    year=year_int,
                    month=int(row["month"]) if "month" in row and pd.notna(row["month"]) else int(args.month),
                    day_idx=day_idx_int,
                    day=day_label,
                    model_variant=model_variant,
                )
            )
        if curves:
            sim_eig.MODEL_SPECS = MODEL_SPECS
            sim_eig.plot_daily_comparison(
                curves,
                sim_eig.clean_json_value(day_summary.to_dict(orient="records")),
                exp_out
                / "daily_plots"
                / f"year_{year_int}"
                / f"sim_{year_int}_day{day_idx_int + 1:02d}_vecchia_conditional_eigen_sort_comparison.png",
                f"Sim July {year_int} day_idx={day_idx_int} ({day_label}): Vecchia conditional eigen diagnostic",
            )

    for year in sorted({int(df["year"].iloc[0]) for df in avg_rows if not df.empty}):
        year_avg = [df for df in avg_rows if not df.empty and int(df["year"].iloc[0]) == year]
        year_summary = [r for r in summary_rows if int(r.get("year")) == year]
        sim_eig.MODEL_SPECS = MODEL_SPECS
        sim_eig.write_monthly_outputs(year_avg, year_summary, exp_out, year)

    print(f"Combined isolated summary: {combined_path}", flush=True)


def run_isolated_experiment(args: argparse.Namespace, data_root: Path, out_root: Path, experiment: str) -> None:
    variants = EXPERIMENT_VARIANTS[experiment]
    exp_out = out_root / experiment
    exp_out.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 100, flush=True)
    if args.split_fit_diagnostic:
        print(f"Running {experiment} with separate fit-only and diag-only Python processes per model: variants={variants}", flush=True)
    else:
        print(f"Running {experiment} with one fresh Python process per model: variants={variants}", flush=True)
    print(f"Output: {exp_out}", flush=True)
    print("#" * 100, flush=True)
    for model_variant in variants:
        if args.split_fit_diagnostic:
            stages = ["fit", "diag"]
            for stage in stages:
                cmd = wrapper_stage_worker_cli(args, data_root, out_root, experiment, model_variant, stage)
                print("\n" + "-" * 100, flush=True)
                print(f"Worker start: {experiment}/{model_variant}/{stage}", flush=True)
                print(" ".join(cmd), flush=True)
                print("-" * 100, flush=True)
                subprocess.run(cmd, check=True)
                gc.collect()
        else:
            cmd = wrapper_worker_cli(args, data_root, out_root, experiment, model_variant)
            print("\n" + "-" * 100, flush=True)
            print(f"Worker start: {experiment}/{model_variant}", flush=True)
            print(" ".join(cmd), flush=True)
            print("-" * 100, flush=True)
            subprocess.run(cmd, check=True)
            gc.collect()
    aggregate_isolated_experiment(args, out_root, experiment, variants)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smooth=0.5 nugget=0 one-day ST Vecchia conditional eigen mismatch simulation.")
    parser.add_argument("--experiment", choices=[*sorted(EXPERIMENT_VARIANTS), "both", "both_dgp1"], default="both")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--generator-script", type=Path, default=default_generator_script())
    parser.add_argument("--generator-input-root", type=Path, default=default_generator_input_root())
    parser.add_argument("--generate-if-missing", action="store_true")
    parser.add_argument("--out-root", type=Path, default=default_output_root())
    parser.add_argument("--years", nargs="+", default=["2023"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0", help="Default is one day: day_idx 0 only.")
    parser.add_argument("--hours-per-day", type=int, default=8)
    parser.add_argument("--truth-nugget", type=float, default=0.0)
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
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-cuda",
        action="store_true",
        help="Honor --device cuda. Without this flag the driver forces CPU to avoid the known 40GB GPU OOM for this one-day mismatch run.",
    )
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--brown-bridge-q", type=float, default=sim_eig.BROWN_BRIDGE_Q95)
    parser.add_argument("--resample-grid", type=int, default=200)
    parser.add_argument("--save-daily-curves", action="store_true")
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--isolate-models", dest="isolate_models", action="store_true", default=True)
    parser.add_argument("--no-isolate-models", dest="isolate_models", action="store_false")
    parser.add_argument("--split-fit-diagnostic", dest="split_fit_diagnostic", action="store_true", default=True)
    parser.add_argument("--no-split-fit-diagnostic", dest="split_fit_diagnostic", action="store_false")
    parser.add_argument("--worker-experiment", choices=sorted(EXPERIMENT_VARIANTS), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-model-variant", choices=sorted(MODEL_SPECS), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-stage", choices=["full", "fit", "diag"], default="full", help=argparse.SUPPRESS)
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
    if str(args.device).startswith("cuda") and not bool(args.allow_cuda):
        print(
            "WARNING: --device cuda was requested, but this mismatch driver forces CPU unless --allow-cuda is passed.",
            flush=True,
        )
        args.device = "cpu"
        args.cuda_fallback = "cpu"
    args.years = [int(y) for y in sim_eig.parse_tokens(args.years)]
    min_hours = required_hours(str(args.days), int(args.hours_per_day))
    data_root = resolve_data_root(args, args.years, min_hours)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ensure_data(args, data_root, args.years, min_hours)

    if args.worker_experiment is not None or args.worker_model_variant is not None:
        if args.worker_experiment is None or args.worker_model_variant is None:
            raise ValueError("--worker-experiment and --worker-model-variant must be passed together.")
        if args.worker_stage == "fit":
            run_fit_only_worker(args, data_root, out_root, str(args.worker_experiment), str(args.worker_model_variant))
        elif args.worker_stage == "diag":
            args.save_daily_curves = True
            run_diag_only_worker(args, data_root, out_root, str(args.worker_experiment), str(args.worker_model_variant))
        else:
            args.save_daily_curves = True
            run_experiment(
                args,
                data_root,
                out_root,
                str(args.worker_experiment),
                variants=[str(args.worker_model_variant)],
                run_stem_suffix=str(args.worker_model_variant),
            )
        return

    manifest = {
        "run_stem": RUN_STEM,
        "driver_script": str(Path(__file__).resolve()),
        "common_engine": str(Path(sim_eig.__file__).resolve()),
        "data_root": str(data_root),
        "out_root": str(out_root),
        "truth": TRUE_INIT_PHYSICAL | {"smooth": 0.5, "nugget": float(args.truth_nugget)},
        "model_specs": sim_eig.clean_json_value(MODEL_SPECS),
        "experiments": EXPERIMENT_VARIANTS,
        "args": sim_eig.clean_json_value(vars(args)),
        "execution": (
            "model-isolated subprocesses with separate fit-only and diagnostic-only Python processes; "
            "fit autograd/optimizer memory is released by process exit before eigen diagnostics"
            if args.isolate_models and args.split_fit_diagnostic
            else "model-isolated subprocesses; data generation is a separate subprocess and each fitted model "
            "runs in a fresh Python process before parent aggregation"
            if args.isolate_models
            else "single process per experiment"
        ),
        "loss_label": f"Vecchia objective per target observation, printed to {LOSS_DECIMALS} decimals",
    }
    (out_root / "experiment_manifest.json").write_text(
        json.dumps(sim_eig.clean_json_value(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.experiment == "both":
        experiments = list(DEFAULT_BOTH_EXPERIMENTS)
    elif args.experiment == "both_dgp1":
        experiments = list(DGP1_BOTH_EXPERIMENTS)
    else:
        experiments = [str(args.experiment)]
    for experiment in experiments:
        if args.isolate_models:
            run_isolated_experiment(args, data_root, out_root, experiment)
        else:
            run_experiment(args, data_root, out_root, experiment)

    print("\nAll requested mismatch experiments completed.", flush=True)


if __name__ == "__main__":
    main()
