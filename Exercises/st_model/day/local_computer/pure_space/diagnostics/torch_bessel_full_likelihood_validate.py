#!/usr/bin/env python3
"""Validate torch smooth-free Bessel full likelihood against SciPy.

This diagnostic does three checks on a small simulated July tile/subset:

1. Fit the existing SciPy direct-Bessel full likelihood with smooth free.
2. Evaluate the new torch direct-Bessel full likelihood at the SciPy estimate.
3. Compare torch autograd gradients against SciPy finite-difference gradients.

The torch module uses SciPy for Bessel K values and custom finite-difference
backward for the scaled-distance and smooth directions.  This is a validation
bridge, not a production GPU implementation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[6]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO.matern_bessel_anisotropic import fit_full_matern, profiled_full_nll
from GEMS_TCO.torch_bessel_full_likelihood import (
    finite_difference_grad_np,
    fit_full_matern_torch,
    raw_from_natural_np,
    torch_value_and_grad,
)


def key_sort_tuple(key: str):
    m = re.search(r"y(\d+)m(\d+)day(\d+)_hm(\d+):(\d+)", str(key))
    if m:
        return tuple(map(int, m.groups())) + (str(key),)
    return (9999, 99, 99, 99, 99, str(key))


def simulation_paths(sim_base: Path, smooth_label: str, year: int, use_real_locations: bool):
    root = sim_base / f"july_st_circulant_realpattern_{smooth_label}" / f"{year}_july_st_circulant"
    suffix = "real_locations" if use_real_locations else "gridded"
    data_path = root / f"sim_july{year}_st_circulant_{suffix}.pkl"
    truth_path = root / f"sim_july{year}_st_circulant_truth.json"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not truth_path.exists():
        raise FileNotFoundError(truth_path)
    return data_path, truth_path


def weighted_monthly_mean(sim_data: dict, col: str = "ColumnAmountO3") -> float:
    total = 0.0
    count = 0
    for df in sim_data.values():
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(arr)
        total += float(arr[ok].sum())
        count += int(ok.sum())
    return total / max(count, 1)


def prepare_hour_frame(
    df: pd.DataFrame,
    monthly_mean: float,
    n_fit: int,
    tile_y: int,
    tile_x: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    d = df[["Latitude", "Longitude", "ColumnAmountO3"]].copy()
    for col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d["Latitude"].between(-3.0, 2.0) & d["Longitude"].between(121.0, 131.0)].copy()
    d = d.groupby(["Latitude", "Longitude"], as_index=False)["ColumnAmountO3"].mean()
    lat = d["Latitude"].to_numpy(dtype=np.float64)
    lon = d["Longitude"].to_numpy(dtype=np.float64)
    y = d["ColumnAmountO3"].to_numpy(dtype=np.float64) - float(monthly_mean)
    coords = np.column_stack([lat, lon])

    y_edges = np.linspace(float(lat.min()), float(lat.max()) + 1e-12, int(tile_y) + 1)
    x_edges = np.linspace(float(lon.min()), float(lon.max()) + 1e-12, int(tile_x) + 1)
    y_idx = np.clip(np.searchsorted(y_edges, lat, side="right") - 1, 0, int(tile_y) - 1)
    x_idx = np.clip(np.searchsorted(x_edges, lon, side="right") - 1, 0, int(tile_x) - 1)
    target_tile = (int(tile_y) // 2) * int(tile_x) + (int(tile_x) // 2)
    tile_id = y_idx * int(tile_x) + x_idx
    keep = np.flatnonzero(tile_id == target_tile)
    if keep.size < max(20, int(n_fit)):
        keep = np.arange(len(y), dtype=int)

    rng = np.random.default_rng(int(seed))
    if keep.size > int(n_fit):
        keep = np.sort(rng.choice(keep, size=int(n_fit), replace=False))
    return y[keep], coords[keep]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate torch Bessel full likelihood against SciPy.")
    p.add_argument("--sim-base", default="/Users/joonwonlee/Documents/GEMS_DATA/simulation")
    p.add_argument("--smooth-label", default="smooth0p5")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--day", type=int, default=1)
    p.add_argument("--hour-slot", type=int, default=0)
    p.add_argument("--n-fit", type=int, default=250)
    p.add_argument("--tile-y", type=int, default=4)
    p.add_argument("--tile-x", type=int, default=4)
    p.add_argument("--seed", type=int, default=20240605)
    p.add_argument("--mean-design", choices=["constant", "lat", "latlon"], default="lat")
    p.add_argument("--smooth-min", type=float, default=0.05)
    p.add_argument("--smooth-max", type=float, default=2.5)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--fd-eps", type=float, default=1e-4)
    p.add_argument("--torch-max-iter", type=int, default=80)
    p.add_argument("--out-dir", default=str(Path(__file__).with_name("torch_bessel_full_likelihood_validation_outputs")))
    p.add_argument("--use-real-locations", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path, truth_path = simulation_paths(Path(args.sim_base), args.smooth_label, int(args.year), bool(args.use_real_locations))
    with data_path.open("rb") as f:
        sim_data = pickle.load(f)
    truth = json.loads(truth_path.read_text())
    keys = [k for k in sorted(sim_data.keys(), key=key_sort_tuple) if key_sort_tuple(k)[2] == int(args.day)]
    if not keys:
        raise RuntimeError(f"No keys found for day={args.day}")
    key = keys[int(args.hour_slot)]
    monthly_mean = weighted_monthly_mean(sim_data)
    y, coords = prepare_hour_frame(
        sim_data[key],
        monthly_mean=monthly_mean,
        n_fit=int(args.n_fit),
        tile_y=int(args.tile_y),
        tile_x=int(args.tile_x),
        seed=int(args.seed),
    )
    smooth_bounds = (float(args.smooth_min), float(args.smooth_max))
    var_y = max(float(np.nanvar(y, ddof=1)), 1e-8)

    t0 = time.time()
    scipy_fit = fit_full_matern(
        y=y,
        coords=coords,
        nugget_mode="free",
        fixed_nugget=0.0,
        mean_design=str(args.mean_design),
        smooth_bounds=smooth_bounds,
        range_bounds=(0.03, 5.0),
        range_lat_init=float(truth.get("range_lat", 0.35)),
        range_lon_init=float(truth.get("range_lon", 0.35)),
        smooth_init=float(truth.get("smooth", 0.5)),
        nugget_init=float(truth.get("nugget", 0.2 * var_y)),
        jitter=1e-6,
        n_restarts=int(args.n_restarts),
        maxiter=int(args.maxiter),
        maxfun=0,
        maxls=20,
        maxcor=20,
        method="L-BFGS-B",
    )
    scipy_s = time.time() - t0
    raw = np.asarray(scipy_fit["raw_params"], dtype=np.float64)
    start_truth_raw = raw_from_natural_np(
        sigmasq=float(truth.get("sigmasq", var_y)),
        range_lat=float(truth.get("range_lat", 0.35)),
        range_lon=float(truth.get("range_lon", 0.35)),
        smooth=float(truth.get("smooth", 0.5)),
        nugget=float(truth.get("nugget", 0.2 * var_y)),
        nugget_mode="free",
        smooth_bounds=smooth_bounds,
    )
    range_bounds = (0.03, 5.0)
    raw_bounds = {
        "log_phi1": (-40.0, 40.0),
        "log_phi2": (np.log(1.0 / range_bounds[1]), np.log(1.0 / range_bounds[0])),
        "log_phi3": (-8.0, 8.0),
        "smooth_raw": (-8.0, 8.0),
        "log_nugget": (np.log(max(var_y * 1e-8, 1e-10)), np.log(max(var_y * 1e3, 1e-8))),
    }

    def scipy_obj(raw_vec):
        param_bounds = {
            "sigmasq": (1e-12, max(var_y * 1e5, 1e-6)),
            "range_lat": (0.03, 5.0),
            "range_lon": (0.03, 5.0),
            "smooth": smooth_bounds,
            "nugget": (0.0, max(var_y * 1e4, 1e-6)),
        }
        return profiled_full_nll(
            raw_vec,
            y=y,
            coords=coords,
            nugget_mode="free",
            fixed_nugget=0.0,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            mean_design=str(args.mean_design),
            jitter=1e-6,
            scale_by_n=True,
        )

    t1 = time.time()
    torch_loss, torch_grad, torch_params = torch_value_and_grad(
        raw,
        y_np=y,
        coords_np=coords,
        nugget_mode="free",
        fixed_nugget=0.0,
        smooth_bounds=smooth_bounds,
        mean_design=str(args.mean_design),
        jitter=1e-6,
    )
    torch_s = time.time() - t1

    t2 = time.time()
    scipy_grad = finite_difference_grad_np(scipy_obj, raw, eps=float(args.fd_eps))
    scipy_grad_s = time.time() - t2

    torch_fit_rows = []
    for start_label, start_raw in [
        ("truth_start", start_truth_raw),
        ("scipy_solution_start", raw),
    ]:
        tt = time.time()
        fit = fit_full_matern_torch(
            y_np=y,
            coords_np=coords,
            start_raw=start_raw,
            nugget_mode="free",
            fixed_nugget=0.0,
            smooth_bounds=smooth_bounds,
            mean_design=str(args.mean_design),
            jitter=1e-6,
            max_iter=int(args.torch_max_iter),
            max_eval=int(args.torch_max_iter),
            history_size=20,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            bounds=raw_bounds,
        )
        fit["fit_s"] = float(time.time() - tt)
        fit["start_label"] = start_label
        torch_fit_rows.append(fit)

    grad_abs = np.abs(torch_grad - scipy_grad)
    grad_rel = grad_abs / np.maximum(np.abs(scipy_grad), 1e-8)
    names = ["log_phi1", "log_phi2", "log_phi3", "smooth_raw", "log_nugget"]
    grad_df = pd.DataFrame({
        "parameter": names[: len(raw)],
        "raw": raw,
        "torch_grad": torch_grad,
        "scipy_fd_grad": scipy_grad,
        "abs_diff": grad_abs,
        "rel_diff": grad_rel,
    })
    grad_path = out_dir / "torch_vs_scipy_bessel_gradient_check.csv"
    grad_df.to_csv(grad_path, index=False)

    scipy_loss = float(scipy_fit["loss"])
    value_row = {
        "key": key,
        "n_fit": int(len(y)),
        "scipy_loss": scipy_loss,
        "torch_loss": float(torch_loss),
        "loss_abs_diff": abs(float(torch_loss) - scipy_loss),
        "loss_rel_diff": abs(float(torch_loss) - scipy_loss) / max(abs(scipy_loss), 1e-8),
        "max_grad_abs_diff": float(np.nanmax(grad_abs)),
        "max_grad_rel_diff": float(np.nanmax(grad_rel)),
        "scipy_fit_s": float(scipy_s),
        "torch_value_grad_s": float(torch_s),
        "scipy_fd_grad_s": float(scipy_grad_s),
        "scipy_success": bool(scipy_fit.get("success", False)),
        "scipy_message": scipy_fit.get("message", ""),
    }
    for p in ["sigmasq", "range_lat", "range_lon", "smooth", "nugget", "phi1", "phi2", "phi3"]:
        value_row[f"scipy_{p}"] = float(scipy_fit.get(p, np.nan))
        value_row[f"torch_{p}"] = float(torch_params.get(p, np.nan))
        value_row[f"truth_{p}"] = float(truth.get(p, np.nan))
    value_path = out_dir / "torch_vs_scipy_bessel_value_check.csv"
    pd.DataFrame([value_row]).to_csv(value_path, index=False)

    fit_compare_rows = []
    for fit in torch_fit_rows:
        row = {
            "method": "torch_full_bessel_lbfgs",
            "start_label": fit["start_label"],
            "success": bool(fit.get("success", False)),
            "message": fit.get("message", ""),
            "n_eval": int(fit.get("n_eval", -1)),
            "fit_s": float(fit.get("fit_s", np.nan)),
            "loss": float(fit.get("loss", np.nan)),
            "loss_minus_scipy": float(fit.get("loss", np.nan)) - scipy_loss,
        }
        for p in ["sigmasq", "range_lat", "range_lon", "smooth", "nugget", "phi1", "phi2", "phi3"]:
            row[p] = float(fit.get(p, np.nan))
            row[f"scipy_{p}"] = float(scipy_fit.get(p, np.nan))
            row[f"diff_{p}"] = row[p] - row[f"scipy_{p}"]
            row[f"truth_{p}"] = float(truth.get(p, np.nan))
        fit_compare_rows.append(row)
    scipy_row = {
        "method": "scipy_full_bessel_lbfgsb",
        "start_label": "best_of_restarts",
        "success": bool(scipy_fit.get("success", False)),
        "message": scipy_fit.get("message", ""),
        "n_eval": int(scipy_fit.get("n_eval", -1)),
        "fit_s": float(scipy_s),
        "loss": scipy_loss,
        "loss_minus_scipy": 0.0,
    }
    for p in ["sigmasq", "range_lat", "range_lon", "smooth", "nugget", "phi1", "phi2", "phi3"]:
        scipy_row[p] = float(scipy_fit.get(p, np.nan))
        scipy_row[f"scipy_{p}"] = float(scipy_fit.get(p, np.nan))
        scipy_row[f"diff_{p}"] = 0.0
        scipy_row[f"truth_{p}"] = float(truth.get(p, np.nan))
    fit_compare = pd.DataFrame([scipy_row] + fit_compare_rows)
    fit_path = out_dir / "torch_vs_scipy_bessel_smooth_free_fit_compare.csv"
    fit_compare.to_csv(fit_path, index=False)

    print("data_path:", data_path)
    print("truth_path:", truth_path)
    print("hour key:", key)
    print("n_fit:", len(y))
    print("SciPy loss:", scipy_loss)
    print("torch loss:", torch_loss)
    print("loss abs diff:", value_row["loss_abs_diff"])
    print("max grad abs diff:", value_row["max_grad_abs_diff"])
    print("max grad rel diff:", value_row["max_grad_rel_diff"])
    print("wrote:", value_path)
    print("wrote:", grad_path)
    print("wrote:", fit_path)
    print("\nSmooth-free optimizer comparison:")
    print(fit_compare[[
        "method", "start_label", "success", "loss", "loss_minus_scipy",
        "sigmasq", "range_lat", "range_lon", "smooth", "nugget", "fit_s", "n_eval",
    ]].to_string(index=False))
    print("\nGradient check at SciPy solution:")
    print(grad_df.to_string(index=False))


if __name__ == "__main__":
    main()
