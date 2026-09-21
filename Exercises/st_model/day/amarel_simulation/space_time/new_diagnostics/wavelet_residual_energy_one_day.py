#!/usr/bin/env python3
"""One-day wavelet residual-energy diagnostic for fitted Vecchia models.

The script is a local-CPU diagnostic for the real 2024-07-03 GEMS field.  It
reuses the regular-grid representation used by the debiased-Whittle workflow:
an observation is retained in a regular cell only when its original location
is within an axis-wise half-cell threshold,

    |source_lat - grid_lat| <= threshold_fraction * delta_lat,
    |source_lon - grid_lon| <= threshold_fraction * delta_lon.

Cells failing the rule remain missing.  Missing cells are zero-filled only for
the linear wavelet transform; every null simulation is passed through the same
mask, so the resulting reference distribution includes the mask effect.

For each fitted model, the observed residual field and simulations from the
fitted stationary covariance are decomposed with a three-level 2-D orthogonal
wavelet transform, independently at each of the eight hours.  The detail
levels define physical resolution scales rather than covariance-eigenvalue
ranks:

    level 3 -> coarse/low-frequency detail,
    level 2 -> middle-frequency detail,
    level 1 -> fine/high-frequency detail.

For coefficient i, the standardized energy is

    e_i = w_i(observed)^2 / E_model[w_i(simulation)^2].

The expected value is one.  Calibration simulations estimate the denominator;
independent envelope simulations give finite-sample 95% reference intervals.
No covariance or precision eigendecomposition is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())

METHODS = ("adapted", "fixed")
METHOD_LABELS = {"adapted": "adapted corridor", "fixed": "fixed center"}
METHOD_COLORS = {"adapted": "#1f77b4", "fixed": "#d62728"}
METHOD_LINESTYLES = {"adapted": "-", "fixed": "-."}
BANDS = ("low", "middle", "high")
BAND_LABELS = {
    "low": "coarse detail (D3)",
    "middle": "middle detail (D2)",
    "high": "fine detail (D1)",
}
ORIENTATIONS = ("H", "V", "D")
ORIENTATION_LABELS = {
    "H": "lat detail (cH)",
    "V": "lon detail (cV)",
    "D": "diagonal (cD)",
    "all": "pooled",
}
EPS = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2024-07-03")
    parser.add_argument(
        "--data-root", type=Path, default=Path("/Users/joonwonlee/Documents/GEMS_DATA")
    )
    parser.add_argument(
        "--fit-csv",
        type=Path,
        default=(
            REPO
            / "outputs/summer_26/vecchia_local_real1_adapted_fixed_lag643_20240703_run01"
            / "task_00_real_20240703/fits.csv"
        ),
    )
    parser.add_argument(
        "--operator-cache-root",
        type=Path,
        default=(
            REPO / "outputs/summer_26/vecchia_local_20240703_lag643_operator_cache"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO
            / "outputs/summer_26/vecchia_local_20240703_adapted_fixed_wavelet_energy"
        ),
    )
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--delta-lat", type=float, default=0.044)
    parser.add_argument("--delta-lon", type=float, default=0.063)
    parser.add_argument("--threshold-fraction", type=float, default=0.5)
    parser.add_argument("--hours-per-day", type=int, default=8)
    parser.add_argument("--wavelet", default="db2")
    parser.add_argument("--wavelet-level", type=int, default=3)
    parser.add_argument("--wavelet-mode", default="periodization")
    parser.add_argument("--calibration-simulations", type=int, default=32)
    parser.add_argument("--envelope-simulations", type=int, default=64)
    parser.add_argument("--simulation-batch-size", type=int, default=4)
    parser.add_argument("--curve-points", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=20260909)
    return parser


def parse_pair(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError(f"Expected increasing pair, got {value!r}")
    return parts[0], parts[1]


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def date_from_key(key: str) -> str | None:
    match = re.search(r"y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})", str(key))
    if match is None:
        return None
    return f"20{match.group('yy')}-{match.group('mm')}-{match.group('dd')}"


def stable_seed(base: int, method: str, stream: str) -> int:
    digest = hashlib.sha256(f"{base}|{method}|{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def load_month_frames(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], list[str]]:
    year, month, _ = (int(part) for part in args.date.split("-"))
    path = args.data_root / f"pickle_{year}" / f"tco_grid_{str(year)[2:]}_{month:02d}.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        raw = pickle.load(handle)

    lat_lo, lat_hi = parse_pair(args.lat_range)
    lon_lo, lon_hi = parse_pair(args.lon_range)
    frames: dict[str, pd.DataFrame] = {}
    for key, frame in raw.items():
        selected = frame[
            frame["Latitude"].between(lat_lo, lat_hi)
            & frame["Longitude"].between(lon_lo, lon_hi)
        ].reset_index(drop=True)
        frames[str(key)] = selected
    day_keys = [key for key in sorted(frames) if date_from_key(key) == args.date]
    if len(day_keys) != int(args.hours_per_day):
        raise RuntimeError(
            f"Expected {args.hours_per_day} hourly fields for {args.date}, got {len(day_keys)}"
        )
    return frames, day_keys


def grid_axes_and_indices(
    frames: dict[str, pd.DataFrame], day_keys: list[str]
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    lats = np.sort(
        np.unique(
            np.concatenate(
                [pd.to_numeric(frames[key]["Latitude"], errors="coerce") for key in day_keys]
            )
        )
    )
    lons = np.sort(
        np.unique(
            np.concatenate(
                [pd.to_numeric(frames[key]["Longitude"], errors="coerce") for key in day_keys]
            )
        )
    )
    if not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise RuntimeError("Regular-grid coordinates contain NaN/Inf")
    indices: list[tuple[np.ndarray, np.ndarray]] = []
    for key in day_keys:
        lat = pd.to_numeric(frames[key]["Latitude"], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(frames[key]["Longitude"], errors="coerce").to_numpy(float)
        ii = np.searchsorted(lats, lat)
        jj = np.searchsorted(lons, lon)
        if (
            np.any(ii < 0)
            or np.any(ii >= len(lats))
            or np.any(jj < 0)
            or np.any(jj >= len(lons))
            or not np.allclose(lats[ii], lat)
            or not np.allclose(lons[jj], lon)
        ):
            raise RuntimeError(f"Could not map {key} exactly to the regular grid")
        if np.unique(ii * len(lons) + jj).size != len(frame := frames[key]):
            raise RuntimeError(f"Duplicate regular cells found in {key}: {len(frame)} rows")
        indices.append((ii, jj))
    return lats, lons, indices


def threshold_mask(
    frame: pd.DataFrame,
    delta_lat: float,
    delta_lon: float,
    fraction: float,
) -> np.ndarray:
    grid_lat = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(float)
    grid_lon = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(float)
    source_lat = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(float)
    source_lon = pd.to_numeric(frame["Source_Longitude"], errors="coerce").to_numpy(float)
    values = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(float)
    finite = (
        np.isfinite(grid_lat)
        & np.isfinite(grid_lon)
        & np.isfinite(source_lat)
        & np.isfinite(source_lon)
        & np.isfinite(values)
    )
    return (
        finite
        & (np.abs(source_lat - grid_lat) <= fraction * delta_lat + 1e-12)
        & (np.abs(source_lon - grid_lon) <= fraction * delta_lon + 1e-12)
    )


def reconstruct_residual_cube(
    residual_path: Path,
    frames: dict[str, pd.DataFrame],
    day_keys: list[str],
    indices: list[tuple[np.ndarray, np.ndarray]],
    shape: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    residual = np.load(residual_path).astype(np.float64, copy=False).reshape(-1)
    cube = np.zeros((len(day_keys), *shape), dtype=np.float64)
    masks = np.zeros((len(day_keys), *shape), dtype=bool)
    cursor = 0
    pre_counts: list[int] = []
    post_counts: list[int] = []

    for hour, (key, (ii, jj)) in enumerate(zip(day_keys, indices)):
        frame = frames[key]
        original_values = pd.to_numeric(
            frame["ColumnAmountO3"], errors="coerce"
        ).to_numpy(float)
        original_valid = np.isfinite(original_values)
        count = int(original_valid.sum())
        if cursor + count > len(residual):
            raise RuntimeError("Cached residual is shorter than the source-data valid rows")
        row_residual = np.full(len(frame), np.nan, dtype=np.float64)
        row_residual[original_valid] = residual[cursor : cursor + count]
        cursor += count

        keep = threshold_mask(
            frame,
            delta_lat=float(args.delta_lat),
            delta_lon=float(args.delta_lon),
            fraction=float(args.threshold_fraction),
        )
        keep &= np.isfinite(row_residual)
        cube[hour, ii[keep], jj[keep]] = row_residual[keep]
        masks[hour, ii[keep], jj[keep]] = True
        pre_counts.append(count)
        post_counts.append(int(keep.sum()))

    if cursor != len(residual):
        raise RuntimeError(
            f"Cached residual/data alignment failed: consumed {cursor}, cache has {len(residual)}"
        )
    summary = {
        "cached_residual_count": len(residual),
        "hourly_valid_before_threshold": pre_counts,
        "hourly_valid_after_threshold": post_counts,
        "valid_before_threshold": int(sum(pre_counts)),
        "valid_after_threshold": int(sum(post_counts)),
        "retained_fraction_of_valid": float(sum(post_counts) / max(sum(pre_counts), 1)),
    }
    return cube, masks, summary


def fitted_parameters(row: pd.Series) -> dict[str, float]:
    return {
        "sigmasq": float(row["est_sigmasq"]),
        "range_lat": float(row["est_range_lat"]),
        "range_lon": float(row["est_range_lon"]),
        "range_time": float(row["est_range_time"]),
        "advec_lat": float(row["est_advec_lat"]),
        "advec_lon": float(row["est_advec_lon"]),
        "nugget": float(row["est_nugget"]),
    }


def circulant_spectrum(
    params: dict[str, float],
    n_lat: int,
    n_lon: int,
    n_time: int,
    delta_lat: float,
    delta_lon: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Odd 2n-1 embeddings avoid an ambiguous signed Nyquist lag, which matters
    # for an advected covariance C(h-v*tau, tau).
    embed_shape = (2 * n_lat - 1, 2 * n_lon - 1, 2 * n_time - 1)
    lags: list[np.ndarray] = []
    for size, step in zip(embed_shape, (delta_lat, delta_lon, 1.0)):
        index = np.arange(size, dtype=np.float64)
        index[index > size // 2] -= size
        lags.append(index * step)
    h_lat, h_lon, h_time = np.meshgrid(*lags, indexing="ij", sparse=True)
    distance = np.sqrt(
        ((h_lat - params["advec_lat"] * h_time) / params["range_lat"]) ** 2
        + ((h_lon - params["advec_lon"] * h_time) / params["range_lon"]) ** 2
        + (h_time / params["range_time"]) ** 2
    )
    covariance = params["sigmasq"] * np.exp(-distance)
    covariance[0, 0, 0] += params["nugget"] + 1e-6
    eigenvalues = np.fft.fftn(covariance).real
    negative = eigenvalues < 0.0
    positive_mass = float(np.maximum(eigenvalues, 0.0).sum())
    negative_mass = float(-np.minimum(eigenvalues, 0.0).sum())
    summary = {
        "embedding_shape": embed_shape,
        "embedding_eigenvalue_min_before_clipping": float(eigenvalues.min()),
        "embedding_negative_fraction": float(negative.mean()),
        "embedding_negative_mass_fraction": negative_mass / max(positive_mass, EPS),
        "model_marginal_variance": float(covariance[0, 0, 0]),
    }
    if summary["embedding_negative_mass_fraction"] > 1e-8:
        raise RuntimeError(
            "Circulant embedding has material negative spectral mass: "
            f"{summary['embedding_negative_mass_fraction']:.3e}"
        )
    return np.sqrt(np.maximum(eigenvalues, 0.0)), summary


def simulation_batches(
    spectral_sqrt: np.ndarray,
    target_shape: tuple[int, int, int],
    count: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    generated = 0
    fft_axes = (1, 2, 3)
    while generated < count:
        current = min(batch_size, count - generated)
        noise = rng.standard_normal((current, *spectral_sqrt.shape))
        transformed = np.fft.fftn(noise, axes=fft_axes)
        transformed *= spectral_sqrt[None, ...]
        fields = np.fft.ifftn(transformed, axes=fft_axes).real
        n_lat, n_lon, n_time = target_shape
        cropped = fields[:, :n_lat, :n_lon, :n_time]
        yield np.moveaxis(cropped, -1, 1)  # batch x time x lat x lon
        generated += current


def wavelet_details(
    cube: np.ndarray,
    masks: np.ndarray,
    wavelet: str,
    level: int,
    mode: str,
) -> dict[str, np.ndarray]:
    if level != 3:
        raise ValueError(
            "This diagnostic currently requires --wavelet-level 3 so D3/D2/D1 "
            "map unambiguously to low/middle/high."
        )
    per_band: dict[str, list[np.ndarray]] = {band: [] for band in BANDS}
    for hour in range(cube.shape[0]):
        observed = np.where(masks[hour], cube[hour], 0.0)
        coeffs = pywt.wavedec2(observed, wavelet=wavelet, mode=mode, level=level)
        details = coeffs[1:]  # D3, D2, D1
        if len(details) != 3:
            raise RuntimeError(f"Expected three detail levels, got {len(details)}")
        for band, (c_h, c_v, c_d) in zip(BANDS, details):
            per_band[band].append(np.stack([c_h, c_v, c_d], axis=0))
    return {band: np.stack(values, axis=0) for band, values in per_band.items()}


def zero_like_details(template: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value, dtype=np.float64) for key, value in template.items()}


def detail_energy_vector(
    details: dict[str, np.ndarray],
    expected_square: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    vectors: list[np.ndarray] = []
    normalized: dict[str, np.ndarray] = {}
    for band in BANDS:
        energy = np.full(details[band].shape, np.nan, dtype=np.float64)
        energy[valid[band]] = (
            details[band][valid[band]] ** 2 / expected_square[band][valid[band]]
        )
        normalized[band] = energy
        vectors.append(energy[valid[band]])
    return np.concatenate(vectors), normalized


def calibration_expectations(
    spectral_sqrt: np.ndarray,
    target_shape: tuple[int, int, int],
    masks: np.ndarray,
    template: dict[str, np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    sum_squares = zero_like_details(template)
    completed = 0
    for batch in simulation_batches(
        spectral_sqrt,
        target_shape,
        int(args.calibration_simulations),
        int(args.simulation_batch_size),
        rng,
    ):
        for cube in batch:
            details = wavelet_details(
                cube, masks, args.wavelet, args.wavelet_level, args.wavelet_mode
            )
            for band in BANDS:
                sum_squares[band] += details[band] ** 2
            completed += 1
        print(
            f"      variance calibration {completed}/{args.calibration_simulations}",
            flush=True,
        )
    # If V_hat is the mean of m independent squared zero-mean Gaussian
    # coefficients, E(1 / V_hat) = [m/(m-2)] / Var for m > 2.  Multiplying the
    # raw second-moment estimate by m/(m-2) therefore removes the inverse-
    # variance bias from independently standardized observed/envelope energies.
    m = float(args.calibration_simulations)
    inverse_variance_bias_correction = m / (m - 2.0)
    return {
        band: (sum_squares[band] / m) * inverse_variance_bias_correction
        for band in BANDS
    }


def valid_expected_masks(expected_square: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    valid: dict[str, np.ndarray] = {}
    for band, values in expected_square.items():
        positive = values[np.isfinite(values) & (values > 0.0)]
        threshold = max(EPS, float(np.median(positive)) * 1e-10) if positive.size else EPS
        valid[band] = np.isfinite(values) & (values > threshold)
    return valid


def summarize_normalized(
    normalized: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for band in BANDS:
        for orientation_index, orientation in enumerate(ORIENTATIONS):
            values = normalized[band][:, orientation_index]
            out[(band, orientation)] = float(np.nanmean(values))
        out[(band, "all")] = float(np.nanmean(normalized[band]))
    return out


def envelope_simulations(
    spectral_sqrt: np.ndarray,
    target_shape: tuple[int, int, int],
    masks: np.ndarray,
    expected_square: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    curve_indices: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    ratios: dict[tuple[str, str], list[float]] = {
        (band, orientation): []
        for band in BANDS
        for orientation in (*ORIENTATIONS, "all")
    }
    curves: list[np.ndarray] = []
    completed = 0
    for batch in simulation_batches(
        spectral_sqrt,
        target_shape,
        int(args.envelope_simulations),
        int(args.simulation_batch_size),
        rng,
    ):
        for cube in batch:
            details = wavelet_details(
                cube, masks, args.wavelet, args.wavelet_level, args.wavelet_mode
            )
            energy_vector, normalized = detail_energy_vector(
                details, expected_square, valid
            )
            summary = summarize_normalized(normalized)
            for key, value in summary.items():
                ratios[key].append(value)
            cumulative = np.cumsum(energy_vector) / len(energy_vector)
            curves.append(cumulative[curve_indices])
            completed += 1
        print(
            f"      null envelope {completed}/{args.envelope_simulations}", flush=True
        )
    return (
        {key: np.asarray(values, dtype=np.float64) for key, values in ratios.items()},
        np.asarray(curves, dtype=np.float64),
    )


def two_sided_monte_carlo_p(observed: float, reference: np.ndarray) -> float:
    center = float(np.median(reference))
    distance = abs(float(observed) - center)
    count = int(np.sum(np.abs(reference - center) >= distance))
    return float((count + 1) / (len(reference) + 1))


def evaluate_method(
    method: str,
    residual_cube: np.ndarray,
    masks: np.ndarray,
    params: dict[str, float],
    lats: np.ndarray,
    lons: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    observed_details = wavelet_details(
        residual_cube, masks, args.wavelet, args.wavelet_level, args.wavelet_mode
    )
    spectral_sqrt, embedding_summary = circulant_spectrum(
        params,
        len(lats),
        len(lons),
        len(masks),
        float(args.delta_lat),
        float(args.delta_lon),
    )
    # Common random-number streams make the adapted/fixed comparison less
    # sensitive to Monte Carlo noise while the two streams below stay mutually
    # independent.
    calibration_rng = np.random.default_rng(
        stable_seed(args.random_seed, "common", "calibration")
    )
    envelope_rng = np.random.default_rng(
        stable_seed(args.random_seed, "common", "envelope")
    )
    expected_square = calibration_expectations(
        spectral_sqrt,
        (len(lats), len(lons), len(masks)),
        masks,
        observed_details,
        args,
        calibration_rng,
    )
    valid = valid_expected_masks(expected_square)
    observed_vector, observed_normalized = detail_energy_vector(
        observed_details, expected_square, valid
    )
    n_coefficients = len(observed_vector)
    if n_coefficients == 0:
        raise RuntimeError("No estimable wavelet coefficients remain after masking")
    curve_indices = np.unique(
        np.linspace(0, n_coefficients - 1, int(args.curve_points)).astype(np.int64)
    )
    reference_ratios, reference_curves = envelope_simulations(
        spectral_sqrt,
        (len(lats), len(lons), len(masks)),
        masks,
        expected_square,
        valid,
        curve_indices,
        args,
        envelope_rng,
    )
    observed_summary = summarize_normalized(observed_normalized)

    rows: list[dict[str, Any]] = []
    for band in BANDS:
        for orientation in (*ORIENTATIONS, "all"):
            ref = reference_ratios[(band, orientation)]
            if orientation == "all":
                count = int(valid[band].sum())
            else:
                orientation_index = ORIENTATIONS.index(orientation)
                count = int(valid[band][:, orientation_index].sum())
            observed = observed_summary[(band, orientation)]
            rows.append(
                {
                    "date": args.date,
                    "method": method,
                    "band": band,
                    "wavelet_detail_level": {"low": 3, "middle": 2, "high": 1}[band],
                    "orientation": orientation,
                    "orientation_label": ORIENTATION_LABELS[orientation],
                    "n_coefficients": count,
                    "observed_energy_ratio": observed,
                    "null_mean": float(np.mean(ref)),
                    "null_q025": float(np.quantile(ref, 0.025)),
                    "null_q975": float(np.quantile(ref, 0.975)),
                    "monte_carlo_two_sided_p": two_sided_monte_carlo_p(observed, ref),
                }
            )
    summary_frame = pd.DataFrame(rows)

    observed_cumulative = np.cumsum(observed_vector) / n_coefficients
    x = (curve_indices + 1) / n_coefficients
    band_counts = {band: int(valid[band].sum()) for band in BANDS}
    cumulative_counts = np.cumsum([band_counts[band] for band in BANDS])
    curve_frame = pd.DataFrame(
        {
            "date": args.date,
            "method": method,
            "mode_fraction": x,
            "observed_cumulative_energy": observed_cumulative[curve_indices],
            "expected_cumulative_energy": x,
            "null_q025": np.quantile(reference_curves, 0.025, axis=0),
            "null_q975": np.quantile(reference_curves, 0.975, axis=0),
        }
    )
    method_summary = {
        **embedding_summary,
        "method": method,
        "parameters": params,
        "n_wavelet_coefficients": n_coefficients,
        "band_coefficient_counts": band_counts,
        "band_fraction_boundaries": (cumulative_counts / n_coefficients).tolist(),
        "overall_observed_mean_energy": float(np.mean(observed_vector)),
        "runtime_seconds": time.perf_counter() - started,
    }
    return summary_frame, curve_frame, observed_normalized, method_summary


def plot_cumulative(
    curves: pd.DataFrame,
    method_summaries: dict[str, dict[str, Any]],
    output: Path,
    date: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    for axis, method in zip(axes, METHODS):
        part = curves[curves["method"].eq(method)].sort_values("mode_fraction")
        axis.fill_between(
            part["mode_fraction"].to_numpy(float),
            part["null_q025"].to_numpy(float),
            part["null_q975"].to_numpy(float),
            color=METHOD_COLORS[method],
            alpha=0.13,
            label="simulation 95% null envelope",
        )
        axis.plot(
            part["mode_fraction"],
            part["observed_cumulative_energy"],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.0,
            label=(
                f"observed; end="
                f"{method_summaries[method]['overall_observed_mean_energy']:.3f}"
            ),
        )
        axis.plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.1, label="expected y=x")
        boundaries = method_summaries[method]["band_fraction_boundaries"][:-1]
        for boundary in boundaries:
            axis.axvline(boundary, color="0.45", linewidth=0.8, alpha=0.7)
        positions = [boundaries[0] / 2, sum(boundaries) / 2, (boundaries[-1] + 1) / 2]
        for position, band in zip(positions, BANDS):
            axis.text(position, 0.98, band, transform=axis.get_xaxis_transform(), ha="center", va="top", color="0.35")
        axis.set(
            xlim=(0, 1),
            xlabel="wavelet-detail coefficient fraction (coarse → fine)",
            ylabel="cumulative standardized residual energy",
            title=METHOD_LABELS[method],
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Real {date}: wavelet residual-energy cumulative diagnostic",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_scale_ratios(summary: pd.DataFrame, output: Path, date: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True, sharey=True)
    orientations = (*ORIENTATIONS, "all")
    x = np.arange(len(orientations), dtype=float)
    offsets = {"adapted": -0.10, "fixed": 0.10}
    for axis, band in zip(axes, BANDS):
        for method in METHODS:
            part = (
                summary[summary["method"].eq(method) & summary["band"].eq(band)]
                .set_index("orientation")
                .loc[list(orientations)]
            )
            observed = part["observed_energy_ratio"].to_numpy(float)
            lower = part["null_q025"].to_numpy(float)
            upper = part["null_q975"].to_numpy(float)
            xpos = x + offsets[method]
            axis.vlines(
                xpos,
                lower,
                upper,
                color=METHOD_COLORS[method],
                linewidth=1.4,
                alpha=0.75,
            )
            axis.scatter(
                xpos,
                observed,
                s=38,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                zorder=3,
            )
        axis.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
        axis.set_xticks(x, [ORIENTATION_LABELS[item] for item in orientations], rotation=18)
        axis.set_title(BAND_LABELS[band])
        axis.set_xlabel("wavelet orientation")
        axis.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("empirical / fitted-model wavelet energy")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Real {date}: localized wavelet-scale residual energy\n"
        "error bars are model-specific simulation 95% intervals",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_energy_maps(
    normalized_by_method: dict[str, dict[str, np.ndarray]],
    output: Path,
    date: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), constrained_layout=True)
    image = None
    for row, method in enumerate(METHODS):
        for col, band in enumerate(BANDS):
            energy = normalized_by_method[method][band]
            # The standardized squared coefficient has expectation one, while
            # its median is below one.  Averaging across hours and orientations
            # therefore makes zero on this log2 map the fitted-model reference.
            pooled = np.nanmean(energy, axis=(0, 1))
            shown = np.log2(np.maximum(pooled, 2.0 ** -3))
            image = axes[row, col].imshow(
                shown,
                origin="lower",
                cmap="RdBu_r",
                vmin=-2.0,
                vmax=2.0,
                interpolation="nearest",
                aspect="auto",
            )
            axes[row, col].set_title(f"{METHOD_LABELS[method]} — {BAND_LABELS[band]}")
            axes[row, col].set_xlabel("longitude coefficient index")
            axes[row, col].set_ylabel("latitude coefficient index")
    if image is not None:
        bar = fig.colorbar(image, ax=axes, shrink=0.82)
        bar.set_label("log2 standardized energy (mean over hour/orientation)")
    fig.suptitle(f"Real {date}: spatial localization of wavelet residual energy", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if args.threshold_fraction <= 0.0:
        raise ValueError("--threshold-fraction must be positive")
    if args.delta_lat <= 0.0 or args.delta_lon <= 0.0:
        raise ValueError("Grid spacings must be positive")
    if args.wavelet_level != 3:
        raise ValueError("Use --wavelet-level 3 for the low/middle/high diagnostic")
    if args.calibration_simulations < 8 or args.envelope_simulations < 8:
        raise ValueError("Use at least 8 calibration and 8 envelope simulations")
    if args.simulation_batch_size <= 0 or args.curve_points < 16:
        raise ValueError("Simulation batch size and curve points must be positive")
    pywt.Wavelet(args.wavelet)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    total_started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Load regular-grid source for {args.date}", flush=True)
    frames, day_keys = load_month_frames(args)
    lats, lons, indices = grid_axes_and_indices(frames, day_keys)
    actual_delta_lat = float(np.median(np.diff(lats)))
    actual_delta_lon = float(np.median(np.diff(lons)))
    if not math.isclose(actual_delta_lat, args.delta_lat, rel_tol=0, abs_tol=1e-8):
        raise RuntimeError(f"Latitude spacing is {actual_delta_lat}, expected {args.delta_lat}")
    if not math.isclose(actual_delta_lon, args.delta_lon, rel_tol=0, abs_tol=1e-8):
        raise RuntimeError(f"Longitude spacing is {actual_delta_lon}, expected {args.delta_lon}")

    fits = pd.read_csv(args.fit_csv)
    fits = fits[fits["date"].astype(str).eq(args.date)]
    if set(fits["geometry"].astype(str)) < set(METHODS):
        raise RuntimeError(f"Fit CSV does not contain both {METHODS} for {args.date}")

    method_summaries: dict[str, dict[str, Any]] = {}
    normalized_by_method: dict[str, dict[str, np.ndarray]] = {}
    all_scale_rows: list[pd.DataFrame] = []
    all_curve_rows: list[pd.DataFrame] = []
    mask_reference: np.ndarray | None = None
    gridding_summary: dict[str, Any] | None = None

    for method in METHODS:
        print(f"\n{method}: reconstruct thresholded residual grids", flush=True)
        residual_path = args.operator_cache_root / method / "residual.npy"
        if not residual_path.is_file():
            raise FileNotFoundError(residual_path)
        residual_cube, masks, grid_summary = reconstruct_residual_cube(
            residual_path,
            frames,
            day_keys,
            indices,
            (len(lats), len(lons)),
            args,
        )
        if mask_reference is None:
            mask_reference = masks
            gridding_summary = grid_summary
        elif not np.array_equal(mask_reference, masks):
            raise RuntimeError("Adapted and fixed residuals unexpectedly use different masks")
        fit_row = fits[fits["geometry"].astype(str).eq(method)].iloc[0]
        params = fitted_parameters(fit_row)
        print(
            f"  grid={len(lats)}x{len(lons)}x{len(day_keys)}, "
            f"retained={grid_summary['valid_after_threshold']:,}/"
            f"{grid_summary['valid_before_threshold']:,}",
            flush=True,
        )
        print(
            f"  {method}: wavelet variance calibration and independent null envelope",
            flush=True,
        )
        scale, curve, normalized, method_summary = evaluate_method(
            method, residual_cube, masks, params, lats, lons, args
        )
        scale["native_nll_per_observation"] = float(fit_row["final_native_nll"])
        method_summary["native_nll_per_observation"] = float(
            fit_row["final_native_nll"]
        )
        all_scale_rows.append(scale)
        all_curve_rows.append(curve)
        normalized_by_method[method] = normalized
        method_summaries[method] = method_summary
        print(
            f"  {method}: overall mean standardized energy="
            f"{method_summary['overall_observed_mean_energy']:.3f}; "
            f"runtime={method_summary['runtime_seconds']:.1f}s",
            flush=True,
        )

    scale_summary = pd.concat(all_scale_rows, ignore_index=True)
    curves = pd.concat(all_curve_rows, ignore_index=True)
    atomic_csv(args.output_root / "wavelet_scale_orientation_energy.csv", scale_summary)
    atomic_csv(args.output_root / "wavelet_cumulative_curves.csv", curves)
    plot_cumulative(
        curves,
        method_summaries,
        args.output_root / "wavelet_cumulative_diagnostic.png",
        args.date,
    )
    plot_scale_ratios(
        scale_summary,
        args.output_root / "wavelet_scale_orientation_diagnostic.png",
        args.date,
    )
    plot_energy_maps(
        normalized_by_method,
        args.output_root / "wavelet_spatial_energy_maps.png",
        args.date,
    )

    run_summary = {
        "date": args.date,
        "diagnostic": "three-level 2-D wavelet standardized residual energy",
        "grid_shape": [len(day_keys), len(lats), len(lons)],
        "grid_spacing": {"latitude": actual_delta_lat, "longitude": actual_delta_lon},
        "gridding": {
            "rule": "axis-wise source-to-regular-cell distance threshold; otherwise missing",
            "threshold_fraction": args.threshold_fraction,
            "latitude_threshold": args.threshold_fraction * args.delta_lat,
            "longitude_threshold": args.threshold_fraction * args.delta_lon,
            **(gridding_summary or {}),
        },
        "wavelet": {
            "name": args.wavelet,
            "level": args.wavelet_level,
            "boundary_mode": args.wavelet_mode,
            "approximation_coefficients_included": False,
            "band_definition": BAND_LABELS,
            "missing_handling": (
                "zero-fill before the linear transform, calibrated with the identical "
                "hour-specific mask in every fitted-model simulation"
            ),
        },
        "simulation": {
            "calibration_count": args.calibration_simulations,
            "envelope_count": args.envelope_simulations,
            "independent_calibration_and_envelope_streams": True,
            "common_random_numbers_across_methods": True,
            "inverse_variance_bias_correction": (
                "calibration mean-square multiplied by m/(m-2)"
            ),
            "covariance_reference": (
                "full stationary fitted Matérn-0.5 covariance on the regular grid; "
                "not the ordering-dependent Vecchia precision"
            ),
        },
        "methods": method_summaries,
        "total_runtime_seconds": time.perf_counter() - total_started,
    }
    write_json(args.output_root / "run_summary.json", run_summary)
    (args.output_root / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"\nComplete: {args.output_root}", flush=True)
    print(f"Total runtime: {run_summary['total_runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
