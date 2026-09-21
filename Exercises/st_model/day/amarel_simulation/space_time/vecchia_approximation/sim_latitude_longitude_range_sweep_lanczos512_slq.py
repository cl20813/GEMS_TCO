#!/usr/bin/env python3
"""Check long-latitude and longitude-only range misspecification.

This extends the fixed-parameter 2024-07-13 smoothness-0.5 simulation with:

* latitude range 3 times truth;
* longitude range 0.5 times truth, with latitude/time ranges held at truth;
* longitude range 2 times truth, with latitude/time ranges held at truth.

The existing latitude-range-2-times result is reused for the comparison plot.
Every assumed physical parameter vector is transformed to the internal phi
parameterization and back before any precision is built; a failed round trip
aborts the run.  Diagnostics use Lanczos 512 and paired SLQ 8x192.
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime

import numpy as np
import pandas as pd

import sim_latitude_short_long_smooth10_lanczos512_slq as prior
import sim_true_vs_misspecified_residual_lanczos_slq as base
import vecchia_local_20240703_adapted_fixed_residual_lanczos_slq as diagnostic


SCENARIOS = ("lat_range_3x", "lon_range_half", "lon_range_2x")
DEFAULT_OUTPUT_ROOT = base.REPO / (
    "outputs/summer_26/"
    "sim_20240713_latitude_longitude_range_sweep_lanczos512_slq8x192"
)


def parser() -> argparse.ArgumentParser:
    out = prior.parser()
    out.description = __doc__
    out.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
    return out


def scenario_parameters(truth: dict[str, float]) -> dict[str, dict[str, float]]:
    keys = (
        "smooth", "sigmasq", "range_lat", "range_lon", "range_time",
        "advec_lat", "advec_lon", "nugget",
    )
    original = {key: float(truth[key]) for key in keys}
    lat3 = dict(original)
    lat3["range_lat"] *= 3.0
    lon_half = dict(original)
    lon_half["range_lon"] *= 0.5
    lon2 = dict(original)
    lon2["range_lon"] *= 2.0
    return {
        "lat_range_3x": lat3,
        "lon_range_half": lon_half,
        "lon_range_2x": lon2,
    }


def validate_parameter_roundtrip(assumed: dict[str, float]) -> dict[str, float]:
    raw = base.sim_data.physical_to_log_phi(assumed)
    recovered = base.sim_data.backmap_params(raw)
    for key in (
        "sigmasq", "range_lat", "range_lon", "range_time",
        "advec_lat", "advec_lon",
    ):
        if not np.isclose(
            float(recovered[key]), float(assumed[key]), rtol=1e-12, atol=1e-12
        ):
            raise RuntimeError(
                f"Parameter round-trip failed for {key}: "
                f"assumed={assumed[key]}, recovered={recovered[key]}"
            )
    return {key: float(recovered[key]) for key in recovered}


def main() -> None:
    args = parser().parse_args()
    if int(args.residual_lanczos_steps) != 512:
        raise ValueError("Use residual-Lanczos 512 for the range sweep")
    if int(args.slq_probes) != 8 or int(args.slq_steps) != 192:
        raise ValueError("Use paired SLQ 8x192 for the range sweep")
    args.output_root.mkdir(parents=True, exist_ok=True)
    diagnostic.DATE = base.SIM_DATE
    workflow_started = time.perf_counter()
    truth = base.load_truth(args.data_root)
    asset = base.load_asset(args)
    assumed_by_scenario = scenario_parameters(truth)
    roundtrip = {
        scenario: validate_parameter_roundtrip(assumed)
        for scenario, assumed in assumed_by_scenario.items()
    }
    for scenario in SCENARIOS:
        recovered = roundtrip[scenario]
        print(
            f"{scenario} round-trip: "
            f"lat={recovered['range_lat']:.6g}, "
            f"lon={recovered['range_lon']:.6g}, "
            f"time={recovered['range_time']:.6g}",
            flush=True,
        )

    model, beta, graph_seconds = prior.prepare_shared_model(asset, truth, args)
    print(f"Shared lag-6/4/3 graph prepared in {graph_seconds:.2f}s", flush=True)
    curve_frames: list[pd.DataFrame] = []
    band_frames: list[pd.DataFrame] = []
    third_frames: list[pd.DataFrame] = []
    slq_frames: list[pd.DataFrame] = []
    summaries = []
    timing_rows = []

    for scenario in SCENARIOS:
        scenario_started = time.perf_counter()
        assumed = assumed_by_scenario[scenario]
        print(f"\n{scenario}: {assumed}", flush=True)
        precision, build_summary = prior.build_precision(
            model, beta, scenario, assumed, args
        )
        slq, curves, bands, result = diagnostic.evaluate_method(
            scenario, precision, args
        )
        elapsed = time.perf_counter() - scenario_started
        summary = {**build_summary, **result["summary"], "scenario_total_seconds": elapsed}
        summaries.append(summary)
        curve_frames.append(curves)
        band_frames.append(bands)
        third_frames.append(result["thirds"])
        slq_frames.append(slq.assign(date=base.SIM_DATE, method=scenario))
        timing_rows.append({
            "scenario": scenario,
            "precision_build_seconds": build_summary["precision_build_seconds"],
            "slq_seconds": result["summary"]["slq_seconds"],
            "residual_lanczos_seconds": result["summary"]["residual_lanczos_seconds"],
            "scenario_total_seconds": elapsed,
        })
        print(
            "  thirds: "
            + ", ".join(
                f"{row.frequency_third}={row.energy_per_mode:.3f}"
                for row in result["thirds"].itertuples(index=False)
            ),
            flush=True,
        )
        del precision
        gc.collect()

    base.atomic_csv(
        args.output_root / "residual_lanczos_cumulative_curves.csv",
        pd.concat(curve_frames, ignore_index=True),
    )
    base.atomic_csv(
        args.output_root / "frequency_band_energy_ratios.csv",
        pd.concat(band_frames, ignore_index=True),
    )
    base.atomic_csv(
        args.output_root / "frequency_third_energy_ratios.csv",
        pd.concat(third_frames, ignore_index=True),
    )
    base.atomic_csv(
        args.output_root / "slq_spectrum.csv",
        pd.concat(slq_frames, ignore_index=True),
    )
    base.atomic_csv(args.output_root / "timings.csv", pd.DataFrame(timing_rows))
    total_seconds = time.perf_counter() - workflow_started
    base.write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "date": base.SIM_DATE,
            "truth": truth,
            "assumed_parameters": assumed_by_scenario,
            "parameter_roundtrip": roundtrip,
            "scenarios": list(SCENARIOS),
            "shared_graph_precompute_seconds": graph_seconds,
            "total_wall_seconds": total_seconds,
            "residual_lanczos_steps": 512,
            "slq_probes": 8,
            "slq_steps": 192,
            "mean_handling": "known DGP mean removed; no parameter fitting",
            "summaries": summaries,
            "individual_eigenvectors_computed": False,
        },
    )
    print(f"\nComplete in {total_seconds:.2f}s: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
