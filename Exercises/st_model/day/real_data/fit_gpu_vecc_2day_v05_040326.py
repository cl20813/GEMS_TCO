"""
fit_gpu_vecc_2day_v05_040326.py  — Amarel (cluster) version

Same as fit_gpu_vecc_day_v05_031826.py but uses **2-day windows** of data:
  window 0 → hours  0–15  (day 1 + day 2)
  window 1 → hours 16–31  (day 3 + day 4)
  ...
  window 13 → hours 208–223 (day 27 + day 28)

This doubles the temporal support (p=16 instead of 8), giving the Vecchia
approximation more cross-day covariance signal without any architecture change.

14 windows × 4 years (2022–2025) = 56 fits.

Output CSV: real_vecc_2day_july_22_23_24_25_mm{mm}.csv
  `day` column format: "YYYY-MM-DD_DD" (e.g. "2022-07-01_02")
"""
import sys
import time
import json
import pandas as pd
import numpy as np
import torch
import typer
from pathlib import Path
from typing import List

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import alg_optimization, BaseLogger
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

# Hours per day (fixed)
HOURS_PER_DAY = 8
DAYS_PER_WIN  = 2
HOURS_PER_WIN = HOURS_PER_DAY * DAYS_PER_WIN   # 16


@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness"),
    lr: float = typer.Option(1.0, help="L-BFGS learning rate"),
    space: List[str] = typer.Option(['1,1'], help="spatial resolution (lat,lon grid bins)"),
    windows: List[str] = typer.Option(['0,14'], help="Window range 'start,end' (0-based, end exclusive, max 14)"),
    mm_cond_number: int = typer.Option(100, help="Max neighbors stored in NNS map"),
    nheads: int = typer.Option(0, help="Number of head points (exact GP)"),
    limit_a: int = typer.Option(20, help="Set A: spatial neighbors at current time step"),
    limit_b: int = typer.Option(20, help="Set B: spatial neighbors at t-1"),
    limit_c: int = typer.Option(20, help="Set C: spatial neighbors at t-daily_stride"),
    daily_stride: int = typer.Option(2, help="Stride for Set C; >= 16 disables Set C"),
    years: List[str] = typer.Option(['2022,2023,2024,2025'], help="Comma-separated years"),
    month: int = typer.Option(7, help="Month to process"),
    keep_exact_loc: bool = typer.Option(True, help="Use actual observation coords (True) or grid centers (False)"),
) -> None:

    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    win_s, win_e = [int(x) for x in windows[0].split(',')]
    win_e = min(win_e, 14)            # July has at most 14 complete 2-day windows
    windows_list = list(range(win_s, win_e))
    years_list = [y.strip() for y in years[0].split(',')]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device    : {DEVICE}")
    print(f"Windows         : {windows_list}  (each = {HOURS_PER_WIN} hours / p=16)")
    print(f"Years           : {years_list}")
    print(f"nheads={nheads}  mm={mm_cond_number}  A={limit_a} B={limit_b} C={limit_c}  stride={daily_stride}")

    LBFGS_LR         = lr
    LBFGS_MAX_STEPS  = 5
    LBFGS_HISTORY    = 10
    LBFGS_MAX_EVAL   = 20
    month_range      = [month]

    output_path = Path(config.amarel_estimates_day_path) / "july_22_23_24_25"
    output_path.mkdir(parents=True, exist_ok=True)

    data_load_instance = load_data_dynamic_processed(config.amarel_data_load_path)

    for year in years_list:
        print(f'\n{"="*60}')
        print(f'=== Processing Year {year} ===')
        print(f'{"="*60}')

        df_map, ord_mm, nns_map, monthly_mean = data_load_instance.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=lat_lon_resolution,
            mm_cond_number=mm_cond_number,
            years_=[year],
            months_=month_range,
            lat_range=[-3, 2],
            lon_range=[121, 131],
            is_whittle=False,
        )

        # ── Pre-load all 14 two-day window maps ───────────────────────────────
        # load_working_data expects at most 8 hours (time keys 0–7).
        # Load each calendar day separately, then shift day-2 keys by +8
        # so the merged map has time keys 0–15.
        print("Pre-loading 2-day window tensors (Vecchia)...")
        win_hourly_maps_vecc = []

        for win_idx in range(14):
            day1_start = win_idx * HOURS_PER_WIN          # e.g. 0
            day2_start = day1_start + HOURS_PER_DAY       # e.g. 8

            map_day1, _ = data_load_instance.load_working_data(
                df_map, monthly_mean,
                [day1_start, day1_start + HOURS_PER_DAY],
                ord_mm=ord_mm,
                dtype=torch.float64,
                keep_ori=keep_exact_loc,
            )
            map_day2, _ = data_load_instance.load_working_data(
                df_map, monthly_mean,
                [day2_start, day2_start + HOURS_PER_DAY],
                ord_mm=ord_mm,
                dtype=torch.float64,
                keep_ori=keep_exact_loc,
            )
            # Keys are timestamp strings (e.g. '2022_07_y22m07day01_hm00:48'),
            # so day1 and day2 keys are already unique — just merge directly.
            merged = {**map_day1, **map_day2}
            win_hourly_maps_vecc.append(merged)

        # ── Fit loop ──────────────────────────────────────────────────────────
        for win_idx in windows_list:
            day1 = 2 * win_idx + 1   # 1-based calendar day (start)
            day2 = 2 * win_idx + 2   # 1-based calendar day (end)
            day_str = f"{year}-{month:02d}-{day1:02d}_{day2:02d}"

            print(f'\n{"="*50}')
            print(f'--- Vecchia 2-day: window {win_idx}  ({day_str}) ---')
            print(f'{"="*50}')

            try:
                win_map_vecc = {
                    k: v.to(DEVICE) for k, v in win_hourly_maps_vecc[win_idx].items()
                }

                if not win_map_vecc:
                    print(f"Skipping window {win_idx}: No data.")
                    continue

                # ── Initial parameters (same as 1-day version) ────────────────
                init_sigmasq    = 13.059
                init_range_lat  = 0.2
                init_range_lon  = 0.25
                init_range_time = 1.5
                init_advec_lat  = 0.0218
                init_advec_lon  = -0.1689
                init_nugget     = 0.247

                init_phi2 = 1.0 / init_range_lon
                init_phi1 = init_sigmasq * init_phi2
                init_phi3 = (init_range_lon / init_range_lat) ** 2
                init_phi4 = (init_range_lon / init_range_time) ** 2

                initial_vals = [
                    np.log(init_phi1), np.log(init_phi2), np.log(init_phi3),
                    np.log(init_phi4), init_advec_lat, init_advec_lon, np.log(init_nugget)
                ]
                params_list_vecc = [
                    torch.tensor([val], requires_grad=True, dtype=torch.float64, device=DEVICE)
                    for val in initial_vals
                ]

                model_instance = kernels_vecchia.fit_vecchia_lbfgs(
                    smooth=v,
                    input_map=win_map_vecc,
                    nns_map=nns_map,
                    mm_cond_number=mm_cond_number,
                    nheads=nheads,
                    limit_A=limit_a, limit_B=limit_b, limit_C=limit_c,
                    daily_stride=daily_stride,
                )

                optimizer_vecc = model_instance.set_optimizer(
                    params_list_vecc,
                    lr=LBFGS_LR,
                    max_iter=LBFGS_MAX_EVAL,
                    max_eval=LBFGS_MAX_EVAL,
                    history_size=LBFGS_HISTORY,
                )

                print(f"--- Starting Vecchia Optimization (mm={mm_cond_number}, p=16) ---")
                start_time = time.time()

                out, steps_ran = model_instance.fit_vecc_lbfgs(
                    params_list_vecc,
                    optimizer_vecc,
                    max_steps=LBFGS_MAX_STEPS,
                    grad_tol=1e-5,
                )

                epoch_time = time.time() - start_time
                print(f"Finished in {epoch_time:.2f}s.  Results: {out}")

                grid_res = len(next(iter(win_map_vecc.values())))

                res = alg_optimization(
                    day=day_str,
                    cov_name=f"Vecc2day_mm{mm_cond_number}_A{limit_a}B{limit_b}C{limit_c}",
                    space_size=grid_res,
                    lr=LBFGS_LR,
                    params=out,
                    time=epoch_time,
                    rmsre=0.0,
                )

                common_filename = f"real_vecc_2day_july_22_23_24_25_mm{mm_cond_number}"

                json_filepath = output_path / f"{common_filename}.json"
                try:
                    current_data = BaseLogger.load_list(json_filepath)
                    if not isinstance(current_data, list):
                        current_data = []
                except Exception:
                    current_data = []
                current_data.append(res.__dict__)
                with json_filepath.open('w', encoding='utf-8') as f:
                    json.dump(current_data, f, separators=(",", ":"), indent=4)

                csv_filepath = output_path / f"{common_filename}.csv"
                pd.DataFrame(current_data).to_csv(csv_filepath, index=False)

                print(f"Saved to {common_filename}.[json/csv]")

            except Exception as e:
                import traceback
                print(f"Window {win_idx} ({day_str}) Failed: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    app()
