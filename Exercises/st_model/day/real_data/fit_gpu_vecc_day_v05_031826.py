import sys
import argparse
import time
import json
import pandas as pd
import numpy as np
import pickle
import torch
import typer
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import alg_optimization, BaseLogger
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness"),
    lr: float = typer.Option(1.0, help="L-BFGS learning rate"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Start and End day index (0-based, e.g. '0,31')"),
    mm_cond_number: int = typer.Option(100, help="Max neighbors stored in NNS map"),
    nheads: int = typer.Option(300, help="Number of head points (exact GP)"),
    limit_a: int = typer.Option(8, help="Set A: spatial neighbors at current time step"),
    limit_b: int = typer.Option(8, help="Set B: spatial neighbors at t-1 (actual size = limit_b+1)"),
    limit_c: int = typer.Option(8, help="Set C: spatial neighbors at t-daily_stride (actual size = limit_c+1)"),
    daily_stride: int = typer.Option(8, help="Stride for Set C; >= n_time_steps(8) disables Set C"),
    years: List[str] = typer.Option(['2022,2024,2025'], help="Comma-separated years to process"),
    month: int = typer.Option(7, help="Month to process"),
    keep_exact_loc: bool = typer.Option(True, help="Use actual observation coordinates (True) or grid centers (False)"),
) -> None:

    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years_list = [y.strip() for y in years[0].split(',')]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Target Days: {days_list}")
    print(f"Target Years: {years_list}")

    limit_A = limit_a
    limit_B = limit_b
    limit_C = limit_c

    LBFGS_LR = lr
    LBFGS_MAX_STEPS = 5
    LBFGS_HISTORY_SIZE = 10
    LBFGS_MAX_EVAL = 20

    month_range = [month]

    output_path = Path(config.amarel_estimates_day_path) / "july_22_23_24_25"
    output_path.mkdir(parents=True, exist_ok=True)

    data_load_instance = load_data_dynamic_processed(config.amarel_data_load_path)

    for year in years_list:
        print(f'\n{"="*60}')
        print(f'=== Processing Year {year} ===')
        print(f'{"="*60}')

        print(f"\nLoading MaxMin Ordered Data for {year}...")
        df_map, ord_mm, nns_map, monthly_mean = data_load_instance.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=lat_lon_resolution,
            mm_cond_number=mm_cond_number,
            years_=[year],
            months_=month_range,
            lat_range=[-3, 2],
            lon_range=[121, 131],
            is_whittle=False
        )

        print("Pre-loading daily tensors (Vecchia)...")
        daily_hourly_maps_vecc = []

        for day_index in range(31):
            hour_indices = [day_index * 8, (day_index + 1) * 8]
            day_hourly_map, _ = data_load_instance.load_working_data(
                df_map,
                monthly_mean,
                hour_indices,
                ord_mm=ord_mm,
                dtype=torch.float64,
                keep_ori=keep_exact_loc
            )
            daily_hourly_maps_vecc.append(day_hourly_map)

        for day_idx in days_list:
            print(f'\n{"="*50}')
            print(f'--- Vecchia: Day {day_idx+1} ({year}-{month:02d}-{day_idx+1}) ---')
            print(f'{"="*50}')

            try:
                daily_hourly_map_vecc = {
                    k: v.to(DEVICE) for k, v in daily_hourly_maps_vecc[day_idx].items()
                }

                if not daily_hourly_map_vecc:
                    print(f"Skipping Day {day_idx+1}: No data.")
                    continue

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
                    input_map=daily_hourly_map_vecc,
                    nns_map=nns_map,
                    mm_cond_number=mm_cond_number,
                    nheads=nheads,
                    limit_A=limit_A, limit_B=limit_B, limit_C=limit_C,
                    daily_stride=daily_stride,
                )

                optimizer_vecc = model_instance.set_optimizer(
                    params_list_vecc,
                    lr=LBFGS_LR,
                    max_iter=LBFGS_MAX_EVAL,
                    max_eval=LBFGS_MAX_EVAL,
                    history_size=LBFGS_HISTORY_SIZE
                )

                print(f"--- Starting Vecchia Optimization (mm={mm_cond_number}) ---")
                start_time = time.time()

                out, steps_ran = model_instance.fit_vecc_lbfgs(
                    params_list_vecc,
                    optimizer_vecc,
                    max_steps=LBFGS_MAX_STEPS,
                    grad_tol=1e-5
                )

                epoch_time = time.time() - start_time
                print(f"Finished in {epoch_time:.2f}s. Results: {out}")

                grid_res = len(next(iter(daily_hourly_map_vecc.values())))

                res = alg_optimization(
                    day=f"{year}-{month:02d}-{day_idx+1}",
                    cov_name=f"Vecc_mm{mm_cond_number}_A{limit_A}B{limit_B}C{limit_C}",
                    space_size=grid_res,
                    lr=LBFGS_LR,
                    params=out,
                    time=epoch_time,
                    rmsre=0.0
                )

                common_filename = f"real_vecc_july_22_23_24_25_h{nheads}_mm{mm_cond_number}"

                json_filepath = output_path / f"{common_filename}.json"
                current_data = BaseLogger.load_list(json_filepath)
                current_data.append(res.__dict__)
                with json_filepath.open('w', encoding='utf-8') as f:
                    json.dump(current_data, f, separators=(",", ":"), indent=4)

                csv_filepath = output_path / f"{common_filename}.csv"
                pd.DataFrame(current_data).to_csv(csv_filepath, index=False)

                print(f"Saved to {common_filename}.[json/csv]")

            except Exception as e:
                import traceback
                print(f"Day {day_idx+1} Failed: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    app()
