"""
fit_dw_nolf_day_v05_040826.py

Debiased Whittle — No-Low-Frequency (NOLF) variant.

  Excludes spatial frequency box {(k1,k2) : k1 <= K1 AND k2 <= K2} from
  the Whittle objective, where K1 = floor(n1*alpha), K2 = floor(n2*alpha).

  Z(i,j,t) = X(i,j,t)       (identity filter, same preprocessing as DW_raw)
  Loss      = DW_raw sum over high-freq only  (low-freq box dropped)

  Default alpha = 0.20  →  same boundary as DW_mixed's Omega_L.
  alpha = 0.10 available for minimal exclusion comparison.

Module: debiased_whittle_nolf

Compare with:
  fit_dw_raw_day_v05_040626.py  — raw baseline (all freq except DC)
  fit_dw_mix_day_v05_040626.py  — mixed-frequency DW (low→raw, high→diff)

Statistical justification:
  Equivalent to spectral REML (Patterson & Thompson 1971; Kitanidis 1983;
  Fuentes 2002 JASA). Removes contamination from large-scale spatial gradients
  (wavelength >> correlation range) without discarding covariance information.

  Domain: lat 5°, lon 10°, 1° resolution → n1≈5, n2≈10
  Excluded scales at alpha=0.20: wavelength ≥ 5° (~25-30× true range_lat)

Usage:
  python fit_dw_nolf_day_v05_040826.py --days "0,28" --years "2022,2023,2024,2025"
  python fit_dw_nolf_day_v05_040826.py --days "0,28" --years "2022,2023,2024,2025" --alpha 0.10
"""

import sys
import time
import json
import pandas as pd
import numpy as np
import torch
import typer
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import alg_optimization, BaseLogger
from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle_nolf as debiased_whittle
from GEMS_TCO.data_loader import load_data_dynamic_processed

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness (unused, API parity)"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '28'], help="Start and End day index (0-based)"),
    keep_exact_loc: bool = typer.Option(True, help="Whether to keep exact observation locations"),
    years: List[str] = typer.Option(['2022,2023,2024,2025'], help="Comma-separated years to process"),
    month: int = typer.Option(7, help="Month to process"),
    alpha: float = typer.Option(0.20, help="Low-freq exclusion fraction: K1=floor(n1*alpha), K2=floor(n2*alpha)"),
) -> None:

    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e  = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years_list = [y.strip() for y in years[0].split(',')]

    # Output directory name encodes alpha, e.g. "nolf_a0p20" for alpha=0.20
    alpha_tag = f"a{alpha:.2f}".replace('.', 'p')   # "a0p20"
    cov_name  = f"DW_nolf_{alpha_tag}"

    DEVICE_DW = torch.device("cpu")
    print(f"Whittle Device  : {DEVICE_DW}")
    print(f"Filter          : NONE (raw passthrough)  Z(i,j)=X(i,j)")
    print(f"Low-freq excl.  : alpha={alpha}  →  K1=floor(n1*{alpha}), K2=floor(n2*{alpha})")
    print(f"                  excludes box {{(k1,k2): k1<=K1, k2<=K2}} from Whittle sum")
    print(f"Target Days     : {days_list}")
    print(f"Target Years    : {years_list}")

    month_range = [month]

    output_path = Path(config.amarel_estimates_day_path) / f"july_22_23_24_25_nolf_{alpha_tag}"
    output_path.mkdir(parents=True, exist_ok=True)

    data_load_instance = load_data_dynamic_processed(config.amarel_data_load_path)

    dwl = debiased_whittle.debiased_whittle_likelihood()
    TAPERING_FUNC = dwl.cgn_hamming
    DWL_MAX_STEPS = 5
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3

    for year in years_list:
        print(f'\n{"="*60}')
        print(f'=== Processing Year {year} ===')
        print(f'{"="*60}')

        df_map, ord_mm, nns_map, monthly_mean = data_load_instance.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=lat_lon_resolution,
            years_=[year],
            months_=month_range,
            lat_range=[-3, 2],
            lon_range=[121, 131],
            is_whittle=True
        )

        print("Pre-loading daily tensors...")
        daily_aggregated_tensors_dw = []
        daily_hourly_maps_dw = []

        for day_index in range(31):
            hour_indices = [day_index * 8, (day_index + 1) * 8]
            day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
                df_map,
                monthly_mean,
                hour_indices,
                ord_mm=None,
                dtype=torch.float64,
                keep_ori=keep_exact_loc
            )
            daily_aggregated_tensors_dw.append(day_aggregated_tensor)
            daily_hourly_maps_dw.append(day_hourly_map)

        for day_idx in days_list:
            print(f'\n{"="*50}')
            print(f'--- {cov_name}: Day {day_idx+1} ({year}-{month:02d}-{day_idx+1}) ---')
            print(f'{"="*50}')

            try:
                daily_hourly_map_dw        = daily_hourly_maps_dw[day_idx]
                daily_aggregated_tensor_dw = daily_aggregated_tensors_dw[day_idx].to(DEVICE_DW)

                if daily_aggregated_tensor_dw.shape[0] == 0:
                    print(f"Skipping Day {day_idx+1}: No data.")
                    continue

                # ── Initial values ────────────────────────────────────────────
                init_sigmasq    = 13.059
                init_range_lat  = 0.154
                init_range_lon  = 0.195
                init_range_time = 1.0
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

                params_list = [
                    torch.tensor([val], requires_grad=True, dtype=torch.float64, device=DEVICE_DW)
                    for val in initial_vals
                ]

                raw_init_floats = [
                    init_sigmasq, init_range_lat, init_range_lon, init_range_time,
                    init_advec_lat, init_advec_lon, init_nugget
                ]

                # ── Preprocess: raw passthrough (demean, no filter) ───────────
                db = debiased_whittle.debiased_whittle_preprocess(
                    daily_aggregated_tensors_dw, daily_hourly_maps_dw,
                    day_idx=day_idx, params_list=raw_init_floats,
                    lat_range=[-3, 2], lon_range=[121.0, 131.0]
                )

                cur_df = db.generate_spatially_filtered_days(-3, 2, 121, 131).to(DEVICE_DW)

                unique_times = torch.unique(cur_df[:, TIME_COL])
                time_slices_list = [cur_df[cur_df[:, TIME_COL] == t_val] for t_val in unique_times]

                print("Pre-computing J-vector (Hamming taper)...")
                J_vec, n1, n2, p_time, taper_grid, obs_masks = dwl.generate_Jvector_tapered_mv(
                    time_slices_list, tapering_func=TAPERING_FUNC,
                    lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL, device=DEVICE_DW
                )

                I_sample = dwl.calculate_sample_periodogram_vectorized(J_vec)
                taper_autocorr_grid = dwl.calculate_taper_autocorrelation_multivariate(
                    taper_grid, obs_masks, n1, n2, DEVICE_DW)
                del obs_masks

                # ── Compute K1, K2 from grid size and alpha ───────────────────
                K1 = int(np.floor(n1 * alpha))
                K2 = int(np.floor(n2 * alpha))
                n_excluded = (K1 + 1) * (K2 + 1)
                n_total    = n1 * n2
                print(f"Grid: {n1}×{n2}, alpha={alpha}  →  K1={K1}, K2={K2}")
                print(f"Excluding {n_excluded}/{n_total} low-freq terms "
                      f"({100*n_excluded/n_total:.1f}%): "
                      f"k1 in [0,{K1}] × k2 in [0,{K2}]")

                # ── Optimize ──────────────────────────────────────────────────
                optimizer_dw = torch.optim.LBFGS(
                    params_list, lr=1.0, max_iter=20, max_eval=100,
                    history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5
                )

                start_time = time.time()

                nat_str, phi_str, raw_str, loss, steps = dwl.run_lbfgs_tapered(
                    params_list=params_list, optimizer=optimizer_dw, I_sample=I_sample,
                    n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=taper_autocorr_grid,
                    max_steps=DWL_MAX_STEPS, device=DEVICE_DW,
                    K1=K1, K2=K2
                )

                epoch_time = time.time() - start_time
                print(f"{cov_name} finished in {epoch_time:.2f}s.")
                print(f"Natural Scale: {nat_str}")
                print(f"Phi Scale    : {phi_str}")

                loss_scaled = loss * n1 * n2 * 8
                dw_estimates_values = [p.item() for p in params_list]
                dw_estimates_loss   = dw_estimates_values + [loss_scaled]

                grid_res = int(daily_aggregated_tensor_dw.shape[0] / 8)

                res = alg_optimization(
                    day=f"{year}-{month:02d}-{day_idx+1}",
                    cov_name=cov_name,
                    space_size=grid_res,
                    lr=1.0,
                    params=dw_estimates_loss,
                    time=epoch_time,
                    rmsre=0.0
                )

                common_filename = f"real_dw_nolf_{alpha_tag}_july_22_23_24_25"

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

                print(f"Saved → {common_filename}.[json/csv]")

            except Exception as e:
                import traceback
                print(f"Day {day_idx+1} Failed: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    app()
