"""
fit_dw_mix_day_v05_040626.py

Debiased Whittle fitting — Mixed-frequency composite likelihood.

  L_mixed(θ) = Σ_{ω∈Ω_L} ℓ_raw(ω; θ)  +  Σ_{ω∈Ω_H} ℓ_diff(ω; θ)

  Ω_L = { (k1,k2) : k1 ≤ K1, k2 ≤ K2 } ∪ {k1=0 row} ∪ {k2=0 col} \\ {(0,0)}
        → identity filter (H=1), Cov: C_X(u·δ1, v·δ2, τ) directly.
        → Preserves low-frequency advection signal.

  Ω_H = complement on 2D-diff grid  (k1>0 AND k2>0, excluding low-freq rectangle)
        → 2D filter [[-1,+1],[+1,-1]], |H(ω)|²=4sin²(ω1/2)·4sin²(ω2/2)
        → Suppresses DC / non-stationarity effects at high frequencies.

  K1 = floor(n1 · α),  K2 = floor(n2 · α)
  Recommended α ∈ {0.15, 0.20, 0.25}.

Module: debiased_whittle_mixed

Compare with:
  fit_dw_2conv_day_v05_040626.py  — full 2D-diff DW (no raw piece)
  fit_dw_lat1d_day_v06_040626.py  — lat-only 1D-diff DW

Usage:
  python fit_dw_mix_day_v05_040626.py --freq-alpha 0.20 --days "0,28" --years "2022,2023,2024,2025"
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
from GEMS_TCO import debiased_whittle_mixed as dwm
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
    freq_alpha: float = typer.Option(0.20,
        help="Low-freq cutoff fraction α: K1=floor(n1·α), K2=floor(n2·α). "
             "Frequencies (k1≤K1, k2≤K2) use raw; rest use 2D-diff. "
             "Recommended: 0.15, 0.20, 0.25."),
) -> None:

    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e  = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years_list = [y.strip() for y in years[0].split(',')]

    DEVICE_DW = torch.device("cpu")
    alpha_tag = str(freq_alpha).replace('.', 'p')

    print(f"Whittle Device  : {DEVICE_DW}")
    print(f"Model           : DW_mixed (composite-likelihood frequency split)")
    print(f"Low-freq (Ω_L)  : raw C_X,  k1≤K1 & k2≤K2  + k1=0 row + k2=0 col \\ DC")
    print(f"High-freq (Ω_H) : 2D-diff [[-1,+1],[+1,-1]],  k1>0 & k2>0, not in rectangle")
    print(f"freq_alpha      : {freq_alpha}  (K1=floor(n1·α), K2=floor(n2·α))")
    print(f"Target Days     : {days_list}")
    print(f"Target Years    : {years_list}")

    month_range = [month]

    output_path = Path(config.amarel_estimates_day_path) / f"july_22_23_24_25_mixed_a{alpha_tag}"
    output_path.mkdir(parents=True, exist_ok=True)

    data_load_instance = load_data_dynamic_processed(config.amarel_data_load_path)

    dwl = dwm.debiased_whittle_likelihood()
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
            print(f'--- DW-mixed a={freq_alpha}: Day {day_idx+1} ({year}-{month:02d}-{day_idx+1}) ---')
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

                # ── RAW preprocessing (identity filter, demean) ───────────────
                db_raw = dwm.debiased_whittle_preprocess_raw(
                    daily_aggregated_tensors_dw, daily_hourly_maps_dw,
                    day_idx=day_idx, params_list=raw_init_floats,
                    lat_range=[-3, 2], lon_range=[121.0, 131.0]
                )
                cur_raw = db_raw.generate_spatially_filtered_days(-3, 2, 121, 131).to(DEVICE_DW)

                unique_times_raw = torch.unique(cur_raw[:, TIME_COL])
                time_slices_raw  = [cur_raw[cur_raw[:, TIME_COL] == t] for t in unique_times_raw]

                print("Pre-computing J-vector (raw, Hamming taper)...")
                J_raw, n1, n2, p_time, taper_raw, obs_masks_raw = dwl.generate_Jvector_tapered_mv(
                    time_slices_raw, tapering_func=TAPERING_FUNC,
                    lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL, device=DEVICE_DW
                )
                I_samp_raw = dwl.calculate_sample_periodogram_vectorized(J_raw)
                taper_auto_raw = dwl.calculate_taper_autocorrelation_multivariate(
                    taper_raw, obs_masks_raw, n1, n2, DEVICE_DW)
                del obs_masks_raw

                # ── DIFF preprocessing (2D filter [[-1,+1],[+1,-1]]) ──────────
                db_diff = dwm.debiased_whittle_preprocess_diff(
                    daily_aggregated_tensors_dw, daily_hourly_maps_dw,
                    day_idx=day_idx, params_list=raw_init_floats,
                    lat_range=[-3, 2], lon_range=[121.0, 131.0]
                )
                cur_diff = db_diff.generate_spatially_filtered_days(-3, 2, 121, 131).to(DEVICE_DW)

                unique_times_diff = torch.unique(cur_diff[:, TIME_COL])
                time_slices_diff  = [cur_diff[cur_diff[:, TIME_COL] == t] for t in unique_times_diff]

                print("Pre-computing J-vector (diff, Hamming taper)...")
                J_diff, n1d, n2d, _, taper_diff, obs_masks_diff = dwl.generate_Jvector_tapered_mv(
                    time_slices_diff, tapering_func=TAPERING_FUNC,
                    lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL, device=DEVICE_DW
                )
                I_samp_diff = dwl.calculate_sample_periodogram_vectorized(J_diff)
                taper_auto_diff = dwl.calculate_taper_autocorrelation_multivariate(
                    taper_diff, obs_masks_diff, n1d, n2d, DEVICE_DW)
                del obs_masks_diff

                # ── Frequency cutoffs ─────────────────────────────────────────
                K1 = int(n1 * freq_alpha)
                K2 = int(n2 * freq_alpha)
                _lm = torch.zeros(n1,  n2,  dtype=torch.bool)
                _lm[:K1+1, :K2+1] = True; _lm[0, :] = True; _lm[:, 0] = True; _lm[0, 0] = False
                _hm = torch.ones(n1d, n2d, dtype=torch.bool)
                _hm[:K1+1, :K2+1] = False; _hm[0, :] = False; _hm[:, 0] = False
                n_low  = int(_lm.sum()); n_high = int(_hm.sum())
                print(f"Raw grid: {n1}×{n2}  Diff grid: {n1d}×{n2d}  "
                      f"K1={K1} K2={K2}  |Ω_L|={n_low}  |Ω_H|={n_high}  total={n_low+n_high}")

                # ── Optimize ──────────────────────────────────────────────────
                optimizer_dw = torch.optim.LBFGS(
                    params_list, lr=1.0, max_iter=20, max_eval=100,
                    history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5
                )

                start_time = time.time()

                nat_str, phi_str, raw_str, loss, steps = dwl.run_lbfgs_mixed(
                    params_list=params_list, optimizer=optimizer_dw,
                    I_samp_raw=I_samp_raw,  I_samp_diff=I_samp_diff,
                    n1=n1,   n2=n2,
                    n1d=n1d, n2d=n2d,
                    p_time=p_time,
                    taper_auto_raw=taper_auto_raw, taper_auto_diff=taper_auto_diff,
                    K1=K1, K2=K2,
                    max_steps=DWL_MAX_STEPS, device=DEVICE_DW
                )

                epoch_time = time.time() - start_time
                print(f"Mixed DW finished in {epoch_time:.2f}s.")
                print(f"Natural Scale: {nat_str}")
                print(f"Phi Scale    : {phi_str}")

                loss_scaled = loss * n1 * n2 * 8
                dw_estimates_values = [p.item() for p in params_list]
                dw_estimates_loss   = dw_estimates_values + [loss_scaled]

                grid_res = int(daily_aggregated_tensor_dw.shape[0] / 8)

                res = alg_optimization(
                    day=f"{year}-{month:02d}-{day_idx+1}",
                    cov_name=f"DW_mixed_a{alpha_tag}",
                    space_size=grid_res,
                    lr=1.0,
                    params=dw_estimates_loss,
                    time=epoch_time,
                    rmsre=0.0
                )

                # append freq metadata to res dict
                res_dict = res.__dict__.copy()
                res_dict['freq_alpha'] = freq_alpha
                res_dict['K1']         = K1
                res_dict['K2']         = K2
                res_dict['n_low']      = n_low
                res_dict['n_high']     = n_high
                res_dict['n1']         = n1
                res_dict['n2']         = n2
                res_dict['n1d']        = n1d
                res_dict['n2d']        = n2d

                common_filename = f"real_dw_mixed_a{alpha_tag}_july_22_23_24_25"

                json_filepath = output_path / f"{common_filename}.json"
                try:
                    current_data = BaseLogger.load_list(json_filepath)
                    if not isinstance(current_data, list):
                        current_data = []
                except Exception:
                    current_data = []
                current_data.append(res_dict)
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
