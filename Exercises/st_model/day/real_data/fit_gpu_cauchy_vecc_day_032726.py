"""
fit_gpu_cauchy_vecc_day_032726.py

Fits the Generalized Cauchy Vecchia model to all GEMS TCO days
(July 2022–2025) using L-BFGS on GPU.

Identical pipeline to fit_gpu_vecc_day_v05_031826.py except:
  - Kernel: C(d) = σ²/(1 + d·φ₂)^β  (Generalized Cauchy)
  - β controlled by --gc-beta (default 1.0)
  - Saves final Vecchia NLL alongside estimates for direct
    likelihood comparison with the Matérn fit

Output CSV columns:
  day, sigma, range_lat, range_lon, range_time,
  advec_lat, advec_lon, nugget, gc_beta,
  vecchia_nll, elapsed, cov_name
"""
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import typer
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia_cauchy
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    gc_beta:        float      = typer.Option(1.0,  help="Cauchy tail exponent β (0.5 / 1.0 / 2.0)"),
    v:              float      = typer.Option(0.5,  help="Smoothness (passed to VecchiaBatched, unused by Cauchy kernel)"),
    lr:             float      = typer.Option(1.0,  help="L-BFGS learning rate"),
    space:          List[str]  = typer.Option(['1,1'], help="Spatial resolution"),
    days:           List[str]  = typer.Option(['0,28'], help="Start,End day index (0-based, exclusive end)"),
    mm_cond_number: int        = typer.Option(16,   help="Vecchia neighbors"),
    nheads:         int        = typer.Option(1000, help="Head points (exact GP block)"),
    limit_a:        int        = typer.Option(16,   help="Set A neighbors"),
    limit_b:        int        = typer.Option(16,   help="Set B neighbors"),
    limit_c:        int        = typer.Option(16,   help="Set C neighbors"),
    daily_stride:   int        = typer.Option(2,    help="Set C stride"),
    years:          List[str]  = typer.Option(['2022,2023,2024,2025'], help="Comma-separated years"),
    month:          int        = typer.Option(7,    help="Month"),
    keep_exact_loc: bool       = typer.Option(True, help="Use actual obs coordinates"),
) -> None:

    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e  = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years_list = [y.strip() for y in years[0].split(',')]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {DEVICE}")
    print(f"Kernel  : Cauchy β={gc_beta}")
    print(f"Years   : {years_list}  month={month}  days={days_list}")
    print(f"nheads={nheads}  mm={mm_cond_number}  A/B/C={limit_a}/{limit_b}/{limit_c}  stride={daily_stride}")

    LBFGS_LR        = lr
    LBFGS_MAX_STEPS = 3
    LBFGS_HIST      = 100
    LBFGS_MAX_EVAL  = 30

    output_path = Path(config.amarel_estimates_day_path) / "july_22_23_24_25"
    output_path.mkdir(parents=True, exist_ok=True)

    beta_tag    = f"b{int(gc_beta * 10):02d}"   # 0.5→b05, 1.0→b10, 2.0→b20
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_file    = output_path / f"real_cauchy_{beta_tag}_july_22_23_24_25_h{nheads}_mm{mm_cond_number}.csv"
    cov_name    = f"Cauchy_b{beta_tag}_mm{mm_cond_number}_A{limit_a}B{limit_b}C{limit_c}"

    data_loader = load_data_dynamic_processed(config.amarel_data_load_path)
    records     = []

    # Load existing CSV to resume if interrupted
    if csv_file.exists():
        records = pd.read_csv(csv_file).to_dict('records')
        done_days = {r['day'] for r in records}
        print(f"Resuming — {len(done_days)} days already done.")
    else:
        done_days = set()

    for year in years_list:
        print(f'\n{"="*60}')
        print(f'=== Year {year} ===')
        print(f'{"="*60}')

        df_map, ord_mm, nns_map, monthly_mean = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=lat_lon_resolution,
            mm_cond_number=mm_cond_number,
            years_=[year], months_=[month],
            lat_range=[-3, 2], lon_range=[121, 131],
            is_whittle=False
        )

        # Pre-load all daily tensors for this year
        print("Pre-loading daily tensors...")
        daily_maps = []
        for day_index in range(31):
            hour_indices = [day_index * 8, (day_index + 1) * 8]
            day_map, _ = data_loader.load_working_data(
                df_map, monthly_mean, hour_indices,
                ord_mm=ord_mm, dtype=torch.float64, keep_ori=keep_exact_loc
            )
            daily_maps.append(day_map)

        for day_idx in days_list:
            day_str = f"{year}-{month:02d}-{day_idx + 1}"

            if day_str in done_days:
                print(f"  [SKIP] {day_str} already done.")
                continue

            print(f'\n{"="*50}')
            print(f'--- Cauchy β={gc_beta}: {day_str} ---')
            print(f'{"="*50}')

            try:
                day_map = {k: v.to(DEVICE) for k, v in daily_maps[day_idx].items()}
                if not day_map:
                    print(f"  No data for {day_str}, skipping.")
                    continue

                # ── Initial values (same as Matérn fit for fair comparison) ──────
                init_sigmasq    = 13.059
                init_range_lat  = 0.2
                init_range_lon  = 0.25
                init_range_time = 1.5
                init_advec_lat  = 0.0218
                init_advec_lon  = -0.1689
                init_nugget     = 0.247

                init_phi2 = 1.0 / init_range_lon
                init_phi1 = init_sigmasq * init_phi2
                init_phi3 = (init_range_lon / init_range_lat)  ** 2
                init_phi4 = (init_range_lon / init_range_time) ** 2

                initial_vals = [
                    np.log(init_phi1), np.log(init_phi2), np.log(init_phi3),
                    np.log(init_phi4), init_advec_lat, init_advec_lon, np.log(init_nugget)
                ]
                params_list = [
                    torch.tensor([val], requires_grad=True, dtype=torch.float64, device=DEVICE)
                    for val in initial_vals
                ]

                model = kernels_vecchia_cauchy.fit_cauchy_vecchia_lbfgs(
                    smooth=v, gc_beta=gc_beta,
                    input_map=day_map, nns_map=nns_map,
                    mm_cond_number=mm_cond_number, nheads=nheads,
                    limit_A=limit_a, limit_B=limit_b, limit_C=limit_c,
                    daily_stride=daily_stride,
                )

                optimizer = model.set_optimizer(
                    params_list, lr=LBFGS_LR,
                    max_iter=LBFGS_MAX_EVAL, history_size=LBFGS_HIST
                )

                t0  = time.time()
                out, _ = model.fit_vecc_lbfgs(params_list, optimizer,
                                              max_steps=LBFGS_MAX_STEPS, grad_tol=1e-7)
                elapsed = time.time() - t0

                # out = [log_phi1, ..., log_phi7_params (7), final_nll]
                raw_params  = out[:-1]   # 7 log-space params
                vecchia_nll = out[-1]    # minimized negative log-likelihood
                est = model._convert_params(raw_params)

                record = {
                    'day':          day_str,
                    'sigma':        round(est['sigma_sq'],   6),
                    'range_lat':    round(est['range_lat'],  6),
                    'range_lon':    round(est['range_lon'],  6),
                    'range_time':   round(est['range_time'], 6),
                    'advec_lat':    round(est['advec_lat'],  6),
                    'advec_lon':    round(est['advec_lon'],  6),
                    'nugget':       round(est['nugget'],     6),
                    'gc_beta':      gc_beta,
                    'vecchia_nll':  round(float(vecchia_nll), 6),
                    'elapsed':      round(elapsed, 2),
                    'cov_name':     cov_name,
                }
                records.append(record)
                done_days.add(day_str)

                pd.DataFrame(records).to_csv(csv_file, index=False)
                print(f"  Saved {day_str}  NLL={vecchia_nll:.4f}  ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  [FAIL] {day_str}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

    print(f"\nDone. {len(records)} days saved to {csv_file.name}")


if __name__ == "__main__":
    app()
