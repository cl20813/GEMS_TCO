# M3 / Q3 / Q5 quick benchmark (090126)

This is a deliberately small screening experiment, not the final confirmatory study.
Q3 uses an unweighted 3x3 fit; Q5 uses a Gaussian-weighted 5x5 fit with sigma=1.25 cells.
Both use cell-scaled coordinates and the same positive-curvature, condition-number, and half-cell safeguards.

## Synthetic initializer summary

| method | n | mean_seed_error_cells | median_seed_error_cells | mean_angle_error_deg | correct_quadrant_rate | safeguard_accept_rate | mean_initializer_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M3_fft | 4 | 0.680389 | 0.559359 | 26.2416 | 0.75 | 0 | 0.00463108 |
| M3_fft_Q3 | 4 | 0.60385 | 0.537985 | 26.0211 | 0.75 | 1 | 0.0047811 |
| M3_fft_Q5 | 4 | 0.530024 | 0.353927 | 26.7451 | 0.75 | 0.75 | 0.0047347 |

## Synthetic short full-fit summary

| method | n_fits | success_rate | convergence_rate | evaluation_budget_hit_rate | mean_common_eval_nll | mean_final_parameter_error | mean_final_advection_error | mean_optimizer_iterations | mean_fit_s | mean_end_to_end_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3_fft | 4 | 1 | 0 | 1 | 1.25814 | 0.591437 | 0.181167 | 10.25 | 4.08557 | 4.18101 |
| M3_fft_Q3 | 4 | 1 | 0 | 1 | 1.25732 | 0.366129 | 0.0307265 | 9.75 | 4.04996 | 4.14624 |
| M3_fft_Q5 | 4 | 1 | 0 | 1 | 1.25699 | 0.396141 | 0.0214037 | 9.25 | 4.04169 | 4.13753 |

## Real-data initializer summary

| method | n_days | mean_initializer_s | safeguard_accept_rate | mean_movement_from_m3_cells | max_movement_from_m3_cells |
| --- | --- | --- | --- | --- | --- |
| M3_fft | 3 | 0.0199233 | 0 | 0 | 0 |
| M3_fft_Q3 | 3 | 0.0200692 | 0.666667 | 0.121157 | 0.311988 |
| M3_fft_Q5 | 3 | 0.0200253 | 0.666667 | 0.114155 | 0.299608 |

## Real-data short full fit

| day | method | status | seed_lat | seed_lon | seed_total_s | subgrid_used | selection_reason | common_eval_nll | common_corridor_seed_lat | common_corridor_seed_lon | common_precompute_s | final_precompute_s | end_to_end_s | own_initial_nll | own_final_nll | own_nll_decrease | final_fit_s | optimizer_n_iter | optimizer_func_evals | final_grad_inf | finite_fit | converged_grad | hit_evaluation_budget | est_sigmasq | est_range_lat | est_range_lon | est_range_time | est_advec_lat | est_advec_lon | est_nugget |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-07-02 | M3_fft | ok | -0.044 | -0.252 | 0.0191717 | False | discrete_fft_argmin | 1.40392 | 0.0218 | -0.1689 | 1.25473 | 1.38961 | 40.411 | 1.49706 | 1.40351 | 0.0935508 | 39.0022 | 7 | 8 | 0.0270677 | True | False | True | 15.4821 | 0.133753 | 0.174934 | 1.18524 | -0.0295782 | -0.234869 | 0.299929 |
| 2024-07-02 | M3_fft_Q3 | ok | -0.0305289 | -0.248219 | 0.0192997 | True | accepted | 1.40363 | 0.0218 | -0.1689 | 1.25473 | 1.5295 | 40.49 | 1.49684 | 1.40327 | 0.0935674 | 38.9412 | 6 | 8 | 0.0220973 | True | False | True | 15.7105 | 0.13379 | 0.174646 | 1.18541 | -0.0219592 | -0.216893 | 0.299757 |
| 2024-07-02 | M3_fft_Q5 | ok | -0.0315212 | -0.245915 | 0.0192719 | True | accepted | 1.40392 | 0.0218 | -0.1689 | 1.25473 | 1.5275 | 40.7218 | 1.49688 | 1.40356 | 0.0933123 | 39.175 | 6 | 8 | 0.0196806 | True | False | True | 16.7757 | 0.143069 | 0.187593 | 1.24294 | -0.012171 | -0.225972 | 0.291928 |

## Screening interpretation

- Synthetic seed accuracy was best on average for M3_fft_Q5 (0.530 grid cells), but n=4 is only a screening sample.
- Short synthetic full fits gave the lowest mean final advection error to M3_fft_Q5 and the lowest seven-parameter error to M3_fft_Q3.
- Every synthetic fit hit the short evaluation budget and none met the gradient criterion; downstream values are budget-matched screening results, not converged estimates.
- On real data, both refinements fell back to M3 on 1/3 days; when accepted, their correction was at most 0.312 grid cells.
- For the one non-fallback real-data short fit, M3_fft_Q3 had the lowest fixed-common-corridor NLL (1.403625); all methods still hit the evaluation budget.

Real-data common NLL is evaluated on the pre-specified production corridor seed (0.0218, -0.1689), independent of the three candidates. Real data have no known advection truth, so initializer accuracy claims come from the synthetic component.
