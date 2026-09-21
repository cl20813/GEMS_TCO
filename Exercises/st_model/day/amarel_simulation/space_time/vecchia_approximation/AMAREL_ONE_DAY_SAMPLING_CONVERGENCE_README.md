# One-day 400/600/900 full-eigen sampling convergence

This experiment uses 2024-07-03 and holds the fitted adapted/fixed lag-6/4/3
parameters and full-data GLS beta fixed. It compares:

- nested common-valid max-min samples of 400, 600, and 900 sites per hour;
- exact-size quasi-regular samples of 400, 600, and 900 sites per hour.

The same spatial sites are used at all eight hours, producing full dense
eigensystems of dimension 3,200, 4,800, and 7,200. If the existing 59-day fit
checkpoint contains 2024-07-03, those fitted results are reused. Otherwise,
only this date's adapted/fixed models are fitted once.

Every covariance eigenvector also receives a physical spatial-roughness proxy
based on a distance-scaled six-nearest-neighbor graph. Consequently, the
result distinguishes a high eigenvalue index from genuinely rapid variation
over latitude/longitude.

## Run

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation
bash scp_vecchia_one_day_sampling_convergence_full_eigen_lag643.sh

ssh jl2815@amarel-new.hpc.rutgers.edu \
  'cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation && sbatch slurm_vecchia_one_day_sampling_convergence_full_eigen_lag643.sh'
```

The job requests four hours, 12 CPU cores, 64 GB RAM, and one GPU. The GPU is
only needed if the fitted checkpoint is missing and the one-day models must be
fitted. Completed design/size/method eigensystems are cached for restart.
The job excludes `gpu017` after a CUDA initialization failure there and runs a
PyTorch CUDA preflight before starting the analysis.

## Outputs

- `sampling_convergence_400_600_900.png`: cumulative curve, D, mean Y-squared,
  and eigenvector spatial-roughness convergence for max-min and regular designs.
- `sampling_geometry_400_600_900.png`: spacing, nominal resolvable wavelength,
  and domain-coverage changes.
- `sampling_layouts_400_600_900.png`: the six actual max-min/regular point layouts.
- `sampling_convergence_metrics.csv`: all scalar diagnostics and timings.
- `sampling_change_tracking.csv`: 400→600, 600→900, and 400→900 curve changes.
- `all_mode_diagnostics.csv`: eigenvalue, residual energy, spatial roughness,
  and temporal roughness for every eigenvector.
- `sampling_selection_manifest.csv`: exact selected grid sites.

Amarel output directory:

```text
/home/jl2815/tco/exercise_output/summer/vecchia_20240703_sampling_convergence_full_eigen_lag643
```
