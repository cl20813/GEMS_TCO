# GIM Local Run Guide

## Setup
```bash
conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
```

## Data paths (local)
| 항목 | 경로 |
|------|------|
| GEMS raw data | `/Users/joonwonlee/Documents/GEMS_DATA/` |
| Fitted estimates (DW) | `outputs/day/july_22_23_24_25/real_dw_july_22_23_24_25.csv` |
| Fitted estimates (Vecc) | `outputs/day/july_22_23_24_25/real_vecc_july_22_23_24_25_h1000_mm16.csv` |
| GIM output | `outputs/day/GIM/GIM_{day}_obsJ_local.csv` |

---

## Observed J (no bootstrap)

J is computed from the real data directly (Varin, Reid & Firth 2011):
- **DW**: per-frequency Jacobian — `jac.T @ jac / n_freq²`  (n1×n2 − 1 terms, DC excluded)
- **Vecchia**: per-unit Jacobian — `jac.T @ jac / N_units²`  (N_heads + N_tails terms, beta fixed at MLE)

No `--num-sims`, `--lat-factor`, `--lon-factor` args needed.

---

## Quick test (day 1)
```bash
python sim_GIM_vecc_irr_dw_local_031926.py \
    --sample-year 2024 \
    --sample-day 1 \
    --month 7
```

## Full run
```bash
python sim_GIM_vecc_irr_dw_local_031926.py \
    --sample-year 2025 \
    --sample-day 2 \
    --month 7 \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 200 \
    --limit-a 6 \
    --limit-b 6 \
    --limit-c 6 \
    --daily-stride 2
```

---

## Transfer to Amarel (after local validation)
```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_GIM_vecc_irr_dw_031926.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco
```
