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
| Fitted estimates (DW) | `outputs/day/july_22_24_25/real_dw_july_22_24_25.csv` |
| Fitted estimates (Vecc) | `outputs/day/july_22_24_25/real_vecc_july_22_24_25_h1000_mm16.csv` |
| GIM output | `outputs/day/GIM/GIM_{day}_nsims{n}_local.csv` |

---

## Bootstrap pipeline (updated)

```
high-res FFT field  (lat × lat_factor,  lon × lon_factor)
        │
        │  nearest high-res grid point per valid obs  (+nugget)
        │
        ├──► irr_map  (src lat/lon 유지, value만 교체)   →  Vecchia-Irr loss
        │
        └──► step3 re-grid  (obs→cell, 1:1, nearest wins)  →  DW loss  [lat, lon, val, t]
```

- **공정성**: DW도 동일한 irr→step3 파이프라인을 통과 (실제 데이터와 동일한 missingness)
- **Vecchia**: bilinear interpolation 대신 nearest-point 샘플링 → smoothing 편향 제거
- `precompute_mapping_indices` 1회 실행, 양 모델 bootstrap에 공유

---

## Quick test (day 1, 20 sims, low-res)
```bash
python sim_GIM_vecc_irr_dw_local_031926.py \
    --sample-year 2024 \
    --sample-day 1 \
    --month 7 \
    --num-sims 20 \
    --lat-factor 4 \
    --lon-factor 2
```

## Full run (100 sims, higher-res — takes longer on CPU)
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
    --daily-stride 2 \
    --num-sims 100 \
    --lat-factor 10 \
    --lon-factor 4
```

---

## Transfer to Amarel (after local validation)
```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_GIM_vecc_irr_dw_031926.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco
```
