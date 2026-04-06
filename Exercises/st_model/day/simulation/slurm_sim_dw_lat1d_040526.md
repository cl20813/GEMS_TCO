# Simulation Study — DW lat-only 1D diff (v040526)

**비교 대상:** `sim_three_model_comparison_031926.py`의 DW 컬럼  
**변경점:** `debiased_whittle` → `debiased_whittle_lat1d`  
  (filter: `[-2,1;1,0]` → `[-1;1]` lat-only, output grid: (nlat-1)×(nlon-1) → (nlat-1)×nlon)

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# src 전체 전송 (debiased_whittle_lat1d.py 포함)
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

# sim 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw_lat1d_040526.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

---

### 2. Amarel 접속 및 환경 설정

```bash
ssh jl2815@amarel.rutgers.edu
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
conda activate faiss_env
```

---

### 3. Slurm 스크립트 생성

```bash
cd ~/jobscript/tco/gp_exercise
nano sim_dw_lat1d_040526.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_dw_lat1d
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_dw_lat1d_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_dw_lat1d_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu035

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_lat1d_040526.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --num-iters 300 \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --init-noise 0.7 \
    --seed 42

echo "Done: $(date)"
```

```bash
sbatch sim_dw_lat1d_040526.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_lat1d_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```

---

### 5. DW vs DW-lat1d 비교 포인트

| 항목 | DW (old, three-model) | DW-lat1d (new) |
|---|---|---|
| Filter | `[-2,1;1,0]` (2D) | `[-1;1]` (1D lat-only) |
| Output grid | (nlat-1)×(nlon-1) | (nlat-1)×nlon |
| 비교 파일 | `sim_three_model_comparison_*.csv` | `sim_dw_lat1d_*.csv` |
| 예상 RMSRE | 기준값 | 낮아야 함 (lon signal 보존) |
| range_lon 추정 | lon filtering으로 bias | 더 정확해야 함 |
| advec_lon 추정 | lon filtering으로 영향 | 더 정확해야 함 |

### 참고: init noise 설명
- `--init-noise 0.7` → log-space에서 ±0.7 uniform → 원래값의 exp(0.7)≈2배 범위
- sigmasq: 10 → [5, 20] 범위 내 랜덤 초기값
- advec_lat: ±2×scale (scale=max(|true|, 0.05)) 범위
