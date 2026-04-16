# Simulation Study — Vecchia vs DW_raw on Complete Grid (sim_vdw_grid_040626)

## 실험 개요

| 항목 | 내용 |
|---|---|
| **목적** | DW 스펙트럼 근사 오차를 gridification 없는 이상적 조건에서 측정 |
| **데이터** | FFT field → target grid 직접 샘플링 (step3 없음, NaN 없음) |
| **Model 1** | Vecchia (irr API, 완전 그리드 — irr ≡ reg) |
| **Model 2** | DW_raw (identity filter, per-slice demean, DC 제외) |
| **해석** | `DW_raw` error = 순수 스펙트럼 근사 편향 (gridification 효과 제거된 상태) |

**3단계 분해:**
```
DW_raw error (여기)          = DW 스펙트럼 근사 오차   (데이터 완전, gridification 없음)
DW_raw_loc - DW_raw_grid     = gridification bias       (sim_dwraw_041626.py)
DW_raw_loc - Vecchia(grid)   = 총 오차 (근사 + gridification)
```

---

### 1. 파일 전송 (mac → Amarel)

```bash
# sim 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dwraw_vdw_grid_041626.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/

# src 패키지 전송 (debiased_whittle_raw, kernels_vecchia 포함)
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/
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
nano sim_vdw_grid_040626.sh
sbatch sim_vdw_grid_040626.sh


```

```bash

#!/bin/bash
#SBATCH --job-name=vdw_grid
#SBATCH --output=/home/jl2815/tco/exercise_output/vdw_grid_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vdw_grid_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu037

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dwraw_vdw_grid_041626.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 0 \
    --limit-a 20 \
    --limit-b 20 \
    --limit-c 20 \
    --daily-stride 2 \
    --num-iters 300 \
    --lat-range "-3,2" \
    --lon-range "121,131" \
    --init-noise 0.7 \
    --seed 42

echo "Done: $(date)"


```

```bash
sbatch sim_vdw_grid_040626.sh
```

---

### 4. 진행 상황 확인

```bash
# 잡 상태
squeue -u jl2815

# 실시간 출력
tail -f /home/jl2815/tco/exercise_output/vdw_grid_<JOBID>.out

# 중간 결과 확인 (CSV 적재 여부)
ls -lh /home/jl2815/tco/exercise_output/estimates/day/sim_vdw_grid_*.csv
```

---

### 5. 결과 전송 (Amarel → mac)

```bash
# CSV
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vdw_grid_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"

# 플롯
scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/plots/vdw/" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/plots/vdw/"
```

---

### 6. 비교 실험 (선택 — mm-cond-number 민감도)

```bash
# mm-cond-number = 50 (빠른 Vecchia)
srun python .../sim_dwraw_vdw_grid_041626.py --mm-cond-number 50  --num-iters 500 --seed 42

# mm-cond-number = 150 (정확한 Vecchia)
srun python .../sim_dwraw_vdw_grid_041626.py --mm-cond-number 150 --num-iters 500 --seed 42
```

---

### 7. 전체 실험 맵

| 파일 | 데이터 | 비교 | 질문 |
|---|---|---|---|
| `sim_dwraw_vdw_grid_041626.py` | 완전 그리드 | Vecchia vs DW_raw | DW 근사 오차 |
| `sim_dwraw_041626.py` | loc vs grid | DW_raw_loc vs DW_raw_grid | gridification bias (raw DW) |
| `sim_dw_2d_040626.py` | loc vs grid | DW_2d_loc vs DW_2d_grid | gridification bias (2D-diff DW) |
| `sim_dw_mixed_040626.py` | loc vs grid | DW_mixed_loc vs DW_mixed_grid | gridification bias (mixed DW) |
