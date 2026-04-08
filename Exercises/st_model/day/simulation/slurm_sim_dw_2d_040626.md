# Simulation Study — DW 2D gridification bias test v040626

## 모델 개요

**연구 질문:** step3 gridification (obs→cell) 이 DW 추정량에 편향을 도입하는가?

| 모델 | 파이프라인 | 데이터 | 설명 |
|---|---|---|---|
| **DW_2d_loc** | FFT high-res → real GEMS obs → step3 (1:1) → target grid | partial NaN | 기존 three-model 파이프라인과 동일 |
| **DW_2d_grid** | FFT high-res → **target grid 직접 샘플** (step3 skip) | complete grid | gridification 없음 → baseline |

두 모델 모두 `debiased_whittle_2d_conv` (2D separable filter `[-1,1;1,-1]`) 사용.  
동일 FFT 필드 + 동일 초기값 → gridification 효과만 분리 가능.

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# src 전체 전송 (debiased_whittle_2d_conv.py 포함)
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

# sim 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw_2d_040626.py" \
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
nano sim_dw_2d_040626.sh
sbatch sim_dw_2d_040626.sh
```

```bash

#!/bin/bash
#SBATCH --job-name=sim_dw_2d
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_dw_2d_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_dw_2d_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_2d_040626.py \
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
sbatch sim_dw_2d_040626.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_2d_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```

---

### 5. 비교 포인트

| 항목 | DW_2d_loc | DW_2d_grid |
|---|---|---|
| 데이터 소스 | GEMS obs 위치 → step3 gridify | target grid 직접 |
| 결측 셀 | 있음 (obs 없는 셀 NaN) | 없음 (완전 격자) |
| 공분산 | 2D diff 16-term | 동일 |
| FFT 필드 | 동일 | 동일 |
| 초기값 | 동일 | 동일 |
| **예상 편향 원인** | step3 1:1 할당 + NaN 패턴 | 없음 (baseline) |

**핵심 확인 사항:**  
- DW_2d_loc vs DW_2d_grid RMSRE 차이 → gridification bias 정량화  
- advec_lat/lon: step3 패턴이 방향성 추정에 미치는 영향  
- nugget: 셀 내 집계 효과 (obs→cell 할당 시 nugget 추정 왜곡 여부)

### 참고: 계산 비용

iteration당 두 모델 순차 실행 → three-model 스크립트 대비 ~2/3 비용.  
300 iters / 48h 할당으로 충분 예상.
