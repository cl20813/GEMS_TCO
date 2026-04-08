# Debiased Whittle — RAW baseline (no filter) Real Data Fit (v05, 040626)

**설계 개요:**

| 항목 | DW_2dconv | DW_mixed | **DW_raw (this)** |
|---|---|---|---|
| 필터 | `[[-1,+1],[+1,-1]]` | low-freq raw + high-freq diff | **없음 (H=1)** |
| 출력 그리드 | (nlat-1)×(nlon-1) | nlat×nlon (raw) + (nlat-1)×(nlon-1) (diff) | **nlat×nlon** |
| Cov 항 수 | 9항 | 1항 (raw) + 9항 (diff) | **1항** |
| DC 처리 | ω₁=0 row + ω₂=0 col 제외 | 동일 | **per-slice demean + ω=(0,0) 제외** |
| 계산 비용 | 중간 | 높음 (~2×) | **낮음** |
| 목적 | 표준 비교 | 저주파 보존 | **Baseline (성능 상한)** |

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# raw 모듈 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/debiased_whittle_raw.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

# fit 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/fit_dw_raw_day_v05_040626.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/
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
nano fit_dw_raw_040626.sh
sbatch fit_dw_raw_040626.sh

```

```bash
#!/bin/bash
#SBATCH --job-name=dw_raw
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_raw_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_raw_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu043

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/fit_dw_raw_day_v05_040626.py \
    --v 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --no-keep-exact-loc

echo "Done: $(date)"
```

```bash
sbatch fit_dw_raw_040626.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25_raw/ \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/july_22_23_24_25_raw/"
```

---

### 5. 세 모델 비교 포인트

| 항목 | DW_2dconv | DW_mixed (α=0.20) | DW_raw |
|---|---|---|---|
| 그리드 감소 | lat-1, lon-1 | 없음 (raw piece) | 없음 |
| 공분산 항 수 | 9 | 1+9 | 1 |
| DC 문제 | axis 제외 | axis 제외 + raw 커버 | demean + DC 제외 |
| 저주파 advection | 손실 가능 | 보존 | 보존 |
| 계산 비용 | 중간 | 높음 | **낮음** |
| 기대 역할 | 표준 | 개선 | **Baseline** |
