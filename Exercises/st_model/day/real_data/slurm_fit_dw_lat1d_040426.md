# Debiased Whittle — Lat-only 1D Diff (v06)

**변경 사항 (v05 대비)**
- 필터: `[-2,1;1,0]` → `[[-1],[1]]` (lat-only 1D diff)
- 출력 그리드: `(nlat-1) × (nlon-1)` → `(nlat-1) × nlon` (lon 차원 유지)
- 공분산: `Cov_Z(u) = 2C_X(u) - C_X(u-δ₁) - C_X(u+δ₁)` (lat shift만)
- 출력 파일: `real_dw_lat1d_july_22_23_24_25.[json/csv]`
- 출력 경로: `.../estimates/day/july_22_23_24_25_lat1d/`

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# src 전체 전송 (debiased_whittle_lat1d.py 포함)
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

# fit 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/fit_D_whittle_day_v06_lat1d_040426.py" \
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
nano fit_dw_lat1d_040426.sh
```

```bash


#!/bin/bash
#SBATCH --job-name=dw_lat1d
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_lat1d_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_lat1d_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu038

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/fit_D_whittle_day_v06_lat1d_040426.py \
    --v 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --no-keep-exact-loc

echo "Done: $(date)"

```






```bash
sbatch fit_dw_lat1d_040426.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25_lat1d/ \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/july_22_23_24_25_lat1d/"
```

---

### 5. v05 vs v06 비교 체크 포인트

| 항목 | v05 (old [-2,1;1,0]) | v06 (lat1d [-1;1]) |
|---|---|---|
| 출력 파일 | `real_dw_july_22_23_24_25` | `real_dw_lat1d_july_22_23_24_25` |
| 그리드 크기 | nlat-1 × nlon-1 | nlat-1 × nlon |
| 예상 loss | 기준값 | 더 낮아야 함 (불필요한 필터링 제거) |
| 추정 range_lat | v05 값 | 비슷하거나 약간 클 수 있음 |
| 추정 range_lon | v05 값 | lon 신호 보존으로 달라질 수 있음 |
