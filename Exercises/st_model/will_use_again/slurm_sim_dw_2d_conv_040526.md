# Simulation Study — DW 2D Conv filter [[-1,1],[1,-1]] (v040526)

**비교 대상:** `sim_dw_lat1d_040526.py`  
**변경점:** `debiased_whittle_lat1d` → `debiased_whittle_2d_conv`  
  (filter: `[-1;1]` lat-only → `[[-1,1],[1,-1]]` 2D separable, output grid: (nlat-1)×nlon → (nlat-1)×(nlon-1))

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# 새 모듈 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/debiased_whittle_2d_conv.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

# sim 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again/sim_dw_2d_conv_040526.py" \
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
nano sim_dw_2d_conv_040526.sh
sbatch sim_dw_2d_conv_040526.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_dw_2d_conv
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_dw_2d_conv_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_dw_2d_conv_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu039

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_2d_conv_040526.py \
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
sbatch sim_dw_2d_conv_040526.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_2d_conv_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```

---

### 5. DW-lat1d vs DW-2d_conv 비교 포인트

| 항목 | DW-lat1d | DW-2d_conv |
|---|---|---|
| Filter | `[-1;1]` (1D lat) | `[[-1,1],[1,-1]]` (2D separable) |
| Transfer fn | `4sin²(ω₁/2)` | `4sin²(ω₁/2)·4sin²(ω₂/2)` |
| Output grid | (nlat-1)×nlon | (nlat-1)×(nlon-1) |
| DC exclusion | ω₁=0 row만 | ω₁=0 row + ω₂=0 col |
| 비교 파일 | `sim_dw_lat1d_*.csv` | `sim_dw_2d_conv_*.csv` |
| 예상 range_time | lat1d보다 발산 가능 | lon 방향도 필터링 → bias 가능 |
| 예상 range_lon | 더 정확 | lon 신호 제거로 bias 가능 |
