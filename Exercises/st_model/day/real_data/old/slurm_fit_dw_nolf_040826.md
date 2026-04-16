# Debiased Whittle — NOLF (No-Low-Frequency) Real Data Fit (v05, 040826)

**설계 개요:**

| 항목 | DW_raw | **DW_nolf (this)** | DW_mixed |
|---|---|---|---|
| 필터 | 없음 (H=1) | **없음 (H=1)** | low→raw + high→diff |
| 저주파 처리 | DC(0,0)만 제외 | **K1×K2 박스 전체 제외** | low-freq → raw piece 사용 |
| K1, K2 | — | **floor(n1·α), floor(n2·α)** | — |
| α=0.20 기준 제외 | 1항 | **(K1+1)×(K2+1)≈6항** | 0항 (raw로 포함) |
| 이론적 근거 | — | **Spectral REML (Fuentes 2002)** | composite likelihood |
| 계산 비용 | 낮음 | **낮음** (raw와 동일 구조) | 높음 |
| 목적 | baseline | **저주파 오염 제거** | 절충안 |

**α 선택:**
- `α=0.20` → K1=floor(n1·0.20), K2=floor(n2·0.20) — DW_mixed Ω_L 경계와 동일
- `α=0.10` → minimal exclusion — gradient 제거만
- 두 버전을 나란히 실행해서 α sensitivity 확인

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# nolf 모듈 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/debiased_whittle_nolf.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

# fit 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/fit_dw_nolf_day_v05_040826.py" \
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

### 3. Slurm 스크립트 생성 및 제출

#### α = 0.20 (권장 — DW_mixed 경계 기준)

```bash
cd ~/jobscript/tco/gp_exercise
nano fit_dw_nolf_a0p20_040826.sh
sbatch fit_dw_nolf_a0p20_040826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=dw_nolf20
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_nolf20_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_nolf20_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu033

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/fit_dw_nolf_day_v05_040826.py \
    --v 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --alpha 0.30 \
    --no-keep-exact-loc

echo "Done: $(date)"
```

```bash
sbatch fit_dw_nolf_a0p20_040826.sh
```

#### α = 0.10 (minimal exclusion — sensitivity check)

```bash
nano fit_dw_nolf_a0p10_040826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=dw_nolf10
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_nolf10_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_nolf10_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/fit_dw_nolf_day_v05_040826.py \
    --v 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --alpha 0.10 \
    --no-keep-exact-loc

echo "Done: $(date)"
```

```bash
sbatch fit_dw_nolf_a0p10_040826.sh
```

---

### 4. 모니터링

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/dw_nolf20_<jobid>.out
tail -f /home/jl2815/tco/exercise_output/dw_nolf10_<jobid>.out
```

---

### 5. 결과 전송 (Amarel → mac)

```bash
# α = 0.20
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25_nolf_a0p20/ \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/july_22_23_24_25_nolf_a0p20/"

# α = 0.10
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25_nolf_a0p10/ \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/july_22_23_24_25_nolf_a0p10/"
```

---

### 6. 비교 포인트

| 항목 | DW_raw | DW_nolf α=0.10 | DW_nolf α=0.20 | DW_mixed α=0.20 |
|---|---|---|---|---|
| 제외 항 수 (5×10 grid) | 1 (DC만) | ~2 | ~6 | 0 (low→raw 포함) |
| 저주파 오염 제거 | 없음 | 최소 | DW_mixed 경계 | 없음 (raw 사용) |
| catastrophic failure | 가끔 | 감소 기대 | 크게 감소 기대 | 가끔 (2023-07-12) |
| 계산 구조 | 단순 | 단순 | 단순 | 복잡 (두 piece) |
| 이론 선례 | — | Fuentes 2002 | Fuentes 2002 | composite lik. |

**기대 효과:**
- DW_raw의 large range/sigmasq overestimation days → 줄어들 것
- DW_mixed의 2023-07-12 catastrophic failure → 없어야 함 (diff piece 없으므로)
- advec 추정: raw piece만 쓰므로 DW_raw와 동등 수준 유지
