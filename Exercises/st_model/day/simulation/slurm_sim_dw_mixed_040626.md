# Simulation Study — DW Mixed-frequency (composite-likelihood) v040626

## 모델 개요

**핵심 아이디어:** 공간 주파수를 두 구간으로 분리, 각 구간에 맞는 periodogram 사용

```
L_mixed(θ) = Σ_{ω∈Ω_L} ℓ_raw(ω; θ)   +   Σ_{ω∈Ω_H} ℓ_diff(ω; θ)
```

| 구간 | 마스크 | 데이터 | 공분산 모델 |
|---|---|---|---|
| **Ω_L (저주파)** | k1≤K1, k2≤K2 | raw X (demean) | C_X(u·δ1, v·δ2, τ) 직접 |
| **Ω_H (고주파)** | 나머지 | 2D diff Z | ΣΣ h·h·C_X cross-terms |

**freq_alpha α:** K1=floor(n1·α), K2=floor(n2·α)  
도메인 n1≈114, n2≈159 기준:  
- α=0.15 → K1=17, K2=23 (|Ω_L|≈390)
- α=0.20 → K1=22, K2=31 (|Ω_L|≈682)  ← **기본값**
- α=0.25 → K1=28, K2=39 (|Ω_L|≈1091)

**통계적 근거:** 각 piece의 estimating equation이 θ_true에서 unbiased → consistent.  
분산: Godambe matrix (composite likelihood, Varin et al. 2011 Stat. Sinica).

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# src 전체 전송 (debiased_whittle_mixed.py 포함)
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

# sim 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw_mixed_040626.py" \
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
nano sim_dw_mixed_040626.sh
sbatch sim_dw_mixed_040626.sh
```

```bash

#!/bin/bash
#SBATCH --job-name=sim_dw_mixed
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_dw_mixed_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_dw_mixed_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_mixed_040626.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --num-iters 300 \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --init-noise 0.7 \
    --seed 42 \
    --freq-alpha 0.20

echo "Done: $(date)"

```

#### freq-alpha 비교 실험 (선택)

alpha 값을 바꾸면서 비교할 경우:

```bash
# alpha = 0.15
srun python .../sim_dw_mixed_040626.py --freq-alpha 0.15 --num-iters 300 --seed 42

# alpha = 0.25  
srun python .../sim_dw_mixed_040626.py --freq-alpha 0.25 --num-iters 300 --seed 42
```

출력 CSV에 alpha 태그가 자동 포함됨:  
`sim_dw_mixed_a0p20_MMDDYY.csv`, `sim_dw_mixed_a0p15_MMDDYY.csv`

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_mixed_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```

---

### 5. 4모델 비교 포인트

| 모델 | 파일 | 그리드 | 저주파 처리 | 핵심 변경 |
|---|---|---|---|---|
| DW_2d | `sim_three_model_...` | (n-1)×(m-1) | diff로 제거 | 기준 |
| DW_lat1d | `sim_dw_lat1d_...` | (n-1)×m | lat-only diff | lon 정보 보존 |
| DW_raw | `sim_dw_raw_...` | n×m | demean only | 필터 없음 |
| **DW_mixed** | `sim_dw_mixed_...` | 두 그리드 | **저주파=raw, 고주파=diff** | 이번 |

**advec_lat/lon 추정 개선 여부가 핵심 확인 대상.**

### 참고: 계산 비용

mixed는 iteration당 **두 번의 expected periodogram 계산** (raw + diff) 이 필요하므로  
lat1d 대비 ~2배 시간 예상 → 72h 할당.
