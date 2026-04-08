# Simulation Study — DW Raw baseline (no spatial filter) v040626

**비교 대상:**
| 모델 | 필터 | 출력 그리드 | 이론 공분산 |
|---|---|---|---|
| DW_2d (three-model) | `[[-1,1],[1,-1]]` (2D) | (nlat-1)×(nlon-1) | `ΣΣ h_ab·h_cd·C_X(u+(a-c)δ1, v+(b-d)δ2, τ)` |
| DW_lat1d | `[-1;1]` (lat-only 1D) | (nlat-1)×nlon | `2C_X - C_X(u-δ1) - C_X(u+δ1)` |
| **DW_raw (this)** | **없음 (identity)** | **nlat×nlon** | **`C_X(u·δ1, v·δ2, τ)` 직접** |

**DC term 처리 (2중 방어):**
1. `apply_raw_passthrough`: 각 time slice의 관측값 spatial mean 차감 (demeaning)
2. `whittle_likelihood_loss_tapered`: ω=(0,0) 제외 (`total_sum - dc_term`)

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# src 전체 전송 (debiased_whittle_raw.py 포함)
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

# sim 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw_raw_040626.py" \
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
nano sim_dw_raw_040626.sh
sbatch sim_dw_raw_040626.sh
```

```bash

#!/bin/bash
#SBATCH --job-name=sim_dw_raw
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_dw_raw_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_dw_raw_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu048

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_raw_040626.py \
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
sbatch sim_dw_raw_040626.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_raw_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```

---

### 5. 세 모델 비교 포인트

| 항목 | DW_2d | DW_lat1d | DW_raw |
|---|---|---|---|
| 그리드 감소 | lat-1, lon-1 | lat-1 | 없음 |
| 공분산 항 수 | 4×4=16 | 2×2=4 | 1 |
| 계산 비용 | 높음 | 중간 | 낮음 |
| Stationarity 가정 | 필요 | 필요 | 필요 |
| DC 문제 | 완전 제거 | lat DC 제거 | demean + 제외 |
| 기대 RMSRE | 기준 | 개선 (lon signal 보존) | ?  (identifiability 핵심 비교) |
| range_lon 편향 | 크다 (lon filter 영향) | 개선 | baseline |
| nugget 추정 | filter 왜곡 | 개선 | baseline |

**핵심 연구 질문:**  
Raw baseline이 differencing 기반 추정자보다 나쁜가, 같은가, 더 좋은가?  
→ Differencing이 non-stationarity 제거의 이점 vs. 정보 손실의 trade-off를 정량화함.

### 참고: init noise 설명
- `--init-noise 0.7` → log-space에서 ±0.7 uniform → 원래값의 exp(0.7)≈2배 범위
- sigmasq: 10 → [5, 20] 범위 내 랜덤 초기값
- advec_lat/lon: ±2×scale (scale=max(|true|, 0.05)) 범위
