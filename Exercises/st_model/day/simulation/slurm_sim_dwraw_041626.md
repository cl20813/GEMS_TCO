# Simulation Study — DW Complex (no spatial filter, complex Whittle) v040826

**핵심 목적:** Real DW (`fft_result.real`) vs Complex DW (`fft_result` 복소수 유지)  
→ advec_lat / advec_lon **부호 복원율** 비교

**비교 대상:**
| 모델 | Expected Periodogram | 부호 구분 |
|---|---|---|
| DW_raw (real) | `fft_result.real * (1/4π²)` | 불가 — cos 대칭 |
| **DW_complex (this)** | **`fft_result * (1/4π²)`, Hermitian-symmetrize** | **가능 — Im 부분 보존** |

**수학적 근거:**
- Real: `F(ω; -α) = F(ω; α)` (cos 대칭) → `L(-α) = L(α)`
- Complex: `F(ω; -α) = conj(F(ω; α)) ≠ F(ω; α)` (Im ≠ 0이면) → `L(-α) ≠ L(α)`

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# src 전체 전송 (debiased_whittle_raw.py 포함)
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

# sim 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dwraw_041626.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

> **Note:** GEMS 실측 obs 패턴 불필요. FFT field를 target regular grid에 직접 생성하므로
> `load_data_dynamic_processed` 미사용. 단일 모델 `DW_complex`.

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
nano sim_dw_complex_040826.sh
sbatch sim_dw_complex_040826.sh
```

```bash

#!/bin/bash
#SBATCH --job-name=sim_dw_complex
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_dw_complex_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_dw_complex_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu031

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dwraw_041626.py \
    --num-iters 300 \
    --t-steps 8 \
    --lat-range "-3,2" \
    --lon-range "121,131" \
    --init-noise 0.7 \
    --seed 42

echo "Done: $(date)"


```



```bash
sbatch sim_dw_complex_040826.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_complex_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```

---

### 5. 핵심 비교 포인트

| 항목 | DW_raw (real) | DW_complex (this) |
|---|---|---|
| Expected periodogram 타입 | real symmetric | complex Hermitian |
| 부호 구분 가능 | 불가 (cos 대칭) | 가능 (Im 보존) |
| advec 부호 복원율 (기대) | ~50% (random) | ~100% |
| 계산 비용 | 기준 | 약간 증가 (complex linalg) |
| F(ω; -α) | = F(ω; α) | = conj(F(ω; α)) ≠ F(ω; α) |

**핵심 출력 컬럼 (sim_dw_complex_*.csv):**
- `advec_lat_sign_ok`: 1 = 부호 맞음, 0 = 틀림
- `advec_lon_sign_ok`: 1 = 부호 맞음, 0 = 틀림
- `advec_lat_sign_rate`, `advec_lon_sign_rate` (summary CSV)

**기대 결과:**
- DW_complex: sign_rate ≈ 0.95–1.00
- DW_raw (비교용): sign_rate ≈ 0.50 (부호 ambiguity 확인)
