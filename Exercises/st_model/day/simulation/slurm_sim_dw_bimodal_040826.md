# sim_dw_bimodal — DW_raw Bimodality Diagnostic (040826)

**목적:**  
advec_lat / advec_lon의 bimodal 추정 분포가  
(a) structural sign ambiguity (두 모드의 DW likelihood가 거의 동일) 인지  
(b) optimization failure (wrong local minimum) 인지 진단.

| 항목 | 내용 |
|---|---|
| 모델 | DW_raw only (Vecchia 없음) |
| 데이터 | direct FFT on target grid (no step3, no NaN) |
| 핵심 출력 | ΔL = L(A) − L(B), A=converged, B=advec sign-flipped |
| bimodal check | `--bimodal-start 50` 이후 매 iteration |
| plots | KDE + scatter(iter×est) + advec 2D + ΔL timeline |

---

### 1. 파일 전송 (mac → Amarel)

```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw_bimodal_040826.py" \
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

```bash
cd ~/jobscript/tco/gp_exercise
nano sim_dw_bimodal_040826.sh
sbatch sim_dw_bimodal_040826.sh
```

```bash

#!/bin/bash
#SBATCH --job-name=dw_bimodal
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_bimodal_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_bimodal_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_bimodal_040826.py \
    --num-iters 1000 \
    --bimodal-start 30 \
    --seed 42

echo "Done: $(date)"


```

```bash
sbatch sim_dw_bimodal_040826.sh
```

---

### 4. 모니터링

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/dw_bimodal_<jobid>.out
```

bimodal check 출력 예시 (iter 50 이후):
```
  ── Bimodal Likelihood Check ──────────────────────────────────
  Point A (converged)  : advec_lat=-0.0193, advec_lon=+0.0812
  Point B (sign-flip)  : advec_lat=+0.0193, advec_lon=-0.0812
  True                 : advec_lat=+0.0218, advec_lon=-0.1689
  L(A) = -1.234567   L(B) = -1.234512   ΔL = L(A)-L(B) = -0.000055
  Verdict: STRUCTURAL SYMMETRY  (|ΔL| < 0.5 — both modes equally valid)
  ──────────────────────────────────────────────────────────────
```

---

### 5. 결과 전송 (Amarel → mac)

```bash
# CSV 결과
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_bimodal_*.csv \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"

# Plots
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/plots/dw_bimodal/ \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/plots/dw_bimodal/"
```

---

### 6. 결과 해석 가이드

| ΔL 범위 | 비율 | 해석 |
|---|---|---|
| `\|ΔL\| < 0.5` | 높으면 | **DW 구조적 sign ambiguity** — 논문에서 "spectral likelihood symmetry" 언급 |
| `ΔL > +1.0` | 높으면 | 최적화 성공 but wrong mode → init 개선 필요 |
| `ΔL < -1.0` | 높으면 | **optimization failure** — LBFGS steps 늘리거나 재시작 필요 |

**기대 결과:**  
bimodal 원인이 sign ambiguity라면 `\|ΔL\|` 값이 0에 가까울 것.  
이는 advec_lat 추정의 불확실성이 optimization 실패가 아닌  
DW periodogram의 구조적 특성임을 의미 → 논문에서 limitation으로 기술 가능.

---

### 7. 전체 시뮬레이션 맵

| 파일 | 목적 | 비교 대상 |
|---|---|---|
| `sim_dw_raw_040626.py` | gridification bias | DW_raw_loc vs DW_raw_grid |
| `sim_dw_2d_040626.py` | 2D differencing bias | DW_2dconv vs DW_raw |
| `sim_dw_mixed_040626.py` | mixed-freq bias | DW_mixed vs DW_raw |
| `sim_vdw_grid_040626.py` | DW spectral approx vs Vecchia | Vecchia vs DW_raw (grid) |
| **`sim_dw_bimodal_040826.py`** | **advec bimodality 진단** | **DW_raw: L(A) vs L(B)** |
