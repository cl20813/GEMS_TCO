# sim_dw_sign — 2-Way Sign Ambiguity Diagnostic (040826)

**목적:**  
4가지 단순화 모델에서 advec 부호 양면성(sign ambiguity)을 진단.  
각 모델은 하나의 advec 방향만 free, 나머지는 0으로 고정.

| Model | 이름 | params | advec | range |
|---|---|---|---|---|
| 1 | DW_iso_lat  | 5 | advec_lat free | isotropic |
| 2 | DW_iso_lon  | 5 | advec_lon free | isotropic |
| 3 | DW_aniso_lat| 6 | advec_lat free | anisotropic |
| 4 | DW_aniso_lon| 6 | advec_lon free | anisotropic |

**핵심 출력:** ΔL = L(converged) − L(sign_flipped)  
- `|ΔL| < 0.5` → **SYM**: +/− 부호 구분 불가 (spectral symmetry)  
- `ΔL < −0.5` → **CORRECT**: 수렴점이 진짜 best  
- `ΔL > +0.5` → **WRONG**: 최적화 실패, 반대 부호가 더 높은 likelihood

**True params:**  
- iso models: range = sqrt(0.154×0.195) ≈ 0.174, sigmasq=13.059, range_t=1.0, nugget=0.247  
- aniso models: range_lat=0.154, range_lon=0.195 (same as DW_raw)  
- advec_lat (models 1,3): +0.0218  
- advec_lon (models 2,4): −0.1689

---

### 1. 파일 전송 (mac → Amarel)

```bash
# 새 DW 모듈
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/debiased_whittle_reduced.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

# 시뮬레이션 스크립트
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw_sign_040826.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/
```

---

### 2. Slurm 스크립트 생성 및 제출

4개 모델을 동시에 각각 다른 노드에서 병렬 실행:

#### Model 1 — DW_iso_lat (advec_lat, isotropic)

```bash
cd ~/jobscript/tco/gp_exercise
nano sign_m1_040826.sh
sbatch sign_m1_040826.sh


```

```bash
#!/bin/bash
#SBATCH --job-name=sign_m1
#SBATCH --output=/home/jl2815/tco/exercise_output/sign_m1_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sign_m1_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu032

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_sign_040826.py \
    --model 1 \
    --num-iters 500 \
    --sign-start 20 \
    --seed 42

echo "Done: $(date)"

```

```bash
sbatch sign_m1_040826.sh
```

---

#### Model 2 — DW_iso_lon (advec_lon, isotropic)

```bash
nano sign_m2_040826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sign_m2
#SBATCH --output=/home/jl2815/tco/exercise_output/sign_m2_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sign_m2_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_sign_040826.py \
    --model 2 \
    --num-iters 300 \
    --sign-start 30 \
    --seed 42

echo "Done: $(date)"
```

```bash
sbatch sign_m2_040826.sh
```

---

#### Model 3 — DW_aniso_lat (advec_lat, anisotropic)

```bash
nano sign_m3_040826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sign_m3
#SBATCH --output=/home/jl2815/tco/exercise_output/sign_m3_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sign_m3_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_sign_040826.py \
    --model 3 \
    --num-iters 300 \
    --sign-start 30 \
    --seed 42

echo "Done: $(date)"
```

```bash
sbatch sign_m3_040826.sh
```

---

#### Model 4 — DW_aniso_lon (advec_lon, anisotropic)

```bash
nano sign_m4_040826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sign_m4
#SBATCH --output=/home/jl2815/tco/exercise_output/sign_m4_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sign_m4_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu044

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_sign_040826.py \
    --model 4 \
    --num-iters 300 \
    --sign-start 30 \
    --seed 42

echo "Done: $(date)"
```

```bash
sbatch sign_m4_040826.sh
```

---

### 3. 모니터링

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/sign_m1_<jobid>.out
tail -f /home/jl2815/tco/exercise_output/sign_m3_<jobid>.out
```

sign check 출력 예시 (iter 30 이후):
```
  ── 2-Way Sign Check (advec_lat) ──────────────────────────────
  Converged advec_lat: +0.0215   True: +0.0218
  L(pp = conv):  -9.548321
  L(mm = flip):  -9.548318
  ΔL = L(pp)-L(mm): -0.0003
  Verdict: STRUCTURAL SYMMETRY  (|ΔL| < 0.5 — +/- signs indistinguishable)
  ─────────────────────────────────────────────────────────────────
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
# 모든 모델 CSV
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/sim_dw_sign_m*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"

# Plots
scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/plots/sign_DW_*" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/plots/"
```

---

### 5. 결과 해석 가이드

| 모델 | 기대 결과 | 의미 |
|---|---|---|
| 1: iso+lat  | SYM 높으면 | isotropic에서 advec_lat 부호 판별 불가 |
| 2: iso+lon  | SYM 높으면 | isotropic에서 advec_lon 부호 판별 불가 |
| 3: aniso+lat| SYM 높으면 | aniso에서도 advec_lat 부호 판별 불가 → 구조적 문제 |
| 4: aniso+lon| SYM 높으면 | aniso에서도 advec_lon 부호 판별 불가 → 구조적 문제 |

이론 예측:  
- 공간 스펙트럼은 advec에 대해 **phase shift** 도입 (크기는 보존)  
- Cross-periodogram I(ω)_{qr}의 복소수 구조가 +/− 부호를 구별하는 유일한 정보원  
- advec이 작을수록 (|advec_lat|=0.022) phase shift 미미 → SYM 더 자주 발생 예측  
- advec이 클수록 (|advec_lon|=0.169) phase shift 뚜렷 → CORRECT 더 자주 발생 예측  

→ Model 2 vs Model 1: advec_lon(=0.169) vs advec_lat(=0.022) 크기 차이로  
   Model 2 CORRECT 비율 > Model 1 CORRECT 비율 예상

---

### 6. 전체 시뮬레이션 맵

| 파일 | 목적 |
|---|---|
| `sim_dw_raw_040626.py`     | gridification bias: DW_raw_loc vs DW_raw_grid |
| `sim_dw_2d_040626.py`      | 2D differencing bias: DW_2dconv vs DW_raw |
| `sim_dw_mixed_040626.py`   | mixed-freq bias: DW_mixed vs DW_raw |
| `sim_dw_bimodal_040826.py` | full 7-param 4-way sign ambiguity |
| **`sim_dw_sign_040826.py`**| **단순화 모델 2-way sign ambiguity (이 파일)** |
