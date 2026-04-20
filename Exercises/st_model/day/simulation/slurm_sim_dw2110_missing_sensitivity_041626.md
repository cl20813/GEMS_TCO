# Simulation Study — Missing Pattern Sensitivity (step3 threshold) v041526

**핵심 목적:** step3 obs→cell 거리 threshold를 변화시키면서 파라미터 추정이 얼마나 민감한지 확인.

**Threshold 룰 (rectangular check):**
```
cell j observed  iff  |obs_lat - cell_lat| ≤ frac × DELTA_LAT  AND
                       |obs_lon - cell_lon| ≤ frac × DELTA_LON
```
- `frac = 0.5` → Voronoi 셀 내부 (standard)
- `frac = 0.45, 0.4, 0.35` → 더 엄격 → 더 많은 missing

**현재 sim_dw2110_three_model_041626.py와의 차이:**
- 기존: threshold 없음 (obs → 가장 가까운 cell에 무조건 배정)
- 신규: `frac` 기반 rectangular cutoff 적용
- Vecc_Irr: threshold 무관 (raw obs location 사용), iter당 1회 실행
- Vecc_Reg + DW: threshold별로 재실행

---

### 1. 패키지 전송 (mac → Amarel)

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw2110_missing_sensitivity_041626.py" \
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
nano sim_dw2110_miss_sens_041626.sh
sbatch sim_dw2110_miss_sens_041626.sh
```



#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu037


#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --partition=mem


```bash
#!/bin/bash
#SBATCH --job-name=sim_miss_sens
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_miss_sens_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_miss_sens_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu037

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw2110_missing_sensitivity_041626.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 0 \
    --limit-a 20 \
    --limit-b 20 \
    --limit-c 20 \
    --daily-stride 2 \
    --num-iters 300 \
    --thresholds "0.5,0.45,0.4,0.35" \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --init-noise 0.7 \
    --seed 42

echo "Done: $(date)"

```

```bash
sbatch sim_dw2110_miss_sens_041626.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_missing_sensitivity_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```

---

### 5. 핵심 해석 포인트

| threshold frac | max_lat 허용 | max_lon 허용 | 예상 missing % |
|---|---|---|---|
| 0.5 | 0.022° | 0.0315° | 낮음 (Voronoi = baseline) |
| 0.45 | 0.0198° | 0.02835° | 낮음~중간 |
| 0.4 | 0.0176° | 0.0252° | 중간 |
| 0.35  | 0.0154° | 0.0221° | 중간~높음 |

**핵심 질문:**
- Vecc_Irr RMSRE는 threshold와 무관 → baseline
- Vecc_Reg / DW RMSRE가 threshold가 작아질수록 얼마나 악화되는가?
- 어느 threshold부터 유의미한 성능 저하가 발생하는가?

**출력 CSV 주요 컬럼:**
- `threshold`: frac 값 (0.5 / 0.45 / 0.4 / 0.35)
- `n_obs_reg`: 해당 threshold에서 관측된 셀 수 (missing 아닌 것)
- `rmsre`: 전체 파라미터 RMSRE
- `model`: Vecc_Irr / Vecc_Reg / DW

**참고 — init noise:**
- `init_noise=0.7` → exp(0.7)=2배, exp(-0.7)=0.5배 범위 내 random init
- advec params: ±2×|true| additive perturbation
