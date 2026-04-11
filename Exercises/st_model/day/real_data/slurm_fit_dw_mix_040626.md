# Debiased Whittle — Mixed-frequency (DW_mixed) Real Data Fit (v05, 040626)

**설계 개요:**

| 항목 | DW_2dconv | **DW_mixed (this)** |
|---|---|---|
| 저주파 Ω_L | 2D-diff (H=0 → 정보 손실) | **raw C_X (H=1, 저주파 보존)** |
| 고주파 Ω_H | 2D-diff | 2D-diff `[[-1,+1],[+1,-1]]` |
| DC 처리 | ω₁=0 row + ω₂=0 col 제외 | 동일 + 저주파 raw로 커버 |
| advection 식별력 | 저주파 소실 → 편향 가능 | **저주파 raw → advection 정보 보존** |
| 수학적 유효성 | 표준 DW | Composite likelihood (Godambe) |

**필터:**
- High-freq piece: `h = {(0,0):-1, (1,0):+1, (0,1):+1, (1,1):-1}` → kernel `[[-1,+1],[+1,-1]]`
- `|H(ω)|² = 4sin²(ω₁/2)·4sin²(ω₂/2)` — `[[1,-1],[-1,1]]`과 부호만 다름, 동일한 필터

**주파수 분할 (--freq-alpha α):**
- `K1 = floor(n1·α)`,  `K2 = floor(n2·α)`
- `Ω_L` (raw): `{k1≤K1, k2≤K2} ∪ {k1=0 행} ∪ {k2=0 열} \ {DC}`
- `Ω_H` (diff): `{k1>K1 OR k2>K2} ∩ {k1>0, k2>0}`

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# mixed 모듈 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/debiased_whittle_mixed.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

# fit 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/fit_dw_mix_day_v05_040626.py" \
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
nano fit_dw_mix_040626.sh
sbatch fit_dw_mix_040626.sh

```

```bash

#!/bin/bash
#SBATCH --job-name=dw_mixed
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_mixed_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_mixed_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/fit_dw_mix_day_v05_040626.py \
    --v 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --freq-alpha 0.65 \
    --no-keep-exact-loc

echo "Done: $(date)"

```

```bash
sbatch fit_dw_mix_040626.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25_mixed_a0p20/ \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/july_22_23_24_25_mixed_a0p20/"
```

---

### 5. 수학적 구조 및 비교

| 항목 | DW_2dconv | DW_mixed (α=0.20) |
|---|---|---|
| 출력 그리드 (raw) | 없음 | n_lat × n_lon |
| 출력 그리드 (diff) | (n_lat-1) × (n_lon-1) | (n_lat-1) × (n_lon-1) |
| Ω_L 크기 | 0 | ~20% + 축 |
| Ω_H 크기 | (n1-1)(n2-1) - 1 | ~80% 내부 |
| Cov 항 수 (raw) | — | 1 (C_X 직접) |
| Cov 항 수 (diff) | 9항 | 9항 (동일) |
| advection 식별력 | 저주파 소실 가능 | **보존** |
| 비용 | O(n1×n2) | O(n1×n2 + n1d×n2d) ≈ 2× |

**α 값 비교 실험 (필요 시):**
```bash
# α=0.15
srun python .../fit_dw_mix_day_v05_040626.py --freq-alpha 0.15 --days "0,28" ...

# α=0.25
srun python .../fit_dw_mix_day_v05_040626.py --freq-alpha 0.25 --days "0,28" ...
```

**출력 파일 구조 (res_dict 추가 필드):**
- `freq_alpha`: α 값
- `K1`, `K2`: 저주파 컷오프
- `n_low`, `n_high`: Ω_L, Ω_H 크기
- `n1`, `n2`: raw 그리드
- `n1d`, `n2d`: diff 그리드
