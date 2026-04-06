# Debiased Whittle — 2D Conv Filter [[-1,1],[1,-1]] (v05, 040626)

**변경 사항 (lat1d v06 대비)**
- 필터: `[[-1],[1]]` (lat-only) → `[[-1,1],[1,-1]]` (2D separable)
- 출력 그리드: `(nlat-1) × nlon` → `(nlat-1) × (nlon-1)`
- Cov_Z: 3항 → 9항 (lon shift ±δ₂ 및 대각 항 추가)
- 주파수 exclusion: ω₁=0 row → ω₁=0 row + ω₂=0 column
- 출력 파일: `real_dw_2dconv_july_22_23_24_25.[json/csv]`
- 출력 경로: `.../estimates/day/july_22_23_24_25_2dconv/`

---

### 1. 패키지 전송 (mac → Amarel)

```bash
# 새 모듈 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/debiased_whittle_2d_conv.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

# fit 스크립트 전송
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/fit_dw_2conv_day_v05_040626.py" \
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
nano fit_dw_2conv_040626.sh
sbatch fit_dw_2conv_040626.sh

```

```bash
#!/bin/bash
#SBATCH --job-name=dw_2dconv
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_2dconv_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_2dconv_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/fit_dw_2conv_day_v05_040626.py \
    --v 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --no-keep-exact-loc

echo "Done: $(date)"
```

```bash
sbatch fit_dw_2conv_040626.sh
```

---

### 4. 결과 전송 (Amarel → mac)

```bash
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25_2dconv/ \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/july_22_23_24_25_2dconv/"
```

---

### 5. 수학적 일관성 체크

| 항목 | v05 (old) | lat1d (v06) | 2dconv (v05) |
|---|---|---|---|
| 필터 | `[-2,1;1,0]` | `[-1;1]` | `[[-1,1],[1,-1]]` |
| `apply_first_diff` kernel | 2×2 | 2×1 | 2×2 |
| 출력 그리드 | (nlat-1)×(nlon-1) | (nlat-1)×nlon | (nlat-1)×(nlon-1) |
| Cov_Z 항 수 | 9항 (cross term 포함) | 3항 | 9항 (대칭, cross 없음) |
| DC exclusion | (0,0)만 | ω₁=0 row만 | ω₁=0 row + ω₂=0 col ✓ (버그수정 040626) |
| `taper_autocorr_grid` | 4D multivariate | 4D multivariate | 4D multivariate ✓ |
| `DELTA_LAT/LON` | 0.044/0.063 | 0.044/0.063 | 0.044/0.063 ✓ |
| API 모듈 | `debiased_whittle` | `debiased_whittle_lat1d` | `debiased_whittle_2d_conv` |

**핵심 검증:**
- `cov_spatial_difference` weights `{(0,0):-1, (1,0):+1, (0,1):+1, (1,1):-1}` 에서
  `(0,1)` 항이 `b_idx=1` → `offset_a2 = 1 * delta2 = 0.063` 올바르게 참조
- lat1d와 달리 lon 방향 shift `δ₂` 가 Cov_Z에 반영됨
- `run_lbfgs_tapered` 내 `DELTA_LAT=0.044, DELTA_LON=0.063` 하드코딩 — 2D 필터에도 동일 적용 ✓
- **[버그수정 040626]** `whittle_likelihood_loss_tapered` 및 `_sum`: DC(0,0)만 제외 → ω₁=0 row + ω₂=0 col 전체 제외로 수정
  - 이전: `sum_loss = total_sum - dc_term`, `num_terms = n1*n2 - 1`
  - 수정: `sum_loss = total_sum - row0_sum - col0_sum + dc_term`, `num_terms = (n1-1)*(n2-1)`
  - 이유: |H(ω)|²=4sin²(ω₁/2)·4sin²(ω₂/2)=0 when ω₁=0 OR ω₂=0 → 해당 frequency에서 f_Z(ω)=0 → 무한대 penalty가 likelihood에 포함되면 추정 편향 발생
