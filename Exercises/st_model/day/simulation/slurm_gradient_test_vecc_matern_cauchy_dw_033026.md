### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_gradient_test_vecc_matern_cauchy_dw_033026.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

---

### Study design
- **Purpose**: Score/gradient test — verify ||∇ℓ(θ_true)|| ≈ 0 at true DGP parameters for simulated data
- **DGP**: Matérn ν=0.5 via FFT circulant embedding (correctly specified for VM)
- **Methods**:
  - VM: Vecchia Matérn ν=0.5 — irregular obs locations (real location matching)
  - VC: Vecchia Cauchy β=1.0 — irregular obs locations
  - DW: Debiased Whittle — regular gridded data (high-res field subsampled)
- **Test metric**: `ratio_inf = ||∇ℓ(θ_true)||∞ / ||∇ℓ(θ_perturbed)||∞` — target: << 1
- **10 parallel jobs × 10 iters each = 100 total iterations** (merge CSVs afterward)

Output files (in `estimates/day/gradient_test/`):
- `gradient_test_{date}_j{0..9}.csv` — raw records per iteration

Merge after all jobs finish:
```python
import pandas as pd, glob

folder = "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/gradient_test/"
df = pd.concat([pd.read_csv(f) for f in glob.glob(folder + "gradient_test_*_j*.csv")])
df.to_csv(folder + "gradient_test_merged.csv", index=False)
```




'''    
import pandas as pd, glob

folder = "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/gradient_test/"
df = pd.concat([pd.read_csv(f) for f in glob.glob(folder + "gradient_test_*_j*.csv")])

df.to_csv(folder + "gradient_test_merged.csv", index=False)                                 
                                                                                            
# ── 1. 핵심 지표: ratio_inf (target << 1)                                                 
print("=== Mean ratio_inf (target << 1) ===")                                              
for m in ['VM', 'VC', 'DW']:                                                               
    r = df[f'{m}_ratio_inf'].mean()                                                        
    print(f"  {m}: {r:.4f}")                                                               
                                                                                            
# ── 2. 메서드별 gradient norm 비교                                                        
print("\n=== Mean ||∇||∞ at θ_true vs perturbed ===")                                      
for m in ['VM', 'VC', 'DW']:                                                               
    g_true = df[f'{m}_grad_inf_true'].mean()                                               
    g_pert = df[f'{m}_grad_inf_pert'].mean()                                               
    print(f"  {m}:  @θ_true={g_true:.6f}  @pert={g_pert:.6f}")                             
                                                                                            
# ── 3. 파라미터별 gradient 크기 (어느 파라미터가 문제인지)                                
P_NAMES = ["SigmaSq","RangeLat","RangeLon","RangeTime","AdvecLat","AdvecLon","Nugget"]     
print("\n=== Mean |∇| per parameter at θ_true ===")                                        
print(f"  {'param':<12} {'VM':>10} {'VC':>10} {'DW':>10}")                                 
for p in P_NAMES:                                                                          
    vm = df[f'VM_grad_{p}_true'].abs().mean()                                              
    vc = df[f'VC_grad_{p}_true'].abs().mean()                                              
    dw = df[f'DW_grad_{p}_true'].abs().mean()                                              
    print(f"  {p:<12} {vm:>10.6f} {vc:>10.6f} {dw:>10.6f}")                                
                                                                                            
# ── 4. iteration 수 확인                                                                  
print(f"\n총 iterations: {len(df)}")                                                       
print(f"Job별 분포:\n{df['job_id'].value_counts().sort_index()}")  
'''


---

### Connect & setup
```
ssh jl2815@amarel.rutgers.edu
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
conda activate faiss_env
```

---

### Job array (10 parallel jobs, each 10 iters)

~3 methods × ~5 min/method × 10 iters ≈ 2.5 h; set time=12h for safety.

```
cd ./jobscript/tco/gp_exercise
nano sim_gradient_test_vecc_matern_cauchy_dw_033026.sh
sbatch sim_gradient_test_vecc_matern_cauchy_dw_033026.sh
```

```bash

#!/bin/bash
#SBATCH --job-name=grad_test_033026
#SBATCH --output=/home/jl2815/tco/exercise_output/grad_test_033026_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/grad_test_033026_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)  array_task=${SLURM_ARRAY_TASK_ID}"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_gradient_test_vecc_matern_cauchy_dw_033026.py \
    --num-iters 30 \
    --job-id ${SLURM_ARRAY_TASK_ID} \
    --years "2022,2023,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --nheads 1000 \
    --limit 16 \
    --mm-cond 30 \
    --daily-stride 2 \
    --pert-scale 0.3 \
    --seed 42

echo "Current date and time: $(date)"


```

---

### Transfer results (Amarel → mac)
```
scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/gradient_test/" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/"
```

---

### Quick local test
```bash
conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer

python sim_gradient_test_local_033026.py \
    --num-iters 2 \
    --lat-factor 10 \
    --lon-factor 4 \
    --nheads 200 \
    --limit 8
```

---

### How to interpret results

Key columns per method (VM / VC / DW):
- `{M}_grad_inf_true`  : L∞ norm of ∇ℓ at θ_true
- `{M}_grad_inf_pert`  : L∞ norm of ∇ℓ at perturbed θ
- `{M}_ratio_inf`      : ratio true/pert — **target << 1** (ideally < 0.1)
- `{M}_grad_{param}_true` : per-parameter gradient component at θ_true

Questions to answer:
1. **Is VM score ≈ 0 at θ_true?** (correctly specified — strongest test)
   `VM_ratio_inf << 1` confirms the Vecchia Matérn score implementation is correct.
2. **Is DW score ≈ 0 at θ_true?**
   DW is consistent for a broad model class; gradient should also be small.
3. **Which parameters have largest gradient components?**
   Large `{M}_grad_{param}_true` after averaging over iters suggests either
   model misspecification or numerical issues in that direction.
4. **VC gradient at Matérn true params**: VC is misspecified here; ratio > 1 is expected
   (gradient points away from θ_true toward the pseudo-true params).
