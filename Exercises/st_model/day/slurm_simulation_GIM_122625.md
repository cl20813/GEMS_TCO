### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### transfer new data into amarel

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/coarse_cen_map_without_decrement_latitude24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024


### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/sim_heads_regular_vecc_GIM_121925.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/sim_GIM_vecc_dw_irregular_122025.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/sim_GIM_vecc_dw_regular_122625.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy estimate file from ```Amarel HPC``` to ```local computer```



# irr vs ir   vecc vs dw
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_re_and_irr_dw_vs_vecc_121225/sim_dW_v050_121225_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_re_and_irr_dw_vs_vecc_121225/sim_irre_dW_v050_121225_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_re_and_irr_dw_vs_vecc_121225/sim_vecc_1212_v050_LBFGS_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_re_and_irr_dw_vs_vecc_121225/sim_irre_vecc_1212_v050_LBFGS_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 

# 아마 이 버전이 0.018 
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_irre_dW_v050_121725_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_irre_vecc_1217_v050_LBFGS_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  module load cuda/11.7.1 ```
```  conda activate gems_gpu   ```

```srun --partition=gpu --gres=gpu:1 --time=00:30:00 --mem=8G --pty bash```
```python /home/jl2815/tco/exercise_25/st_model/torch_gpu_test.py```

```conda list | grep torch ```

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/sim_gpu_veccDWlbfgs_day_v05_121125.py```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_112025.py --v 0.5 --lr 0.03 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 10 --no-keep-exact-loc    ```


### simulation regular grid + GIM

``` cd ./jobscript/tco/gp_exercise ```
```  nano sim_GIM_122625.sh  ``` 
```  sbatch sim_GIM_122625.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=sim_GIM_122625       # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_GIM_122625_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_GIM_122625_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat           # 💥 파티션 이름 확인 필요
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu038               # 💥 확인하신 idle 노드 중 하나 입력

#### Load Modules
module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on High-End AdaLovelace Node: $(hostname)"
nvidia-smi

# Run the script
srun python /home/jl2815/tco/exercise_25/st_model/sim_GIM_vecc_dw_regular_122625.py --start-day 1 --end-day 28 --no-keep-exact-loc 

echo "Current date and time: $(date)"

```


```  nano sim_GIM_no_heads_121925.sh  ``` 
```  sbatch sim_GIM_no_heads_121925.sh  ```
 

``` 
#!/bin/bash
#SBATCH --job-name=sim_GIM_no_heads__121225       # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_GIM_no_heads__121225_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_GIM_no_heads__121225_%j.err
#SBATCH --time=24:00:00                                 # Reduced time (GPU is faster)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                               # CHANGED: Reduced from 48 (GPU does the work now)
#SBATCH --mem=64G                                       # CHANGED: Reduced from 400G (GPU handles the matrices)
#SBATCH --partition=gpu                                 # 💥 CRITICAL: Change to your cluster's GPU partition name
#SBATCH --gres=gpu:1                                    # 💥 CRITICAL: Request 1 GPU

#### Load Modules
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 
module load cuda/12.1.0

#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env  # Ensure this env has PyTorch with CUDA installed!

echo "Current date and time: $(date)"
echo "Running GPU Batched Vecchia Optimization"
echo "Node: $(hostname)"

# Check if GPU is actually visible
nvidia-smi

# Run the script
srun python /home/jl2815/tco/exercise_25/st_model/sim_heads_regular_vecc_GIM_121925.py \
    --v 0.5 \
    --lr 0.03 \
    --step 80 \
    --epochs 100 \
    --space "1, 1" \
    --days "20,30" \
    --mm-cond-number 8 \
    --nheads 0 \
    --no-keep-exact-loc 

echo "Current date and time: $(date)"

```




### simulation irregular grid + GIM  for VECC AND DW

``` cd ./jobscript/tco/gp_exercise ```
```  nano sim_GIM_irr_veccDW_122025.sh  ``` 
```  sbatch sim_GIM_irr_veccDW_122025.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=sim_GIM_veccDW_122025       # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_GIM_irr_veccDW_122025_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_GIM_irr_veccDW_122025_%j.err
#SBATCH --time=24:00:00                                 # Reduced time (GPU is faster)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40                               # CHANGED: Reduced from 48 (GPU does the work now)
#SBATCH --mem=400G                                       # CHANGED: Reduced from 400G (GPU handles the matrices)
#SBATCH --partition=mem                                 # 💥 

#### Load Modules
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 


#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env  # Ensure this env has PyTorch with CUDA installed!

echo "Current date and time: $(date)"
echo "Running GPU Batched Vecchia Optimization"
echo "Node: $(hostname)"

# Run the script
srun python /home/jl2815/tco/exercise_25/st_model/sim_GIM_vecc_dw_irregular_122025.py \
    --v 0.5 \
    --lr 0.03 \
    --step 80 \
    --epochs 100 \
    --space "1, 1" \
    --days "20,30" \
    --mm-cond-number 8 \
    --nheads 300 \
    --no-keep-exact-loc 

echo "Current date and time: $(date)"

```

### simulation regular grid + GIM  for VECC AND DW

``` cd ./jobscript/tco/gp_exercise ```
```  nano sim_GIM_reg_veccDW_122025.sh  ``` 
```  sbatch sim_GIM_reg_veccDW_122025.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=sim_GIM_reg_veccDW_122025       # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_GIM_reg_veccDW_122025_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_GIM_reg_veccDW_122025_%j.err
#SBATCH --time=24:00:00                                 # Reduced time (GPU is faster)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40                               # CHANGED: Reduced from 48 (GPU does the work now)
#SBATCH --mem=400G                                       # CHANGED: Reduced from 400G (GPU handles the matrices)
#SBATCH --partition=mem                                 # 💥 

#### Load Modules
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 


#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env  # Ensure this env has PyTorch with CUDA installed!

echo "Current date and time: $(date)"
echo "Running GPU Batched Vecchia Optimization"
echo "Node: $(hostname)"

# Run the script
srun python /home/jl2815/tco/exercise_25/st_model/sim_GIM_vecc_dw_regular_122025.py \
    --v 0.5 \
    --lr 0.03 \
    --step 80 \
    --epochs 100 \
    --space "1, 1" \
    --days "20,30" \
    --mm-cond-number 8 \
    --nheads 300 \
    --no-keep-exact-loc 

echo "Current date and time: $(date)"

```