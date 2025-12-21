### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### transfer new data into amarel

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/coarse_cen_map_without_decrement_latitude24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024


### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/sim_regular_veccDWlbfgs_day_v05_122125.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

#### 12:20 nheads 400 LOC_ERR_STD 0.02. 12:21 nheads 300 LOC_ERR_STD 0.02
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/sim_irregular_veccDW_day_v05_122125.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/sim_heads_regular_vecc_testing_day_v05_122125.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


### Copy estimate file from ```Amarel HPC``` to ```local computer```

# irregular
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_irre_dW_v050_1220_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_irre_vecc_1220_v050_LBFGS_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 

# regular

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dW_v050_122025_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_reg_vecc_1220_v050_LBFGS_18126.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day" 


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


```    srun --cpus-per-task=3 --mem=8G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_veccDW_day_v05_112125.py --v 0.5 --lr 0.03 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 10 --no-keep-exact-loc    ```


```    srun --cpus-per-task=3 --mem=8G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_veccDWadams_day_v05_112225.py --v 0.5 --lr 0.03 --epochs 100 --space "20, 20" --days "12,13" --mm-cond-number 8 --nheads 10 --no-keep-exact-loc    ```



### simulation regular grid

``` cd ./jobscript/tco/gp_exercise ```
```  nano sim_reg_cpu_122125.sh  ``` 
```  sbatch sim_reg_cpu_122125.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=sim_reg_cpu_122125       # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_reg_cpu_122125_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/simc_reg_cpu_122125_%j.err
#SBATCH --time=24:00:00                                 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40                               
#SBATCH --mem=400G                                       
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
srun python /home/jl2815/tco/exercise_25/st_model/sim_regular_veccDWlbfgs_day_v05_122125.py \
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



############# irregular grid

# 1220: nheads 400 LOC_ERR_STR 0.02


``` cd ./jobscript/tco/gp_exercise ```
```  nano sim_irr_122125.sh  ``` 
```  sbatch sim_irr_122125.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=sim_irr_122125       # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_irr_122125_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_irr_122125_%j.err
#SBATCH --time=24:00:00                                 # Reduced time (GPU is faster)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40                               # CHANGED: Reduced from 48 (GPU does the work now)
#SBATCH --mem=400G                                       
#SBATCH --partition=mem                                 # 💥 

#### Load Modules
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env  # Ensure this env has PyTorch with CUDA installed!

echo "Current date and time: $(date)"
echo "Node: $(hostname)"


# Run the script
srun python /home/jl2815/tco/exercise_25/st_model/sim_irregular_veccDW_day_v05_122125.py \
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



## regular grid testing heads and tail separation setting

``` cd ./jobscript/tco/gp_exercise ```
```  nano sim_vecc_heads_testing_cpu_122125.sh  ``` 
```  sbatch sim_vecc_heads_testing_cpu_122125.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=sim_vecc_heads_testing_cpu_122125      # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_vecc_heads_testing_cpu_122125_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_vecc_heads_testing_cpu_122125_%j.err
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
srun python /home/jl2815/tco/exercise_25/st_model/sim_heads_regular_vecc_testing_day_v05_122125.py \
    --v 0.5 \
    --lr 0.03 \
    --step 80 \
    --epochs 100 \
    --space "1, 1" \
    --days "20,30" \
    --mm-cond-number 8 \
    --nheads 100 \
    --no-keep-exact-loc 

echo "Current date and time: $(date)"

```