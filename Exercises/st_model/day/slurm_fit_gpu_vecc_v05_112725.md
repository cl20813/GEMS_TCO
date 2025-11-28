### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### transfer new data into amarel

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/coarse_cen_map_without_decrement_latitude24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024


### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/torch_gpu_test.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_gpu_veccDWlbfgs_day_v05_112725.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model



### Copy estimate file from ```Amarel HPC``` to ```local computer```

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_r2s10_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/df_cv_smooth_05" 


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  module load cuda/11.7.1 ```
```  conda activate gems_gpu   ```

```srun --partition=gpu --gres=gpu:1 --time=00:30:00 --mem=8G --pty bash```
```python /home/jl2815/tco/exercise_25/st_model/torch_gpu_test.py```

```conda list | grep torch ```

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/torch_gpu_test.py ```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_112025.py --v 0.5 --lr 0.03 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 10 --no-keep-exact-loc    ```


```    srun --cpus-per-task=3 --mem=8G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_veccDW_day_v05_112125.py --v 0.5 --lr 0.03 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 10 --no-keep-exact-loc    ```


```    srun --cpus-per-task=3 --mem=8G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_veccDWadams_day_v05_112225.py --v 0.5 --lr 0.03 --epochs 100 --space "20, 20" --days "12,13" --mm-cond-number 8 --nheads 10 --no-keep-exact-loc    ```



### vecc lbfgs + dw initiate 

``` cd ./jobscript/tco/gp_exercise ```
```  nano fit_gpu_veccDWlbfgs_v05_nov24_18126.sh  ``` 
```  sbatch fit_gpu_veccDWlbfgs_v05_nov24_18126.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vecDWlbfgs_v05_nov24_18126_GPU       # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/gpu_veccDWlbfgs_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/gpu_veccDWlbfgs_%j.err
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
srun python /home/jl2815/tco/exercise_25/st_model/fit_gpu_veccDWlbfgs_day_v05_112725.py \
    --v 0.5 \
    --lr 0.03 \
    --step 80 \
    --epochs 100 \
    --space "1, 1" \
    --days "0,3" \
    --mm-cond-number 8 \
    --nheads 400 \
    --no-keep-exact-loc 

echo "Current date and time: $(date)"

```

