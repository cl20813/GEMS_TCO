### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco


### transfer new data into amarel

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/coarse_cen_map_rect24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024


### Copy run file from ```local``` to ```Amarel HPC```
# mac


# vecchia

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_gpu_vecc_day_v05_010126.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/sim_heads_regular_vecc_testing_day_v05_010126.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

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



### vecc gpu l-bfgs

``` cd ./jobscript/tco/gp_exercise ```
```  nano fit_vecc_gpu_011026.sh  ``` 
```  sbatch fit_vecc_gpu_011026.sh  ``` 


```
#!/bin/bash
#SBATCH --job-name=fit_vecc_h1000m16
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_vecc_h1000m16.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_vecc_h1000m16.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G                      #  gpu30:80
#SBATCH --partition=gpu-redhat                 # 'gpu' 파티션 사용
#SBATCH --gres=gpu:1                    # GPU 1개 요청
#SBATCH --nodelist=gpu034      # 💥 여기를 gpu030으로 변경! (idle 상태임)

#### Load Modules
module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on High-End Node: $(hostname)"
nvidia-smi

# Run the script
# A100/V100급이므로 nheads 1000으로 과감하게 갑니다.
srun python /home/jl2815/tco/exercise_25/st_model/fit_gpu_vecc_day_v05_010126.py \
    --v 0.5 \
    --space "1, 1" \
    --days "0,31" \
    --mm-cond-number 16 \
    --nheads 1000 \
    --no-keep-exact-loc

```


#gpu30:1000, mm16 거뜬 4분




``` cd ./jobscript/tco/gp_exercise ```
```  nano fit_vecc_cpu010126.sh  ``` 
```  sbatch fit_vecc_cpu010126.sh  ``` 

### vecc cpu


``` 

#!/bin/bash
#SBATCH --job-name=fit_vecc_cpu010126     # Job name (Added GPU tag)
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_vecc_cpu010126.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_vecc_cpu010126.err
#SBATCH --time=24:00:00                                 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40                               
#SBATCH --mem=300G                                       
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
srun python /home/jl2815/tco/exercise_25/st_model/fit_gpu_vecc_day_v05_010126.py \
    --v 0.5 \
    --mm-cond-number 8 \
    --nheads 300 \
    --v 0.5 \
    --space "1, 1" \
    --days "0,31" \
    --no-keep-exact-loc 

echo "Current date and time: $(date)"

```



# heads testing

``` cd ./jobscript/tco/gp_exercise ```
```  nano heads_testing_010126.sh  ``` 
```  sbatch heads_testing_010126.sh  ``` 


```
#!/bin/bash
#SBATCH --job-name=heads_testing_010126
#SBATCH --output=/home/jl2815/tco/exercise_output/heads_testing_010126_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/heads_testing_010126_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat           # 💥 파티션 이름 확인 필요
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpu033               # 💥 확인하신 idle 노드 중 하나 입력

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_heads_regular_vecc_testing_day_v05_010126.py \
    --v 0.5 \
    --mm-cond-number 8 \
    --no-keep-exact-loc

```