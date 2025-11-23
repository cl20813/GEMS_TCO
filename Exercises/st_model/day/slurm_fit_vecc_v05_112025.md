### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### transfer new data into amarel

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/coarse_cen_map_without_decrement_latitude24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024


### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_vecc_day_v05_112025.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_veccDW_day_v05_112125.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_veccDWadams_day_v05_112225.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


# window
scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_vecc_day_v05_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_vecc_day_v10_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy estimate file from ```Amarel HPC``` to ```local computer```

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_r2s10_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/df_cv_smooth_05" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v150_r2s10_1127.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/df_cv_smooth_15" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v150_r2s10_4508.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/df_cv_smooth_15" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v150_r2s10_18033.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/df_cv_smooth_15" 


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_112025.py --v 0.5 --lr 0.03 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 10 --no-keep-exact-loc    ```


```    srun --cpus-per-task=3 --mem=8G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_veccDW_day_v05_112125.py --v 0.5 --lr 0.03 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 10 --no-keep-exact-loc    ```


```    srun --cpus-per-task=3 --mem=8G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_veccDWadams_day_v05_112225.py --v 0.5 --lr 0.03 --epochs 100 --space "20, 20" --days "12,13" --mm-cond-number 8 --nheads 10 --no-keep-exact-loc    ```




### Job Order SLURM for both vecchia 

# 18126   nov 20 2025
``` cd ./jobscript/tco/gp_exercise ```
```  nano fit_day_vecc_v05_nov20_18126.sh  ``` 
```  sbatch fit_day_vecc_v05_nov20_18126.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vec_v05_nov20_18126                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_nov20_18126_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_nov20_18126_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=248G                                                 # Memory per node
#SBATCH --partition=main                                           # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"
echo "fit_vecc_v05_nov_18126_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_112025.py --v 0.5 --lr 0.03 --step 80 --epochs 100 --space "1, 1" --days "5,28" --mm-cond-number 8 --nheads 400 --no-keep-exact-loc 

```

## Exact location 

``` cd ./jobscript/tco/gp_exercise ```
```  nano el_fit_day_vecc_v05_nov20_18126.sh  ``` 
```  sbatch el_fit_day_vecc_v05_nov20_18126.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vec_v05_nov20_18126                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_nov20_18126_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_nov20_18126_%j.err # Standard error file (%j = JobID)
#SBATCH --time=40:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=16                                        # Number of CPU cores per task
#SBATCH --mem=248G                                                 # Memory per node
#SBATCH --partition=main                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"
echo "fit_vecc_v05_nov_18126_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_112025.py --v 0.5 --lr 0.03 --step 80 --epochs 100 --space "1, 1" --days "0,6" --mm-cond-number 8 --nheads 400 --keep-exact-loc 

```


### vecc lbfgs + dw initiate 

``` cd ./jobscript/tco/gp_exercise ```
```  nano fit_day_veccDW_v05_nov21_18126.sh  ``` 
```  sbatch fit_day_veccDW_v05_nov21_18126.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vecDW_v05_nov22_18126                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vecDW_nov21_18126_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vecDW_nov21_18126_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=48                                       # Number of CPU cores per task
#SBATCH --mem=400G                                                 # Memory per node
#SBATCH --partition=mem                                           # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"
echo "fit_vecc_v05_nov_18126_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_veccDW_day_v05_112125.py --v 0.5 --lr 0.03 --step 80 --epochs 100 --space "1, 1" --days "0,10" --mm-cond-number 8 --nheads 400 --no-keep-exact-loc 

```


### vecc adams+ dw initiate 

``` cd ./jobscript/tco/gp_exercise ```
```  nano fit_day_veccDWadams_v05_nov22_18126.sh  ``` 
```  sbatch fit_day_veccDWadams_v05_nov22_18126.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vecDWadams_v05_nov22_18126                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vecDWadams_nov21_18126_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vecDWadams_nov21_18126_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=48                                       # Number of CPU cores per task
#SBATCH --mem=400G                                                 # Memory per node
#SBATCH --partition=mem                                           # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"
echo "fit_veccDWadams_v05_nov_18126_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_veccDWadams_day_v05_112225.py --v 0.5 --lr 0.01 --step 80 --epochs 80 --space "1, 1" --days "0,10" --mm-cond-number 8 --nheads 400 --no-keep-exact-loc 

echo "Current date and time: $(date)"

```



