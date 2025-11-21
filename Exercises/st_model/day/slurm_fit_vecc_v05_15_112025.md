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


```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_112025.py --v 0.5 --lr 0.03 --step 80 --epochs 100 --space "1, 1" --days "0,31" --mm-cond-number 8 --nheads 300 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```



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



### v15 4508
```mkdir -p ./jobscript/tco/gp_exercise```     

```  cd ./jobscript/tco/gp_exercise  ```          
```  nano fit_day_vecc_v15_may30_4508.sh  ``` 
```  sbatch fit_day_vecc_v15_may30_4508.sh  ``` 


``` 
#!/bin/bash
#SBATCH --job-name=vec_v15_may30_4508                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_may30_4508_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_may30_4508_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                        # Number of CPU cores per task
#SBATCH --mem=400G                                                 # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"
echo "fit_vecc_v15_may30_1126_save_estimates"


srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v15_530.py --v 1.5 --lr 0.03 --step 100 --gamma-par 0.3 --epochs 1500 --space "2, 2" --days "0,31" --mm-cond-number 10 --nheads 300
```

### v15 18033
```mkdir -p ./jobscript/tco/gp_exercise```     

```  cd ./jobscript/tco/gp_exercise  ```          
```  nano fit_day_vecc_v15_may30_18033.sh  ``` 
```  sbatch fit_day_vecc_v15_may30_18033.sh  ``` 


``` 
#!/bin/bash
#SBATCH --job-name=vec_v15_may30_18033                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_may30_18033_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_may30_18033_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                        # Number of CPU cores per task
#SBATCH --mem=400G                                                 # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"
echo "fit_vecc_v15_may30_18033_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v15_530.py --v 1.5 --lr 0.03 --step 100 --gamma-par 0.3 --epochs 1500 --space "1, 1" --days "18,31" --mm-cond-number 10 --nheads 300
```

## May 30, 2025 I ran days 0,15 and 15,31  43976579 for 15 and 31 43976576 for 0 and 15
