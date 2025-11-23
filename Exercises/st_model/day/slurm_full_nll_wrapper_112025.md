### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### transfer new data into amarel

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/coarse_cen_map_without_decrement_latitude24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024


### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/full_nll_wrapper_112025.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


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

## 

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "20, 20" --days "0,1" --mm-cond-number 8 --nheads 30```


```    srun --cpus-per-task=3 --mem=80G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "1, 1" --days "0,1" --mm-cond-number 8 --nheads 300```




### smooth 0.5
```mkdir -p ./jobscript/tco/gp_exercise```     


```  cd ./jobscript/tco/gp_exercise  ```          
```  nano full_nll_1125.sh  ``` 
```  sbatch full_nll_1125.sh  ``` 


``` 
#!/bin/bash
#SBATCH --job-name=full_nll_1125                       # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/full_nll_1125_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/full_nll_1125_%j.err # Standard error file (%j = JobID)
#SBATCH --time=6:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=20                                        # Number of CPU cores per task
#SBATCH --mem=248G                                              # Memory per node
#SBATCH --partition=main                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"

srun python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "1, 1" --days "0,2" --mm-cond-number 8 --nheads 300 --no-keep-exact-loc 

echo "Current date and time: $(date)"

```
