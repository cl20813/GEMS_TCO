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

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/full_nll_1122four_03 "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/full_nll_1122four_24 "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/" 




### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```

## 

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "20, 20" --days "0,1" --mm-cond-number 8 --nheads 30 --no-keep-exact-loc    ```


```    srun --cpus-per-task=3 --mem=8G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "14, 14" --days "0,1" --mm-cond-number 8 --nheads 30  --lat-range "0,3" --lon-range "125,127" --no-keep-exact-loc ```




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


```  nano full_nll_112225.sh  ``` 
```  sbatch full_nll_112225.sh  ``` 


``` 
#!/bin/bash
#SBATCH --job-name=full_nll_1125                       # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/full_nll_1122four_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/full_nll_1122four_%j.err # Standard error file (%j = JobID)
#SBATCH --time=6:00:00                                            # Time limit
#SBATCH --ntasks=2                                              # Number of tasks
#SBATCH --cpus-per-task=20                                      # Number of CPU cores per task
#SBATCH --mem=400G                                              # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"


srun python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "1, 1" --days "0,4" --mm-cond-number 8 --nheads 300  --lat-range "0,3" --lon-range "123,128" --no-keep-exact-loc 

srun python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "1, 1" --days "0,4" --mm-cond-number 8 --nheads 300  --lat-range "0,3" --lon-range "128,133" --no-keep-exact-loc 

echo "Current date and time: $(date)"

```

```  nano full_nll_112325.sh  ``` 
```  sbatch full_nll_112325.sh  ``` 


``` 
#!/bin/bash
#SBATCH --job-name=full_nll_1125                       # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/full_nll_1122four_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/full_nll_1122four_%j.err # Standard error file (%j = JobID)
#SBATCH --time=6:00:00                                            # Time limit
#SBATCH --ntasks=2                                              # Number of tasks
#SBATCH --cpus-per-task=20                                      # Number of CPU cores per task
#SBATCH --mem=400G                                              # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"


srun python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "1, 1" --days "0,4" --mm-cond-number 8 --nheads 300  --lat-range "2,4" --lon-range "123,128" --no-keep-exact-loc 

srun python /home/jl2815/tco/exercise_25/st_model/full_nll_wrapper_112025.py --v 0.5 --space "1, 1" --days "0,4" --mm-cond-number 8 --nheads 300  --lat-range "2,4" --lon-range "128,133" --no-keep-exact-loc 

echo "Current date and time: $(date)"

```





