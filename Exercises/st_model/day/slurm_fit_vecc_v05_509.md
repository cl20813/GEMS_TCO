### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_vecc_day_v05_may9.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_vecc_day_v10_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

# window
scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_vecc_day_v05_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_vecc_day_v10_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy estimate file from ```Amarel HPC``` to ```local computer```

# 1250
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_r2s10_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 

# 5000

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_r2s10_5000.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 

# 20000

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_r2s10_20000.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 

# tmp

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/full_day_v05_r2s10_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/full_day_r2s10_v045_spline1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 




scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/may9_estimates/full_day_v05_1250.0plus2estimates.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/may9_estimates/full_day_v03_spline1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/may9_estimates/full_day_v04_spline1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/may9_estimates/full_day_v045_spline1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/may9_estimates/full_day_v055_spline1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/may12" 





### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_may9.py --v 0.5 --lr 0.03 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 10 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```


```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v10_416.py --v 0.5 --lr 0.03 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```



### Job Order SLURM for both vecchia and full
### smooth 0.5
```mkdir -p ./jobscript/tco/gp_exercise```     

```  cd ./jobscript/tco/gp_exercise  ```          
```  nano fit_day_vecc_v05_may21_1120.sh  ``` 
```  sbatch fit_day_vecc_v05_may21_1120.sh  ``` 


``` 
#!/bin/bash
#SBATCH --job-name=vec_v05_may9_1250                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_may9_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_may9_1250_%j.err # Standard error file (%j = JobID)
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
echo "fit_vecc_v05_may9_1250_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_may9.py --v 0.5 --lr 0.03 --step 100 --gamma-par 0.3 --epochs 1500 --space "4, 4" --days "0,31" --mm-cond-number 10 --nheads 200

```
``` cd ./jobscript/tco/gp_exercise ```
```  nano fit_day_vecc_v05_may21_4560.sh  ``` 
```  sbatch fit_day_vecc_v05_may21_4560.sh  ``` 

``` 

#!/bin/bash
#SBATCH --job-name=vec_v05_may21_4560                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_may21_4560_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_may21_4560_%j.err # Standard error file (%j = JobID)
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
echo "fit_vecc_v05_may09_5000_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_may9.py --v 0.5 --lr 0.03 --step 100 --gamma-par 0.3 --epochs 1500 --space "2, 2" --days "0,31" --mm-cond-number 10 --nheads 300
```


# 18126
``` cd ./jobscript/tco/gp_exercise ```
```  nano fit_day_vecc_v05_may21_18126.sh  ``` 
```  sbatch fit_day_vecc_v05_may21_18126.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vec_v05_may9_18126                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_may21_18126_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_may21_18126_%j.err # Standard error file (%j = JobID)
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
echo "fit_vecc_v05_may09_20000_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_may9.py --v 0.5 --lr 0.03 --step 100 --gamma-par 0.3 --epochs 1500 --space "1, 1" --days "24,30" --mm-cond-number 10 --nheads 300


```

## May 22 I did 0,6 6, 12, 12, 18 (18,24) (24,30)

