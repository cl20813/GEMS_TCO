### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### Copy run file from ```local``` to ```Amarel HPC```
# mac

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_vecc_day_v04_may28.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


# window

scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_vecc_day_v10_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy estimate file from ```Amarel HPC``` to ```local computer```

# 1250
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_reord_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"  

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_ori_ord_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"  

# 5000
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_reord_5000.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_ori_ord5000.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/" 



### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100


```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v10_416.py --v 0.5 --lr 0.0005 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```



### Job Order SLURM for both vecchia and full
```mkdir -p ./jobscript/tco/gp_exercise```     


### v04

```  cd ./jobscript/tco/gp_exercise  ``` 
```  nano fit_day_vecc_v04_may28.sh  ``` 
```  sbatch fit_day_vecc_v04_may28.sh  ``` 

``` 

#!/bin/bash
#SBATCH --job-name=fit_day_vecc_v04_may28                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_day_vecc_v04_may28_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_day_vecc_v04_may28_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=16                                        # Number of CPU cores per task
#SBATCH --mem=240G                                                 # Memory per node
#SBATCH --partition=main                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"
echo "fit_day_vecc_v04_may28"


srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v04_may28.py --v 0.4 --lr 0.02 --step 100 --gamma-par 0.3 --epochs 1000 --space "2, 2" --days "0, 31" --mm-cond-number 10 --nheads 300   


```




