### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

For significant updates or installations, use pip install --force-reinstall or python setup.py install after copying the files.

### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/GEMS_TCO_EDA/semivariograms/directional_emp_sem.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/GEMS_TCO_EDA/semivariograms/empirical_sem_map_short_lag.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25

# window
scp "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\st_model\fit_st_torch_327.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


### Copy estimate file from ```Amarel HPC``` to ```local computer```
##### change name for directional semivariograms
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/empirical_n135_45_sem_20000_july24.pkl "/Users/joonwonlee/Documents/GEMS_TCO-1/GEMS_TCO_EDA/outputs"  

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/empirical_lon_sem_20000_july24.pkl "/Users/joonwonlee/Documents/GEMS_TCO-1/GEMS_TCO_EDA/outputs"


scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/empirical_lat_sem_20000_july24.pkl "/Users/joonwonlee/Documents/GEMS_TCO-1/GEMS_TCO_EDA/outputs"






### Run this part
```  ssh jl2815@amarel.rutgers.edu  ```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```


## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

``` srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/directional_emp_sem.py --space 20 20 --days 2    ```



### Job Order SLURM for both vecchia and full
```mkdir -p ./jobscript/tco/gp_exercise```     

```   sbatch dir_emp.sh   ```


```  cd ./jobscript/tco/gp_exercise  ```                             
```  nano dir_emp.sh  ```        
 

``` 
#!/bin/bash
#SBATCH --job-name=dir_emp                            # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/dir_emp%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/dir_emp%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
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

echo "compute empirical semivariograms by lat,lon distance h"


srun python /home/jl2815/tco/exercise_25/directional_emp_sem.py --space 1 1 --days 31 

```


```  nano dir_emp.sh  ```        
 

``` 
#!/bin/bash
#SBATCH --job-name=dir_emp                            # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/dir_emp%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/dir_emp%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
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

echo "compute empirical semivariograms by lat,lon distance h"


srun python /home/jl2815/tco/exercise_25/directional_emp_sem.py --space 4 4 --days 31 

```

