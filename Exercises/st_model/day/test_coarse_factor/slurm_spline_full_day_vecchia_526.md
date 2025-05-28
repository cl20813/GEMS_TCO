### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### Copy run file from ```local``` to ```Amarel HPC```
# mac

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/test_coarse_factor/spline_vecchia_testing526.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


### Run this part
```  ssh jl2815@amarel.rutgers.edu  ```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```


## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100


```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/spline_vecchia_testing526.py --v 0.4 --coarse-factor 4 --space "16, 16" --days "0,1" --mm-cond-number 10 --nheads 20    ```


### Job Order SLURM for both vecchia and full

```mkdir -p ./jobscript/tco/gp_exercise```     

```  cd ./jobscript/tco/gp_exercise  ```   
```  nano spline_vecc_coarse_testing.sh  ```        
 ```   sbatch spline_vecc_coarse_testing.sh  ```

``` 
#!/bin/bash
#SBATCH --job-name=spline_vecc_coarse_testing                            # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/spline_vecc_coarse_testing%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/spline_vecc_coarse_testing%j.err # Standard error file (%j = JobID)
#SBATCH --time=24:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=350G                                                 # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"

echo "spline_vecc_coarse_testing"

srun python /home/jl2815/tco/exercise_25/st_model/spline_vecchia_testing526.py --v 0.4 --coarse-factor 4 --space "2, 2" --days "0,10" --mm-cond-number 10 --nheads 300

```


```  cd ./jobscript/tco/gp_exercise  ```   
```  nano spline_vecc_coarse_testing4k.sh  ```        
 ```   sbatch spline_vecc_coarse_testing4k.sh  ```

``` 
#!/bin/bash
#SBATCH --job-name=spline_vecc_coarse_testing4k                            # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/spline_vecc_coarse_testing4k%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/spline_vecc_coarse_testing4k%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=200G                                                 # Memory per node
#SBATCH --partition=main                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"

echo "spline_vecc_coarse_testing"

srun python /home/jl2815/tco/exercise_25/st_model/spline_vecchia_testing526.py --v 0.4 --coarse-factor 4 --space "2, 2" --days "0,31" --mm-cond-number 10 --nheads 300

```




```  cd ./jobscript/tco/gp_exercise  ```   
```  nano spline_vecc_coarse_testing18k.sh  ```        
 ```   sbatch spline_vecc_coarse_testing18k.sh  ```

``` 
#!/bin/bash
#SBATCH --job-name=spline_vecc_coarse_testing18k                            # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/spline_vecc_coarse_testing18k%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/spline_vecc_coarse_testing18k%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=16                                       # Number of CPU cores per task
#SBATCH --mem=350G                                                 # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"

echo "spline_vecc_coarse_testing"

srun python /home/jl2815/tco/exercise_25/st_model/spline_vecchia_testing526.py --v 0.4 --coarse-factor 4 --space "1, 1" --days "0,31" --mm-cond-number 10 --nheads 400

```







