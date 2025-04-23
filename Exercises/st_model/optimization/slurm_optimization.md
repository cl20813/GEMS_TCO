### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\tco\GEMS_TCO-2\src\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\optimization\vecc_alg_opt_ama.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model 

scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\optimization\vecc_opt_hyper_ama.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


### Copy run file from ```local``` to ```Amarel HPC```
# mac

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/optimization/vecc_alg_opt_ama.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/optimization/vecc_opt_hyper_ama.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


### Copy estimate file from ```Amarel HPC``` to ```local computer```

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/full_v(0.5)_estimation_1250_july24.pkl "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"


## window
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/optimization/output/hyper_parm_opt_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"  "C:\\Users\\joonw\\tco\\GEMS_TCO-2\\Exercises\\from_amarel"


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```


## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100


```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/vecc_alg_opt_ama.py --v 0.5 --lr 0.02 --step 80 --epochs 3000 --space "20, 20" --days 1 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/vecc_opt_hyper_ama.py --v 0.5 --lr 0.02 --epochs 2500 --space "20, 20" --days 1 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34" ```


### Job Order SLURM for both vecchia and full
```mkdir -p ./jobscript/tco/gp_exercise```     


```  cd ./jobscript/tco/gp_exercise  ```   
```  nano vecc_alg_opt.sh  ```        
 ```   sbatch vecc_alg_opt.sh   ```

``` 
#!/bin/bash
#SBATCH --job-name=vecc_alg_opt2                             # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vecc_alg_opt_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vecc_alg_opt_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=300G                                                 # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"

echo "which data type is more efficient for vecchia?"

srun python /home/jl2815/tco/exercise_25/st_model/vecc_alg_opt_ama.py --v 0.5 --lr 0.02 --step 80 --epochs 3000 --space "20, 20" --days 1 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34" 

srun python /home/jl2815/tco/exercise_25/st_model/vecc_alg_opt_ama.py --v 0.5 --lr 0.25 --step 80 --epochs 3000 --space "8, 8" --days 15 --mm-cond-number 10 --nheads 100 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34" 

srun python /home/jl2815/tco/exercise_25/st_model/vecc_alg_opt_ama.py --v 0.5 --lr 0.03 --step 80 --epochs 3000 --space "6, 6" --days 15 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34" 



```

```  cd ./jobscript/tco/gp_exercise  ```  
```  nano vecc_hyp_opt.sh  ```        
 ```   sbatch vecc_hyp_opt.sh   ```


``` 
#!/bin/bash
#SBATCH --job-name=vecc_hyp_opt                             # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vecc_hyp_opt_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vecc_hyp_opt_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=300G                                                 # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"

echo "fit_hyper_opt"

srun python /home/jl2815/tco/exercise_25/st_model/vecc_opt_hyper_ama.py --v 0.5 --lr 0.02 --epochs 2500 --space "4, 4" --days 1 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34" 

```






