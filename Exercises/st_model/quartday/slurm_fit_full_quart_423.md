### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\tco\GEMS_TCO-2\src\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 


### Copy run file from ```local``` to ```Amarel HPC```
# mac

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/quartday/fit_full_quart_v05_423.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/quartday/fit_full_quart_v10_423.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/quartday/fit_full_quart_v15_423.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

# window

scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\quartday\fit_full_quart_v05_423.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\quartday\fit_full_quart_v15_423.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy estimate file from ```Amarel HPC``` to ```local computer```

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/halfday/full_v(0.5)_estimation_1250_july24.pkl "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```


## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_full_quart_v05_423.py --v 0.5 --lr 0.02 --step 60 --avg-days 3 --epochs 3000 --space "20, 20" --days 1 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_full_quart_v10_423.py --v 1.0 --lr 0.01 --epochs 3000 --space "20, 20" --days 1 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"   ```

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_full_quart_v15_423.py --v 1.5 --lr 0.01 --epochs 3000 --space "20, 20" --days 1 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34" ```


### Job Order SLURM for both vecchia and full
```mkdir -p ./jobscript/tco/gp_exercise```     


 ```  cd ./jobscript/tco/gp_exercise  ```   

 ```  nano fit_full_quart_1250_estimates.sh  ```        
 ```   sbatch fit_full_quart_1250_estimates.sh   ```


``` 
#!/bin/bash
#SBATCH --job-name=fit_quart_full_1250                             # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_full_quart_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_full_quart_1250_%j.err # Standard error file (%j = JobID)
#SBATCH --time=48:00:00                                            # Time limit
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
echo "fit_full_quart_v05_v15 1250"


srun python /home/jl2815/tco/exercise_25/st_model/fit_full_quart_v05_423.py --v 0.5 --lr 0.02 --step 60 --avg-days 3 --epochs 3000 --space "4, 4" --days 31 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"

srun python /home/jl2815/tco/exercise_25/st_model/fit_full_quart_v15_423.py --v 1.5 --lr 0.02 --step 60 --avg-days 3 --epochs 3000 --space "4, 4" --days 31 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"


```


