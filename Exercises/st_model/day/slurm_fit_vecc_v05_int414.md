### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

For significant updates or installations, use pip install --force-reinstall or python setup.py install after copying the files.

### Copy run file from ```local``` to ```Amarel HPC```
# mac

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_vecc_day_v05_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


# window
scp "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\st_model\fit_st_torch_327.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy estimate file from ```Amarel HPC``` to ```local computer```

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/vecc_extra_estimates_1250_july24.pkl "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"

### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```


## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_st_vecc_v05_int414.py --v 0.5 --lr 0.01 --epochs 3000 --space "20, 20" --days 1 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```



### Job Order SLURM for both vecchia and full
```mkdir -p ./jobscript/tco/gp_exercise```     


```  cd ./jobscript/tco/gp_exercise  ```          


```  rm fit_vecc_1250.sh  ``` 

```  nano fit_day_vecc_v05_1250.sh  ``` 
```  sbatch fit_day_vecc_v05_1250.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=fit_day_vecc_v05_1250                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_day_vecc_v05_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_day_vecc_v05_1250_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                        # Number of CPU cores per task
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
echo "fit_st_vecc_v05_1250_save_estimates"


srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_416.py --v 0.5 --lr 0.01 --epochs 3000 --space "4,4" --days 31 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"  

```


#######  5000 data

```  nano fit_vecc_5000.sh  ```    

``` 

#!/bin/bash
#SBATCH --job-name=fit_v5000                        # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_v5000_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_v5000_%j.err  # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                 # Number of tasks
#SBATCH --cpus-per-task=40                                         # Number of CPU cores per task
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
echo "fit_st_vecc_5000_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_v_int414.py --v 0.5 --lr 0.01 --epochs 3000 --space "2, 2" --days 31 --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"  

```

