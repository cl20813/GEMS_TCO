### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### Copy run file from ```local``` to ```Amarel HPC```
# mac

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_full_day_spline_511.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

# window

scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_full_day_v10_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_full_day_v15_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


### Copy estimate file from ```Amarel HPC``` to ```local computer```

# mac

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/full_v(0.5)_estimation_1250_july24.pkl "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/saved/full_v15_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/full_day_v10_spline1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"


scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/full_day_v10_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/full_day_v05_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"


# window
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/full_v10_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"  "C:\\Users\\joonw\\tco\\GEMS_TCO-2\\Exercises\\from_amarel"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/full_v15_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/"  "C:\\Users\\joonw\\tco\\GEMS_TCO-2\\Exercises\\from_amarel"


### Run this part
```  ssh jl2815@amarel.rutgers.edu  ```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```


## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100


```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_full_day_spline_506.py --v 0.4 --lr 0.03 --step 100 --coarse-factor 10 --gamma-par 0.5 --epochs 1000 --space "20, 20" --days "1,2" --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"   ```


```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_full_day_spline_506.py --v 0.45 --lr 0.03 --step 100 --coarse-factor 10 --gamma-par 0.5 --epochs 1000 --space "20, 20" --days "1,2" --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34" ```


### Job Order SLURM for both vecchia and full
```mkdir -p ./jobscript/tco/gp_exercise```     

```  cd ./jobscript/tco/gp_exercise  ```   
```  nano fit_full_day_v04_1250_spline.sh  ```        
 ```   sbatch fit_full_day_v04_1250_spline.sh   ```

``` 
#!/bin/bash
#SBATCH --job-name=ful_v04_day_1250                             # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_full_day_v04_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_full_day_v04_1250_%j.err # Standard error file (%j = JobID)
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

echo "fit_full_day_v04_1250 using cubic spline"

srun python /home/jl2815/tco/exercise_25/st_model/fit_full_day_v04_506.py --v 0.4 --lr 0.02 --step 100 --coarse-factor 1000 --gamma-par 0.2 --epochs 1500 --space "4, 4" --days "0,31"

```


```  cd ./jobscript/tco/gp_exercise  ```   
```  nano fit_full_day_v03_1250_estimates.sh  ```        
 ```   sbatch fit_full_day_v03_1250_estimates.sh   ```

``` 
#!/bin/bash
#SBATCH --job-name=ful_v03_day_1250                             # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_full_day_v03_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_full_day_v03_1250_%j.err # Standard error file (%j = JobID)
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

echo "fit_full_day_v03_1250 using cubic spline"

srun python /home/jl2815/tco/exercise_25/st_model/fit_full_day_v04_506.py --v 0.3 --lr 0.02 --step 100 --coarse-factor 1000 --gamma-par 0.2 --epochs 1500 --space "4, 4" --days "0,31"

```


```  cd ./jobscript/tco/gp_exercise  ```   
```  nano fit_full_day_v045_1250.sh  ```        
 ```   sbatch fit_full_day_v045_1250.sh   ```

``` 

#!/bin/bash
#SBATCH --job-name=ful_v045_day_1250                             # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_full_day_v045_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_full_day_v045_1250_%j.err # Standard error file (%j = JobID)
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

echo "fit_full_day_v045_1250 using cubic spline"


srun python /home/jl2815/tco/exercise_25/st_model/fit_full_day_spline_511.py --v 0.45 --lr 0.03 --step 100 --coarse-factor 1000 --gamma-par 0.3 --epochs 1500 --space "4, 4" --days "0,31" 

```



```  cd ./jobscript/tco/gp_exercise  ```   
```  nano fit_full_day_v055_1250.sh  ```        
 ```   sbatch fit_full_day_v055_1250.sh   ```

``` 

#!/bin/bash
#SBATCH --job-name=ful_v055_day_1250                             # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_full_day_v055_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_full_day_v055_1250_%j.err # Standard error file (%j = JobID)
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

echo "fit_full_day_v055_1250 using cubic spline"


srun python /home/jl2815/tco/exercise_25/st_model/fit_full_day_spline_506.py --v 0.55 --lr 0.02 --step 100 --coarse-factor 1000 --gamma-par 0.2 --epochs 1500 --space "4, 4" --days "0,31" 

```





