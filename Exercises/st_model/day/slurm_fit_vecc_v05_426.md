### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_vecc_day_v05_ori_ord_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_vecc_day_v10_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

# window
scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_vecc_day_v05_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
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

```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_416.py --v 0.5 --lr 0.0005 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```


```    srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v10_416.py --v 0.5 --lr 0.0005 --step 80 --epochs 1000 --space "20, 20" --days "12,13" --mm-cond-number 10 --nheads 200 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"    ```



### Job Order SLURM for both vecchia and full
### smooth 0.5
```mkdir -p ./jobscript/tco/gp_exercise```     

```  cd ./jobscript/tco/gp_exercise  ```          
```  nano fit_day_vecc_v05_reorder1250.sh  ``` 
```  sbatch fit_day_vecc_v05_reorder1250.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vec_Drev05_1250                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_Drev05_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_Drev05_1250_%j.err # Standard error file (%j = JobID)
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
echo "fit_vecc_v05_reorder_1250_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_reord_416.py --v 0.5 --lr 0.02 --step 80 --epochs 3000 --space "4, 4" --days 31 --mm-cond-number 10 --nheads 300

```

## original ordering

```  nano fit_day_vecc_v05_oriorder1250.sh  ``` 
```  sbatch fit_day_vecc_v05_oriorder1250.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vec_Doriv05_1250                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_Doriv05_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_Doriv05_1250_%j.err # Standard error file (%j = JobID)
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
echo "fit_vecc_v05_order_1250_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_ori_ord_416.py --v 0.5 --lr 0.02 --step 80 --epochs 3000 --space "4, 4" --days 31 --mm-cond-number 10 --nheads 300

```



### v10

```  cd ./jobscript/tco/gp_exercise  ``` 
```  nano fit_day_vecc_v10_1250.sh  ``` 
```  sbatch fit_day_vecc_v10_1250.sh  ``` 


``` 
#!/bin/bash
#SBATCH --job-name=vecc_Dv10_1250                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vecc_Dv10_1250_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vecc_Dv10_1250_%j.err # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                        # Number of CPU cores per task
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
echo "fit_vecc_v10_1250_save_estimates"


srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v10_416.py --v 1.0 --lr 0.0005 --step 200 --epochs 1000 --space "2, 2" --days "12, 13" --mm-cond-number 10 --nheads 300 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34"  

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v10_416.py --v 1.0 --lr 0.0005 --step 200 --epochs 1000 --space "1, 1" --days "12, 13" --mm-cond-number 10 --nheads 300 --params "24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34" 

```



#######  5000 data

```  nano fit_day_vecc_v05_reorder5000.sh  ``` 
```  sbatch fit_day_vecc_v05_reorder5000.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vec_Drev05_5000                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_Drev05_5000_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_Drev05_5000_%j.err # Standard error file (%j = JobID)
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
echo "fit_vecc_v05_reorder_1250_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_reord_416.py --v 0.5 --lr 0.02 --step 80 --epochs 3000 --space "2, 2" --days "0,  31" --mm-cond-number 10 --nheads 300

```

## original ordering

```  nano fit_day_vecc_v05_oriorder5000.sh  ``` 
```  sbatch fit_day_vecc_v05_oriorder5000.sh  ``` 

``` 
#!/bin/bash
#SBATCH --job-name=vec_Doriv05_5000                         # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_Doriv05_5000_%j.out     # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_Doriv05_5000_%j.err # Standard error file (%j = JobID)
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
echo "fit_vecc_v05_order_1250_save_estimates"

srun python /home/jl2815/tco/exercise_25/st_model/fit_vecc_day_v05_ori_ord_416.py --v 0.5 --lr 0.02 --step 80 --epochs 3000 --space "2, 2" --days 31 --mm-cond-number 10 --nheads 300


```

