### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

For significant updates or installations, use pip install --force-reinstall or python setup.py install after copying the files.

### Copy run file from ```local``` to ```Amarel HPC```

# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/fit_st_torch_322.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/fit_st_torch_vecc_322.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model



# window
scp "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\st_model\fit_st_torch_322.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy from ```Amarel HPC``` to ```local computer```

# window
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\st_model\"

### Run this part
```ssh jl2815@amarel.rutgers.edu```
```  module use /projects/community/modulefiles  ```           
```  module load anaconda/2024.06-ts840  ``` 
```  conda activate faiss_env   ```


## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100


``` srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 20 20 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34   ```


``` srun --cpus-per-task=3 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --epochs 5 --space 20 20 --keys 0 8 --mm_cond_number=5 --params 20 20 5 .2 .2 .05 5  ```

### Job Order SLURM for both vecchia and full
```mkdir -p ./jobscript/tco/gp_exercise```     

```   sbatch fit_st_torch_vec.sh   ```
```    sbatch fit_st_torch_full.sh    ``` 


```  cd ./jobscript/tco/gp_exercise  ```                             
```  nano fit_st_torch_full.sh  ```        
```  nano fit_st_torch_vec.sh  ```    

``` 
#!/bin/bash
#SBATCH --job-name=fit_st_torch_full                                  # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_st_torch_full_%j.out    # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_st_torch_full_%j.err     # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=250G                                                 # Memory per node
#SBATCH --partition=mem                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"

#### Run the Python script { (20,20):(5,10), (5,5):(20,40) }
echo "fit_st_torch_fv"

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 20 20 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.001 --epochs 3000 --space 20 20 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 10 10 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.001 --epochs 3000 --space 10 10 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 4 4 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.001 --epochs 3000 --space 4 4 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 


srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 2 2 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --lr 0.001 --epochs 3000 --space 2 2 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 


```


```   cd ./jobscript/tco/gp_exercise   ```                          
```  nano fit_st_torch_vec.sh  ```         


``` 
#!/bin/bash
#SBATCH --job-name=fit_st_torch_vec2                                  # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_st_torch_vec2_%j.out    # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_st_torch_vec2_%j.err     # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
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

#### Run the Python script { (20,20):(5,10), (5,5):(20,40) }
echo "fit_st_torch_fv"

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 20 20 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --lr 0.001 --epochs 3000 --space 20 20 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 10 10 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --lr 0.001 --epochs 3000 --space 10 10 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 4 4 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --lr 0.001 --epochs 3000 --space 4 4 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --lr 0.01 --epochs 3000 --space 2 2 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34 

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_vecc_322.py --v 0.5 --lr 0.001 --epochs 3000 --space 2 2 --keys 0 8 --mm_cond_number=5 --params 24.42 1.92 1.92 0.001 -0.045 0.237 3.34


```