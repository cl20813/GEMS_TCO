### Update my packages
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

For significant updates or installations, use pip install --force-reinstall or python setup.py install after copying the files.

### Copy from ```local``` to ```Amarel HPC```
scp "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\likelihood_exercise\testing\full_vecc_matern_separable.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/likelihood_exercise

### Copy from ```Amarel HPC``` to ```local computer```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_matern_separable.py "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\likelihood_exercise\testing"


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```module use /projects/community/modulefiles```           
```module load anaconda/2024.06-ts840``` 
```conda activate gems_tco```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

#  15 1.25 1.25 12 1 0.2   my random number this guarantee good approximation. 

# best search on 2/26/2025 10 10 10 5 5 0.01
# To avoid singular, I need to find good initial point
#  4.32 11.99 6.52 2.21 2.39 0.22    estimated from full likelihood  --> approximation does not work 
#  4.85 20 0.01 2.40 3.61 0.002    estimated from vecchia likelihood

```  srun --cpus-per-task=2 --mem=10G --time=01:00:00 python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_matern_separable.py --keys 0 8 --params 15 1.25 1.25 12 1 0.2 --space 15 15 --mm_cond_number 5  ```


```  srun --cpus-per-task=2 --mem=10G --time=01:00:00 python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_matern_separable.py --keys 0 8 --params 4.32 11.99 6.52 2.21 2.39 0.22 --space 15 15 --mm_cond_number 5  ```


# the bset number so far  15 1.25 1.25 12 1 0.2 

### Job Order SLURM: fv_onelag.py    
```mkdir -p ./jobscript/tco/gp_exercise```      

```cd ./jobscript/tco/gp_exercise```                          
```nano fv_onelag.sh```         (rm cnn_lstm1.sh)        # open a new text editor                         

```
#!/bin/bash
#SBATCH --job-name=fv_like_onelag                                       # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fv_onelag_%j.out    # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fv_onelag_%j.err     # Standard error file (%j = JobID)
#SBATCH --time=48:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=300G                                                 # Memory per node
#SBATCH --partition=main                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate gems_tco

echo "Current date and time: $(date)"

#### Run the Python script { (20,20):(5,1), (5,5):(20,40) }
echo "Compare full likelihood vs Vecchia approximation; one t lag"

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 50 --params 60 8.25 8.25 0.5 0.5 5 --space 5 5 --mm_cond_number 10
srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 50 --params 60 5 5 0.7 0.7 5 --space 5 5 --mm_cond_number 10

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 50 --params 60 8.25 8.25 0.5 0.5 5 --space 4 4 --mm_cond_number 10
srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 50 --params 60 5 5 0.7 0.7 5 --space 4 4 --mm_cond_number 10

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 50 --params 60 8.25 8.25 0.5 0.5 5 --space 3 3 --mm_cond_number 10
srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 50 --params 60 5 5 0.7 0.7 5 --space 3 3 --mm_cond_number 10

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 50 --params 60 8.25 8.25 0.5 0.5 5 --space 2 2 --mm_cond_number 10
srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 50 --params 60 5 5 0.7 0.7 5 --space 2 2 --mm_cond_number 10

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 2 --params 60 8.25 8.25 0.5 0.5 5 --space 1 1 --mm_cond_number 10
srun python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 2 --params 60 5 5 0.7 0.7 5 --space 1 1 --mm_cond_number 10

```

```cd ./jobscript/tco/gp_exercise```                          
```sbatch fv_onelag.sh```           

############################################################################################################# 
