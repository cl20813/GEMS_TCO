### Update my packages
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

For significant updates or installations, use pip install --force-reinstall or python setup.py install after copying the files.

### Copy from ```local``` to ```Amarel HPC```
scp "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\likelihood_exercise\vecc_like_search_params.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/likelihood_exercise

### Copy from ```Amarel HPC``` to ```local computer```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/likelihood_exercise/vecc_like_search_params.py "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\likelihood_exercise\"


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```module use /projects/community/modulefiles```           
```module load anaconda/2024.06-ts840``` 
```conda activate gems_tco```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

```srun --cpus-per-task=8 --mem=20G --time=05:00:00 python /home/jl2815/tco/exercise_25/likelihood_exercise/full_vecc_st_lagone.py --key 2 --params 60 8.25 8.25 0.5 0.5 5 --space 20 20 --mm_cond_number 5```

`

### Job Order SLURM: fv_onelag.py    
```mkdir -p ./jobscript/tco/gp_exercise```      

```cd ./jobscript/tco/gp_exercise```                          
```nano vecc_per_search.sh```         (rm vecc_per_search.sh)        # open a new text editor                         

```
#!/bin/bash
#SBATCH --job-name=vecc_per_search                                      # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/vecc_per_search_%j.out    # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/vecc_per_search_%j.err     # Standard error file (%j = JobID)
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
conda activate gems_tco

echo "Current date and time: $(date)"

#### Run the Python script { (20,20):(5,10), (5,5):(20,40) }
echo "Parameter search using Vecchia approximation; one t lag"

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/vecc_like_search_params.py --key 8 --params 60 8.25 8.25 0.5 0.5 5 --space 20 20 --mm_cond_number 10

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/vecc_like_search_params.py --key 8 --params 60 8.25 8.25 0.5 0.5 5 --space 15 15 --mm_cond_number 10

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/vecc_like_search_params.py --key 8 --params 60 8.25 8.25 0.5 0.5 5 --space 10 10 --mm_cond_number 10

srun python /home/jl2815/tco/exercise_25/likelihood_exercise/vecc_like_search_params.py --key 8 --params 60 8.25 8.25 0.5 0.5 5 --space 5 5 --mm_cond_number 10

```

```cd ./jobscript/tco/gp_exercise```                          
```sbatch vecc_per_search.sh```           

############################################################################################################# 
