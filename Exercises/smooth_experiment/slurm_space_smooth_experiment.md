### Update my packages
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

For significant updates or installations, use pip install --force-reinstall or python setup.py install after copying the files.

### Copy from ```local``` to ```Amarel HPC```

scp "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\smooth_experiment\space_smooth.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/smooth_experiment

### Copy from ```Amarel HPC``` to ```local computer```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/smooth_experiment/space_smooth.py "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\smooth_experiment\"


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```module use /projects/community/modulefiles```           
```module load anaconda/2024.06-ts840``` 
```conda activate gems_tco```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100

```srun --cpus-per-task=8 --mem=20G --time=05:00:00 python /home/jl2815/tco/exercise_25/smooth_experiment/space_smooth.py --key 2 --params 60 8.25 8.25 0.5 0.5 5 --space 20 20 --mm_cond_number 10```

`

### Job Order SLURM: fv_onelag.py    
```mkdir -p ./jobscript/tco/gp_exercise```      

```cd ./jobscript/tco/gp_exercise```                          
```nano space_smooth.sh```         (rm vecc_per_search.sh)        # open a new text editor                         

```
#!/bin/bash
#SBATCH --job-name=space_smooth                                      # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/space_smooth_%j.out    # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/space_smooth_%j.err     # Standard error file (%j = JobID)
#SBATCH --time=24:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=100G                                                 # Memory per node
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
echo "Experiment space smoothness using Vecchia approximation"





```

```cd ./jobscript/tco/gp_exercise```                          
```sbatch vecc_per_search.sh```           

############################################################################################################# 
