### Job Order SLURM: groupdata_by_center.py 

### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### Copy run file from ```local``` to ```Amarel HPC```
# mac

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_full_day_spline_511.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model




cd ./jobscript/tco/dl          
nano tmp_dataprocess.sh          (rm tmp_dataprocess.sh)      # open a new text editor         

```
#!/bin/bash
#SBATCH --job-name=tmp_dataprocess.sh                                       # Job name
#SBATCH --output=/home/jl2815/tco/model_output/data_preprocess1_%j.out            # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/model_output/data_preprocess1_%j.err              # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=8                 
#SBATCH --mem=200G                          
#SBATCH --partition=mem                    

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate gems_tco

echo "Current date and time: $(date)"

#### Run the Python script  
srun python /home/jl2815/tco/data_preprocessing/groupdata_by_center.py 
srun python /home/jl2815/tco/data_preprocessing/groupdata_by_center.py  
```
cd ./jobscript/tco/dl
sbath tmp_datapreprocess.sh
