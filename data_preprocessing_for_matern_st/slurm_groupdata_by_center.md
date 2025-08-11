### Job Order SLURM: groupdata_by_center.py 

### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 


## I need to upload cvs files

scp "/Users/joonwonlee/Documents/GEMS_DATA/data_2024/data_24_07_0131_N05_E123133.csv" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/data_2024

### Copy run file from ```local``` to ```Amarel HPC```
# mac

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing/groupdata_by_center.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data_preprocessing/



``` cd ./jobscript/tco/gp_exercise ```
```  nano groupdata_bycenter.sh  ``` 
```  sbatch groupdata_bycenter.sh  ```    

``` 
#!/bin/bash
#SBATCH --job-name=tmp_dataprocess.sh                                       # Job name
#SBATCH --output=/home/jl2815/tco/model_output/data_preprocess1_%j.out            # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/model_output/data_preprocess1_%j.err              # Standard error file (%j = JobID)
#SBATCH --time=5:00:00                
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=16                
#SBATCH --mem=40G                          
#SBATCH --partition=mem                    

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Current date and time: $(date)"
echo "data preprocess to make pickle files"

#### Run the Python script  
srun python /home/jl2815/tco/data_preprocessing/groupdata_by_center.py

```

