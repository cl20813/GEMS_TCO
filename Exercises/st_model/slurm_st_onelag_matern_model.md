### Update my packages
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

For significant updates or installations, use pip install --force-reinstall or python setup.py install after copying the files.

### Copy from ```local``` to ```Amarel HPC```

scp "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\st_model\fit_st_1_27.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy from ```Amarel HPC``` to ```local computer```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\st_model\"


### Run this part
```ssh jl2815@amarel.rutgers.edu```
```module use /projects/community/modulefiles```           
```module load anaconda/2024.06-ts840``` 
```conda activate gems_tco```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100


``` srun --cpus-per-task=32 --mem=20G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 20 20 --keys 0 8 --mm_cond_number=5 --params 40 5.25 5.25 0.3 0.3 0.5 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5 ```


### Job Order SLURM: fv_onelag.py    
```mkdir -p ./jobscript/tco/gp_exercise```      

```cd ./jobscript/tco/gp_exercise```                          
```nano fit_st_onelag.sh```         (rm vecc_per_search.sh)        # open a new text editor     

``` 
#!/bin/bash
#SBATCH --job-name=fit_st_onelag                                      # Job name
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_st_onelag_%j.out    # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_st_onelag_%j.err     # Standard error file (%j = JobID)
#SBATCH --time=24:00:00                                            # Time limit
#SBATCH --ntasks=1                                                # Number of tasks
#SBATCH --cpus-per-task=40                                       # Number of CPU cores per task
#SBATCH --mem=200G                                                 # Memory per node
#SBATCH --partition=main                                            # Partition name

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate gems_tco

echo "Current date and time: $(date)"

#### Run the Python script { (20,20):(5,10), (5,5):(20,40) }
echo "fitting one lage st_model"

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 20 20 --keys 0 8 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 20 20 --keys 8 16 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 20 20 --keys 16 24 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 15 15 --keys 0 8 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 15 15 --keys 8 16 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 15 15 --keys 16 24 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 10 10 --keys 0 8 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 10 10 --keys 8 16 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 10 10 --keys 16 24 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 5 5 --keys 0 8 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 5 5 --keys 8 16 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5

srun python /home/jl2815/tco/exercise_25/st_model/fit_st_1_27.py --v 0.5 --space 5 5 --keys 16 24 --mm_cond_number 5 --params 20 1 1 0.1 0.5 0.1 --bounds 0.05 40 0.05 15 0.05 15 -15 15 0.25 20 0.05 0.5


```