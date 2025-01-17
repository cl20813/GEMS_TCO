### Copy files

Below includes: original csv files by month, dictionary map for original data, and processed data with 20,000 rows per hour. 

```scp -r "C:\\Users\\joonw\\TCO\\data_engineering\\data_2023" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_data```   
```scp -r "C:\\Users\\joonw\\TCO\\data_engineering\\data_2024" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_data```         

```scp "C:\Users\joonw\TCO\pipeline_2025\job_scripts\deep_learning_cnn_lstm.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25```

### Run

```ssh jl2815@amarel.rutgers.edu```     
```module use /projects/community/modulefiles```     
```module load anaconda/2024.06-ts840```     
```conda activate gems_tco```     
```srun --cpus-per-task=16 --partition gpu --mem=20G --time=05:00:00 python /home/jl2815/tco/exercise_25/deep_learning_cnn_lstm.py```       


nano cnn_lstm1.sh                # open a new text editor

'''
#!/bin/bash
#SBATCH --job-name=cnn_lstm1                                       # Job name
#SBATCH --output=/home/jl2815/GEMS/cnn_lstm1_%j.out            # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/GEMS/cnn_lstm1_%j.err              # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=40                 
#SBATCH --mem=350G                          
#SBATCH --partition=gpu                     

# Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

# Initialize Conda
eval "$(conda shell.bash hook)"                           # Initialize Conda for SLURM environment
conda activate gems_tco

echo "Current date and time: $(date)"

# Run the Python script
# 

echo " lat_idx 5,6,7,8,9 --key 1 --rho 3 --v 0.5 --params 20 10 5 0.5 0.5 0.2 --mm_cond_number 10 --bounds 0.05 60 0.05 15 0.05 15 -15 15 0.05 20 0.05 1.5 "

srun python /home/jl2815/tco/exercise/fit_st_bylat_11_16.py --key 1 --lat_idx 5 --rho 3 --v 0.5 --params 20 10 5 0.5 0.5 0.2 --mm_cond_number 10 --bounds 0.05 60 0.05 15 0.05 15 -15 15 0.05 20 0.05 1.5 

```

sbatch fit_st_11_lat6.sh 
