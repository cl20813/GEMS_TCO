### Copy files

Below includes: original csv files by month, dictionary map for original data, and processed data with 20,000 rows per hour. 

```scp -r "C:\\Users\\joonw\\TCO\\data_engineering\\data_2023" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_data```   
```scp -r "C:\\Users\\joonw\\TCO\\data_engineering\\data_2024" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_data```         

I found errors when I try to load pickle files in Amarel that are created from my local computer. Hence I upload csv files and then
remake pickle files in Amaral HPC.
       
```scp "C:\Users\joonw\TCO\GEMS_TCO-1\data_preprocessing\groupdata_by_center.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/pipeline```            
```srun --cpus-per-task=4 --partition main --mem=20G --time=05:00:00 python /home/jl2815/tco/pipeline/groupdata_by_center.py```                     
```scp "C:\Users\joonw\TCO\pipeline_2025\job_scripts\deep_learning_cnn_lstm.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25```


### Run

```ssh jl2815@amarel.rutgers.edu```     
```module use /projects/community/modulefiles```     
```module load anaconda/2024.06-ts840```     
```conda activate gems_tco```     
```srun --cpus-per-task=4 --partition main --mem=20G --time=05:00:00 python /home/jl2815/tco/exercise_25/deep_learning_cnn_lstm.py```       


cd ./jobscript/tco/dl
nano cnn_lstm1.sh                # open a new text editor

'''
#!/bin/bash
#SBATCH --job-name=cnn_lstm1                                       # Job name
#SBATCH --output=/home/jl2815/GEMS/cnn_lstm1_%j.out            # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/GEMS/cnn_lstm1_%j.err              # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=8                 
#SBATCH --mem=200G                          
#SBATCH --partition=gpu                     

#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 
conda activate gems_tco

echo "Current date and time: $(date)"

#### Run the Python script

echo "testing cnn_lstm 1"

srun python /home/jl2815/tco/exercise_25/deep_learning_cnn_lstm.py
```
cd ./jobscript/tco/dl
sbatch cnn_lstm1.sh  
