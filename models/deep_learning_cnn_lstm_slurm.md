### Copy files

I found errors when I try to load pickle files in Amarel that are created from my local computer. Hence I upload csv files and then
remake pickle files in Amaral HPC. Hence, transfer csv files first and then process the data in Amarel directly to make 20,000 rows per hour. 

```scp -r "C:\\Users\\joonw\\TCO\\data_engineering\\data_2023" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_data```   
```scp -r "C:\\Users\\joonw\\TCO\\data_engineering\\data_2024" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_data```         
```scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\data_preprocessing" jl2815@amarel.rutgers.edu:/home/jl2815/tco```        

```srun --cpus-per-task=4 --partition main --mem=20G --time=05:00:00 python /home/jl2815/tco/data_preprocessing/groupdata_by_center.py```                     
```scp "C:\Users\joonw\TCO\GEMS_TCO-1\models\deep_learning_cnn_lstm.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/models```

### Run deep_learning_cnn_lstm.py

```ssh jl2815@amarel.rutgers.edu```     
```module use /projects/community/modulefiles```     
```module load anaconda/2024.06-ts840```     
```conda activate gems_tco```     
```srun --cpus-per-task=4 --partition main --mem=20G --time=05:00:00 python /home/jl2815/tco/models/deep_learning_cnn_lstm.py```       



### Job Order SLURM: deep_learning_cnn_lstm.py 

```cd ./jobscript/tco/dl```                       
```nano cnn_lstm1.sh```         (rm cnn_lstm1.sh)        # open a new text editor                      

```
#!/bin/bash
#SBATCH --job-name=cnn_lstm1                                       # Job name
#SBATCH --output=/home/jl2815/tco/model_output/cnn_lstm1_%j.out    # Standard output file (%j = JobID)
#SBATCH --error=/home/jl2815/tco/model_output/cnn_lstm1_%j.err     # Standard error file (%j = JobID)
#SBATCH --time=72:00:00                                            # Time limit
#SBATCH --ntasks=2                                                 # Number of tasks
#SBATCH --cpus-per-task=8                                          # Number of CPU cores per task
#SBATCH --mem=200G                                                 # Memory per node
#SBATCH --partition=gpu                                            # Partition name
#SBATCH --gres=gpu:1                                               # Number of GPUs per node


#### Load the Anaconda module to use srun 
module purge                                              
module use /projects/community/modulefiles                 
module load anaconda/2024.06-ts840 

#### Initialize conda for the current shell session if not already done for the current shell session.
eval "$(conda shell.bash hook)"
conda activate gems_tco

echo "Current date and time: $(date)"

#### Run the Python script
echo "testing cnn_lstm 1"

srun python /home/jl2815/tco/models/deep_learning_cnn_lstm.py
srun python /home/jl2815/tco/models/deep_learning_cnn_lstm_cpu.py
```

cd ./jobscript/tco/dl       
sbatch cnn_lstm1.sh         

#############################################################################################################

### Job Order SLURM: groupdata_by_center.py (make 20,000 rows per hour)

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
```
cd ./jobscript/tco/dl
sbath tmp_datapreprocess.sh




