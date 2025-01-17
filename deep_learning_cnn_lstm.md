### Copy files

Below includes: original csv files by month, dictionary map for original data, and processed data with 20,000 rows per hour. 

```scp -r "C:\\Users\\joonw\\TCO\\data_engineering\\data_2023" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_data```   
```scp -r "C:\\Users\\joonw\\TCO\\data_engineering\\data_2024" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_data```         

```scp "C:\Users\joonw\TCO\pipeline_2025\job_scripts\deep_learning_cnn_lstm.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25```

### Run

```srun --cpus-per-task=16 --mem=20G --time=05:00:00 python /home/jl2815/tco/exercise_25/deep_learning_cnn_lstm.py``` 
