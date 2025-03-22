### Update my packages
# mac
scp -r "/Users/joonwonlee/GEMS_TCO/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

For significant updates or installations, use pip install --force-reinstall or python setup.py install after copying the files.

### Copy from ```local``` to ```Amarel HPC```

# mac
scp "/Users/joonwonlee/GEMS_TCO/Exercises/st_model/fit_st_torch_322.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

# window
scp "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\st_model\fit_st_torch_322.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy from ```Amarel HPC``` to ```local computer```

# window
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py "C:\Users\joonw\TCO\GEMS_TCO-1\Exercises\st_model\"

### Run this part
```ssh jl2815@amarel.rutgers.edu```
```module use /projects/community/modulefiles```           
```module load anaconda/2024.06-ts840``` 
```conda activate gems_tco```

## space 5 5: 5x10, 4 4: 25x50, 2 2: 50x100


``` srun --cpus-per-task=1 --mem=5G --time=05:00:00 python /home/jl2815/tco/exercise_25/st_model/fit_st_torch_322.py --v 0.5 --space 20 20 --keys 0 8 --mm_cond_number=5 --params 20 10 5 .2 .2 .05 5  ```