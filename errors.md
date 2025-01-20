## ModuleNotFoundError: No module named 'numpy._core'

This happens when I try to open a pkl file using pandas. This error is in python 311\Lib\pickle.py file.
My environments where I created and loaded the pickle file were different. I used the exact same dependency version on both environments, still getting the issue.

Check in my local computer:
```print(pd.__version__)```   
```print(np.__version__)```   
Then in Amarel Rutgers 
python -c "import pandas as pd; print(pd.__version__)"      
python -c "import numpy as np; print(np.__version__)"    

 Maybe I should askthe creator to use the pandas version which my project is using.
 
## 1. Mismatch python and conda
This happens when I can find package using ```conda list torch``` but cannot find the package in python from         
```python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"```            
```which python``` should return ```/home/jl2815/.conda/envs/gems_tco/bin/python```                
Problem is solved. See -[Install mypackage on Amarel](install_mypackage_amarel.md)     

## 2. Cannot find GEMS_TCO package

when using sbatch, I need to add sys.path.append("/cache/home/jl2815/tco/") in .py file otherwise it cannot read GEMS package
But I think this is unnecessary after solving 1. 

## 2-2. Cannot find skgstat package
Solved: -[Create an environment on Rutgers Amarel](amarel_environment.md)

## 3. "can only concatenate str (not int) to str" when doing f'{key+1}' where  key is a string 

## 4. float is not interable: this could happen when parameter size changes ex) I did --params 0 1 1 1 1 1  where my function only requires 3. 

## 5. Error when using scipy minimize function
Note that ```bounds = [tuple(args.bounds)]```` has to be of this form to be used in ```scipy.minimize()```  

## 6. Object type is not subscriptable 
```def maxmin_naive(dist: np.ndarray, first: np.intp) -> tuple[np.ndarray, np.ndarray]:```  
```def maxmin_naive(dist: np.ndarray, first: int) -> Tuple[np.ndarray, np.ndarray]:```   

```from typing import Tuple```           
```first: np.intp --> first: int```     
```tuple --> Tuple```

## 7. The error "unsupported operand type(s) for |: '_variadicGenericAlias' and 'type'" suggests that the type annotation is incorrect.

```def find_nns_naive(locs: np.ndarray, dist_fun: Callable | str = "euclidean", max_nn: int = 10, **kwargs) -> np.ndarray:```      
```def find_nns_naive(locs: np.ndarray, dist_fun: Union[Callable, str] = "euclidean", max_nn: int = 10, **kwargs) -> np.ndarray:```    

1) ```from typing import Callable, Union```   
2) Replace ```Callable | str = "euclidean", max_nn: int = 10, **kwargs``` by ```Union[Callable, str] = "euclidean", max_nn: int = 10, **kwargs```    (worked on 2024-10-29)    

## 8. Deprecated np.int
Use ```int``` or ```np.int64``` instead

## 9. print(f'Full likelihood using {params} is {neg_log_likelihood(params, data, data['ColumnAmountO3'] ) }')                                                                                   ^^^^^^^^^^^^^^
SyntaxError: f-string: unmatched '['       
make ```''``` to ```""``` to match nested quotations. Use ```data["ColumnAmountO3"]```   

## 10. export OMP_NUM_THREADS=1 showed "command not found"
Even though ```export OMP_NUM_THREADS=1``` showed "command not found", the rest of your code continues to run. But to ensure proper thread control, you can add this line directly in your Python script:

Remove it and ```re-type it wihtout copy-paste.```

## vecchia poor performance
1. don't apply reordering twice! (once in .py file once in likelihood function) it will change the order again!   
2. when reordering, don't forget to add  .reset_index(drop=True)     

## full vs vecchia likelihood discrepancy
It happens when I reorder data for vecchia approximation. Obviously, the order affects the conditioning set.


## sbatch preempted: 
add this line into nano file:  
```sacct --units=G --format=MaxRSS,MaxDiskRead,MaxDiskWrite,Elapsed,NodeList -j $SLURM_JOBID```  

It will capture the peak RAM usage and some other useful info about your job  

Also you can use:   
```sstat --format=MaxRSS,MaxDiskRead,MaxDiskWrite -j 39534244```  
