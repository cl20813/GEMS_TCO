Faiss requires python 3.12 so we need a new environment.

### Create enviornment
Go to terminal in VSCode.        
```conda create -n faiss_env python=3.12```       
```conda activate faiss_env```       
```conda install -c pytorch faiss-cpu```      
```conda install pybind11```      # comfile c++ file       
```conda install numpy pandas matplotlib seaborn scikit-learn```         
```conda install pytorch::pytorch torchvision torchaudio -c pytorch```  

### Install gems_tco packge on local computer. 
```cd /Users/joonwonlee/Documents/GEMS_TCO-1/src```    
```/opt/anaconda3/envs/gems_tco/bin/python -m pip install -e . --use-pep517```   

When activate the new environment for the first time, system will ask to install ```ipkernel```.

### Comfile c++ file
```python3 -c "import platform; print(platform.machine())"```   Verify yourself if it says ```arm64``` or others. My mac computer says arm64.
```cd /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/cpp_src/```  This is the location for .cpp files. Note that compiled c++ .so files are recommended to located at the package folder with other modules.

```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin_ancestor.cpp -o ../maxmin_ancestor_cpp$(python3-config --extension-suffix) -undefined dynamic_lookup```  

```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin.cpp -o ../maxmin_cpp$(python3-config --extension-suffix) -undefined dynamic_lookup```   


# Amarel
Make the same environment as above.

cd /home/jl2815/tco/GEMS_TCO/src_cpp

``` c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin_ancestor.cpp -o ../maxmin_ancestor.so ```
``` c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin.cpp -o ../maxmin.so  ```

