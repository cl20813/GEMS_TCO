Faiss requires python 3.12 so we need a new environment.

### Create enviornment
Go to terminal in VSCode.        
```conda create -n faiss_env python=3.12```       
```conda activate faiss_env```       
```conda install -c pytorch faiss-cpu```      
```conda install pybind11```      # comfile c++ file       
```conda install numpy pandas matplotlib seaborn scikit-learn```         
```conda install pytorch::pytorch torchvision torchaudio -c pytorch```  

### Install gems_tco packge on my macbook. 
```cd /Users/joonwonlee/Documents/GEMS_TCO-1/src```    
```/opt/anaconda3/envs/gems_tco/bin/python -m pip install -e . --use-pep517```   
When activate the new environment for the first time, system will ask to install ```ipkernel```.

### Install gems_tco packge on my window laptop. 
```cd "C:\Users\joonw\tco\GEMS_TCO-2\src"```
``` C:\Users\joonw\anaconda3\envs\faiss_env\python.exe -m pip install -e . --use-pep517```

### Check python interpreter because the environment in VS code may not linked to the python.
For example, python in window computer is located at ```C:\Users\joonw\AppData\Local\Programs\Python\Python312\python.exe```.

Press Ctrl + Shift + P (or Cmd + Shift + P on macOS) to open the Command Palette.        
Select Interpreter: Type Python: Select Interpreter and select it from the dropdown list.        
You will see a list of available Python interpreters. Choose the one associated with your faiss_env environment.        


### Comfile c++ file
mac: ``` python3 -c "import platform; print(platform.machine())" ```   Verify yourself if it says ```arm64``` or others. My mac computer says arm64.
window: ``` python -c "import platform; print(platform.machine())" ```   Verify yourself if it says ```arm64``` or others. My mac computer says arm64.

mac:   
```cd /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/cpp_src/```  This is the location for .cpp files. Note that compiled c++ .so files are recommended to located at the package folder with other modules.

window:
``` cd "C:\Users\joonw\tco\GEMS_TCO-2\src/GEMS_TCO/cpp_src/" ```
  
mac:  I am using pybind11 to compile c++ for mac.
```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin_ancestor.cpp -o ../maxmin_ancestor_cpp$(python3-config --extension-suffix) -undefined dynamic_lookup```
```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin.cpp -o ../maxmin_cpp$(python3-config --extension-suffix) -undefined dynamic_lookup```  

window:   
```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) maxmin_ancestor.cpp -o ../maxmin_ancestor_cpp$(python-config --extension-suffix) -undefined dynamic_lookup```
```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin.cpp -o ../maxmin_cpp$(python3-config --extension-suffix) -undefined dynamic_lookup```  

 



# Amarel
Make the same environment as above.

cd /home/jl2815/tco/GEMS_TCO/cpp_src

NOTE: It should be ```maxmin_ancestor_cpp.so``` instead of ```maxmin_ancestor.so.``` The file should be compiled within the Linux system rather than compiling it on a local computer and transferring the file to Amarel, which is Linux-based.
      
``` c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin_ancestor.cpp -o ../maxmin_ancestor_cpp.so ```      
``` c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin.cpp -o ../maxmin_cpp.so  ```         


Now I need to add path so that python can find my libraries.
   
nano ~/.bashrc   
export PATH="/home/jl2815/.conda/envs/faiss_env/bin:$PATH"   
source ~/.bashrc   

conda activate faiss_env   

