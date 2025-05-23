Faiss requires python 3.12 so we need a new environment.

### Create enviornment
Go to terminal in VSCode.        
```conda create -n faiss_env python=3.12```       
```conda activate faiss_env```       
```conda install -c pytorch faiss-cpu```       # order matters, install this first because it's a core dependency.    
```conda install pybind11```      # comfile c++ file       
```conda install numpy pandas matplotlib seaborn scikit-learn xarray netCDF4 ```            
```conda install pytorch::pytorch torchvision torchaudio -c pytorch```     
``` pip install typer```   conda manages dependencies for a wide range of software, not just Python packages. This can sometimes lead to conflicts or delays in updating packages. pip   
Also I need to install torch spline library.   
``` pip install git+https://github.com/patrick-kidger/torchcubicspline.git ```   

### Install gems_tco packge on my macbook. 
```cd /Users/joonwonlee/Documents/GEMS_TCO-1/src```    
```/opt/anaconda3/envs/gems_tco/bin/python -m pip install -e . --use-pep517```   
When activate the new environment for the first time, system will ask to install ```ipkernel```.

### Install gems_tco packge on my window laptop. 
``` cd "C:\Users\joonw\tco\GEMS_TCO-2\src" ```
``` C:\Users\joonw\anaconda3\envs\faiss_env\python.exe -m pip install -e . --use-pep517  ```
``` C:\Users\joonw\AppData\Local\Programs\Python\Python312\python.exe -m pip install -e . --use-pep517 ```   I have to choose base python interpreter so that change is pulled.
### Check python interpreter because there might be a mismatch between the Python interpreter set in VS Code and the one in your conda environment.   
WINDOW:

For example, my ```python interpreter linked to VS code``` is located at ```C:\Users\joonw\AppData\Local\Programs\Python\Python312\python.exe```.
- You can check this by typing ```where python``` in ```DEVELOPER COMMAND PROMPT FOR VS CODE```.
- VS code terminal: Type ```conda activate environment``` first and then type ```python --v``` this will tell your ```python interpreter linked to the ENVIRONMENT``` in window computer. It should match with the environment.

MAC:
Just type ```where python``` or ```which python``` in terminal to check python linked to VS code. Then activate conda environment and do ```which python``` again to see python linked to your environment.   

Press Ctrl + Shift + P (or Cmd + Shift + P on macOS) to open the Command Palette.        
Select Interpreter: Type Python: Select Interpreter and select it from the dropdown list.        
You will see a list of available Python interpreters. Choose the one associated with your faiss_env environment.        

Sometimes, it takes a time for adjustment. If above doesn't work, then consider adding following path to my environment if my environment cannot read conda:
```C:\Users\joonw\Anaconda3\Scripts```
```C:\Users\joonw\Anaconda3\condabin```

### Comfile c++ file (after finishing above steps, you can do this)
mac: ``` python3 -c "import platform; print(platform.machine())" ```   Verify yourself if it says ```arm64``` or others. My mac computer says arm64.
mac:   
```cd /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/cpp_src/```  This is the location for .cpp files. Note that compiled c++ .so files are recommended to located at the package folder with other modules.

mac:  I am using pybind11 to compile c++ for mac.
```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin_ancestor.cpp -o ../maxmin_ancestor_cpp$(python3-config --extension-suffix) -undefined dynamic_lookup```
```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maxmin.cpp -o ../maxmin_cpp$(python3-config --extension-suffix) -undefined dynamic_lookup```  

window: ``` python -c "import platform; print(platform.machine())" ```   Verify yourself if it says ```arm64``` or others. My mac computer says arm64.
For window, I should use MinGW. Download MinGW installing manager and then check below then apply changes.

Essential Packages for C++ Development. Also consider installing gdb 
mingw32-base-bin: This package includes the basic tools required for MinGW.
mingw32-gcc-g++-bin: This package includes the G++ compiler for C++.
mingw32-gcc-bin: This package includes the GCC compiler for C.

#### Verify 
``` "C:\MinGW\bin\gcc.exe" --version ```    this should work
``` gcc --version ```  this should also work if the PATH  "C:\MinGW\bin" is added under PATH system variable. Making other variable won't work!!


window:
``` cd "C:\Users\joonw\tco\GEMS_TCO-2\src/GEMS_TCO/cpp_src/" ```

window:   You need to install MingW for c++ compiler. Install this and add path ```C:\MinGW\bin```.

#### Now check the location of -I<path_to_python_include> -I<path_to_pybind11_include> manually      
Type below in VS code terminal.
```conda activate faiss_env```    
```python -m pybind11 --includes```

Go to powershell and check
```g++ --version```  then proceed with below
```g++ -O3 -Wall -shared -std=c++11 -fPIC -IC:\Users\joonw\anaconda3\envs\faiss_env\Include -IC:\Users\joonw\anaconda3\envs\faiss_env\Lib\site-packages\pybind11\include maxmin_ancestor.cpp -o ../maxmin_ancestor_cpp```

```g++ -O3 -Wall -shared -std=c++11 -fPIC -IC:\Users\joonw\anaconda3\envs\faiss_env\Include -IC:\Users\joonw\anaconda3\envs\faiss_env\Lib\site-packages\pybind11\include maxmin.cpp -o ../maxmin_cpp```

#### It turns out that MinGW is not good for compiling my c++ file. 
- Open Developer Command Prompt for VS code (cmd and then click + to find this)
- locate the working directory ``` cd "C:\Users\joonw\tco\GEMS_TCO-2\src/GEMS_TCO/cpp_src/" ```
-  ``` cl /O2 /LD /I"C:\Users\joonw\anaconda3\envs\faiss_env\Include" ^
   /I"C:\Users\joonw\anaconda3\envs\faiss_env\Lib\site-packages\pybind11\include" ^
   maxmin_ancestor.cpp /link ^
   /OUT:..\maxmin_ancestor_cpp.pyd ^
   /LIBPATH:"C:\Users\joonw\anaconda3\envs\faiss_env\libs" python312.lib
 ```

```cl /O2 /LD /I"C:\Users\joonw\anaconda3\envs\faiss_env\Include" ^
   /I"C:\Users\joonw\anaconda3\envs\faiss_env\Lib\site-packages\pybind11\include" ^
   maxmin.cpp /link ^
   /OUT:..\maxmin_cpp.pyd ^
   /LIBPATH:"C:\Users\joonw\anaconda3\envs\faiss_env\libs" python312.lib
```  

This work but note that file name is .pyd for window and .so for linux and mac os. So we have to change this carefully.


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

