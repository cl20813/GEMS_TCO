### Update my packages
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco 

### transfer new data into amarel

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/coarse_cen_map_without_decrement_latitude24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024


### Copy run file from ```local``` to ```Amarel HPC```
# mac
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_vecc_day_v05_112025.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_veccDWlbfgs_day_v05_112425.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_veccDWadams_day_v05_112225.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


# window
scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_vecc_day_v05_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
scp "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\st_model\day\fit_vecc_day_v10_416.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

### Copy estimate file from ```Amarel HPC``` to ```local computer```

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v05_r2s10_1250.0.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/df_cv_smooth_05" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v150_r2s10_1127.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/df_cv_smooth_15" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/vecchia_v150_r2s10_4508.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/df_cv_smooth_15" 

# real data fitted, dw and vecc estimates

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/real_fit_dw_and_vecc_122325/real_dw_18126.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_dw_vecc_estimates_122425/" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/real_fit_dw_and_vecc_122325/real_vecc_h1000_mm16_18126.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_dw_vecc_estimates_122425/" 


# simulation results

# regular grid
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_re_and_irr_dw_vs_vecc_122325/perturb0.2_nheads300_mm16_122325/sim_reg_dW_v05_18126.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/sim_dw_vecc_100times_122425/" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_re_and_irr_dw_vs_vecc_122325/perturb0.2_nheads300_mm16_122325/sim_reg_vecc_v05_18126.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/sim_dw_vecc_100times_122425/" 

# regular gridded irregular

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_re_and_irr_dw_vs_vecc_122325/perturb0.2_nheads300_mm16_122325/sim_irre_dW_v05_18126.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/sim_dw_vecc_100times_122425/" 

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_re_and_irr_dw_vs_vecc_122325/perturb0.2_nheads300_mm16_122325/sim_irre_vecc_v05_18126.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/sim_dw_vecc_100times_122425/" 





