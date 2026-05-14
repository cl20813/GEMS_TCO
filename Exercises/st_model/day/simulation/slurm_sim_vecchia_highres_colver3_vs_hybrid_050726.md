# Column V3 vs Hybrid High-Res Simulation

Purpose: compare equal-sized conditioning on a complete regular-grid simulation.

- Column V3: reverse-L/downward-right template reuse, nominal m = 42
- Hybrid Lean: batched GPU Vecchia, nominal m = 41

## Files

Local source files:

```bash
/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernel_vecchia_col_batch.py
/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_highres_colver3_vs_hybrid_050726.py
/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_highres_colver3_vs_hybrid_050726.sh
```

Amarel target paths:

```bash
/home/jl2815/tco/GEMS_TCO/kernel_vecchia_col_batch.py
/home/jl2815/tco/exercise_25/st_model/sim_vecchia_highres_colver3_vs_hybrid_050726.py
/home/jl2815/tco/exercise_25/st_model/sim_vecchia_highres_colver3_vs_hybrid_050726.sh
```

## Transfer

Run these from the local Mac terminal, not inside Amarel:

```bash
scp /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernel_vecchia_col_batch.py jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

scp /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_highres_colver3_vs_hybrid_050726.py jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/

scp /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_highres_colver3_vs_hybrid_050726.sh jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/
```

## Submit

```bash
cd /home/jl2815/tco/exercise_25/st_model
nano sim_vecchia_highres_colver3_vs_hybrid_050726.sh

sbatch sim_vecchia_highres_colver3_vs_hybrid_050726.sh
```

## Output

CSV outputs are written to:

```bash
/home/jl2815/tco/exercise_output/estimates/day/
```

Main files:

```bash
sim_vecchia_highres_colver3_vs_hybrid_050726_raw.csv
sim_vecchia_highres_colver3_vs_hybrid_050726_model_summary.csv
sim_vecchia_highres_colver3_vs_hybrid_050726_param_summary.csv
```

The script rounds saved CSV values and printed summaries to 4 decimal places.
