#!/usr/bin/env python3
"""Pure-space nugget fit entrypoint for generated simulation assets.

The simulation data are generated separately under:

    Exercises/st_model/simulate_data

This file is only a thin simulation-specific entrypoint.  It delegates to the
real-data pure-space fitter so simulated and real-data experiments use the same
likelihood/model implementation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_real_data_eda_to_path() -> None:
    env_path = os.environ.get("REAL_DATA_EDA_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve()
    candidates.extend([
        here.parents[2] / "real_data" / "eda",  # .../day/amarel_simulation/pure_space -> .../day/real_data/eda
        here.parents[3] / "real_data" / "eda",  # .../st_model/real_data/eda, old Amarel layout
        Path("/home/jl2815/tco/exercise_25/st_model/day/real_data/eda"),
        Path("/home/jl2815/tco/exercise_25/st_model/real_data/eda"),
        Path("/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/eda"),
    ])

    for path in candidates:
        if (path / "fit_july_spatial_nugget_tiles.py").exists():
            sys.path.insert(0, str(path))
            return
    checked = "\n".join(str(p) for p in candidates)
    raise SystemExit(f"Could not find fit_july_spatial_nugget_tiles.py. Checked:\n{checked}")


def main() -> None:
    add_real_data_eda_to_path()
    from fit_july_spatial_nugget_tiles import main as real_fit_main

    real_fit_main()


if __name__ == "__main__":
    main()
