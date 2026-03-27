"""
step5_transfer_to_amarel_032626.py
===================================
Transfer preprocessed GEMS_TCO data from mac to Amarel.

Transfers TWO sets of files by default:

  [A] Merged Apr-Sep pkl  (step4 outputs)
      GEMS_DATA/Apr_to_Sep/tco_grid_apr_sep_{year}.pkl   (~1.4 GB each)
      GEMS_DATA/Apr_to_Sep/day_index_apr_sep_2022_2025.csv
      GEMS_DATA/Apr_to_Sep/monthly_means_apr_sep_2022_2025.csv
      → Amarel: /home/jl2815/tco/data/Apr_to_Sep/

  [B] Individual monthly pkl  (step3 outputs)
      GEMS_DATA/pickle_{year}/tco_grid_{yy}_{mm}.pkl     (~200 MB each)
      → Amarel: /home/jl2815/tco/data/pickle_{year}/

Usage
-----
# Dry-run (print commands only):
python step5_transfer_to_amarel_032626.py --dry-run

# Transfer everything (both A and B), all years, Apr-Sep:
python step5_transfer_to_amarel_032626.py

# Specific years only:
python step5_transfer_to_amarel_032626.py --years 2023

# Merged pkl only (skip monthly):
python step5_transfer_to_amarel_032626.py --skip-monthly

# Monthly pkl only (skip merged):
python step5_transfer_to_amarel_032626.py --skip-merged

Prerequisites
-------------
  - SSH key auth to Amarel: ssh-copy-id jl2815@amarel.rutgers.edu
  - rsync installed locally (macOS: pre-installed)
  - step3/step4 must have run so pkl files exist locally
"""

import sys
import subprocess
import argparse
from pathlib import Path

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
from GEMS_TCO import configuration as config

# ── Paths ──────────────────────────────────────────────────────────────────────
MAC_DATA    = Path(config.mac_data_load_path)           # /Users/.../GEMS_DATA/
AMAREL_HOST = "jl2815@amarel.rutgers.edu"
AMAREL_DATA = config.amarel_data_load_path.rstrip("/")  # /home/jl2815/tco/data

DEFAULT_YEARS  = [2022, 2023, 2024, 2025]
DEFAULT_MONTHS = [4, 5, 6, 7, 8, 9]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _size_str(path: Path) -> str:
    if not path.exists():
        return "NOT FOUND"
    b = path.stat().st_size
    if b >= 1e9:  return f"{b/1e9:.2f} GB"
    if b >= 1e6:  return f"{b/1e6:.1f} MB"
    return f"{b/1e3:.1f} KB"


def _run(cmd: list, dry_run: bool) -> bool:
    if dry_run:
        print(f"    [DRY] {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd)
    return result.returncode == 0


def _ensure_remote_dir(remote_dir: str, dry_run: bool):
    _run(["ssh", AMAREL_HOST, f"mkdir -p {remote_dir}"], dry_run)


def _rsync(src: Path, remote_dir: str, dry_run: bool) -> bool:
    cmd = ["rsync", "-avh", "--progress", str(src), f"{AMAREL_HOST}:{remote_dir}/"]
    return _run(cmd, dry_run)


def _transfer_list(files: list, remote_dir: str, dry_run: bool) -> tuple:
    """Transfer a list of (Path, label) tuples to remote_dir. Returns (ok, fail)."""
    _ensure_remote_dir(remote_dir, dry_run)
    ok = fail = 0
    for path, label in files:
        if not path.exists():
            print(f"    [SKIP]  {label}  — not found locally")
            fail += 1
            continue
        print(f"    → {label}  ({_size_str(path)})")
        if _rsync(path, remote_dir, dry_run):
            ok += 1
        else:
            print(f"    [ERROR] rsync failed for {label}")
            fail += 1
    return ok, fail


# ── Section A: merged Apr-Sep pkl ─────────────────────────────────────────────

def transfer_merged(years: list, dry_run: bool) -> tuple:
    remote_dir = f"{AMAREL_DATA}/Apr_to_Sep"
    local_dir  = MAC_DATA / "Apr_to_Sep"

    files = [(local_dir / f"tco_grid_apr_sep_{y}.pkl", f"tco_grid_apr_sep_{y}.pkl")
             for y in years]
    files += [
        (local_dir / "day_index_apr_sep_2022_2025.csv",    "day_index_apr_sep_2022_2025.csv"),
        (local_dir / "monthly_means_apr_sep_2022_2025.csv","monthly_means_apr_sep_2022_2025.csv"),
    ]

    print(f"\n  [A] Merged Apr-Sep pkl  →  {AMAREL_HOST}:{remote_dir}/")
    print(f"      {len(files)} files  (missing locally will be skipped)")
    return _transfer_list(files, remote_dir, dry_run)


# ── Section B: individual monthly pkl ─────────────────────────────────────────

def transfer_monthly(years: list, months: list, dry_run: bool) -> tuple:
    ok = fail = 0
    for year in years:
        remote_dir = f"{AMAREL_DATA}/pickle_{year}"
        local_dir  = MAC_DATA / f"pickle_{year}"
        yy = str(year)[2:]

        files = [
            (local_dir / f"tco_grid_{yy}_{m:02d}.pkl",
             f"pickle_{year}/tco_grid_{yy}_{m:02d}.pkl")
            for m in months
        ]

        print(f"\n  [B] {year} monthly pkl  →  {AMAREL_HOST}:{remote_dir}/")
        a, b = _transfer_list(files, remote_dir, dry_run)
        ok += a; fail += b

    return ok, fail


# ── Main ──────────────────────────────────────────────────────────────────────

def main(years, months, dry_run, skip_merged, skip_monthly):
    print(f"\n{'='*62}")
    print(f"  step5: Transfer TCO data → Amarel")
    print(f"  years        : {years}")
    print(f"  months       : {months}  (Apr–Sep)")
    print(f"  transfer [A] merged pkl : {not skip_merged}")
    print(f"  transfer [B] monthly pkl: {not skip_monthly}")
    print(f"  dry-run      : {dry_run}")
    print(f"{'='*62}")

    total_ok = total_fail = 0

    if not skip_merged:
        ok, fail = transfer_merged(years, dry_run)
        total_ok += ok; total_fail += fail

    if not skip_monthly:
        ok, fail = transfer_monthly(years, months, dry_run)
        total_ok += ok; total_fail += fail

    print(f"\n{'='*62}")
    print(f"  Done.  transferred={total_ok}  skipped/failed={total_fail}")
    print(f"{'='*62}\n")

    if not dry_run and total_ok > 0:
        print("  Verify on Amarel:")
        print(f"    ssh {AMAREL_HOST}")
        print(f"    ls -lh {AMAREL_DATA}/Apr_to_Sep/")
        for y in years:
            print(f"    ls -lh {AMAREL_DATA}/pickle_{y}/")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer GEMS_TCO preprocessed files from mac to Amarel"
    )
    parser.add_argument("--years",  type=int, nargs="+", default=DEFAULT_YEARS,
                        help="Years (default: 2022 2023 2024 2025)")
    parser.add_argument("--months", type=int, nargs="+", default=DEFAULT_MONTHS,
                        help="Months (default: 4 5 6 7 8 9)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print rsync commands without executing")
    parser.add_argument("--skip-merged",  action="store_true",
                        help="Skip section A (merged Apr-Sep pkl)")
    parser.add_argument("--skip-monthly", action="store_true",
                        help="Skip section B (individual monthly pkl)")
    args = parser.parse_args()
    main(args.years, args.months, args.dry_run, args.skip_merged, args.skip_monthly)
