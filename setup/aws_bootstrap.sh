#!/bin/bash
# aws_bootstrap.sh
# Run this ONCE on a fresh EC2 instance to set up the full environment.
# Usage: bash aws_bootstrap.sh [S3_DATA_BUCKET]
# Example: bash aws_bootstrap.sh s3://my-bucket/gems_data

set -e
S3_DATA_BUCKET=${1:-""}
AWS_HOME="/home/ec2-user/gems_tco"
CONDA_ENV="faiss_env"

echo "================================================================"
echo "GEMS TCO — AWS Bootstrap"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "================================================================"

# ── 1. Directory structure ────────────────────────────────────────────
mkdir -p ${AWS_HOME}/{src,data,exercise_25/st_model,exercise_output/estimates/day,jobscript}

# ── 2. Conda ──────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[1/6] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /home/ec2-user/miniconda3
    eval "$(/home/ec2-user/miniconda3/bin/conda shell.bash hook)"
    conda init bash
else
    echo "[1/6] Conda already present."
    eval "$(conda shell.bash hook)"
fi

# ── 3. Create faiss_env ───────────────────────────────────────────────
if conda env list | grep -q "^${CONDA_ENV}"; then
    echo "[2/6] ${CONDA_ENV} already exists."
else
    echo "[2/6] Creating conda env: ${CONDA_ENV}..."
    conda create -y -n ${CONDA_ENV} python=3.12
fi
conda activate ${CONDA_ENV}

# ── 4. Install Python packages ────────────────────────────────────────
echo "[3/6] Installing Python packages..."
conda install -y -c pytorch -c nvidia faiss-gpu pytorch torchvision torchaudio pytorch-cuda=12.1 2>/dev/null || \
    pip install faiss-gpu-cu12

pip install -q pybind11 numpy scipy scikit-learn pandas typer

# ── 5. Compile maxmin C++ extensions ─────────────────────────────────
echo "[4/6] Compiling maxmin C++ extensions..."
CPP_DIR="${AWS_HOME}/src/GEMS_TCO/cpp_src"
OUT_DIR="${AWS_HOME}/src/GEMS_TCO"

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND11_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "   Python include: ${PYTHON_INCLUDE}"
echo "   pybind11 include: ${PYBIND11_INCLUDE}"
echo "   Extension suffix: ${EXT_SUFFIX}"

g++ -O3 -Wall -shared -std=c++14 -fPIC \
    -I${PYTHON_INCLUDE} -I${PYBIND11_INCLUDE} \
    ${CPP_DIR}/maxmin.cpp \
    -o ${OUT_DIR}/maxmin_cpp${EXT_SUFFIX}

g++ -O3 -Wall -shared -std=c++14 -fPIC \
    -I${PYTHON_INCLUDE} -I${PYBIND11_INCLUDE} \
    ${CPP_DIR}/maxmin_ancestor.cpp \
    -o ${OUT_DIR}/maxmin_ancestor_cpp${EXT_SUFFIX}

echo "   maxmin_cpp${EXT_SUFFIX} compiled OK"
echo "   maxmin_ancestor_cpp${EXT_SUFFIX} compiled OK"

# ── 6. Data sync from S3 ──────────────────────────────────────────────
if [ -n "${S3_DATA_BUCKET}" ]; then
    echo "[5/6] Syncing GEMS data from ${S3_DATA_BUCKET}..."
    aws s3 sync ${S3_DATA_BUCKET} ${AWS_HOME}/data/ --no-progress
else
    echo "[5/6] No S3 bucket provided — skipping data sync."
    echo "      Run manually: aws s3 sync s3://YOUR_BUCKET/gems_data ${AWS_HOME}/data/"
fi

# ── 7. Verify imports ─────────────────────────────────────────────────
echo "[6/6] Verifying imports..."
python -c "
import sys
sys.path.insert(0, '${AWS_HOME}/src')
from GEMS_TCO.orderings import maxmin_cpp, find_nns_l2
import faiss, torch
print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'  faiss GPU available: {faiss.get_num_gpus()} GPU(s)')
print('  maxmin_cpp: OK')
print('  find_nns_l2 (faiss): OK')
"

echo ""
echo "================================================================"
echo "Bootstrap complete. To run simulation:"
echo "  conda activate ${CONDA_ENV}"
echo "  bash ${AWS_HOME}/jobscript/run_hybrid_compare.sh"
echo "================================================================"
