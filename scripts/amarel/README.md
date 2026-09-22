# Amarel native CUDA covariance

This directory builds and checks the optional fused CUDA covariance used by
the grouped corridor Vecchia likelihood.  The build contains native cubins for
both GPUs used by this project:

- NVIDIA A100: compute capability `sm_80`
- NVIDIA L40S: compute capability `sm_89`

The statistical code remains float64.  A100 should be the first choice for the
full likelihood because it has substantially stronger FP64 throughput than
L40S; L40S is supported and is still useful when A100 queue time dominates.

## Submit the validation job

From the repository root on Amarel:

```bash
sbatch scripts/amarel/submit_cuda_smoke.slurm
```

The batch script resolves the checkout through Slurm's submission directory.
When submitting elsewhere, set `GEMS_TCO_REPOSITORY_ROOT` to the checkout path.

The checked-in job defaults to Amarel's `gpu` partition, which is the preferred
A100 route for this float64 workload.  To target a known A100 node, add its
current hostname, for example:

```bash
sbatch --partition=gpu --nodelist=gpu015 \
    scripts/amarel/submit_cuda_smoke.slurm
```

To validate the same binary on the L40S resource, override the partition at
submission time:

```bash
sbatch --partition=gpu-redhat scripts/amarel/submit_cuda_smoke.slurm
```

Command-line Slurm options override the `#SBATCH` defaults in the script.
Confirm a specific hostname with `sinfo`/`scontrol` before adding
`--nodelist`, because assignments can change.  The job prints the actual GPU
model and compute capability and refuses a device other than `sm_80` or
`sm_89` by default.

The job assumes the existing `gems_gpu` conda environment.  Override it with:

```bash
GEMS_TCO_CONDA_ENV=my_environment \
    sbatch scripts/amarel/submit_cuda_smoke.slurm
```

The build script sets `TORCH_CUDA_ARCH_LIST="8.0;8.9"`, compiles the CPU and
CUDA native extensions in place through an editable install, runs CUDA parity
tests, and prints a representative forward/backward microbenchmark.  CUDA
11.8 or newer is needed to compile `sm_89`; the supplied job loads Amarel's
CUDA 12.1 module.  The active PyTorch installation must be CUDA-enabled and
compatible with that toolkit.

Do not copy a locally compiled macOS `.so` file to Amarel.  Compile on the
Linux cluster so the Python ABI, PyTorch ABI, CUDA runtime, and GPU
architectures all match.

## Runtime selection

No notebook-side model change is required.  With tensors on CUDA and
`covariance_backend="auto"`, the package selects
`GEMS_TCO._vecchia_covariance_cuda`.  `covariance_backend="native"` makes a
missing/incompatible CUDA build a hard error; use it for the smoke test.
`covariance_backend="torch"` remains the reference implementation used for
parity checks.

The native backward provides first-order gradients for the seven covariance
parameters.  Coordinate derivatives and higher-order derivatives intentionally
remain on the Torch path.
