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

To validate the same binary on a currently available L40S node, stay on the
`gpu` partition and select a verified `gpu029-048` candidate from the current
`sinfo` output, for example:

```bash
sbatch --partition=gpu --nodelist=gpu029 scripts/amarel/submit_cuda_smoke.slurm
```

Command-line Slurm options override the `#SBATCH` defaults in the script.
There is no `gpu-redhat` partition in the current Amarel configuration.
Confirm a specific hostname and GPU model with `sinfo`/`scontrol` before adding
`--nodelist`, because assignments can change.  The job prints the actual GPU
model and compute capability and refuses a device other than `sm_80` or
`sm_89` by default.

An idle `gpuk` or `volta` node is not an L40S substitute.  `volta` is a V100
family name, and the current production extension intentionally targets only
A100 (`sm_80`) and L40S (`sm_89`).  A mixed-state `gpu029-048` node may still
have a free GPU and is the correct pool for the L40S compatibility test.

The job assumes the existing `gems_gpu` conda environment.  Override it with:

```bash
GEMS_TCO_CONDA_ENV=my_environment \
    sbatch scripts/amarel/submit_cuda_smoke.slurm
```

The build script sets `TORCH_CUDA_ARCH_LIST="8.0;8.9"`, compiles the CUDA
native extension in place through an editable install, runs Matérn and
generalized-Cauchy CUDA parity tests, and prints representative
forward/backward microbenchmarks. The local macOS workflow builds its CPU
extension separately; platform-specific binaries are never copied between
machines. CUDA
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

The native backward provides first-order gradients for the seven-parameter
smoothness-0.5 Matérn model and for six/seven-parameter fixed-shape
generalized-Cauchy models. Coordinate derivatives and higher-order derivatives
intentionally remain on the Torch path.
