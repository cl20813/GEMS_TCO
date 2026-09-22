import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "GEMS_TCO._maxmin",
        ["cpp/maxmin_order.cpp"],
        cxx_std=11,
    ),
]
cmdclass = {"build_ext": build_ext}

# The Torch extensions are optional accelerators.  Keeping them behind explicit
# build flags avoids making Torch/CUDA heavyweight PEP 517 build-time
# dependencies for users who only need the portable reference implementation.
build_cpu_covariance = os.environ.get("GEMS_TCO_BUILD_TORCH_EXT", "0") == "1"
build_cuda_covariance = os.environ.get("GEMS_TCO_BUILD_CUDA_EXT", "0") == "1"
if build_cpu_covariance or build_cuda_covariance:
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    except ImportError as error:  # pragma: no cover - exercised by installation environments.
        raise RuntimeError(
            "building a native covariance extension requires PyTorch in the "
            "active build environment"
        ) from error

    host_compile_args = ["-O3"]
    if sys.platform == "darwin":
        # Apple Clang 21 diagnoses a specialization used by Torch 2.5 headers
        # as an error; Torch itself supports this specialization.
        host_compile_args.append("-Wno-invalid-specialization")

    if build_cpu_covariance:
        ext_modules.append(
            CppExtension(
                "GEMS_TCO._vecchia_covariance_cpu",
                ["cpp/vecchia_covariance_cpu.cpp"],
                extra_compile_args=host_compile_args,
            )
        )

    if build_cuda_covariance:
        if sys.platform == "darwin":
            raise RuntimeError("the CUDA covariance extension requires Linux with CUDA")
        # This project's Amarel production targets are A100 (sm_80) and L40S
        # (sm_89).  Callers can still override the list for another cluster.
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9")
        ext_modules.append(
            CUDAExtension(
                "GEMS_TCO._vecchia_covariance_cuda",
                [
                    "cpp/vecchia_covariance_cuda.cpp",
                    "cpp/vecchia_covariance_cuda_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    # Deliberately omit --use_fast_math: the likelihood and
                    # analytic gradient are evaluated in IEEE float64.
                    "nvcc": ["-O3"],
                },
            )
        )

    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=False)}


setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
