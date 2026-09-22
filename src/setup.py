from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "GEMS_TCO.maxmin_cpp",
        ["GEMS_TCO/cpp_src/maxmin.cpp"],
        cxx_std=11,
    ),
    Pybind11Extension(
        "GEMS_TCO.maxmin_ancestor_cpp",
        ["GEMS_TCO/cpp_src/maxmin_ancestor.cpp"],
        cxx_std=11,
    ),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
