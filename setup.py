"""
Build the C_net_functions Cython extension.

Runs automatically as part of `pip install -e .` because the build
requirements (Cython, numpy) are declared in pyproject.toml.

Manual build (rare — for in-place dev without an editable install):
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "net3.C_net_functions",
        ["src/net3/C_net_functions.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
