from __future__ import annotations

from pathlib import Path

from setuptools import Extension, find_packages, setup

import numpy as np
from Cython.Build import cythonize


ROOT = Path(__file__).resolve().parent

extensions = [
    Extension(
        "utils._route_speedups",
        [str(ROOT / "utils" / "_route_speedups.pyx")],
        include_dirs=[np.get_include()],
    )
]


setup(
    name="thesis-route-speedups",
    version="0.0.0",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
