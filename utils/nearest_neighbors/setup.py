from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension(
       "nearest_neighbors",
       sources=["knn.pyx", "knn_.cxx"],
       include_dirs=["./", numpy.get_include()],
       language="c++",
       extra_compile_args=["/std:c++11", "/O2"],
       extra_link_args=[],
       define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
  )]

setup(
    name = "nearest_neighbors",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)