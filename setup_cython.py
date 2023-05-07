from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='dirinv_cython',
      ext_modules=cythonize("hn2016_falwa/cython_modules/dirinv_cython.pyx"),
      include_dirs=[numpy.get_include()])


