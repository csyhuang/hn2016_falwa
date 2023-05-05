from distutils.core import setup
from Cython.Build import cythonize

setup(name='dirinv_cython',
      package_dir={'pyx_modules': ''},
      ext_modules=cythonize("pyx_modules/dirinv_cython.pyx"))
