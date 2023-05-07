"""
Run
$ python setup_cython.py build_ext --inplace
in repo root level
"""
from hn2016_falwa import cython_modules

ans = cython_modules.dirinv_cython.x_sq_minus_x(19)
print(f"ans = {ans}")
ans = cython_modules.dirinv_cython.sin_func(19)
print(f"ans = {ans}")
