from setuptools import setup, find_packages
      
VERSION='0.2.1'
DISTNAME='hn2016_falwa'
URL='https://github.com/csyhuang/hn2016_falwa' # how can we make download_url automatically get the right version?
DOWNLOAD_URL='https://github.com/csyhuang/hn2016_falwa/'
AUTHOR='Shao Ying (Clare) Huang'
AUTHOR_EMAIL='csyhuang@uchicago.edu'
LICENSE='MIT'
DESCRIPTION='python package to compute finite-amplitude local wave activity (Huang and Nakamura 2016, JAS)'
install_requires=[
   'numpy',
   'scipy',
]
LONG_DESCRIPTION="""
hn2016_falwa is a package that contains modules to compute the finite-amplitude 
local wave activity (FALWA) and reference state (U_ref) in the following papers:
Huang and Nakamura (2016, JAS): http://dx.doi.org/10.1175/JAS-D-15-0194.1
Huang and Nakamura (2017, GRL): http://onlinelibrary.wiley.com/doi/10.1002/2017GL073760/full
The current version of the library handles calculation of FALWA in a spherical 
barotropic model and QGPV fields on isobaric surfaces.

The functions in this library can compute the tracer equivalent-latitude relationship 
proposed in Nakamura (1996) (Also, see Allen and Nakamura (2003)) and the (zonal mean)
finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

v0.2.0:
- Functions are structred in 4 different modules: basis, wrapper, utilities and beta_version. See documentation on Github for details.
v0.2.1 (bug fixing):
- hn2016_falwa/beta_version.py: In *solve_uref_both_bc*, a plotting option is added.
- hn2016_falwa/wrapper.py: In all functions, when n_points are not specified, it is taken to be nlat_s (input). Also fixed a bug of missing argument n_points in *theta_lwa*.
- hn2016_falwa/utilities.py: In *static_stability*, make s_et and n_et an integer if they are not input. In *compute_qgpv_givenvort*, remove the bug that nlat_s being hard-coded by mistake.


Links:
-----
- Source code: http://github.com/csyhuang/hn2016_falwa/
"""

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      packages=['hn2016_falwa'],
      test_suite = 'tests.my_module_suite',
	  zip_safe=False
	  )
                  
