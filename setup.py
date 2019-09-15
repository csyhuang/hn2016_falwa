from setuptools import find_packages
from numpy.distutils.core import setup, Extension

install_requires=['numpy', 'scipy']

LONG_DESCRIPTION="""
hn2016_falwa is a package that contains modules to compute the finite-amplitude
local wave activity (FALWA) and reference state (U_ref) in the following papers:
Huang and Nakamura (2016, JAS): http://dx.doi.org/10.1175/JAS-D-15-0194.1
Huang and Nakamura (2017, GRL): http://onlinelibrary.wiley.com/doi/10.1002/2017GL073760/full
Nakamura and Huang (2018, Science): https://doi.org/10.1126/science.aat0721
The current version of the library handles calculation of FALWA in a spherical barotropic model and QGPV fields on isobaric surfaces.

The functions in this library can compute the tracer equivalent-latitude relationship
proposed in Nakamura (1996) (Also, see Allen and Nakamura (2003)) and the (zonal mean)
finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

Minor update in v0.3.6:
- Fixed a bug in oopinterface.QGField that interpolation is done on a grid larger than the Northern hemispheric grid size when northern_hemisphere_results_only=True.

Minor update in v0.3.5:
- Added new functionality in the class 'QGField' to accept latitude grids with even number of grid points.

Minor update in v0.3.4:
- Fixed an issue in compute_reference_states.f90 that can potentially lead to segmentation fault error.

Minor update in v0.3.3:
- Added functionality to compute convergence of zonal advective flux.

Minor update in v0.3.2:
- Modified a procedure in eqvlat in basis.py such that division by a zero differential area is avoided.

Minor update in v0.3.1:
- The method compute_local_fluxes in the object QGField is renamed compute_lwa_and_barotropic_fluxes.
- Fixed some typos in the documentation.

Major update in v0.3.0:
- The interface for computing local wave activity in QG framework has been released.
  See QGField in oopinterface.py.


Links:
-----
- Source code: http://github.com/csyhuang/hn2016_falwa/
"""

ext1 = Extension(name='interpolate_fields',
                 sources=['hn2016_falwa/interpolate_fields.f90'],
                 f2py_options=['--quiet'])

ext2 = Extension(name='compute_reference_states',
                 sources=['hn2016_falwa/compute_reference_states.f90'],
                 f2py_options=['--quiet'])

ext3 = Extension(name='compute_lwa_and_barotropic_fluxes',
                 sources=['hn2016_falwa/compute_lwa_and_barotropic_fluxes.f90'],
                 f2py_options=['--quiet'])


setup(
    name='hn2016_falwa',
    version='0.3.6',
    description='python package to compute finite-amplitude local wave activity (Huang and Nakamura 2016, JAS)',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/csyhuang/hn2016_falwa',
    author='Clare S. Y. Huang',
    author_email='csyhuang@uchicago.edu',
    license='MIT',
    packages=find_packages(),
    ext_modules=[ext1, ext2, ext3],
    zip_safe=False
)
