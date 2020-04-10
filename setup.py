from setuptools import find_packages
from numpy.distutils.core import setup, Extension

LONG_DESCRIPTION =\
    """
    hn2016_falwa is a package that contains modules to compute the finite-amplitude
    local wave activity (FALWA) and reference state (U_ref) in the following papers:
    Huang and Nakamura (2016, JAS): http://dx.doi.org/10.1175/JAS-D-15-0194.1
    Huang and Nakamura (2017, GRL): http://onlinelibrary.wiley.com/doi/10.1002/2017GL073760/full
    Nakamura and Huang (2018, Science): https://doi.org/10.1126/science.aat0721
    The current version of the library handles calculation of FALWA in a spherical barotropic model and QGPV fields on isobaric surfaces.
    
    The functions in this library can compute the tracer equivalent-latitude relationship
    proposed in Nakamura (1996) (Also, see Allen and Nakamura (2003)) and the (zonal mean)
    finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).
    
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
    version='0.4.0',
    description='python package to compute finite-amplitude local wave activity (Huang and Nakamura 2016, JAS)',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/csyhuang/hn2016_falwa',
    author='Clare S. Y. Huang',
    author_email='csyhuang@uchicago.edu',
    license='MIT',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    ext_modules=[ext1, ext2, ext3],
    zip_safe=False
)
