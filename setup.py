from numpy.distutils.core import setup

LONG_DESCRIPTION = \
    """
    Important: this python package has been renamed from `hn2016_falwa` to `falwa` since version v1.0.0.
    
    This python package contains modules to compute the finite-amplitude
    local wave activity (FALWA) and reference state (U_ref) in the following papers:
    Huang and Nakamura (2016, JAS): http://dx.doi.org/10.1175/JAS-D-15-0194.1
    Huang and Nakamura (2017, GRL): http://onlinelibrary.wiley.com/doi/10.1002/2017GL073760/full
    Nakamura and Huang (2018, Science): https://doi.org/10.1126/science.aat0721
    Neal et al (2022, GRL): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL097699

    The current version of the library handles calculation of FALWA in a spherical barotropic model and QGPV fields on 
    isobaric surfaces.

    The functions in this library can compute the tracer equivalent-latitude relationship
    proposed in Nakamura (1996) (Also, see Allen and Nakamura (2003)) and the (zonal mean)
    finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

    Links:    
    - Source code: http://github.com/csyhuang/hn2016_falwa/
    """

setup(
    name='hn2016_falwa',
    version='0.7.3',
    description='This python package has been renamed from `hn2016_falwa` to `falwa` since version v1.0.0.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/csyhuang/hn2016_falwa',
    author='Clare S. Y. Huang',
    author_email='csyhuang@uchicago.edu',
    license='MIT',
    python_requires='>=3',
    install_requires=['falwa'],
    classifiers=["Development Status :: 7 - Inactive"])

