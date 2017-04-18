from setuptools import setup, find_packages
      
VERSION='0.1.7'
DISTNAME='hn2016_falwa'
URL='https://github.com/csyhuang/hn2016_falwa' # how can we make download_url automatically get the right version?
DOWNLOAD_URL='https://github.com/csyhuang/hn2016_falwa/'
AUTHOR='Shao Ying (Clare) Huang'
AUTHOR_EMAIL='clare1068@gmail.com'
LICENSE='MIT'
DESCRIPTION='python package to compute finite-amplitude local wave activity (Huang and Nakamura 2016, JAS)'
install_requires=[
   'numpy',
   'scipy',
]
LONG_DESCRIPTION="""
hn2016_falwa is a package that contains modules to compute the finite-amplitude 
local wave activity (FALWA) proposed in Huang and Nakamura (2016 JAS):
http://dx.doi.org/10.1175/JAS-D-15-0194.1
The current version of the library handles calculation of FALWA in a spherical 
barotropic model and QGPV fields on isobaric surfaces.

The functions in this library can compute the tracer equivalent-latitude relationship 
proposed in Nakamura (1996) (Also, see Allen and Nakamura (2003)) and the (zonal mean)
finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

v0.1.6: Included LWA computation subroutines with *prescribed* equivalent latitude profile:
- barotropic_input_Qref_to_compute_LWA (for barotropic vorticity)
- qgpv_input_Qref_to_compute_LWA (for QGPV)
 
v0.1.7: added 'theta_lwa' 
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
                  
