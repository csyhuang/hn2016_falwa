from setuptools import setup
      
VERSION='0.1.5c'
DISTNAME='hn2016_falwa'
URL='https://github.com/csyhuang/hn2016_falwa' # how can we make download_url automatically get the right version?
DOWNLOAD_URL='https://github.com/csyhuang/hn2016_falwa/'
AUTHOR='Clare S. Y. Huang'
AUTHOR_EMAIL='clare1068@gmail.com'
LICENSE='MIT'

DESCRIPTION='python package to compute finite-amplitude local wave activity (Huang and Nakamura 2016, JAS)'

LONG_DESCRIPTION="""
hn2016_falwa is a package that contains modules to compute the finite-amplitude 
local wave activity (FALWA) proposed in Huang and Nakamura (2016 JAS):
http://dx.doi.org/10.1175/JAS-D-15-0194.1
The current version of the library handles calculation of FALWA in a spherical 
barotropic model and QGPV fields on isobaric surfaces.

The functions in this library can compute the tracer equivalent-latitude relationship 
proposed in Nakamura (1996) (Also, see Allen and Nakamura (2003)) and the (zonal mean)
finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

v0.1.5: A function 'Solve_URef_noslip_hemisphere' has been added. This calculates the reference state introduced in Nakamura and Solomon (2010, JAS) but in a hemispheric domain. Documentations and examples to be uploaded soon.
v0.1.5b: Include a function 'static_stability' to compute static stability from potential temperature field.
v0.1.5c: (1) Fixed a bug in outputing equivalent latitude in the function 'qgpv_Eqlat_LWA'. The sign was flipped.
(2) A beta version of function 'Solve_URef_noslip_hemisphere' is updated. It can only be used to compute eddy-free reference state in no-slip boundary conditions.
 

Links
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
      zip_safe=False)
                  
