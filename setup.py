from setuptools import setup
      
VERSION='0.1.0'
DISTNAME='hn2016_falwa'
URL='https://github.com/csyhuang/hn2016_falwa' # how can we make download_url automatically get the right version?
DOWNLOAD_URL='https://github.com/csyhuang/hn2016_falwa/'
AUTHOR='Clare S. Y. Huang'
AUTHOR_EMAIL='clare1068@gmail.com'
LICENSE='MIT'

DESCRIPTION='python package to compute finite-amplitude local wave activity (Huang and Nakamura 2016, JAS)'

LONG_DESCRIPTION="""
HN2016_FALWA is a package that contains modules to compute the finite-amplitude 
local wave activity (FALWA) proposed in Huang and Nakamura (2016 JAS)
http://dx.doi.org/10.1175/JAS-D-15-0194.1
The current version of the library handles calculation of FALWA in a spherical 
barotropic model and QGPV fields on isobaric surfaces.

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
                  