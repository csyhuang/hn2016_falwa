''' The python package "hn2016_falwa" contains a function "barotropic_Eqlat_LWA" 
that computes the finite-amplitude local wave activity (LWA) in a barotropic model 
with spherical geometry according to the definition in Huang & Nakamura (2016,JAS) 
equation (13). This sample code demonstrates how the function in this package can 
be used to reproduce LWA plots (Fig.4 in HN15) from an absolute vorticity map.

Please email Clare S. Y. Huang if you have any inquiries/suggestions: clare1068@gmail.com
'''

import hn2016_falwa # Module for plotting local wave activity (LWA) plots and 
                      # the corresponding equivalent-latitude profile
from math import *
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype

# --- Parameters --- #
a = 6.378e+6 # Earth's radius

# --- Load the absolute vorticity field [256x512] --- #
readFile = Dataset('barotropic_vorticity.nc', mode='r')

# --- Read in longitude and latitude arrays --- #
xlon = readFile.variables['longitude'][:]
ylat = readFile.variables['latitude'][:]
nlon = xlon.size
nlat = ylat.size

# --- Parameters needed to use the module HN2015_LWA --- #
dphi = (ylat[2]-ylat[1])*pi/180. # Equal spacing between latitude grid points, in radian
area = 2.*pi*a*a*(np.cos(ylat[:,np.newaxis]*pi/180.)*dphi)/float(nlon) * np.ones((nlat,nlon))
area = np.abs(area) # To make sure area element is always positive (given floating point errors). 

# --- Read in the absolute vorticity field from the netCDF file --- #
absVorticity = readFile.variables['absolute_vorticity'][:]

# --- Obtain equivalent-latitude relationship and also the LWA from the absolute vorticity snapshot ---
Q_ref,LWA = hn2016_falwa.barotropic_Eqlat_LWA(ylat,absVorticity,area,a*dphi) # Full domain included
''' def Eqlat_LWA(ylat,vort,area,dphi):
    Assume area element = a**2 cos(lat)d(lat)d(lon)
    dx = a * cos(lat) * d(lon)
    dy = a * d(lat)
    dphi = d(lat) = pi/float(nlat-1)
    
    Input variables:
        ylat: 1-d numpy array with equal spacing in ascending order; dimension = nlat
        vort: 2-d numpy array of vorticity values; dimension = (nlat,nlon)
        area: 2-d numpy array specifying differential areal element of each grid point; dimension = (nlat,nlon)
        dphi: scalar specifying differential length element in meridional direction. 
              dphi = pi/float(nlat-1) if assuming equally-spaced y-grid of range [-90:90],                 
    Output variables:
        Qref: Equivalent latitude relationship Q(y), where y is given by ylat. Values of Q in excluded domain is zero.
        LWA_result: 2-d numpy array of Local Wave Activity (LWA); dimension = (nlat,nlon)                                
    '''

# --- Color axis for plotting LWA --- #
LWA_caxis = np.linspace(0,LWA.max(),31,endpoint=True)

# --- Plot the abs. vorticity field, LWA and equivalent-latitude relationship and LWA --- #
fig = plt.subplots(figsize=(14,6))

plt.subplot(1,3,1) # Absolute vorticity map
c=plt.contourf(xlon,ylat,absVorticity,31)
cb = plt.colorbar(c)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')                         
cb.update_ticks()
plt.title('Absolute vorticity [1/s]')
plt.xlabel('Longitude (degree)')
plt.ylabel('Latitude (degree)')

plt.subplot(1,3,2) # LWA (full domain)
plt.contourf(xlon,ylat,LWA,LWA_caxis)
plt.colorbar()
plt.title('Local Wave Activity [m/s]')
plt.xlabel('Longitude (degree)')
plt.ylabel('Latitude (degree)')

plt.subplot(1,3,3) # Equivalent-latitude relationship Q(y)
plt.plot(Q_ref,ylat,'b',label='Equivalent-latitude relationship')
plt.plot(np.mean(absVorticity,axis=1),ylat,'g',label='zonal mean abs. vorticity')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylim(-90,90)
plt.legend(loc=4,fontsize=10)
plt.title('Equivalent-latitude profile')
plt.ylabel('Latitude (degree)')
plt.xlabel('Q(y) [1/s] | y = latitude')
plt.tight_layout()
plt.show()
plt.savefig('Example_barotropic.pdf')
