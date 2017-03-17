''' eqv_lat contains 2 modules that compute equivalent-latitude relationship Q(y)
(See Nakamura 1996, Allen and Nakamura 2003, etc) with a global domain (EqvLat) 
or hemispheric domain (EqvLat_hemispheric) as in Huang and Nakamura (2016).

The computation of Q(y) with hemispheric domain is preferrable when studying
a QGPV field, in which the meridional gradient of QGPV near equator is vanishing.

The use of hemispheric domain is necessary to compute the surface wave activity 
(B in Nakamura and Solomon (2010) equation (15)) from potential temperature field
because there is a reversal of meridional gradient at the equator.
	
Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
'''

from math import *
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
a = 6.378e+6 # Earth's radius [m]

# --- Calculation of equivalent latitude ---
def EqvLat(ylat,vort,area,n_points):
    '''
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in 
              ascending order; dimension = nlat
        vort: 2-d numpy array of vorticity values; dimension = [nlat_S x nlon]
        area: 2-d numpy array specifying differential areal element of each 
              grid point; dimension = [nlat_S x nlon]
        n_points: analysis resolution to calculate equivalent latitude.
        
    Output variables:
        Q_part: 1-d numpy array of value Q(y) where latitude y is given by ylat.
    '''
    vort_min = np.min([vort.min(),vort.min()])
    vort_max = np.max([vort.max(),vort.max()])
    Q_part_u = np.linspace(vort_min,vort_max,n_points,endpoint=True)
    aa = np.zeros(Q_part_u.size) # to sum up area
    vort_flat = vort.flatten() # Flatten the 2D arrays to 1D
    area_flat = area.flatten()    
    # Find equivalent latitude:
    inds = np.digitize(vort_flat, Q_part_u) 
    for i in np.arange(0,aa.size):  # Sum up area in each bin
        aa[i] = np.sum( area_flat[np.where(inds==i)] )
    aq = np.cumsum(aa)
    Y_part = aq/(2*pi*a**2) - 1.0
    lat_part = np.arcsin(Y_part)*180/pi    
    Q_part = np.interp(ylat, lat_part, Q_part_u)    
    return Q_part

def EqvLat_hemispheric(ylat,vort,area,nlat_S=61):                                        
    '''
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in 
              ascending order; dimension = nlat
        vort: 2-d numpy array of vorticity values; dimension = [nlat_S x nlon]
        area: 2-d numpy array specifying differential areal element of each 
              grid point; dimension = [nlat_S x nlon]
        nlat_S: the index of grid point that defines the extent of hemispheric 
                domain from the pole. The default is 61 for ERA-Interim data of 
                latitudinal resolution of 1.5 deg.
        
    Output variables:
        Q_part: 1-d numpy array of value Q(y) where latitude y is given by ylat.
    '''

    nlat = vort.shape[0]
    Qref = np.zeros(nlat)

    # --- Southern Hemisphere ---
    Qref1 = EqvLat(ylat[:nlat_S],vort[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[:nlat_S] = Qref1

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1,:] # Added the minus sign, but gotta see if NL_North is affected
    Qref2 = EqvLat(ylat[:nlat_S],vort2[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[-nlat_S:] = Qref2[::-1]

    return Qref
