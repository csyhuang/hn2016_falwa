''' HN2015_LWA is a python module developed for readers to implement local finite-amplitude wave activity
    analysis proposed in Huang & Nakamura (2016, JAS) (http://dx.doi.org/10.1175/JAS-D-15-0194.1) on cli-
	mate datasets. See/Run Example_plotting_script.py for example usage of this module to reproduce figures 
	in HN2016.
	
	Current modules included (Last update: Oct 4, 2015)
	barotropic_Eqlat_LWA: computes equivalent-latitude profile Q(y) and the corresponding LWA from a 2D-vorticity map on a spherical surface.
	barotropic_EqvLat: a function used by Eqlat_LWA to obtain Q(y) from a 2D-vorticity map.
	
	Please email Clare S. Y. Huang if you have any inquiries/suggestions: clare1068@gmail.com
'''
from math import *
from lwa import LWA
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
a = 6.378e+6 # Earth's radius [m]

# --- Calculation of local wave activity on a 2-D vorticity map ---
def barotropic_Eqlat_LWA(ylat,vort,area,dy): # used to be Eqlat_LWA
    ''' Assume area element = a**2 cos(lat)d(lat)d(lon)
    dx = a * cos(lat) * d(lon)
    dy = a * d(lat)
    
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
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    Qref = barotropic_EqvLat(ylat,vort,area,ylat.size)
    LWA_result = LWA(nlon,nlat,nlat,vort,Qref,dy)
    return Qref, LWA_result

# --- Calculation of equivalent latitude ---
def barotropic_EqvLat(ylat,vort,area,n_points):
    '''
    Input variables:
        ylat: 1-d numpy array with equal spacing in ascending order; dimension = nlat_S
        vort: 2-d numpy array of vorticity values; dimension = (nlat_S,nlon)
        area: 2-d numpy array specifying differential areal element of each grid point; dimension = (nlat_S,nlon)
        n_points: analysis resolution to calculate equivalent latitude.
    Output variables:
        Q_part: 1-d numpy array of value Q(y) where y is given by ylat.
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
