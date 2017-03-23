''' The function "barotropic_Eqlat_LWA" in this module computes the equivalent-
latitude relationship and finite-amplitude local wave activity (LWA) from the 
vorticity fields on a global spherical domain according to the definition in 
Huang & Nakamura (2016,JAS) equation (13). 

Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
'''
from math import *
from lwa import LWA
from eqv_lat import EqvLat
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
a = 6.378e+6 # Earth's radius [m]

# --- Calculation of local wave activity on a 2-D vorticity map ---
def barotropic_Eqlat_LWA(ylat,vort,area,dmu): # used to be Eqlat_LWA
    ''' Assume area element = a**2 cos(lat)d(lat)d(lon)
    dx = a * cos(lat) * d(lon)
    dmu = a cos(lat) * d(lat)
    
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
    Qref = EqvLat(ylat,vort,area,ylat.size)
    LWA_result = LWA(nlon,nlat,vort,Qref,dmu)
    return Qref, LWA_result


def barotropic_input_Qref_to_compute_LWA(ylat,Qref,vort,area,dmu): # used to be Eqlat_LWA
    ''' 
	This function computes LWA based on a *prescribed* Qref instead of Qref
	obtained from the vorticity field.
	--------------------------------------------------------------------------
	Assume area element = a**2 cos(lat)d(lat)d(lon)
    dx = a * cos(lat) * d(lon)
    dmu = a cos(lat) * d(lat)
    
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
    LWA_result = LWA(nlon,nlat,vort,Qref,dmu)
    return LWA_result
