''' The function "barotropic_Eqlat_LWA" in this module computes the equivalent-
latitude relationship and finite-amplitude local wave activity (LWA) from the 
vorticity fields on a global spherical domain according to the definition in 
Huang & Nakamura (2016,JAS) equation (13). 

Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
'''
from math import pi,exp
from lwa import lwa
from eqv_lat import eqvlat
import numpy as np

# --- Calculation of local wave activity on a 2-D vorticity map ---
def barotropic_eqlat_lwa(ylat,vort,area,dmu,planet_radius = 6.378e+6):
    ''' Assume area element = a**2 cos(lat)d(lat)d(lon)
    dx = a * cos(lat) * d(lon)
    dmu = a cos(lat) * d(lat)
    
    Input variables:
        ylat: 1-d numpy array with equal spacing in ascending order; dimension = nlat
        vort: 2-d numpy array of vorticity values; dimension = (nlat,nlon)
        area: 2-d numpy array specifying differential areal element of each grid point; dimension = (nlat,nlon)
        dmu: 1-d numpy array of differential element in latitudinal direction; dimension = nlat
        planet_radius: scalar; radius of spherical planet of interest consistent with input 'area'
    Output variables:
        qref: Equivalent latitude relationship Q(y), where y is given by ylat. Values of Q in excluded domain is zero.
        lwa_result: 2-d numpy array of Local Wave Activity (LWA); dimension = (nlat,nlon)                                
    '''
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    qref = eqvlat(ylat,vort,area,ylat.size)
    lwa_result = lwa(nlon,nlat,vort,qref,dmu)
    return qref, lwa_result

def barotropic_input_qref_to_compute_lwa(ylat,qref,vort,area,dmu,planet_radius = 6.378e+6): # used to be Eqlat_LWA
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
        dmu: 1-d numpy array of differential element in latitudinal direction; dimension = nlat    
        planet_radius: scalar; radius of spherical planet of interest consistent with input 'area'             
    Output variables:
        Qref: Equivalent latitude relationship Q(y), where y is given by ylat. Values of Q in excluded domain is zero.
        LWA_result: 2-d numpy array of Local Wave Activity (LWA); dimension = (nlat,nlon)                                
    '''
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    lwa_result = lwa(nlon,nlat,vort,qref,dmu)
    return lwa_result
