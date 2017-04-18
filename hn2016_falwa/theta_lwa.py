''' The function "theta_lwa" computes the longitudinally local surface wave activity (its zonal mean is B* in Nakamura and Solomon (2010)) and reference state of potential temperature.

Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
'''
from math import *
from lwa import lwa
from eqv_lat import eqvlat
import numpy as np

def theta_lwa(ylat,theta,area,dmu,nlat_s=None,planet_radius=6.378e+6):
    '''
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in 
                ascending order; dimension = nlat
        theta: 2-d numpy array of potential temperature values; dimension = [nlat_s x nlon]
        area: 2-d numpy array specifying differential areal element of each 
                grid point; dimension = [nlat_s x nlon]
        nlat_s: the index of grid point that defines the extent of hemispheric 
                domain from the pole. If the value of nlat_s is not input, it 
                will be set to nlat/2, where nlat is the size of ylat.
        planet_radius: scalar; radius of spherical planet of interest consistent 
                with input 'area'
        
    Output variables:
        qref: 1-d numpy array of value Q(y) where latitude y is given by ylat. 
        lwa_result: 2-d numpy array of local wave activity values; 
                    dimension = [nlat_s x nlon]    
    '''
    
    nlat = theta.shape[0]
    nlon = theta.shape[1]
    if nlat_s == None:
        nlat_s = nlat/2
        
    qref = np.zeros(nlat)
    lwa_result = np.zeros((nlat,nlon))
    
    # --- southern Hemisphere ---
    qref1 = eqvlat(ylat[:nlat_s],theta[:nlat_s,:],area[:nlat_s,:],nlat_s,planet_radius=6.378e+6)
    qref[:nlat_s] = qref1
    lwa_south = lwa(nlon,nlat_s,theta[:nlat_s,:],qref1,dmu[:nlat_s])
    lwa_result[:nlat_s,:] = lwa_south

    # --- northern Hemisphere ---
    theta2 = theta[::-1,:] # Added the minus sign, but gotta see if NL_north is affected
    qref2 = eqvlat(ylat[:nlat_s],theta2[:nlat_s,:],area[:nlat_s,:],nlat_s,planet_radius=6.378e+6)
    qref[-nlat_s:] = qref2[::-1]
    lwa_north = lwa(nlon,nlat_s,theta2[:nlat_s,:],qref2,dmu[:nlat_s])
    lwa_result[-nlat_s:,:] = lwa_north[::-1,:]

    return qref, lwa_result