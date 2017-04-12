''' The function "theta_lwa" computes the longitudinally local surface wave activity (its zonal mean is B* in Nakamura and Solomon (2010)) and reference state of potential temperature.

Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
'''
from math import *
from lwa import LWA
from eqv_lat import EqvLat
import numpy as np

def theta_lwa(ylat,theta,area,dmu,nlat_S=None):
    '''
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in 
                ascending order; dimension = nlat
        theta: 2-d numpy array of potential temperature values; dimension = [nlat_S x nlon]
        area: 2-d numpy array specifying differential areal element of each 
                grid point; dimension = [nlat_S x nlon]
        nlat_S: the index of grid point that defines the extent of hemispheric 
                domain from the pole. If the value of nlat_S is not input, it 
                will be set to nlat/2, where nlat is the size of ylat.
        
    Output variables:
        Qref: 1-d numpy array of value Q(y) where latitude y is given by ylat. 
        LWA_result: 2-d numpy array of local wave activity values; 
                    dimension = [nlat_S x nlon]    
    '''
    
    nlat = theta.shape[0]
    nlon = theta.shape[1]
    if nlat_S == None:
        nlat_S = nlat/2
        
    Qref = np.zeros(nlat)
    LWA_result = np.zeros((nlat,nlon))
    
    # --- Southern Hemisphere ---
    Qref1 = EqvLat(ylat[:nlat_S],theta[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[:nlat_S] = Qref1
    LWA_South = LWA(nlon,nlat_S,theta[:nlat_S,:],Qref1,dmu[:nlat_S])
    LWA_result[:nlat_S,:] = LWA_South

    # --- Northern Hemisphere ---
    theta2 = theta[::-1,:] # Added the minus sign, but gotta see if NL_North is affected
    Qref2 = EqvLat(ylat[:nlat_S],theta2[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[-nlat_S:] = Qref2[::-1]
    LWA_North = LWA(nlon,nlat_S,theta2[:nlat_S,:],Qref2,dmu[:nlat_S])
    LWA_result[-nlat_S:,:] = LWA_North[::-1,:]

    return Qref, LWA_result