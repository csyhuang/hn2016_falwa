''' The function "qgpv_Eqlat_LWA" in this module computes the equivalent-
latitude relationship and finite-amplitude local wave activity (LWA) from the 
vorticity fields on a hemispheric domain as shown in the example in fig. 8 and 9 
of Huang & Nakamura (2016) equation (13). 

Please email Clare S. Y. Huang if you have any inquiries/suggestions: clare1068@gmail.com
'''
from math import *
from lwa import LWA
from eqv_lat import EqvLat
import numpy as np

def qgpv_Eqlat_LWA(ylat,vort,area,dmu,nlat_S=61):
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
        Qref: 1-d numpy array of value Q(y) where latitude y is given by ylat. 
        LWA_result: 2-d numpy array of local wave activity values; 
                    dimension = [nlat_S x nlon]    
    '''
    
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    Qref = np.zeros(nlat)
    LWA_result = np.zeros((nlat,nlon))
    NL = np.zeros((nlat,nlon))
    
    # --- Southern Hemisphere ---
    Qref1 = EqvLat(ylat[:nlat_S],vort[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[:nlat_S] = Qref1
    LWA_South = LWA(nlon,nlat_S,nlat_S,vort[:nlat_S,:],Qref1,dmu[:nlat_S])
    LWA_result[:nlat_S,:] = LWA_South

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1,:] # Added the minus sign, but gotta see if NL_North is affected
    Qref2 = EqvLat(ylat[:nlat_S],vort2[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[-nlat_S:] = Qref2[::-1]
    LWA_North = LWA(nlon,nlat_S,nlat_S,vort2[:nlat_S,:],Qref2,dmu[:nlat_S])
    LWA_result[-nlat_S:,:] = LWA_North[::-1,:]

    return Qref, LWA_result


