''' The function "qgpv_Eqlat_LWA" in this module computes the equivalent-
latitude relationship and finite-amplitude local wave activity (LWA) from the 
vorticity fields on a hemispheric domain as shown in the example in fig. 8 and 9 
of Huang & Nakamura (2016) equation (13). 

Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
'''
from math import *
from lwa import LWA
from eqv_lat import EqvLat
import numpy as np

def qgpv_Eqlat_LWA(ylat,vort,area,dmu,nlat_S=None):
    '''
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in 
                ascending order; dimension = nlat
        vort: 2-d numpy array of vorticity values; dimension = [nlat_S x nlon]
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
    
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    if nlat_S == None:
        nlat_S = nlat/2
        
    Qref = np.zeros(nlat)
    LWA_result = np.zeros((nlat,nlon))
    
    # --- Southern Hemisphere ---
    Qref1 = EqvLat(ylat[:nlat_S],vort[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[:nlat_S] = Qref1
    LWA_South = LWA(nlon,nlat_S,vort[:nlat_S,:],Qref1,dmu[:nlat_S])
    LWA_result[:nlat_S,:] = LWA_South

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1,:] # Added the minus sign, but gotta see if NL_North is affected
    Qref2 = EqvLat(ylat[:nlat_S],vort2[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[-nlat_S:] = -Qref2[::-1]
    LWA_North = LWA(nlon,nlat_S,vort2[:nlat_S,:],Qref2,dmu[:nlat_S])
    LWA_result[-nlat_S:,:] = LWA_North[::-1,:]

    return Qref, LWA_result


def qgpv_input_Qref_to_compute_LWA(ylat,Qref,vort,area,dmu,nlat_S=None):
    '''
	This function computes LWA based on a *prescribed* Qref instead of Qref
	obtained from the QGPV field.
	--------------------------------------------------------------------------
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in 
                ascending order; dimension = nlat
        Qref: 1-d numpy array of value Q(y) where latitude y is given by ylat. 
        vort: 2-d numpy array of vorticity values; dimension = [nlat_S x nlon]
        area: 2-d numpy array specifying differential areal element of each 
                grid point; dimension = [nlat_S x nlon]
        nlat_S: the index of grid point that defines the extent of hemispheric 
                domain from the pole. If the value of nlat_S is not input, it 
                will be set to nlat/2, where nlat is the size of ylat.
        
    Output variables:
        LWA_result: 2-d numpy array of local wave activity values; 
                    dimension = [nlat_S x nlon]    
    '''
    
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    if nlat_S == None:
        nlat_S = nlat/2
        
    LWA_result = np.zeros((nlat,nlon))
    
    # --- Southern Hemisphere ---
    LWA_result[:nlat_S,:] = LWA(nlon,nlat_S,vort[:nlat_S,:],Qref[:nlat_S],dmu[:nlat_S])

    # --- Northern Hemisphere ---
    LWA_result[-nlat_S:,:] = LWA(nlon,nlat_S,vort[-nlat_S:,:],Qref[-nlat_S:],dmu[-nlat_S:])

    return LWA_result