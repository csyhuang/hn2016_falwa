''' The function "qgpv_Eqlat_lwa" in this module computes the equivalent-
latitude relationship and finite-amplitude local wave activity (lwa) from the 
vorticity fields on a hemispheric domain as shown in the example in fig. 8 and 9 
of Huang & Nakamura (2016) equation (13). 

Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
'''
from math import pi,exp
from lwa import lwa
from eqv_lat import eqvlat
import numpy as np

def qgpv_eqlat_lwa(ylat,vort,area,dmu,nlat_s=None,planet_radius=6.378e+6):
    '''
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in 
                ascending order; dimension = nlat
        vort: 2-d numpy array of vorticity values; dimension = [nlat_s x nlon]
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
    
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    if nlat_s == None:
        nlat_s = nlat/2
        
    qref = np.zeros(nlat)
    lwa_result = np.zeros((nlat,nlon))
    
    # --- Southern Hemisphere ---
    qref1 = eqvlat(ylat[:nlat_s],vort[:nlat_s,:],area[:nlat_s,:],nlat_s,planet_radius=planet_radius)
    qref[:nlat_s] = qref1
    lwa_South = lwa(nlon,nlat_s,vort[:nlat_s,:],qref1,dmu[:nlat_s])
    lwa_result[:nlat_s,:] = lwa_South

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1,:] # Added the minus sign, but gotta see if NL_North is affected
    qref2 = eqvlat(ylat[:nlat_s],vort2[:nlat_s,:],area[:nlat_s,:],nlat_s,planet_radius=planet_radius)
    qref[-nlat_s:] = -qref2[::-1]
    lwa_North = lwa(nlon,nlat_s,vort2[:nlat_s,:],qref2,dmu[:nlat_s])
    lwa_result[-nlat_s:,:] = lwa_North[::-1,:]

    return qref, lwa_result


def qgpv_input_qref_to_compute_lwa(ylat,qref,vort,area,dmu,nlat_s=None,planet_radius=6.378e+6):
    '''
	This function computes lwa based on a *prescribed* qref instead of qref
	obtained from the QGPV field.
	--------------------------------------------------------------------------
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in 
                ascending order; dimension = nlat
        qref: 1-d numpy array of value Q(y) where latitude y is given by ylat. 
        vort: 2-d numpy array of vorticity values; dimension = [nlat_s x nlon]
        area: 2-d numpy array specifying differential areal element of each 
                grid point; dimension = [nlat_s x nlon]
        nlat_s: the index of grid point that defines the extent of hemispheric 
                domain from the pole. If the value of nlat_s is not input, it 
                will be set to nlat/2, where nlat is the size of ylat.
        
    Output variables:
        lwa_result: 2-d numpy array of local wave activity values; 
                    dimension = [nlat_s x nlon]    
    '''
    
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    if nlat_s == None:
        nlat_s = nlat/2
        
    lwa_result = np.zeros((nlat,nlon))
    
    # --- Southern Hemisphere ---
    lwa_result[:nlat_s,:] = lwa(nlon,nlat_s,vort[:nlat_s,:],qref[:nlat_s],dmu[:nlat_s])

    # --- Northern Hemisphere ---
    lwa_result[-nlat_s:,:] = lwa(nlon,nlat_s,vort[-nlat_s:,:],qref[-nlat_s:],dmu[-nlat_s:])

    return lwa_result