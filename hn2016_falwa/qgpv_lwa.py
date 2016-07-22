''' The function qgpv_Eqlat_LWA computes local wave activity with hemispheric domain.

    Please email Clare S. Y. Huang if you have any inquiries/suggestions: clare1068@gmail.com
'''
from math import *
from lwa import LWA
from barotropic_lwa import barotropic_EqvLat
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
#a = 6.378e+6 # Earth's radius [m]
# dy = a*dphi

def qgpv_Eqlat_LWA(ylat,vort,area,dy,nlat_S=61): # Calculation of local wave activity on a 2-D vorticity map

    nlat = vort.shape[0]
    nlon = vort.shape[1]
    Qref = np.zeros(nlat)
    LWA_result = np.zeros((nlat,nlon))
    NL = np.zeros((nlat,nlon))

    # --- Southern Hemisphere ---
    Qref1 = barotropic_EqvLat(ylat[:nlat_S],vort[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[:nlat_S] = Qref1
    LWA_South = LWA(nlon,nlat_S,nlat_S,vort[:nlat_S,:],Qref1,dy)
    LWA_result[:nlat_S,:] = LWA_South

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1,:] # Added the minus sign, but gotta see if NL_North is affected
    Qref2 = barotropic_EqvLat(ylat[:nlat_S],vort2[:nlat_S,:],area[:nlat_S,:],nlat_S)
    Qref[-nlat_S:] = Qref2[::-1]
    LWA_North = LWA(nlon,nlat_S,nlat_S,vort2[:nlat_S,:],Qref2,dy)
    LWA_result[-nlat_S:,:] = LWA_North[::-1,:]

    return Qref, LWA_result


