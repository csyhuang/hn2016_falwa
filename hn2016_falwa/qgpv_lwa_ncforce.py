''' The function "qgpv_eqlat_lwa_ncforce" in this module computes the equivalent-
latitude relationship and finite-amplitude local wave activity (lwa) from the
vorticity fields on a hemispheric domain as shown in the example in fig. 8 and 9
of Huang & Nakamura (2016) equation (13), and also the contribution of local non-
conservative force (Sigma in NZ10 eq. (23a) and (23b)).

Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
'''
from lwa_ncforce import lwa_ncforce
from eqv_lat import eqvlat
import numpy as np


def qgpv_eqlat_lwa_ncforce(ylat, vort, ncforce, area, dmu, nlat_s=None, planet_radius=6.378e+6):
    '''
    Input variables:
        ylat: 1-d numpy array of latitude (in degree) with equal spacing in
                ascending order; dimension = nlat
        vort: 2-d numpy array of vorticity values; dimension = [nlat_s x nlon]
        ncforce: 2-d numpy array of non-conservative force field (i.e. theta
                 in NZ10(a) in equation (23a) and (23b)); dimension = [nlat_s x nlon]
        area: 2-d numpy array specifying differential areal element of each
                grid point; dimension = [nlat_s x nlon]
        nlat_s: the index of grid point that defines the extent of hemispheric
                domain from the pole. If the value of nlat_s is not input,  it
                will be set to nlat/2,  where nlat is the size of ylat.
        planet_radius: scalar; radius of spherical planet of interest consistent
                with input 'area'

    Output variables:
        qref: 1-d numpy array of value Q(y) where latitude y is given by ylat.
        lwa_result: 2-d numpy array of local wave activity values;
                    dimension = [nlat_s x nlon]
        bigsigma_result: 2-d numpy array of non-conservative force contribution value;
                    dimension = [nlat_s x nlon]
    '''

    nlat = vort.shape[0]
    nlon = vort.shape[1]
    if nlat_s == None:
        nlat_s = nlat/2

    qref = np.zeros(nlat)
    lwa_result = np.zeros((nlat, nlon))
    bigsigma_result = np.zeros((nlat, nlon))

    # lwa_ncforce(nlon,  nlat,  vort,  q_part,  ncforce,  dy)

    # --- Southern Hemisphere ---
    qref1 = eqvlat(ylat[:nlat_s], vort[:nlat_s, :], area[:nlat_s, :], nlat_s, planet_radius=planet_radius)
    qref[:nlat_s] = qref1
    lwa_South, ncforce_South = lwa_ncforce(nlon, nlat_s, vort[:nlat_s, :],
                                           qref[:nlat_s], ncforce[:nlat_s, :],
                                           dmu[:nlat_s])
    lwa_result[:nlat_s, :] = lwa_South
    bigsigma_result[:nlat_s, :] = ncforce_South

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1, :]  # Added the minus sign,  but gotta see if NL_North is affected
    qref2 = eqvlat(ylat[:nlat_s], vort2[:nlat_s, :], area[:nlat_s, :], nlat_s, planet_radius=planet_radius)
    qref[-nlat_s:] = -qref2[::-1]
    # Do the same operation but without flipping QGPV field. Do it on the Northern Hemisphere
    lwa_North, ncforce_North = lwa_ncforce(nlon, nlat_s, vort[-nlat_s:, :],
                                           qref[-nlat_s:], ncforce[-nlat_s:, :],
                                           dmu[-nlat_s:])
    lwa_result[-nlat_s:, :] = lwa_North
    bigsigma_result[-nlat_s:, :] = ncforce_North

    return qref,  lwa_result, bigsigma_result
