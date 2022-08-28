"""
------------------------------------------
File name: basis.py
Author: Clare Huang
"""
import numpy as np
from math import pi


def eqvlat(ylat, vort, area, n_points, planet_radius=6.378e+6, vgrad=None):

    """
    Compute equivalent latitude, and optionally <...>_Q in Nakamura and Zhu (2010).

    Parameters
    ----------
    ylat : sequence or array_like
        1-d numpy array of latitude (in degree) with equal spacing in ascending order; dimension = nlat.
    vort : ndarray
        2-d numpy array of vorticity values; dimension = (nlat, nlon).
    area : ndarray
        2-d numpy array specifying differential areal element of each grid point; dimension = (nlat, nlon).
    n_points: int
        Analysis resolution to calculate equivalent latitude.
    planet_radius: float
        Radius of spherical planet of interest consistent with input *area*. Default: earth's radius 6.378e+6
    vgrad: ndarray, optional
        2-d numpy array of laplacian (or higher-order laplacian) values; dimension = (nlat, nlon)

    Returns
    -------
    q_part : ndarray
        1-d numpy array of value Q(y) where latitude y is given by ylat; dimension = (nlat).
    brac : ndarray or None
        1-d numpy array of averaged vgrad in the square bracket.
        If *vgrad* = None, *brac* = None.

    """

    vort_min = np.min([vort.min(), vort.min()])
    vort_max = np.max([vort.max(), vort.max()])
    q_part_u = np.linspace(vort_min, vort_max, n_points, endpoint=True)
    aa = np.zeros(q_part_u.size)  # to sum up area
    vort_flat = vort.flatten()  # Flatten the 2D arrays to 1D
    area_flat = area.flatten()

    if vgrad is not None:
        dp = np.zeros_like(aa)
        vgrad_flat = vgrad.flatten()

    # Find equivalent latitude:
    inds = np.digitize(vort_flat, q_part_u)
    for i in np.arange(0, aa.size):  # Sum up area in each bin
        aa[i] = np.sum(area_flat[np.where(inds == i)])
        if vgrad is not None:
            # This is to avoid the divided-by-zero error
            if aa[i] == 0.:
                continue
            else:
                dp[i] = np.sum(area_flat[np.where(inds == i)] *
                               vgrad_flat[np.where(inds == i)]) / aa[i]

    aq = np.cumsum(aa)
    if vgrad is not None:
        brac = np.zeros_like(aa)
        brac[1:-1] = 0.5 * (dp[:-2] + dp[2:])

    y_part = aq / (2 * pi * planet_radius**2) - 1.0
    lat_part = np.rad2deg(np.arcsin(y_part))
    q_part = np.interp(ylat, lat_part, q_part_u)

    if vgrad is not None:
        brac_return = np.interp(ylat, lat_part, brac)
    else:
        brac_return = None

    return q_part, brac_return


def lwa(nlon, nlat, vort, q_part, dmu, ncforce=None):

    """
    At each grid point of vorticity q(x,y) and reference state vorticity Q(y),
    this function calculate the difference between the line integral of [q(x,y+y')-Q(y)]
    (and ncforce, if given) over the domain {y+y'>y,q(x,y+y')<Q(y)} and {y+y'<y,q(x,y+y')>Q(y)}.
    See fig. (1) and equation (13) of Huang and Nakamura (2016).
    dmu is a vector of length nlat: dmu = cos(phi) d(phi) such that phi is the latitude.


    Parameters
    ----------
    nlon : int
        Longitudinal dimension of vort (i.e. vort.shape[1]).
    nlat : int
        Latitudinal dimension of vort (i.e. vort.shape[0]).
    vort : ndarray
        2-d numpy array of vorticity values; dimension = (nlat,nlon).
    Q_part: sequence or array_like
        1-d numpy array of Q (vorticity reference state) as a function of latitude. Size = nlat.
    dmu: sequence or array_like
        1-d numpy array of latitudinal differential length element (e.g. dmu = planet_radius * cos(lat) d(lat)). Size = nlat.
    ncforce: ndarray or None, optional
        2-d numpy array of non-conservative force field (i.e. theta in NZ10(a) in equation (23a) and (23b))


    Returns
    -------
    lwact : ndarray
        2-d numpy array of local wave activity calculated; dimension = (nlat,nlon).
    bigsigma : ndarray or None
        2-d numpy array of the nonconservative forces acting on local wave activity computed from *ncforce*.
        If *ncforce* = None, *bigsigma* = None.

    """

    lwact = np.zeros((nlat, nlon))
    if ncforce is not None:
        bigsigma = np.zeros((nlat, nlon))

    for j in np.arange(0, nlat - 1):
        vort_e = vort[:, :] - q_part[j]
        vort_boo = np.zeros((nlat, nlon))
        vort_boo[np.where(vort_e[:, :] < 0)] = -1
        vort_boo[:j + 1, :] = 0
        vort_boo[np.where(vort_e[:j + 1, :] > 0)] = 1
        lwact[j, :] = np.sum(vort_e * vort_boo *
                             dmu[:, np.newaxis], axis=0)
        if ncforce is not None:
            bigsigma[j, :] = np.sum(ncforce * vort_boo *
                                    dmu[:, np.newaxis], axis=0)

    if ncforce is None:
        bigsigma = None

    return lwact, bigsigma
