"""
------------------------------------------
File name: wrapper.py
Author: Clare Huang
"""
import numpy as np
from falwa.constant import EARTH_RADIUS
from falwa import basis
from typing import Optional, Tuple, Dict, Any


def barotropic_eqlat_lwa(
        ylat: np.ndarray, vort: np.ndarray, area: np.ndarray, dmu: np.ndarray,
        n_points: int, planet_radius: float = EARTH_RADIUS,
        return_partitioned_lwa: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equivalent latitude and wave activity on a barotropic sphere.

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    vort : np.ndarray
        2D numpy array of vorticity values; dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    dmu : np.ndarray
        1D numpy array of latitudinal differential length element
        (e.g. dmu = planet_radius * cos(lat) d(lat)). Size = nlat.
    n_points : int
        Analysis resolution to calculate equivalent latitude. If None, it will
        be initialized as nlat.
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.
    return_partitioned_lwa : bool, optional
        If True, return local wave activity as a stacked field of cyclonic and
        anticyclonic components. If False, return a single field of local wave
        activity of dimension (nlat, nlon). Default is False.

    Returns
    -------
    qref : np.ndarray
        1D numpy array of value Q(y) where latitude y is given by ylat;
        dimension = (nlat,).
    lwa_result : np.ndarray
        If return_partitioned_lwa is True, 3D numpy array of dimension
        (2, nlat_s, nlon), first dimension being anticyclonic and cyclonic
        components. If False, 2D numpy array of local wave activity values;
        dimension = (nlat_s, nlon).

    Examples
    --------
    See the :doc:`barotropic example notebook <notebooks/Example_barotropic>`.
    """
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    if n_points is None:
        n_points = nlat

    qref, fawa = basis.eqvlat_fawa(
        ylat, vort, area, n_points, planet_radius=planet_radius)
    lwa_result, _ = basis.lwa(nlon, nlat, vort, qref, dmu, return_partitioned_lwa=return_partitioned_lwa)

    return qref, lwa_result


def barotropic_input_qref_to_compute_lwa(
        ylat: np.ndarray, qref: np.ndarray, vort: np.ndarray, area: np.ndarray,
        dmu: np.ndarray, planet_radius: float = EARTH_RADIUS) -> np.ndarray:
    """Compute LWA based on a prescribed Qref on a barotropic sphere.

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    qref : np.ndarray
        1D numpy array of prescribed reference value of vorticity at each
        latitude; dimension = (nlat,).
    vort : np.ndarray
        2D numpy array of vorticity values; dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    dmu : np.ndarray
        1D numpy array of latitudinal differential length element
        (e.g. dmu = planet_radius * cos(lat) d(lat)). Size = nlat.
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.

    Returns
    -------
    np.ndarray
        2D numpy array of local wave activity values;
        dimension = (nlat_s, nlon).
    """
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    lwa_result, _ = basis.lwa(nlon, nlat, vort, qref, dmu)
    return lwa_result


def eqvlat_hemispheric(
        ylat: np.ndarray, vort: np.ndarray, area: np.ndarray,
        nlat_s: Optional[int] = None, n_points: Optional[int] = None,
        planet_radius: float = EARTH_RADIUS) -> np.ndarray:
    """Compute equivalent latitude in a hemispheric domain.

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    vort : np.ndarray
        2D numpy array of vorticity values; dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    nlat_s : int, optional
        The index of grid point that defines the extent of hemispheric domain
        from the pole. Defaults to nlat // 2.
    n_points : int, optional
        Analysis resolution to calculate equivalent latitude.
        Defaults to nlat_s.
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.

    Returns
    -------
    np.ndarray
        1D numpy array of value Q(y) where latitude y is given by ylat;
        dimension = (nlat,).
    """
    nlat = vort.shape[0]
    qref = np.zeros(nlat)

    if nlat_s is None:
        nlat_s = nlat // 2

    if n_points is None:
        n_points = nlat_s

    # --- Southern Hemisphere ---
    qref1, _ = basis.eqvlat_fawa(ylat[:nlat_s], vort[:nlat_s, :], area[:nlat_s, :],
                            n_points, planet_radius=planet_radius)
    qref[:nlat_s] = qref1

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1, :]
    qref2, _ = basis.eqvlat_fawa(ylat[:nlat_s], vort2[:nlat_s, :], area[:nlat_s, :],
                            n_points, planet_radius=planet_radius)
    qref[-nlat_s:] = qref2[::-1]

    return qref


def eqvlat_bracket_hemispheric(
        ylat: np.ndarray, vort: np.ndarray, area: np.ndarray,
        nlat_s: Optional[int] = None, n_points: Optional[int] = None,
        planet_radius: float = EARTH_RADIUS,
        vgrad: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equivalent latitude and <...>_Q in a hemispheric domain.

    The second quantity is from Nakamura and Zhu (2010).

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    vort : np.ndarray
        2D numpy array of vorticity values; dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    nlat_s : int, optional
        The index of grid point that defines the extent of hemispheric domain
        from the pole. Defaults to nlat // 2.
    n_points : int, optional
        Analysis resolution to calculate equivalent latitude.
        Defaults to nlat_s.
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.
    vgrad : np.ndarray, optional
        2D numpy array of laplacian (or higher-order laplacian) values;
        dimension = (nlat, nlon).

    Returns
    -------
    q_part : np.ndarray
        1D numpy array of value Q(y) where latitude y is given by ylat;
        dimension = (nlat,).
    brac : np.ndarray
        1D numpy array of averaged vgrad in the square bracket.
        If vgrad is None, this is an array of zeros.
    """
    nlat = vort.shape[0]
    qref = np.zeros(nlat)
    brac = np.zeros(nlat)

    if nlat_s is None:
        nlat_s = nlat // 2

    if n_points is None:
        n_points = nlat_s

    # --- Southern Hemisphere ---
    qref1, brac1 = basis.eqvlat_fawa(ylat[:nlat_s], vort[:nlat_s, :],
                                area[:nlat_s, :],
                                n_points, planet_radius=planet_radius,
                                vgrad=vgrad)
    qref[:nlat_s] = qref1

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1, :]
    qref2, brac2 = basis.eqvlat_fawa(ylat[:nlat_s], vort2[:nlat_s, :],
                                area[:nlat_s, :],
                                n_points, planet_radius=planet_radius,
                                vgrad=vgrad)

    qref[-nlat_s:] = qref2[::-1]

    brac[:nlat_s] = brac1
    brac[-nlat_s:] = brac2[::-1]

    return qref, brac


def qgpv_eqlat_lwa(
        ylat: np.ndarray, vort: np.ndarray, area: np.ndarray, dmu: np.ndarray,
        nlat_s: Optional[int] = None, n_points: Optional[int] = None,
        planet_radius: float = EARTH_RADIUS) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equivalent latitude and LWA from QGPV.

    This is based on the method outlined in Huang and Nakamura (2017).

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    vort : np.ndarray
        2D numpy array of Quasi-geostrophic potential vorticity field;
        dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    dmu : np.ndarray
        1D numpy array of latitudinal differential length element
        (e.g. dmu = planet_radius * cos(lat) d(lat)). Size = nlat.
    nlat_s : int, optional
        The index of grid point that defines the extent of hemispheric domain
        from the pole. Defaults to nlat // 2.
    n_points : int, optional
        Analysis resolution to calculate equivalent latitude.
        Defaults to nlat_s.
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.

    Returns
    -------
    qref : np.ndarray
        1D numpy array of value Q(y) where latitude y is given by ylat;
        dimension = (nlat,).
    lwa_result : np.ndarray
        2D numpy array of local wave activity values;
        dimension = (nlat_s, nlon).

    Examples
    --------
    See the :doc:`QGPV example notebook <notebooks/Example_qgpv>`.
    """
    nlat = vort.shape[0]
    nlon = vort.shape[1]

    if nlat_s is None:
        nlat_s = nlat // 2

    if n_points is None:
        n_points = nlat_s

    qref = np.zeros(nlat)
    lwa_result = np.zeros((nlat, nlon))

    # --- Southern Hemisphere ---
    qref1, _ = basis.eqvlat_fawa(
        ylat[:nlat_s], vort[:nlat_s, :], area[:nlat_s, :], n_points, planet_radius=planet_radius)
    qref[:nlat_s] = qref1
    lwa_result[:nlat_s, :], _ = basis.lwa(nlon, nlat_s,
                                          vort[:nlat_s, :],
                                          qref1, dmu[:nlat_s])

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1, :]
    # Added the minus sign, but gotta see if NL_North is affected
    qref2, _ = basis.eqvlat_fawa(
        ylat[:nlat_s], vort2[:nlat_s, :], area[:nlat_s, :], n_points, planet_radius=planet_radius)
    qref[-nlat_s:] = -qref2[::-1]
    lwa_result[-nlat_s:, :], _ = basis.lwa(nlon, nlat_s,
                                           vort[-nlat_s:, :],
                                           qref[-nlat_s:],
                                           dmu[-nlat_s:])
    return qref, lwa_result


def qgpv_eqlat_lwa_ncforce(
        ylat: np.ndarray, vort: np.ndarray, ncforce: np.ndarray, area: np.ndarray,
        dmu: np.ndarray, nlat_s: Optional[int] = None, n_points: Optional[int] = None,
        planet_radius: float = EARTH_RADIUS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute equivalent latitude, LWA, and non-conservative force from QGPV.

    This is based on the method outlined in Huang and Nakamura (2017).

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    vort : np.ndarray
        2D numpy array of Quasi-geostrophic potential vorticity field;
        dimension = (nlat, nlon).
    ncforce : np.ndarray
        2D numpy array of non-conservative force field (i.e. theta in NZ10(a)
        in equation (23a) and (23b)); dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    dmu : np.ndarray
        1D numpy array of latitudinal differential length element
        (e.g. dmu = planet_radius * cos(lat) d(lat)). Size = nlat.
    nlat_s : int, optional
        The index of grid point that defines the extent of hemispheric domain
        from the pole. Defaults to nlat // 2.
    n_points : int, optional
        Analysis resolution to calculate equivalent latitude.
        Defaults to nlat_s.
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.

    Returns
    -------
    qref : np.ndarray
        1D numpy array of value Q(y) where latitude y is given by ylat;
        dimension = (nlat,).
    lwa_result : np.ndarray
        2D numpy array of local wave activity values; dimension = (nlat, nlon).
    capsigma : np.ndarray
        2D numpy array of non-conservative force contribution value;
        dimension = (nlat, nlon).
    """
    nlat = vort.shape[0]
    nlon = vort.shape[1]

    if nlat_s is None:
        nlat_s = nlat // 2

    if n_points is None:
        n_points = nlat_s

    qref = np.zeros(nlat)
    lwa_result = np.zeros((nlat, nlon))
    capsigma = np.zeros((nlat, nlon))

    # --- Southern Hemisphere ---
    qref1, _ = basis.eqvlat_fawa(ylat[:nlat_s],
                            vort[:nlat_s, :], area[:nlat_s, :],
                            n_points, planet_radius=planet_radius)
    qref[:nlat_s] = qref1
    lwa_result[:nlat_s, :], \
    capsigma[:nlat_s, :] = basis.lwa(nlon, nlat_s, vort[:nlat_s, :],
                                     qref1, dmu[:nlat_s],
                                     ncforce=ncforce[:nlat_s, :])

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1, :]
    # Added the minus sign, but gotta see if NL_North is affected
    qref2, _ = basis.eqvlat_fawa(ylat[:nlat_s], vort2[:nlat_s, :], area[:nlat_s, :],
                            n_points, planet_radius=planet_radius)
    qref[-nlat_s:] = -qref2[::-1]
    lwa_result[-nlat_s:, :], \
    capsigma[-nlat_s:, :] = basis.lwa(nlon, nlat_s, vort[-nlat_s:, :],
                                      qref[-nlat_s:], dmu[-nlat_s:],
                                      ncforce=ncforce[-nlat_s:, :])
    return qref, lwa_result, capsigma


def qgpv_eqlat_lwa_options(
        ylat: np.ndarray, vort: np.ndarray, area: np.ndarray, dmu: np.ndarray,
        nlat_s: Optional[int] = None, n_points: Optional[int] = None,
        vgrad: Optional[np.ndarray] = None, ncforce: Optional[np.ndarray] = None,
        planet_radius: float = EARTH_RADIUS) -> Dict[str, np.ndarray]:
    """Compute equivalent latitude, LWA, and other optional terms from QGPV.

    This is based on the method outlined in Huang and Nakamura (2017).

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    vort : np.ndarray
        2D numpy array of Quasi-geostrophic potential vorticity field;
        dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    dmu : np.ndarray
        1D numpy array of latitudinal differential length element
        (e.g. dmu = planet_radius * cos(lat) d(lat)). Size = nlat.
    nlat_s : int, optional
        The index of grid point that defines the extent of hemispheric domain
        from the pole. Defaults to nlat // 2.
    n_points : int, optional
        Analysis resolution to calculate equivalent latitude.
        Defaults to nlat_s.
    vgrad : np.ndarray, optional
        2D numpy array of vgrad field.
    ncforce : np.ndarray, optional
        2D numpy array of non-conservative force field (i.e. theta in NZ10(a)
        in equation (23a) and (23b)); dimension = (nlat, nlon).
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.

    Returns
    -------
    dict
        A dictionary that consists of the computed outputs. It always contains
        'qref' and 'lwa_result'. It contains 'brac_result' if `vgrad` is
        provided, and 'capsigma' if `ncforce` is provided.
    """
    nlat = vort.shape[0]
    nlon = vort.shape[1]

    if nlat_s is None:
        nlat_s = nlat // 2

    if n_points is None:
        n_points = nlat_s

    qref = np.zeros(nlat)
    lwa_result = np.zeros((nlat, nlon))
    if ncforce is not None:
        capsigma = np.zeros((nlat, nlon))
    if vgrad is not None:
        brac_result = np.zeros(nlat)

    # --- Southern Hemisphere ---
    if vgrad is None:
        qref1, brac = basis.eqvlat_fawa(ylat[:nlat_s], vort[:nlat_s, :],
                                   area[:nlat_s, :],
                                   n_points, planet_radius=planet_radius)
    else:
        qref1, brac = basis.eqvlat_fawa(ylat[:nlat_s], vort[:nlat_s, :],
                                   area[:nlat_s, :],
                                   n_points, planet_radius=planet_radius,
                                   vgrad=vgrad[:nlat_s, :])

    qref[:nlat_s] = qref1
    if vgrad is not None:
        brac_result[:nlat_s] = brac

    if ncforce is not None:
        lwa_result[:nlat_s, :], capsigma[:nlat_s, :] = \
        basis.lwa(nlon, nlat_s,
                  vort[:nlat_s, :],
                  qref1, dmu[:nlat_s],
                  ncforce=ncforce[:nlat_s, :])
    else:
        lwa_result[:nlat_s, :], _ = basis.lwa(nlon, nlat_s, vort[:nlat_s, :],
                                              qref1, dmu[:nlat_s])

    # --- Northern Hemisphere ---
    vort2 = -vort[::-1, :]
    # Added the minus sign, but gotta see if NL_North is affected

    if vgrad is None:
        qref2, brac = basis.eqvlat_fawa(ylat[:nlat_s], vort2[:nlat_s, :],
                                   area[:nlat_s, :],
                                   n_points, planet_radius=planet_radius)
    else:
        vgrad2 = -vgrad[::-1, :]  # Is this correct?
        qref2, brac = basis.eqvlat_fawa(ylat[:nlat_s], vort2[:nlat_s, :],
                                   area[:nlat_s, :],
                                   n_points, planet_radius=planet_radius,
                                   vgrad=vgrad2[:nlat_s, :])

    qref[-nlat_s:] = -qref2[::-1]
    if vgrad is not None:
        brac_result[-nlat_s:] = -brac[::-1]

    if ncforce is not None:
        lwa_result[-nlat_s:, :], \
        capsigma[-nlat_s:, :] = basis.lwa(nlon, nlat_s, vort[-nlat_s:, :],
                                          qref[-nlat_s:],
                                          dmu[-nlat_s:],
                                          ncforce=ncforce[-nlat_s:, :])
    else:
        lwa_result[-nlat_s:, :], _ = basis.lwa(nlon, nlat_s, vort[-nlat_s:, :],
                                               qref[-nlat_s:], dmu[-nlat_s:])

    # Return things as dictionary
    return_dict = dict()
    return_dict['qref'] = qref
    return_dict['lwa_result'] = lwa_result

    if vgrad is not None:
        return_dict['brac_result'] = brac_result
    if ncforce is not None:
        return_dict['capsigma'] = capsigma

    return return_dict


def qgpv_input_qref_to_compute_lwa(
        ylat: np.ndarray, qref: np.ndarray, vort: np.ndarray, area: np.ndarray,
        dmu: np.ndarray, nlat_s: Optional[int] = None,
        planet_radius: float = EARTH_RADIUS) -> np.ndarray:
    """Compute LWA from QGPV with a prescribed Qref.

    This function computes LWA based on a prescribed `qref` instead of `qref`
    obtained from the QGPV field, as outlined in Huang and Nakamura (2017).

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    qref : np.ndarray
        1D numpy array of value Q(y) where latitude y is given by ylat;
        dimension = (nlat,).
    vort : np.ndarray
        2D numpy array of Quasi-geostrophic potential vorticity field;
        dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    dmu : np.ndarray
        1D numpy array of latitudinal differential length element
        (e.g. dmu = planet_radius * cos(lat) d(lat)). Size = nlat.
    nlat_s : int, optional
        The index of grid point that defines the extent of hemispheric domain
        from the pole. Defaults to nlat // 2.
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.

    Returns
    -------
    np.ndarray
        2D numpy array of local wave activity values; dimension = (nlat, nlon).
    """
    nlat = vort.shape[0]
    nlon = vort.shape[1]
    if nlat_s is None:
        nlat_s = nlat // 2

    lwa_result = np.zeros((nlat, nlon))

    # --- Southern Hemisphere ---
    lwa_result[:nlat_s, :], dummy = basis.lwa(nlon, nlat_s, vort[:nlat_s, :],
                                              qref[:nlat_s], dmu[:nlat_s])

    # --- Northern Hemisphere ---
    lwa_result[-nlat_s:, :], dummy = basis.lwa(nlon, nlat_s, vort[-nlat_s:, :],
                                               qref[-nlat_s:], dmu[-nlat_s:])

    return lwa_result


def theta_lwa(
        ylat: np.ndarray, theta: np.ndarray, area: np.ndarray, dmu: np.ndarray,
        nlat_s: Optional[int] = None, n_points: Optional[int] = None,
        planet_radius: float = EARTH_RADIUS) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the surface wave activity B based on surface potential temperature.

    See Nakamura and Solomon (2010a) for details.

    Parameters
    ----------
    ylat : np.ndarray
        1D numpy array of latitude (in degree) with equal spacing in
        ascending order; dimension = (nlat,).
    theta : np.ndarray
        2D numpy array of surface potential temperature field;
        dimension = (nlat, nlon).
    area : np.ndarray
        2D numpy array specifying differential areal element of each grid
        point; dimension = (nlat, nlon).
    dmu : np.ndarray
        1D numpy array of latitudinal differential length element
        (e.g. dmu = planet_radius * cos(lat) d(lat)). Size = nlat.
    nlat_s : int, optional
        The index of grid point that defines the extent of hemispheric domain
        from the pole. Defaults to nlat // 2.
    n_points : int, optional
        Analysis resolution to calculate equivalent latitude.
        Defaults to nlat_s.
    planet_radius : float, optional
        Radius of spherical planet of interest consistent with input 'area'.
        Default is EARTH_RADIUS.

    Returns
    -------
    qref : np.ndarray
        1D numpy array of value reference potential temperature Theta(y)
        approximated by box counting method, where latitude y is given by ylat;
        dimension = (nlat,).
    lwa_result : np.ndarray
        2D numpy array of local surface wave activity values;
        dimension = (nlat, nlon).
    """
    nlat = theta.shape[0]
    nlon = theta.shape[1]
    if nlat_s is None:
        nlat_s = nlat // 2
    if n_points is None:
        n_points = nlat_s

    qref = np.zeros(nlat)
    lwa_result = np.zeros((nlat, nlon))

    # --- southern Hemisphere ---
    qref[:nlat_s], brac = basis.eqvlat_fawa(ylat[:nlat_s], theta[:nlat_s, :],
                                       area[:nlat_s, :], n_points,
                                       planet_radius=planet_radius)
    lwa_result[:nlat_s, :], dummy = basis.lwa(nlon, nlat_s, theta[:nlat_s, :],
                                              qref[:nlat_s], dmu[:nlat_s])

    # --- northern Hemisphere ---
    theta2 = theta[::-1, :]
    # Added the minus sign, but gotta see if NL_north is affected
    qref2, brac = basis.eqvlat_fawa(ylat[:nlat_s], theta2[:nlat_s, :],
                               area[:nlat_s, :],
                               n_points, planet_radius=planet_radius)
    qref[-nlat_s:] = qref2[::-1]
    lwa_result[-nlat_s:, :], dummy = basis.lwa(nlon, nlat_s,
                                               theta[-nlat_s:, :],
                                               qref[-nlat_s:],
                                               dmu[-nlat_s:])

    return qref, lwa_result
