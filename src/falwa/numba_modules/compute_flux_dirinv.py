"""
Numba implementation of compute_flux_dirinv_nshem.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `compute_flux_dirinv_nshem` for computing LWA and various flux terms
using direct inversion for either hemisphere.

.. versionadded:: 2.4.0

Notes
-----
Arrays use C-order indexing:
- 3D arrays: [k, j, i] where k=height, j=latitude, i=longitude
- 2D lat-height arrays: [k, j]
- 2D lon-lat arrays: [j, i]
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def _compute_flux_dirinv_nshem_core(
    pv: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    pt: np.ndarray,
    ncforce: np.ndarray,
    tn0: np.ndarray,
    qref: np.ndarray,
    uref: np.ndarray,
    tref: np.ndarray,
    imax: int,
    jmax: int,
    kmax: int,
    nd: int,
    jb: int,
    jd: int,
    is_nhem: bool,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float,
    prefac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core implementation of compute_flux_dirinv_nshem.
    """
    pi = np.arccos(-1.0)
    dp = pi / float(jmax - 1)
    rkappa = rr / cp
    
    # Initialize height array
    z = np.zeros(kmax, dtype=np.float64)
    for k in range(kmax):
        z[k] = dz * float(k)
    
    # Use tn0 as hemispheric-mean potential temperature
    tg = tn0.copy()
    
    # Initialize output arrays (C-order: [k, j, i] for 3D, [j, i] for 2D)
    astar1 = np.zeros((kmax, nd, imax), dtype=np.float64)
    astar2 = np.zeros((kmax, nd, imax), dtype=np.float64)
    ncforce3d = np.zeros((kmax, nd, imax), dtype=np.float64)
    ua1 = np.zeros((kmax, nd, imax), dtype=np.float64)
    ua2 = np.zeros((kmax, nd, imax), dtype=np.float64)
    ep1 = np.zeros((kmax, nd, imax), dtype=np.float64)
    ep2 = np.zeros((kmax, nd, imax), dtype=np.float64)
    ep3 = np.zeros((kmax, nd, imax), dtype=np.float64)
    ep4 = np.zeros((nd, imax), dtype=np.float64)
    
    # Working arrays (C-order: [j, i])
    qe = np.zeros((nd, imax), dtype=np.float64)
    ue = np.zeros((nd, imax), dtype=np.float64)
    ncforce2d = np.zeros((nd, imax), dtype=np.float64)
    
    dc = dz / prefac
    
    # Bounds of y-indices in N/SHem
    # Fortran uses 1-indexed, Python uses 0-indexed
    # Fortran do loop is inclusive, Python range is exclusive on upper bound
    if is_nhem:
        jstart = jb      # Fortran jb+1 (1-indexed) -> jb (0-indexed)
        jend = nd - 1    # Fortran nd-1 (1-indexed, inclusive) -> nd-1 (0-indexed, exclusive)
    else:
        jstart = 1       # Fortran 2 (1-indexed) -> 1 (0-indexed)
        jend = nd - jb   # Fortran nd-jb (1-indexed, inclusive) -> nd-jb (0-indexed, exclusive)
    
    # Main computation loop
    for k in range(1, kmax - 1):  # k = 2 to kmax-1 in Fortran
        zk = dz * float(k)
        
        for i in range(imax):
            for j in range(jstart, jend):  # Adjusted for 0-indexed
                astar1[k, j, i] = 0.0
                astar2[k, j, i] = 0.0
                ncforce3d[k, j, i] = 0.0
                ua2[k, j, i] = 0.0
                
                if is_nhem:
                    phi0 = dp * float(j)  # j-1 in Fortran with 1-indexed -> j in 0-indexed
                else:
                    phi0 = dp * float(j) - 0.5 * pi
                
                cor = 2.0 * om * np.sin(phi0)
                ab = a * dp
                
                for jj in range(nd):  # jj = 1 to nd in Fortran
                    if is_nhem:
                        phi1 = dp * float(jj)
                        qe[jj, i] = pv[k, jj + nd - 1, i] - qref[k, j]
                        ncforce2d[jj, i] = ncforce[k, jj + nd - 1, i]
                        ue[jj, i] = uu[k, jj + nd - 1, i] * np.cos(phi0) - uref[k, j - jb] * np.cos(phi1)
                    else:
                        phi1 = dp * float(jj) - 0.5 * pi
                        qe[jj, i] = pv[k, jj, i] - qref[k, j]
                        ncforce2d[jj, i] = ncforce[k, jj, i]
                        ue[jj, i] = uu[k, jj, i] * np.cos(phi0) - uref[k, j] * np.cos(phi1)
                    
                    aa = a * dp * np.cos(phi1)
                    
                    if qe[jj, i] <= 0.0 and jj >= j:
                        if is_nhem:
                            astar2[k, j, i] = astar2[k, j, i] - qe[jj, i] * aa
                        else:
                            astar1[k, j, i] = astar1[k, j, i] - qe[jj, i] * aa
                        ncforce3d[k, j, i] = ncforce3d[k, j, i] - ncforce2d[jj, i] * aa
                        ua2[k, j, i] = ua2[k, j, i] - qe[jj, i] * ue[jj, i] * ab
                    
                    if qe[jj, i] > 0.0 and jj < j:
                        if is_nhem:
                            astar1[k, j, i] = astar1[k, j, i] + qe[jj, i] * aa
                        else:
                            astar2[k, j, i] = astar2[k, j, i] + qe[jj, i] * aa
                        ncforce3d[k, j, i] = ncforce3d[k, j, i] + ncforce2d[jj, i] * aa
                        ua2[k, j, i] = ua2[k, j, i] + qe[jj, i] * ue[jj, i] * ab
                
                # Other fluxes
                if is_nhem:
                    ua1[k, j, i] = uref[k, j - jb] * (astar1[k, j, i] + astar2[k, j, i])
                    ep1[k, j, i] = -0.5 * (uu[k, j + nd - 1, i] - uref[k, j - jb]) ** 2
                    ep1[k, j, i] = ep1[k, j, i] + 0.5 * vv[k, j + nd - 1, i] ** 2
                    ep11 = 0.5 * (pt[k, j + nd - 1, i] - tref[k, j - jb]) ** 2
                else:
                    ua1[k, j, i] = uref[k, j] * (astar1[k, j, i] + astar2[k, j, i])
                    ep1[k, j, i] = -0.5 * (uu[k, j, i] - uref[k, j]) ** 2
                    ep1[k, j, i] = ep1[k, j, i] + 0.5 * vv[k, j, i] ** 2
                    ep11 = 0.5 * (pt[k, j, i] - tref[k, j]) ** 2
                
                zz = dz * float(k)
                ep11 = ep11 * (rr / h) * np.exp(-rkappa * zz / h)
                ep11 = ep11 * 2.0 * dz / (tg[k + 1] - tg[k - 1])
                ep1[k, j, i] = ep1[k, j, i] - ep11
                
                if is_nhem:
                    phip = dp * float(j + 1)
                    phi0_local = dp * float(j)
                    phim = dp * float(j - 1)
                else:
                    phip = dp * float(j + 1) - 0.5 * pi
                    phi0_local = dp * float(j) - 0.5 * pi
                    phim = dp * float(j - 1) - 0.5 * pi
                
                cosp = np.cos(phip)
                cos0 = np.cos(phi0_local)
                cosm = np.cos(phim)
                sin0 = np.sin(phi0_local)
                ep1[k, j, i] = ep1[k, j, i] * cos0
                
                # Meridional eddy momentum flux
                if is_nhem:
                    ep2[k, j, i] = (uu[k, j + nd, i] - uref[k, j - jb + 1]) * cosp * cosp * vv[k, j + nd, i]
                    ep3[k, j, i] = (uu[k, j + nd - 2, i] - uref[k, j - jb - 1]) * cosm * cosm * vv[k, j + nd - 2, i]
                else:
                    ep2[k, j, i] = (uu[k, j + 1, i] - uref[k, j + 1]) * cosp * cosp * vv[k, j + 1, i]
                    ep3[k, j, i] = (uu[k, j - 1, i] - uref[k, j - 1]) * cosm * cosm * vv[k, j - 1, i]
                
                # Low-level meridional eddy heat flux
                if k == 1:  # k=2 in Fortran
                    ep41 = 2.0 * om * sin0 * cos0 * dz / prefac
                    if is_nhem:
                        ep42 = np.exp(-dz / h) * vv[1, j + nd - 1, i] * (pt[1, j + nd - 1, i] - tref[1, j - jb]) / (tg[2] - tg[0])
                        ep43 = vv[0, j + nd - 1, i] * (pt[0, j + nd - 1, i] - tref[0, j - jb])
                    else:
                        ep42 = np.exp(-dz / h) * vv[1, j, i] * (pt[1, j, i] - tref[1, j]) / (tg[2] - tg[0])
                        ep43 = vv[0, j, i] * (pt[0, j, i] - tref[0, j])
                    ep43 = 0.5 * ep43 / (tg[1] - tg[0])
                    ep4[j, i] = ep41 * (ep42 + ep43)
            
            # Boundary values at jb
            phip = dp * float(jb)
            phi0_bc = dp * float(jb - 1)
            cosp = np.cos(phip)
            cos0 = np.cos(phi0_bc)
            ep2[k, jb, i] = (uu[k, nd + jb, i] - uref[k, 1]) * cosp * cosp * vv[k, nd + jb, i]
            ep3[k, jb, i] = (uu[k, nd + jb - 1, i] - uref[k, 0]) * cos0 * cos0 * vv[k, nd + jb - 1, i]
    
    return astar1, astar2, ncforce3d, ua1, ua2, ep1, ep2, ep3, ep4


def compute_flux_dirinv_nshem(
    pv: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    pt: np.ndarray,
    ncforce: np.ndarray,
    tn0: np.ndarray,
    qref: np.ndarray,
    uref: np.ndarray,
    tref: np.ndarray,
    jb: int,
    is_nhem: bool,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float,
    prefac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LWA and various flux terms using direct inversion.
    
    Parameters
    ----------
    pv : np.ndarray
        Potential vorticity, shape (kmax, jmax, imax).
    uu : np.ndarray
        Zonal wind, shape (kmax, jmax, imax).
    vv : np.ndarray
        Meridional wind, shape (kmax, jmax, imax).
    pt : np.ndarray
        Potential temperature, shape (kmax, jmax, imax).
    ncforce : np.ndarray
        Non-conservative forcing, shape (kmax, jmax, imax).
    tn0 : np.ndarray
        Reference temperature profile, shape (kmax,).
    qref : np.ndarray
        Reference QGPV, shape (kmax, nd).
    uref : np.ndarray
        Reference zonal wind, shape (kmax, jd).
    tref : np.ndarray
        Reference temperature, shape (kmax, jd).
    jb : int
        Lower bounding latitude index.
    is_nhem : bool
        True for Northern Hemisphere, False for Southern.
    a : float
        Earth radius.
    om : float
        Earth rotation rate.
    dz : float
        Vertical grid spacing.
    h : float
        Scale height.
    rr : float
        Gas constant.
    cp : float
        Specific heat.
    prefac : float
        Prefactor for vertical averaging.
        
    Returns
    -------
    astar1 : np.ndarray
        Cyclonic LWA*cos(phi), shape (kmax, nd, imax).
    astar2 : np.ndarray
        Anticyclonic LWA*cos(phi), shape (kmax, nd, imax).
    ncforce3d : np.ndarray
        Non-conservative forcing 3D, shape (kmax, nd, imax).
    ua1 : np.ndarray
        Flux F1, shape (kmax, nd, imax).
    ua2 : np.ndarray
        Flux F2, shape (kmax, nd, imax).
    ep1 : np.ndarray
        Flux F3, shape (kmax, nd, imax).
    ep2 : np.ndarray
        Meridional momentum flux (north), shape (kmax, nd, imax).
    ep3 : np.ndarray
        Meridional momentum flux (south), shape (kmax, nd, imax).
    ep4 : np.ndarray
        Low-level heat flux, shape (nd, imax).
    """
    ncforce = np.ascontiguousarray(ncforce, dtype=np.float64)

    kmax, jmax, imax = pv.shape
    nd = qref.shape[1]
    jd = uref.shape[1]
    
    return _compute_flux_dirinv_nshem_core(
        pv, uu, vv, pt, ncforce, tn0, qref, uref, tref,
        imax, jmax, kmax, nd, jb, jd, is_nhem,
        float(a), float(om), float(dz), float(h), float(rr), float(cp), float(prefac)
    )
