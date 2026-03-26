"""
Numba implementation of compute_flux_dirinv_nshem.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `compute_flux_dirinv_nshem` for computing LWA and various flux terms
using direct inversion for either hemisphere.

.. versionadded:: 2.4.0
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
    
    # Initialize output arrays
    astar1 = np.zeros((imax, nd, kmax), dtype=np.float64)
    astar2 = np.zeros((imax, nd, kmax), dtype=np.float64)
    ncforce3d = np.zeros((imax, nd, kmax), dtype=np.float64)
    ua1 = np.zeros((imax, nd, kmax), dtype=np.float64)
    ua2 = np.zeros((imax, nd, kmax), dtype=np.float64)
    ep1 = np.zeros((imax, nd, kmax), dtype=np.float64)
    ep2 = np.zeros((imax, nd, kmax), dtype=np.float64)
    ep3 = np.zeros((imax, nd, kmax), dtype=np.float64)
    ep4 = np.zeros((imax, nd), dtype=np.float64)
    
    # Working arrays
    qe = np.zeros((imax, nd), dtype=np.float64)
    ue = np.zeros((imax, nd), dtype=np.float64)
    ncforce2d = np.zeros((imax, nd), dtype=np.float64)
    
    dc = dz / prefac
    
    # Bounds of y-indices in N/SHem
    if is_nhem:
        jstart = jb + 1  # 6 in Fortran 1-indexed -> jb in 0-indexed
        jend = nd - 1    # nd-1 in Fortran -> nd-2 in 0-indexed (exclusive)
    else:
        jstart = 2       # 2 in Fortran -> 1 in 0-indexed
        jend = nd - jb   # nd-jb in Fortran -> nd-jb-1 in 0-indexed (exclusive)
    
    # Main computation loop
    for k in range(1, kmax - 1):  # k = 2 to kmax-1 in Fortran
        zk = dz * float(k)
        
        for i in range(imax):
            for j in range(jstart, jend):  # Adjusted for 0-indexed
                astar1[i, j, k] = 0.0
                astar2[i, j, k] = 0.0
                ncforce3d[i, j, k] = 0.0
                ua2[i, j, k] = 0.0
                
                if is_nhem:
                    phi0 = dp * float(j)  # j-1 in Fortran with 1-indexed -> j in 0-indexed
                else:
                    phi0 = dp * float(j) - 0.5 * pi
                
                cor = 2.0 * om * np.sin(phi0)
                ab = a * dp
                
                for jj in range(nd):  # jj = 1 to nd in Fortran
                    if is_nhem:
                        phi1 = dp * float(jj)
                        qe[i, jj] = pv[i, jj + nd - 1, k] - qref[j, k]
                        ncforce2d[i, jj] = ncforce[i, jj + nd - 1, k]
                        ue[i, jj] = uu[i, jj + nd - 1, k] * np.cos(phi0) - uref[j - jb, k] * np.cos(phi1)
                    else:
                        phi1 = dp * float(jj) - 0.5 * pi
                        qe[i, jj] = pv[i, jj, k] - qref[j, k]
                        ncforce2d[i, jj] = ncforce[i, jj, k]
                        ue[i, jj] = uu[i, jj, k] * np.cos(phi0) - uref[j, k] * np.cos(phi1)
                    
                    aa = a * dp * np.cos(phi1)
                    
                    if qe[i, jj] <= 0.0 and jj >= j:
                        if is_nhem:
                            astar2[i, j, k] = astar2[i, j, k] - qe[i, jj] * aa
                        else:
                            astar1[i, j, k] = astar1[i, j, k] - qe[i, jj] * aa
                        ncforce3d[i, j, k] = ncforce3d[i, j, k] - ncforce2d[i, jj] * aa
                        ua2[i, j, k] = ua2[i, j, k] - qe[i, jj] * ue[i, jj] * ab
                    
                    if qe[i, jj] > 0.0 and jj < j:
                        if is_nhem:
                            astar1[i, j, k] = astar1[i, j, k] + qe[i, jj] * aa
                        else:
                            astar2[i, j, k] = astar2[i, j, k] + qe[i, jj] * aa
                        ncforce3d[i, j, k] = ncforce3d[i, j, k] + ncforce2d[i, jj] * aa
                        ua2[i, j, k] = ua2[i, j, k] + qe[i, jj] * ue[i, jj] * ab
                
                # Other fluxes
                if is_nhem:
                    ua1[i, j, k] = uref[j - jb, k] * (astar1[i, j, k] + astar2[i, j, k])
                    ep1[i, j, k] = -0.5 * (uu[i, j + nd - 1, k] - uref[j - jb, k]) ** 2
                    ep1[i, j, k] = ep1[i, j, k] + 0.5 * vv[i, j + nd - 1, k] ** 2
                    ep11 = 0.5 * (pt[i, j + nd - 1, k] - tref[j - jb, k]) ** 2
                else:
                    ua1[i, j, k] = uref[j, k] * (astar1[i, j, k] + astar2[i, j, k])
                    ep1[i, j, k] = -0.5 * (uu[i, j, k] - uref[j, k]) ** 2
                    ep1[i, j, k] = ep1[i, j, k] + 0.5 * vv[i, j, k] ** 2
                    ep11 = 0.5 * (pt[i, j, k] - tref[j, k]) ** 2
                
                zz = dz * float(k)
                ep11 = ep11 * (rr / h) * np.exp(-rkappa * zz / h)
                ep11 = ep11 * 2.0 * dz / (tg[k + 1] - tg[k - 1])
                ep1[i, j, k] = ep1[i, j, k] - ep11
                
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
                ep1[i, j, k] = ep1[i, j, k] * cos0
                
                # Meridional eddy momentum flux
                if is_nhem:
                    ep2[i, j, k] = (uu[i, j + nd, k] - uref[j - jb + 1, k]) * cosp * cosp * vv[i, j + nd, k]
                    ep3[i, j, k] = (uu[i, j + nd - 2, k] - uref[j - jb - 1, k]) * cosm * cosm * vv[i, j + nd - 2, k]
                else:
                    ep2[i, j, k] = (uu[i, j + 1, k] - uref[j + 1, k]) * cosp * cosp * vv[i, j + 1, k]
                    ep3[i, j, k] = (uu[i, j - 1, k] - uref[j - 1, k]) * cosm * cosm * vv[i, j - 1, k]
                
                # Low-level meridional eddy heat flux
                if k == 1:  # k=2 in Fortran
                    ep41 = 2.0 * om * sin0 * cos0 * dz / prefac
                    if is_nhem:
                        ep42 = np.exp(-dz / h) * vv[i, j + nd - 1, 1] * (pt[i, j + nd - 1, 1] - tref[j - jb, 1]) / (tg[2] - tg[0])
                        ep43 = vv[i, j + nd - 1, 0] * (pt[i, j + nd - 1, 0] - tref[j - jb, 0])
                    else:
                        ep42 = np.exp(-dz / h) * vv[i, j, 1] * (pt[i, j, 1] - tref[j, 1]) / (tg[2] - tg[0])
                        ep43 = vv[i, j, 0] * (pt[i, j, 0] - tref[j, 0])
                    ep43 = 0.5 * ep43 / (tg[1] - tg[0])
                    ep4[i, j] = ep41 * (ep42 + ep43)
            
            # Boundary values at jb
            phip = dp * float(jb)
            phi0_bc = dp * float(jb - 1)
            cosp = np.cos(phip)
            cos0 = np.cos(phi0_bc)
            ep2[i, jb, k] = (uu[i, nd + jb, k] - uref[1, k]) * cosp * cosp * vv[i, nd + jb, k]
            ep3[i, jb, k] = (uu[i, nd + jb - 1, k] - uref[0, k]) * cos0 * cos0 * vv[i, nd + jb - 1, k]
    
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
        Potential vorticity, shape (imax, jmax, kmax).
    uu : np.ndarray
        Zonal wind, shape (imax, jmax, kmax).
    vv : np.ndarray
        Meridional wind, shape (imax, jmax, kmax).
    pt : np.ndarray
        Potential temperature, shape (imax, jmax, kmax).
    ncforce : np.ndarray
        Non-conservative forcing, shape (imax, jmax, kmax).
    tn0 : np.ndarray
        Reference temperature profile, shape (kmax,).
    qref : np.ndarray
        Reference QGPV, shape (nd, kmax).
    uref : np.ndarray
        Reference zonal wind, shape (jd, kmax).
    tref : np.ndarray
        Reference temperature, shape (jd, kmax).
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
        Cyclonic LWA*cos(phi), shape (imax, nd, kmax).
    astar2 : np.ndarray
        Anticyclonic LWA*cos(phi), shape (imax, nd, kmax).
    ncforce3d : np.ndarray
        Non-conservative forcing 3D, shape (imax, nd, kmax).
    ua1 : np.ndarray
        Flux F1, shape (imax, nd, kmax).
    ua2 : np.ndarray
        Flux F2, shape (imax, nd, kmax).
    ep1 : np.ndarray
        Flux F3, shape (imax, nd, kmax).
    ep2 : np.ndarray
        Meridional momentum flux (north), shape (imax, nd, kmax).
    ep3 : np.ndarray
        Meridional momentum flux (south), shape (imax, nd, kmax).
    ep4 : np.ndarray
        Low-level heat flux, shape (imax, nd).
    """
    pv = np.ascontiguousarray(pv, dtype=np.float64)
    uu = np.ascontiguousarray(uu, dtype=np.float64)
    vv = np.ascontiguousarray(vv, dtype=np.float64)
    pt = np.ascontiguousarray(pt, dtype=np.float64)
    ncforce = np.ascontiguousarray(ncforce, dtype=np.float64)
    tn0 = np.ascontiguousarray(tn0, dtype=np.float64)
    qref = np.ascontiguousarray(qref, dtype=np.float64)
    uref = np.ascontiguousarray(uref, dtype=np.float64)
    tref = np.ascontiguousarray(tref, dtype=np.float64)
    
    imax, jmax, kmax = pv.shape
    nd = qref.shape[0]
    jd = uref.shape[0]
    
    return _compute_flux_dirinv_nshem_core(
        pv, uu, vv, pt, ncforce, tn0, qref, uref, tref,
        imax, jmax, kmax, nd, jb, jd, is_nhem,
        float(a), float(om), float(dz), float(h), float(rr), float(cp), float(prefac)
    )

