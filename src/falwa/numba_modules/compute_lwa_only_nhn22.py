"""
Numba implementation of compute_lwa_only_nhn22.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `compute_lwa_only_nhn22` for computing LWA and barotropic averages
for the NHN22 algorithm.

.. versionadded:: 2.4.0
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def _compute_lwa_only_nhn22_core(
    pv: np.ndarray,
    uu: np.ndarray,
    qref: np.ndarray,
    imax: int,
    jmax: int,
    kmax: int,
    nd: int,
    jb: int,
    is_nhem: bool,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float,
    prefac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core implementation of compute_lwa_only_nhn22.
    """
    pi = np.arccos(-1.0)
    dp = pi / float(jmax - 1)
    rkappa = rr / cp
    
    # Initialize height array
    z = np.zeros(kmax, dtype=np.float64)
    for k in range(kmax):
        z[k] = dz * float(k)
    
    # Initialize output arrays
    astarbaro = np.zeros((imax, nd), dtype=np.float64)
    ubaro = np.zeros((imax, nd), dtype=np.float64)
    astar1 = np.zeros((imax, nd, kmax), dtype=np.float64)
    astar2 = np.zeros((imax, nd, kmax), dtype=np.float64)
    
    # Working array
    qe = np.zeros((imax, nd), dtype=np.float64)
    
    dc = dz / prefac
    
    # Bounds of y-indices in N/SHem
    if is_nhem:
        jstart = jb + 1  # jb+1 in Fortran 1-indexed -> jb in 0-indexed
        jend = nd - 1    # nd-1 in Fortran -> nd-2 in 0-indexed (exclusive range)
    else:
        jstart = 2       # 2 in Fortran -> 1 in 0-indexed
        jend = nd - jb   # nd-jb in Fortran -> nd-jb-1 in 0-indexed (exclusive range)
    
    # Main computation loop
    for k in range(1, kmax - 1):  # k = 2 to kmax-1 in Fortran
        zk = dz * float(k)
        
        for i in range(imax):
            for j in range(jstart, jend):  # Adjusted for 0-indexed
                astar1[i, j, k] = 0.0
                astar2[i, j, k] = 0.0
                
                if is_nhem:
                    phi0 = dp * float(j)  # j-1 in Fortran 1-indexed -> j in 0-indexed
                else:
                    phi0 = dp * float(j) - 0.5 * pi
                
                cor = 2.0 * om * np.sin(phi0)
                ab = a * dp
                
                for jj in range(nd):  # jj = 1 to nd in Fortran
                    if is_nhem:
                        phi1 = dp * float(jj)
                        qe[i, jj] = pv[i, jj + nd - 1, k] - qref[j, k]
                    else:
                        phi1 = dp * float(jj) - 0.5 * pi
                        qe[i, jj] = pv[i, jj, k] - qref[j, k]
                    
                    aa = a * dp * np.cos(phi1)
                    
                    if qe[i, jj] <= 0.0 and jj >= j:
                        if is_nhem:
                            astar2[i, j, k] = astar2[i, j, k] - qe[i, jj] * aa
                        else:
                            astar1[i, j, k] = astar1[i, j, k] - qe[i, jj] * aa
                    
                    if qe[i, jj] > 0.0 and jj < j:
                        if is_nhem:
                            astar1[i, j, k] = astar1[i, j, k] + qe[i, jj] * aa
                        else:
                            astar2[i, j, k] = astar2[i, j, k] + qe[i, jj] * aa
        
        # Column average: (25) of SI-HN17
        for j in range(nd):
            for i in range(imax):
                astarbaro[i, j] = astarbaro[i, j] + (astar1[i, j, k] + astar2[i, j, k]) * np.exp(-zk / h) * dc
        
        if is_nhem:
            for j in range(jstart, jend):
                for i in range(imax):
                    ubaro[i, j] = ubaro[i, j] + uu[i, nd - 1 + j, k] * np.exp(-zk / h) * dc
        else:
            for j in range(jstart, jend):
                for i in range(imax):
                    ubaro[i, j] = ubaro[i, j] + uu[i, j, k] * np.exp(-zk / h) * dc
    
    return astarbaro, ubaro, astar1, astar2


def compute_lwa_only_nhn22(
    pv: np.ndarray,
    uu: np.ndarray,
    qref: np.ndarray,
    jb: int,
    is_nhem: bool,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float,
    prefac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LWA and barotropic averages for NHN22 algorithm.
    
    Parameters
    ----------
    pv : np.ndarray
        Potential vorticity, shape (imax, jmax, kmax).
    uu : np.ndarray
        Zonal wind, shape (imax, jmax, kmax).
    qref : np.ndarray
        Reference QGPV, shape (nd, kmax).
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
    astarbaro : np.ndarray
        Barotropic LWA, shape (imax, nd).
    ubaro : np.ndarray
        Barotropic zonal wind, shape (imax, nd).
    astar1 : np.ndarray
        Cyclonic LWA*cos(phi), shape (imax, nd, kmax).
    astar2 : np.ndarray
        Anticyclonic LWA*cos(phi), shape (imax, nd, kmax).
    """
    pv = np.ascontiguousarray(pv, dtype=np.float64)
    uu = np.ascontiguousarray(uu, dtype=np.float64)
    qref = np.ascontiguousarray(qref, dtype=np.float64)
    
    imax, jmax, kmax = pv.shape
    nd = qref.shape[0]
    
    return _compute_lwa_only_nhn22_core(
        pv, uu, qref,
        imax, jmax, kmax, nd, jb, is_nhem,
        float(a), float(om), float(dz), float(h), float(rr), float(cp), float(prefac)
    )

