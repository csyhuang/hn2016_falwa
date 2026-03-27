"""
Numba implementation of compute_lwa_only_nhn22.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `compute_lwa_only_nhn22` for computing LWA and barotropic averages
for the NHN22 algorithm.

.. versionadded:: 2.4.0

.. versionchanged:: 2.4.0
   Added explicit type signatures for eager compilation at import time.

Notes
-----
Arrays use C-order indexing:
- 3D arrays: [k, j, i] where k=height, j=latitude, i=longitude
- 2D lat-height arrays: [k, j]
- 2D lon-lat arrays: [j, i]
"""

import numpy as np
from numba import njit, float64, int64, boolean
from numba.core.types import Tuple as NbTuple
from typing import Tuple

# Type aliases for readability
f8 = float64
i8 = int64
b1 = boolean
f8_1d = float64[:]
f8_2d = float64[:, :]
f8_3d = float64[:, :, :]


@njit(NbTuple((f8_2d, f8_2d, f8_3d, f8_3d))(
    f8_3d, f8_3d, f8_2d,
    i8, i8, i8, i8, i8, b1,
    f8, f8, f8, f8, f8, f8, f8), cache=True)
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
    
    # Initialize output arrays (C-order: [j, i] for 2D, [k, j, i] for 3D)
    astarbaro = np.zeros((nd, imax), dtype=np.float64)
    ubaro = np.zeros((nd, imax), dtype=np.float64)
    astar1 = np.zeros((kmax, nd, imax), dtype=np.float64)
    astar2 = np.zeros((kmax, nd, imax), dtype=np.float64)
    
    # Working array (C-order: [j, i])
    qe = np.zeros((nd, imax), dtype=np.float64)
    
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
                
                if is_nhem:
                    phi0 = dp * float(j)  # j-1 in Fortran 1-indexed -> j in 0-indexed
                else:
                    phi0 = dp * float(j) - 0.5 * pi
                
                cor = 2.0 * om * np.sin(phi0)
                ab = a * dp
                
                for jj in range(nd):  # jj = 1 to nd in Fortran
                    if is_nhem:
                        phi1 = dp * float(jj)
                        qe[jj, i] = pv[k, jj + nd - 1, i] - qref[k, j]
                    else:
                        phi1 = dp * float(jj) - 0.5 * pi
                        qe[jj, i] = pv[k, jj, i] - qref[k, j]
                    
                    aa = a * dp * np.cos(phi1)
                    
                    if qe[jj, i] <= 0.0 and jj >= j:
                        if is_nhem:
                            astar2[k, j, i] = astar2[k, j, i] - qe[jj, i] * aa
                        else:
                            astar1[k, j, i] = astar1[k, j, i] - qe[jj, i] * aa
                    
                    if qe[jj, i] > 0.0 and jj < j:
                        if is_nhem:
                            astar1[k, j, i] = astar1[k, j, i] + qe[jj, i] * aa
                        else:
                            astar2[k, j, i] = astar2[k, j, i] + qe[jj, i] * aa
        
        # Column average: (25) of SI-HN17
        for j in range(nd):
            for i in range(imax):
                astarbaro[j, i] = astarbaro[j, i] + (astar1[k, j, i] + astar2[k, j, i]) * np.exp(-zk / h) * dc
        
        if is_nhem:
            for j in range(jstart, jend):
                for i in range(imax):
                    # Fortran: uu(:, nd-1+j, k) with j 1-indexed
                    # Python: j is 0-indexed, so use nd + j to get same array position
                    ubaro[j, i] = ubaro[j, i] + uu[k, nd + j, i] * np.exp(-zk / h) * dc
        else:
            for j in range(jstart, jend):
                for i in range(imax):
                    ubaro[j, i] = ubaro[j, i] + uu[k, j, i] * np.exp(-zk / h) * dc
    
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
        Potential vorticity, shape (kmax, jmax, imax).
    uu : np.ndarray
        Zonal wind, shape (kmax, jmax, imax).
    qref : np.ndarray
        Reference QGPV, shape (kmax, nd).
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
        Barotropic LWA, shape (nd, imax).
    ubaro : np.ndarray
        Barotropic zonal wind, shape (nd, imax).
    astar1 : np.ndarray
        Cyclonic LWA*cos(phi), shape (kmax, nd, imax).
    astar2 : np.ndarray
        Anticyclonic LWA*cos(phi), shape (kmax, nd, imax).
    """
    kmax, jmax, imax = pv.shape
    nd = qref.shape[1]
    
    return _compute_lwa_only_nhn22_core(
        pv, uu, qref,
        imax, jmax, kmax, nd, jb, is_nhem,
        float(a), float(om), float(dz), float(h), float(rr), float(cp), float(prefac)
    )
