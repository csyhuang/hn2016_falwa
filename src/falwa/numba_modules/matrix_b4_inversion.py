"""
Numba implementation of matrix_b4_inversion.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `matrix_b4_inversion` for setting up matrices before inversion
in the direct solver algorithm.

.. versionadded:: 2.4.0

Notes
-----
Arrays use C-order indexing:
- 2D lat-height arrays: [k, j]
- 3D arrays: [k, j1, j2]
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def _matrix_b4_inversion_core(
    k: int,
    jmax: int,
    kmax: int,
    nd: int,
    jb: int,
    jd: int,
    z: np.ndarray,
    statn: np.ndarray,
    qref: np.ndarray,
    ckref: np.ndarray,
    sjk: np.ndarray,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core implementation of matrix_b4_inversion.
    """
    rkappa = rr / cp
    pi = np.arccos(-1.0)
    dp = pi / float(jmax - 1)
    
    zp = 0.5 * (z[k] + z[k - 1])
    zm = 0.5 * (z[k - 2] + z[k - 1])
    statp = 0.5 * (statn[k] + statn[k - 1])
    statm = 0.5 * (statn[k - 2] + statn[k - 1])
    
    # Initialize output arrays
    cjj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
    djj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
    qjj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
    rj = np.zeros(jd - 2, dtype=np.float64)
    
    # Copy sjk slice (C-order: sjk[k, j1, j2])
    sjj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
    for i in range(jd - 2):
        for j in range(jd - 2):
            sjj[i, j] = sjk[k - 1, i, j]
    
    # Initialize u array for boundary conditions (C-order: [k, j])
    u = np.zeros((kmax, jd), dtype=np.float64)
    
    # Fortran: do jj = jb+2, (nd-1)
    for jj in range(jb + 2, nd):
        j = jj - jb
        phi0 = float(jj - 1) * dp
        phip = (float(jj) - 0.5) * dp
        phim = (float(jj) - 1.5) * dp
        cos0 = np.cos(phi0)
        cosp = np.cos(phip)
        cosm = np.cos(phim)
        sin0 = np.sin(phi0)
        sinp = np.sin(phip)
        sinm = np.sin(phim)
        
        fact = 4.0 * om * om * h * a * a * sin0 * dp * dp / (dz * dz * rr * cos0)
        amp = np.exp(-zp / h) * np.exp(rkappa * zp / h) / statp
        amp = amp * fact * np.exp(z[k - 1] / h)
        amm = np.exp(-zm / h) * np.exp(rkappa * zm / h) / statm
        amm = amm * fact * np.exp(z[k - 1] / h)
        
        # Specify A, B, C, D, E, F (Eqs. 4-9)
        ajk = 1.0 / (sinp * cosp)
        bjk = 1.0 / (sinm * cosm)
        cjk = amp
        djk = amm
        ejk = ajk + bjk + cjk + djk
        fjk = -0.5 * a * dp * (qref[k - 1, jj] - qref[k - 1, jj - 2])  # C-order: qref[k, j]
        
        # North-south boundary conditions
        u[k - 1, jd - 1] = 0.0
        phi0_bc = dp * float(jb)
        u[k - 1, 0] = ckref[k - 1, jb] / (2.0 * pi * a) - om * a * np.cos(phi0_bc)
        
        rj[j - 2] = fjk
        if j == 2:
            rj[j - 2] = fjk - bjk * u[k - 1, 0]
        if j == jd - 1:
            rj[j - 2] = fjk - ajk * u[k - 1, jd - 1]
        
        # Specify Ck & Dk (Eqs. 18-19)
        cjj[j - 2, j - 2] = cjk
        djj[j - 2, j - 2] = djk
        
        # Specify Qk (Eq. 17)
        qjj[j - 2, j - 2] = -ejk
        if j - 1 >= 1 and j - 1 < jd - 2:
            qjj[j - 2, j - 1] = ajk
        if j - 1 > 1 and j - 1 <= jd - 2:
            qjj[j - 2, j - 3] = bjk
    
    # Compute Qk + Ck Sk
    xjj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
    for i in range(jd - 2):
        for j in range(jd - 2):
            xjj[i, j] = 0.0
            for kk in range(jd - 2):
                xjj[i, j] += cjj[i, kk] * sjj[kk, j]
            qjj[i, j] = qjj[i, j] + xjj[i, j]
    
    return qjj, djj, cjj, rj


def matrix_b4_inversion(
    k: int,
    jmax: int,
    jb: int,
    jd: int,
    z: np.ndarray,
    statn: np.ndarray,
    qref: np.ndarray,
    ckref: np.ndarray,
    sjk: np.ndarray,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up matrices before inversion in the direct solver algorithm.
    
    Parameters
    ----------
    k : int
        Current vertical level index (1-indexed as in Fortran).
    jmax : int
        Number of latitude points.
    jb : int
        Lower bounding latitude index.
    jd : int
        nd - lower bounding latitude.
    z : np.ndarray
        Height levels, shape (kmax,).
    statn : np.ndarray
        Static stability, shape (kmax,).
    qref : np.ndarray
        Reference QGPV, shape (kmax, nd).
    ckref : np.ndarray
        Reference Kelvin circulation, shape (kmax, nd).
    sjk : np.ndarray
        S matrix from previous iteration, shape (kmax-1, jd-2, jd-2).
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
        
    Returns
    -------
    qjj : np.ndarray
        Q matrix, shape (jd-2, jd-2).
    djj : np.ndarray
        D matrix, shape (jd-2, jd-2).
    cjj : np.ndarray
        C matrix, shape (jd-2, jd-2).
    rj : np.ndarray
        R vector, shape (jd-2,).
    """
    kmax, nd = qref.shape
    
    return _matrix_b4_inversion_core(
        int(k), int(jmax), int(kmax), int(nd), int(jb), int(jd),
        z, statn, qref, ckref, sjk,
        float(a), float(om), float(dz), float(h), float(rr), float(cp)
    )
