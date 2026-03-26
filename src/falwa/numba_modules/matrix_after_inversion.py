"""
Numba implementation of matrix_after_inversion.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `matrix_after_inversion` for updating matrices after inversion
in the direct solver algorithm.

.. versionadded:: 2.4.0
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def _matrix_after_inversion_core(
    k: int,
    kmax: int,
    jd: int,
    qjj: np.ndarray,
    djj: np.ndarray,
    cjj: np.ndarray,
    rj: np.ndarray,
    sjk: np.ndarray,
    tjk: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core implementation of matrix_after_inversion.
    
    Updates sjk and tjk in place and returns them.
    """
    # Copy tj from tjk
    tj = np.zeros(jd - 2, dtype=np.float64)
    for i in range(jd - 2):
        tj[i] = tjk[i, k - 1]  # tjk(:,k) in Fortran -> tjk[:,k-1] in 0-indexed
    
    # Compute -Qk * Dk and store in sjk(:,:,k-1)
    xjj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
    for i in range(jd - 2):
        for j in range(jd - 2):
            xjj[i, j] = 0.0
            for kk in range(jd - 2):
                xjj[i, j] += qjj[i, kk] * djj[kk, j]
            sjk[i, j, k - 2] = -xjj[i, j]  # sjk(i,j,k-1) in Fortran -> sjk[i,j,k-2] in 0-indexed
    
    # Evaluate rk - Ck Tk
    yj = np.zeros(jd - 2, dtype=np.float64)
    for i in range(jd - 2):
        yj[i] = 0.0
        for kk in range(jd - 2):
            yj[i] += cjj[i, kk] * tj[kk]
        yj[i] = rj[i] - yj[i]
    
    # Evaluate Eq. 23: tj = Qk * yj
    for i in range(jd - 2):
        tj[i] = 0.0
        for kk in range(jd - 2):
            tj[i] += qjj[i, kk] * yj[kk]
        tjk[i, k - 2] = tj[i]  # tjk(i,k-1) in Fortran -> tjk[i,k-2] in 0-indexed
    
    return sjk, tjk


def matrix_after_inversion(
    k: int,
    qjj: np.ndarray,
    djj: np.ndarray,
    cjj: np.ndarray,
    rj: np.ndarray,
    sjk: np.ndarray,
    tjk: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update matrices after inversion in the direct solver algorithm.
    
    Parameters
    ----------
    k : int
        Current vertical level index (1-indexed as in Fortran).
    qjj : np.ndarray
        Inverted Q matrix, shape (jd-2, jd-2).
    djj : np.ndarray
        D matrix, shape (jd-2, jd-2).
    cjj : np.ndarray
        C matrix, shape (jd-2, jd-2).
    rj : np.ndarray
        R vector, shape (jd-2,).
    sjk : np.ndarray
        S matrix array, shape (jd-2, jd-2, kmax-1). Modified in place.
    tjk : np.ndarray
        T vector array, shape (jd-2, kmax-1). Modified in place.
        
    Returns
    -------
    sjk : np.ndarray
        Updated S matrix array, shape (jd-2, jd-2, kmax-1).
    tjk : np.ndarray
        Updated T vector array, shape (jd-2, kmax-1).
        
    Notes
    -----
    This function modifies sjk and tjk in place.
    """
    qjj = np.ascontiguousarray(qjj, dtype=np.float64)
    djj = np.ascontiguousarray(djj, dtype=np.float64)
    cjj = np.ascontiguousarray(cjj, dtype=np.float64)
    rj = np.ascontiguousarray(rj, dtype=np.float64)
    
    jd_minus_2 = qjj.shape[0]
    jd = jd_minus_2 + 2
    kmax = tjk.shape[1] + 1
    
    return _matrix_after_inversion_core(
        int(k), int(kmax), int(jd),
        qjj, djj, cjj, rj, sjk, tjk
    )

