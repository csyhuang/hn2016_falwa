"""
Numba implementation of matrix_after_inversion.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `matrix_after_inversion` for updating matrices after inversion
in the direct solver algorithm.

.. versionadded:: 2.4.0

.. versionchanged:: 2.4.0
   Added explicit type signatures for eager compilation at import time.

Notes
-----
Arrays use C-order indexing:
- 2D arrays: [k, j] for lat-height
- 3D arrays: [k, j1, j2]
"""

import numpy as np
from numba import njit, float64, int64
from numba.core.types import Tuple as NbTuple
from typing import Tuple

# Type aliases for readability
f8 = float64
i8 = int64
f8_1d = float64[:]
f8_2d = float64[:, :]
f8_3d = float64[:, :, :]


@njit(NbTuple((f8_3d, f8_2d))(
    i8, i8, i8,
    f8_2d, f8_2d, f8_2d, f8_1d, f8_3d, f8_2d), cache=True)
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
    # Copy tj from tjk (C-order: tjk[k, j])
    tj = np.zeros(jd - 2, dtype=np.float64)
    for i in range(jd - 2):
        tj[i] = tjk[k - 1, i]
    
    # Compute -Qk * Dk and store in sjk[k-2, :, :] (C-order: sjk[k, j1, j2])
    xjj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
    for i in range(jd - 2):
        for j in range(jd - 2):
            xjj[i, j] = 0.0
            for kk in range(jd - 2):
                xjj[i, j] += qjj[i, kk] * djj[kk, j]
            sjk[k - 2, i, j] = -xjj[i, j]
    
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
        tjk[k - 2, i] = tj[i]
    
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
        S matrix array, shape (kmax-1, jd-2, jd-2). Modified in place.
    tjk : np.ndarray
        T vector array, shape (kmax-1, jd-2). Modified in place.
        
    Returns
    -------
    sjk : np.ndarray
        Updated S matrix array, shape (kmax-1, jd-2, jd-2).
    tjk : np.ndarray
        Updated T vector array, shape (kmax-1, jd-2).
        
    Notes
    -----
    This function modifies sjk and tjk in place.
    """
    jd_minus_2 = qjj.shape[0]
    jd = jd_minus_2 + 2
    kmax = tjk.shape[0] + 1
    
    return _matrix_after_inversion_core(
        int(k), int(kmax), int(jd),
        qjj, djj, cjj, rj, sjk, tjk
    )
