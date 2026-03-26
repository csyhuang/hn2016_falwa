"""
Numba-based implementations of FALWA computational modules.

This module provides JIT-compiled Python implementations of the Fortran
subroutines previously accessed via F2PY. These implementations use Numba
for near-Fortran performance without requiring a Fortran compiler.

.. versionadded:: 2.4.0
"""

from .compute_qgpv import compute_qgpv
from .compute_qgpv_direct_inv import compute_qgpv_direct_inv
from .compute_reference_states import compute_reference_states

__all__ = [
    "compute_qgpv",
    "compute_qgpv_direct_inv",
    "compute_reference_states",
]

