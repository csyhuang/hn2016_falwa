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
from .compute_qref_and_fawa_first import compute_qref_and_fawa_first
from .matrix_b4_inversion import matrix_b4_inversion
from .matrix_after_inversion import matrix_after_inversion
from .upward_sweep import upward_sweep
from .compute_flux_dirinv import compute_flux_dirinv_nshem
from .compute_lwa_only_nhn22 import compute_lwa_only_nhn22

__all__ = [
    "compute_qgpv",
    "compute_qgpv_direct_inv",
    "compute_reference_states",
    "compute_qref_and_fawa_first",
    "matrix_b4_inversion",
    "matrix_after_inversion",
    "upward_sweep",
    "compute_flux_dirinv_nshem",
    "compute_lwa_only_nhn22",
]

