"""
-------------------------------------------------------------------------------------------------------------------
File name: compute_qref_fawa.py
Author: Clare Huang
Created on: 2023/4/12
Description: This module contains functions that replace the f2py modules:
  - compute_qref_and_fawa_first.f90
  - part of compute_reference_states.f90
The main function in this module is:
  -
The helper functions are:
  -
-------------------------------------------------------------------------------------------------------------------
"""
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from hn2016_falwa.constant import P_GROUND, SCALE_HEIGHT, CP, DRY_GAS_CONSTANT, EARTH_RADIUS, EARTH_OMEGA
import numpy as np


def compute_qref_for_one_level(pv: np.ndarray, uu: np.ndarray, nnd: int, area: np.ndarray):
    """
    Compute qref for an isobaric surface GLOBALLY

    - nnd: put in nlat
    - area (jmax): a**2 (pi/float(jmax-1))**2  * cos(phi)

    """
    # *** Initialize arrays ***
    qn, an, cn = np.zeros(nnd), np.zeros(nnd), np.zeros(nnd)  # This is the bin

    # *** Find min-max bound of QGPV ***
    qmin, qmax = np.amax(pv), np.amin(pv)




