"""
------------------------------------------
File name: preprocessing.py
Author: Clare Huang
"""
import numpy as np
import gridfill


def gridfill_each_level(lat_lon_field: np.ndarray, itermax: int = 1000, verbose: bool = False) -> np.ndarray:
    """Fill missing values in a lat-lon grid.

    The filling is done by solving Poisson's equation using a relaxation scheme.

    Parameters
    ----------
    lat_lon_field : np.ndarray
        2D array to apply gridfill on.
    itermax : int, optional
        Maximum iteration for the Poisson solver, by default 1000.
    verbose : bool, optional
        Verbose level of the Poisson solver, by default False.

    Returns
    -------
    np.ndarray
        A 2D array of the same dimension with all NaNs filled.
    """
    if np.isnan(lat_lon_field).sum() == 0:
        return lat_lon_field

    lat_lon_filled, converged = gridfill.fill(
        grids=np.ma.masked_invalid(lat_lon_field), xdim=1, ydim=0, eps=0.01,
        cyclic=True, itermax=itermax, verbose=verbose)

    return lat_lon_filled
