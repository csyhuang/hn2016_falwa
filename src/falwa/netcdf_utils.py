"""
------------------------------------------
File name: netcdf_utils.py
Author: Clare Huang
"""
from typing import List
import xarray as xr


def extract_one_timeslice(nc_file: str, output_fname: str, timeslice: List[int] = [0]) -> None:
    """Extract specific snapshots (in time) from large netCDF files.

    The extracted snapshots are saved to another netCDF file specified by `output_fname`.

    Parameters
    ----------
    nc_file : str
        Path to the netCDF file that contains the time slice to be extracted.
    output_fname : str
        Filename of the output netCDF file.
    timeslice : list of int, optional
        Time index of slice to be extracted, by default [0].
    """
    dataset = xr.open_dataset(nc_file).isel(time=timeslice)
    print(f"Outputing {output_fname} with 1 timeslice.")
    dataset.to_netcdf(output_fname)
    print(f"Finished saving {output_fname} with 1 timeslice.")


if __name__ == "__main__":
    extract_one_timeslice(nc_file="cesm_mdtfv3_timeslice.U.1hr.nc", output_fname="cesm_1tslice.nc")
