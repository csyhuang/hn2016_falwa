"""
------------------------------------------
File name: netcdf_utils.py
Author: Clare Huang
"""
import xarray as xr


def extract_one_timeslice(nc_file: str, output_fname: str, timeslice : int = 0) -> None:
    """
    Parameters
    ----------
    nc_file : str
        netCDF file that contains the time slice to be extracted
    output_fname : str
        filename of the netCDF file that contains the time slice extracted

    Returns
    -------

    """
    dataset = xr.open_dataset(nc_file).isel(time=timeslice)
    print(f"Outputing {output_fname} with 1 timeslice.")
    dataset.to_netcdf(output_fname)
    print(f"Finished saving {output_fname} with 1 timeslice.")



