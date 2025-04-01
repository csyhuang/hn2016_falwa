"""
Create sample test data for integration test of ncforce computation
"""
import os
import xarray as xr


def extract_data_from_notebook_dir(dir: str, test_data_dir: str):
    for var in ['U', 'V', 'T', 'DTDTLWR']:
        ds_disk = xr.open_dataset(f"{dir}{var}.1980.daily.nc")
        field = ds_disk.isel(time=0).expand_dims(dim={"time": 1})
        if var == 'T':
            cached_time_var = field.time
        if var == 'DTDTLWR':
            field['time'] = cached_time_var
        output_fname = f"{test_data_dir}{var}.daily.merra.sample.nc"
        field.to_netcdf(output_fname)
        print(f"Successfully output {output_fname}")

original_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../notebooks/lubis_et_al_2024/MERRA2/"
test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/"

if __name__ == "__main__":
    extract_data_from_notebook_dir(dir=original_data_dir, test_data_dir=test_data_dir)