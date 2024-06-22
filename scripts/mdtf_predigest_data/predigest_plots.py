import json
import datetime
import xarray as xr
import matplotlib.pyplot as plt
from falwa.xarrayinterface import QGDataset

# *** Path to data ***
with open('path.json', "r") as f:
    paths = json.load(f)
vol1_loc = paths['vol1']  # 1979-2018
vol2_loc = paths['vol2']  # 2019-2023

year = 2023
for month in range(1, 13):
    datapath = f"{vol2_loc}{year}_{month:02d}_[uvt].nc"
    data = xr.open_mfdataset(datapath)
    qgds = QGDataset(data)
    qgds.interpolate_fields(return_dataset=False)
    qgds.compute_reference_states(return_dataset=False)
    qgds.compute_lwa_and_barotropic_fluxes(return_dataset=False)

    # *** Output dataset to compute predigest plots ***
    output_ds = xr.Dataset({
        "uref": qgds.uref,
        "ubar": qgds.interpolated_u.mean(axis=-1),
        "u_baro": qgds.u_baro,
        "fawa": qgds.lwa.mean(axis=-1),
        "lwa_baro": qgds.lwa_baro})
    output_ds.to_netcdf(f"for_predigest_{year}_{month}.nc")
    print(f"{datetime.datetime.now()}: Finished output for_predigest_{year}_{month}.nc")



