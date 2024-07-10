import os
import xarray as xr
import matplotlib.pyplot as plt
from falwa.stat_utils import calculate_covariance


data_path = f"{os.environ['HOME']}/Dropbox/GitHub/hn2016_falwa/github_data_storage/predigest/output_2023_01.nc"

df = xr.open_mfdataset(data_path, decode_times=False)
print("Start computing covariance")
output = calculate_covariance(df.variables['lwa_baro'].data, df.variables['u_baro'])
print("Finished computing covariance")
plt.contourf(df.coords['longitude'], df.coords['latitude'].data, output)
plt.colorbar()
plt.show()
