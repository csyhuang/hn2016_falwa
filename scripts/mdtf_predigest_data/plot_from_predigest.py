import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


df = xr.open_dataset(
    f"{os.environ['HOME']}/Dropbox/GitHub/hn2016_falwa/github_data_storage/predigest/output_2023_01.nc",
    decode_times=False)
lat = df.coords['latitude']
lon = df.coords['longitude']
height = np.arange(0, 49000, 1000)

lwa_baro = df.variables['lwa_baro'].data
plt.contourf(lon, lat, lwa_baro.mean(axis=0))
plt.title('lwa_baro')
plt.colorbar()
plt.show()

u_baro = df.variables['u_baro'].data
plt.contourf(lon, lat, u_baro.mean(axis=0))
plt.title('u_baro')
plt.colorbar()
plt.show()

uref = df.variables['uref'].data
plt.contourf(lat, height, uref.mean(axis=0))
plt.title('uref')
plt.colorbar()
plt.show()

ubar = df.variables['ubar'].data
plt.contourf(lat, height, ubar.mean(axis=0))
plt.title('ubar')
plt.colorbar()
plt.show()

du = df.variables['uref'].data - df.variables['ubar'].data
plt.contourf(lat, height, du.mean(axis=0))
plt.title('Du')
plt.colorbar()
plt.show()

print("d")