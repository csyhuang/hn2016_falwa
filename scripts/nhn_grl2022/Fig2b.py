#This script reads in netCDF data for ERA5  ==> Fig2b
from netCDF4 import Dataset 
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plot
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#-----------------
# read netCDF files
#-----------------
data_dir = "grl2021_data/"
filename3 = data_dir + "2021_06_str.nc"
filename5 = data_dir + "2021_06_ssr.nc"
filename6 = data_dir + "2021_06_sshf.nc"
filename7 = data_dir + "2021_06_slhf.nc"


ncin3 = Dataset(filename3, 'r', format='NETCDF4')
ncin5 = Dataset(filename5, 'r', format='NETCDF4')
ncin6 = Dataset(filename6, 'r', format='NETCDF4')
ncin7 = Dataset(filename7, 'r', format='NETCDF4')


irt = ncin3.variables["str"]
srad = ncin5.variables["ssr"]
shff = ncin6.variables["sshf"]
lhff = ncin7.variables["slhf"]

irtt = np.zeros(720)
ssrad = np.zeros(720)
shf = np.zeros(720)
lhf = np.zeros(720)
zr = np.zeros(720)

irtt[:] = irt[:,41,241]/3600.
ssrad[:] = srad[:,41,241]/3600.
shf[:] = shff[:,41,241]/3600.
lhf[:] = lhff[:,41,241]/3600.
zr[:] = 0.

print()

x = np.arange(0,720)/24.+1

plot.rcParams.update({'font.size': 16})
fig = plot.figure(figsize=(8,4))
plot.title('Surface Heat Fluxes at 49$^\circ$N 119$^\circ$W') 
plot.xlabel('Day')
plot.ylabel('Flux (Wm$^{-2}$)') 
plot.xlim(20,31) 
plot.ylim(-800,1000) 
fig = plot.plot(x,ssrad,color='red')
fig = plot.plot(x,irtt,'r--')
fig = plot.plot(x,shf,color='blue')
fig = plot.plot(x,lhf,'b--')
plot.savefig('surface_F.png',bbox_inches='tight',dpi =600)

