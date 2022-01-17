#This script reads in netCDF data for ERA5
from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plot
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#-----------------
# read netCDF files
#-----------------

data_dir = "grl2022_data/"
filename1 = data_dir + "2021_06_z.nc"
filename2 = data_dir + "2021_06_u.nc"
filename3 = data_dir + "2021_06_v.nc"

ncin1 = Dataset(filename1, 'r', format='NETCDF4')
ncin2 = Dataset(filename2, 'r', format='NETCDF4')
ncin3 = Dataset(filename3, 'r', format='NETCDF4')

zmean = ncin1.variables['z'] 
zmean = (np.array(zmean))
umean = ncin2.variables['u'] 
umean = (np.array(umean))
vmean = ncin3.variables['v'] 
vmean = (np.array(vmean))

print(zmean.shape)

zz = np.zeros((44,181,360))
z = np.zeros((181,360))
uu = np.zeros((44,181,360))
u = np.zeros((181,360))
vv = np.zeros((44,181,360))
v = np.zeros((181,360))
m = 76
while(m < 120):
    zz[m-76,:,:] = zmean[m,16,:,:]
    uu[m-76,:,:] = umean[m,16,:,:]
    vv[m-76,:,:] = vmean[m,16,:,:]
    m = m+1
day = ['00 UTC 20 June 2021','00 UTC 21 June 2021','00 UTC 22 June 2021','00 UTC 23 June 2021','00 UTC 24 June 2021','00 UTC 25 June 2021','00 UTC 26 June 2021','00 UTC 27 June 2021','00 UTC 28 June 2021','00 UTC 29 June 2021','00 UTC 30 June 2021']
b = ['0620.png','0621.png','0622.png','0623.png','0624.png','0625.png','0626.png','0627.png','0628.png','0629.png','0630.png']
n = 0
while(n < 11):
    nn = n*4
    j = 0
    while(j < 181):
        z[j,:]=zz[nn,180-j,:]/9.81
        u[j,:]=uu[nn,180-j,:]
        v[j,:]=vv[nn,180-j,:]
        j = j+1
    cl1 = np.arange(9600,11300,100)
    cl2 = np.arange(0,95,5)
    x = np.arange(0,360)
    y = np.arange(0,181)-90.
    plot.rcParams.update({'font.size':14,'text.usetex': False})
    fig = plot.figure(figsize=(8,4))
    ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plot.xlim(140,280) 
    plot.ylim(10,80) 
    plot.title(''+day[n]) 
    plot.xlabel('Longitude')
    plot.ylabel('Latitude') 
    ax5.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
    ax5.coastlines(color='white',alpha = 0.7)
    ax5.set_aspect('auto', adjustable=None)
    ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax5.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax5.xaxis.set_major_formatter(lon_formatter)
    ax5.yaxis.set_major_formatter(lat_formatter)
    ott = ax5.contourf(x,y,np.sqrt(u*u+v*v),levels=cl2,transform=ccrs.PlateCarree(),cmap='rainbow') 
    fig.colorbar(ott,ax=ax5,label='wind speed (m/s)')
    ott = ax5.contour(x,y,z,levels=cl1,colors='black',transform=ccrs.PlateCarree(),linewidths=1) 
    ax5.clabel(ott, ott.levels,fmt='%5i')
    plot.savefig(b[n],bbox_inches='tight',dpi =600)
    # plot.show()
    
    n = n+1