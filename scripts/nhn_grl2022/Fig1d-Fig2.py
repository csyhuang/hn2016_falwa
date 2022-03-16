# This is not used now. (March 16)
#This script reads in netCDF data for ERA5
from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plot
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#-----------------
# read netCDF files
#-----------------

data_dir = "grl2021_data/"
filename1 = data_dir + "2021_06_t.nc"

ncin1 = Dataset(filename1, 'r', format='NETCDF4')

tmean = ncin1.variables['t'] 
tmean = (np.array(tmean))


print(tmean.shape)

ttheta = np.zeros((44,37,360))
theta = np.zeros((37,360))
p = np.zeros(37)
z = np.zeros(37)
p[36] = 1000.
p[35] = 975.
p[34] = 950.
p[33] = 925.
p[32] = 900.
p[31] = 875.
p[30] = 850.
p[29] = 825.
p[28] = 800.
p[27] = 775.
p[26] = 750.
p[25] = 700.
p[24] = 650.
p[23] = 600.
p[22] = 550.
p[21] = 500.
p[20] = 450.
p[19] = 400.
p[18] = 350.
p[17] = 200.
p[16] = 250.
p[15] = 225.
p[14] = 200.
p[13] = 175.
p[12] = 150.
p[11] = 125.
p[10] = 100.
p[9] = 70.
p[8] = 50.
p[7] = 30.
p[6] = 20.
p[5] = 10.
p[4] = 7.
p[3] = 5.
p[2] = 3.
p[1] = 2.
p[0] = 1.

r = 287.
cp = 1004.
kappa = r/cp
m = 76
while(m < 120):
    k = 0
    while(k < 37):
        z[36-k] = -8000.*np.log(p[k]/1000.)  #pseudoheight
        ttheta[m-76,36-k,:] = tmean[m,k,41,:]
        ttheta[m-76,36-k,:] = ttheta[m-76,36-k,:]*np.power((1000./p[k]),kappa)  # potential temp
        k = k+1
    m = m+1
day = ['00 UTC 20 June 2021','00 UTC 21 June 2021','00 UTC 22 June 2021','00 UTC 23 June 2021','00 UTC 24 June 2021','00 UTC 25 June 2021','00 UTC 26 June 2021','00 UTC 27 June 2021','00 UTC 28 June 2021','00 UTC 29 June 2021','00 UTC 30 June 2021']
b = ['THETA_0620.png','THETA_0621.png','THETA_0622.png','THETA_0623.png','THETA_0624.png','THETA_0625.png','THETA_0626.png','THETA_0627.png','THETA_0628.png','THETA_0629.png','THETA_0630.png']
n = 0
while(n < 11):
    nn = n*4
    theta[:,:]=ttheta[nn,:,:]
    cl1 = np.arange(250,365,5)
    x = np.arange(0,360)
    plot.rcParams.update({'font.size':14,'text.usetex': False})
    fig = plot.figure(figsize=(8,4))
    ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plot.xlim(140,280) 
    plot.ylim(0,10) 
    plot.title('49$^\circ$N '+day[n]) 
    plot.xlabel('Longitude')
    plot.ylabel('pseudoheight (km)') 
    ax5.set_extent([-220, -80, 0, 10], ccrs.PlateCarree())
    ax5.set_aspect('auto', adjustable=None)
    ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax5.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax5.xaxis.set_major_formatter(lon_formatter)
    ott = ax5.contourf(x,z/1000.,theta,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow') 
    fig.colorbar(ott,ax=ax5,label='Kelvin')
    ott = ax5.contour(x,z/1000.,theta,levels=cl1,transform=ccrs.PlateCarree(),colors='black',linewidths=0.5)
    ott = ax5.contour(x,z/1000.,theta,levels=np.arange(320,325,5),transform=ccrs.PlateCarree(),colors='black',linewidths=1)
    ax5.clabel(ott, ott.levels,fmt='%5i')
    plot.savefig(b[n],bbox_inches='tight',dpi =600)
    n = n+1

plot.rcParams.update({'font.size': 16})
fig = plot.figure(figsize=(6,4))
ax5.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plot.title('49$^\circ$N 119$^\circ$W  00 UTC') 
plot.xlabel('Kelvin')
plot.ylabel('pseudoheight (km)') 
plot.xlim(290,360) 
plot.ylim(0,10) 

lcolor = np.array(['blue', 'red','green', 'black'])
lstyle = np.array(['dotted', 'dashed','dashdot', 'solid'])
lalpha = np.array([1,1,1,1])
n = 2
while(n < 6):
    thetaz = np.zeros(37)
    nn = n*8
    thetaz[:] = ttheta[nn,:,241]
    i = 0
    while(i < 37):
        if(z[i] < 1000.):
            thetaz[i] = np.nan
        i = i+1
    fig = plot.plot(thetaz,z/1000.,color=lcolor[n-2],linestyle = lstyle[n-2],alpha=lalpha[n-2])
    n = n+1
plot.savefig('t_profile.png',bbox_inches='tight',dpi =600)