## This script reads in netCDF LWAb data for ERA5
from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plot

#-----------------
# read netCDF files
#-----------------

data_dir = "grl2022_data/"
filename1 = data_dir + "2021_06_LWAb_N.nc"

ncin1 = Dataset(filename1, 'r', format='NETCDF4')

lwa = ncin1.variables['lwa'] 
lwa = (np.array(lwa))

print(lwa.shape)

z = np.zeros((120,360))
m = 0
while(m < 120):
    z[m,:] = lwa[m,49,:]   # 49N LWA for June 1-30
    m = m+1

zs = np.zeros((120,360))  # smoothed z #

#### smoothing in longitude ####
n = 5     # smoothing width #
m = 0
while(m < 120):
    zx = np.zeros(360)
    zx[:] = z[m,:]
    nn = -n
    while(nn < n+1):
        zy = np.roll(zx,nn)
        zs[m,:] = zs[m,:] + zy[:]/(2*n+1)
        nn = nn+1   
    m = m+1
    
zx[:] = zs[100,:]-zs[76,:]
zy[:] = 0


filename1 = data_dir + "2021_06_ua1_N.nc"
filename2 = data_dir + "2021_06_ua2_N.nc"
filename3 = data_dir + "2021_06_ep1_N.nc"
filename4 = data_dir + "2021_06_ep2_N.nc"
filename5 = data_dir + "2021_06_ep3_N.nc"
filename6 = data_dir + "2021_06_ep4_N.nc"

ncin1 = Dataset(filename1, 'r', format='NETCDF4')
ua1 = ncin1.variables['ua1'] 
ua1 = (np.array(ua1))
ncin2 = Dataset(filename2, 'r', format='NETCDF4')
ua2 = ncin2.variables['ua2'] 
ua2 = (np.array(ua2))
ncin3 = Dataset(filename3, 'r', format='NETCDF4')
ep1 = ncin3.variables['ep1'] 
ep1 = (np.array(ep1))
ncin4 = Dataset(filename4, 'r', format='NETCDF4')
ep2 = ncin4.variables['ep2'] 
ep2 = (np.array(ep2))
ncin5 = Dataset(filename5, 'r', format='NETCDF4')
ep3 = ncin5.variables['ep3'] 
ep3 = (np.array(ep3))
ncin6 = Dataset(filename6, 'r', format='NETCDF4')
ep4 = ncin6.variables['ep4'] 
ep4 = (np.array(ep4))

f1 = np.zeros((120,360))
f2 = np.zeros((120,360))
f11 = np.zeros((120,360))
f22 = np.zeros((120,360))

z1 = np.zeros((120,360))
z2 = np.zeros((120,360))
z3 = np.zeros((120,360))
dt = 3600.*6.
a = 6378000.
dl = 2.*np.pi/360.
dp = 2.*np.pi/360.
m = 0              # m = 76 is 20 June 2021 00 UTC
while(m < 120):     # m = 100 is 26 June 2021 00 UTC
    z3[m,:] = ep4[m,49,:]
    f1[m,:] = ua1[m,49,:]+ua2[m,49,:]+ep1[m,49,:]
    j = 49
    phi = dp*j
    const = 1./(2.*a*np.cos(phi)*dl)
    z2[m,:]=const*(ep2[m,49,:]-ep3[m,49,:])
    f2[m,:] = 0.5*(ep2[m,49,:]+ep3[m,49,:])/np.cos(phi)
    i = 1
    while(i < 359): 
        z1[m,i] = -const*(f1[m,i+1]-f1[m,i-1])
        i = i+1
    z1[m,0] = -const*(f1[m,1]-f1[m,359])
    z1[m,359] = -const*(f1[m,0]-f1[m,358])
    m = m+1
    
z1s = np.zeros((120,360))  # smoothed z1 #
z2s = np.zeros((120,360))  # smoothed z2 #
z3s = np.zeros((120,360))  # smoothed z3 #
f1s = np.zeros((120,360))  # smoothed f1 #

#### smoothing in longitude ####
m = 0
while(m < 120):
    z1x = np.zeros(360)
    z1x[:] = z1[m,:]
    z2x = np.zeros(360)
    z2x[:] = z2[m,:]
    z3x = np.zeros(360)
    z3x[:] = z3[m,:]
    f1x = np.zeros(360)
    f1x[:] = f1[m,:]

    nn = -n
    while(nn < n+1):
        z1y = np.roll(z1x,nn)
        z1s[m,:] = z1s[m,:] + z1y[:]/(2*n+1)
        z2y = np.roll(z2x,nn)
        z2s[m,:] = z2s[m,:] + z2y[:]/(2*n+1)
        z3y = np.roll(z3x,nn)
        z3s[m,:] = z3s[m,:] + z3y[:]/(2*n+1)
        f1y = np.roll(f1x,nn)
        f1s[m,:] = f1s[m,:] + f1y[:]/(2*n+1)
        nn = nn+1   
    m = m+1

##### Time integration test ######

lwa = np.zeros(360)
res = np.zeros((120,360))
dlwa = np.zeros((120,360))
gama = np.zeros((120,360))
gama1 = np.zeros((120,360))
gama2 = np.zeros((120,360))
cgx = np.zeros((120,360))

m = 0
while(m < 119):
    dlwa[m,:] = zs[m+1,:]-zs[m,:]
    cgx[m,:] = (f1s[m,:]+f1s[m+1,:])/(zs[m,:]+zs[m+1,:])
    res[m,:] = (dlwa[m,:] - 0.5*dt*(z1s[m,:]+z1s[m+1,:]+z2s[m,:]+z2s[m+1,:]+z3s[m,:]+z3s[m+1,:]))/dt
    gama[m,:] = 2.*res[m,:]/(zs[m,:]+zs[m+1,:])
    m = m+1

m = 76
while(m < 100):
    lwa[:] = lwa[:] + 0.5*dt*(z1s[m,:]+z1s[m+1,:]+z2s[m,:]+z2s[m+1,:]+z3s[m,:]+z3s[m+1,:]) + dt*res[m,:]
    m = m+1
    
    
m = 76
j = 49
phi = dp*j
const = 1./(2.*a*np.cos(phi)*dl)
lwa0 = np.zeros((120,360))
lwa1 = np.zeros((120,360))

lwa0[:,:] = zs[:,:]
lwa1[:,:] = zs[:,:]


while(m < 100):
    i = 1
    while(i < 359):
        d0 = -const*0.5*((lwa0[m,i+1]+lwa0[m+1,i+1])*cgx[m,i+1]-(lwa0[m,i-1]+lwa0[m+1,i-1])*cgx[m,i-1])
        d2 = 0.5*(z2s[m+1,i]+z2s[m,i])
        d3 = 0.5*(z3s[m+1,i]+z3s[m,i])
        d4 = gama[m,i]*0.5*(zs[m,i]+zs[m+1,i])
        lwa1[m+1,i] = lwa1[m,i]+dt*(d0+d2+d3+d4)
        i = i+1
    d0 = -const*0.5*((lwa0[m,0]+lwa0[m+1,0])*cgx[m,0]-(lwa0[m,358]+lwa0[m+1,358])*cgx[m,358])
    d2 = 0.5*(z2s[m+1,359]+z2s[m,359])
    d3 = 0.5*(z3s[m+1,359]+z3s[m,359])
    d4 = gama[m,359]*0.5*(zs[m,359]+zs[m+1,359])
    lwa1[m+1,359] = lwa1[m,359]+dt*(d0+d2+d3+d4)
        
    d0 = -const*0.5*((lwa0[m,1]+lwa0[m+1,1])*cgx[m,1]-(lwa0[m,359]+lwa0[m+1,359])*cgx[m,359])
    d2 = 0.5*(z2s[m+1,0]+z2s[m,0])
    d3 = 0.5*(z3s[m+1,0]+z3s[m,0])
    d4 = gama[m,0]*0.5*(zs[m,0]+zs[m+1,0])
    lwa1[m+1,0] = lwa1[m,0]+dt*(d0+d2+d3+d4)    
    m = m+1

####  Modify gamma  ####

gama1[:,:] = gama[:,:]
gama2[:,:] = gama[:,:]
m = 76
while(m < 100):
    if((m >= 84) and (m <= 92)):
        i = 200
        while(i < 221):
            if(gama[m,i] > 0):
                gama1[m,i] = gama[m,i]*0.7
                gama2[m,i] = gama[m,i]*0.
            i = i+1
    m = m+1

gama1[:,:] = gama1[:,:]-gama[:,:]
gama2[:,:] = gama2[:,:]-gama[:,:]

##### Interpolate gama, gama1, cgx onto finer tiem mesh dt = 30 min #####

gamap = np.zeros((8640,360))
gama1p = np.zeros((8640,360))
gama2p = np.zeros((8640,360))
cgxp = np.zeros((8640,360))
forcep = np.zeros((8640,360))
lwap = np.zeros((8640,360))

m = 76
while(m < 101):
    i = 0
    while(i < 360):
        mm = (m-76)*360+i-180
        if(mm >= 0):
            if(mm < 8640):
                gamap[mm,:] = gama[m,:]*(i/360.)+gama[m-1,:]*(1.-i/360.)
                gama1p[mm,:] = gama1[m,:]*(i/360.)+gama1[m-1,:]*(1.-i/360.)
                gama2p[mm,:] = gama2[m,:]*(i/360.)+gama2[m-1,:]*(1.-i/360.)
                cgxp[mm,:] = cgx[m,:]*(i/360.)+cgx[m-1,:]*(1.-i/360.)
        i = i+1
    m = m+1

m = 76
while(m < 100):
    i = 0
    while(i < 360):
        mm = (m-76)*360+i
        forcep[mm,:] = (z2s[m+1,:]+z3s[m+1,:])*(i/360.)+(z2s[m,:]+z3s[m,:])*(1.-i/360.)
        lwap[mm,:] = zs[m+1,:]*(i/360.)+zs[m,:]*(1.-i/360.)
        i = i+1
    m = m+1

##### Time integration ####

dt = 60.
j = 49
phi = dp*j
const = 1./(2.*a*np.cos(phi)*dl)
al = 0.
diff = 200000.
dlwap = np.zeros((8641,360))
dlwap2 = np.zeros((8641,360))

m = 1
while(m < 8640):
    i = 1
    while(i < 359):
        d1 = -const*(dlwap[m,i+1]*(cgxp[m,i+1]-al*(dlwap[m,i+1]+lwap[m,i+1]))-dlwap[m,i-1]*(cgxp[m,i-1]-al*(dlwap[m,i-1]+lwap[m,i-1])))
        d2 = gama1p[m,i]*dlwap[m,i]
        d3 = gamap[m,i]*dlwap[m,i]
        d4 = gama1p[m,i]*lwap[m,i]
        d5 = diff*(dlwap[m-1,i+1]+dlwap[m-1,i-1]-2.*dlwap[m-1,i])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
        dlwap[m+1,i] = dlwap[m-1,i]+2.*dt*(d1+d2+d3+d4+d5)
        
        d12 = -const*(dlwap2[m,i+1]*(cgxp[m,i+1]-al*(dlwap2[m,i+1]+lwap[m,i+1]))-dlwap2[m,i-1]*(cgxp[m,i-1]-al*(dlwap2[m,i-1]+lwap[m,i-1])))
        d22 = gama2p[m,i]*dlwap2[m,i]
        d32 = gamap[m,i]*dlwap2[m,i]
        d42 = gama2p[m,i]*lwap[m,i]
        d52 = diff*(dlwap2[m-1,i+1]+dlwap2[m-1,i-1]-2.*dlwap2[m-1,i])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
        dlwap2[m+1,i] = dlwap2[m-1,i]+2.*dt*(d12+d22+d32+d42+d52)
        i = i+1
        
    d1 = -const*(dlwap[m,0]*(cgxp[m,0]-al*(dlwap[m,0]+lwap[m,0]))-dlwap[m,358]*(cgxp[m,358]-al*(dlwap[m,358]+lwap[m,358])))
    d2 = gama1p[m,359]*dlwap[m,359]
    d3 = gamap[m,359]*dlwap[m,359]
    d4 = gama1p[m,359]*lwap[m,359]
    d5 = diff*(dlwap[m-1,0]+dlwap[m-1,358]-2.*dlwap[m-1,359])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
    dlwap[m+1,359] = dlwap[m-1,359]+2.*dt*(d1+d2+d3+d4+d5)
    
    d12 = -const*(dlwap2[m,0]*(cgxp[m,0]-al*(dlwap2[m,0]+lwap[m,0]))-dlwap2[m,358]*(cgxp[m,358]-al*(dlwap2[m,358]+lwap[m,358])))
    d22 = gama2p[m,359]*dlwap2[m,359]
    d32 = gamap[m,359]*dlwap2[m,359]
    d42 = gama2p[m,359]*lwap[m,359]
    d52 = diff*(dlwap2[m-1,0]+dlwap2[m-1,358]-2.*dlwap2[m-1,359])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
    dlwap2[m+1,359] = dlwap[m-1,359]+2.*dt*(d12+d22+d32+d42+d52)

    d1 = -const*(dlwap[m,1]*(cgxp[m,1]-al*(dlwap[m,1]+lwap[m,1]))-dlwap[m,359]*(cgxp[m,359]-al*(dlwap[m,359]+lwap[m,359])))            
    d2 = gama1p[m,0]*dlwap[m,0]
    d3 = gamap[m,0]*dlwap[m,0]
    d4 = gama1p[m,0]*lwap[m,0]
    d5 = diff*(dlwap[m-1,1]+dlwap[m-1,359]-2.*dlwap[m-1,0])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
    dlwap[m+1,0] = dlwap[m-1,0]+2.*dt*(d1+d2+d3+d4+d5)
    
    d12 = -const*(dlwap2[m,1]*(cgxp[m,1]-al*(dlwap2[m,1]+lwap[m,1]))-dlwap2[m,359]*(cgxp[m,359]-al*(dlwap2[m,359]+lwap[m,359])))            
    d22 = gama2p[m,0]*dlwap2[m,0]
    d32 = gamap[m,0]*dlwap2[m,0]
    d42 = gama2p[m,0]*lwap[m,0]
    d52 = diff*(dlwap2[m-1,1]+dlwap2[m-1,359]-2.*dlwap2[m-1,0])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
    dlwap2[m+1,0] = dlwap2[m-1,0]+2.*dt*(d12+d22+d32+d42+d52)
    m = m+1    

gm = np.zeros(360)
gm2 = np.zeros(360)
gm[:] = lwa[:]+dlwap[8640,:]
gm2[:] = lwa[:]+dlwap2[8640,:]
x = np.arange(0,360)
plot.rcParams.update({'font.size':14})
fig = plot.figure(figsize=(8,4))
plot.xlim(140,280) 
plot.ylim(-80,80) 
plot.title('Column LWA Change  49$^\circ$N  June 20 - 26  00 UTC') 
plot.xlabel('Longitude')
plot.ylabel('$\Delta$LWA (m/s)') 
ott = plot.plot(x,zx)
ott = plot.plot(x,zy,color='black',alpha = 0.3)
ott = plot.plot(x,gm,color='red',alpha = 0.5)
ott = plot.plot(x,gm2,'r--',alpha = 0.5)
plot.savefig('dLWAp.png',bbox_inches='tight',dpi =600)
#plot.show()

print(lwa[242],lwa[252]+dlwap[8640,252])