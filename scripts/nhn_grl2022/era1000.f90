   Program era

! Read ERA5 binary files and computes QG fields.
! *** 6 hourly data ***

   integer,parameter  :: kmax = 97
   character*30 :: file1,file2,file3,file4,file5
   character*34 :: file6
   character*33 :: file7
   character*32 :: file8,file9,file10,file11,file12
   character*30 :: fn

   common /arry/  tt(360,181,37),tzd(181,kmax)
   common /crry/  uu(360,181,37)
   common /drry/  vv(360,181,37)
   common /brry/  ww(360,181,37)
   common /erry/  zz(360,181,37)
   common /brry/  xlon(360),ylat(181),plev(37)
   common /frry/  height(kmax),statn(kmax),stats(kmax)
   common /ffry/  ts0(kmax),tn0(kmax),zlev(37)
   common /grry/  st(360,181),zmst(181),uq(360,181,kmax)
   common /irry/  vq(360,181,kmax)
   common /jrry/  wq(360,181,kmax)
   common /krry/  tq(360,181,kmax)
   common /lrry/  zq(360,181,kmax)
   common /hhry/  tt0(360,181,kmax)
   common /irry/  avort(360,181,kmax),zmav(181,kmax)
   common /jrry/  pv(360,181,kmax),zmpv(181,kmax)
   integer :: dsadata,mm(12),inverse(12,37)
   integer :: k0(kmax),kp(kmax)
   real :: dd2(kmax),dd1(kmax),pks(kmax)
   character*5 :: yy
   character*4 :: y0(44),y00
   character*3 :: mn(12)
   character*8 :: yr

   y0(1) = '1978'
   y0(2) = '1979'
   y0(3) = '1980'
   y0(4) = '1981'
   y0(5) = '1982'
   y0(6) = '1983'
   y0(7) = '1984'
   y0(8) = '1985'
   y0(9) = '1986'
   y0(10) = '1987'
   y0(11) = '1988'
   y0(12) = '1989'
   y0(13) = '1990'
   y0(14) = '1991'
   y0(15) = '1992'
   y0(16) = '1993'
   y0(17) = '1994'
   y0(18) = '1995'
   y0(19) = '1996'
   y0(20) = '1997'
   y0(21) = '1998'
   y0(22) = '1999'
   y0(23) = '2000'
   y0(24) = '2001'
   y0(25) = '2002'
   y0(26) = '2003'
   y0(27) = '2004'
   y0(28) = '2005'
   y0(29) = '2006'
   y0(30) = '2007'
   y0(31) = '2008'
   y0(32) = '2009'
   y0(33) = '2010'
   y0(34) = '2011'
   y0(35) = '2012'
   y0(36) = '2013'
   y0(37) = '2014'
   y0(38) = '2015'
   y0(39) = '2016'
   y0(40) = '2017'
   y0(41) = '2018'
   y0(42) = '2019'
   y0(43) = '2020'
   y0(44) = '2021'

   dz = 500.
   cp = 1004.
   rr = 287.
   rkappa = rr/cp
   grav = 9.81
   pi = acos(-1.)
   omega = 7.29e-5
   aa = 6378000.
   dphi = pi/180.
   hh = 7000.

! ====== Assign pseudoheight =====

   do k = 1,kmax
     height(k) = float(k-1)*dz
     pks(k) = exp(rkappa*height(k)/hh)
   enddo

  do mmm = 44,44
   mf = 28   ! 29 for leap year
   if(mod(mmm,4).eq.3) mf = 29
   y00 = y0(mmm)
   yy = y00//'_'      ! Year to extract
   mm(1) = 31
   mm(2) = mf        ! Adjust for leap years
   mm(3) = 31
   mm(4) = 30
   mm(5) = 31
   mm(6) = 30
   mm(7) = 31
   mm(8) = 31
   mm(9) = 30
   mm(10) = 31
   mm(11) = 30
   mm(12) = 31

   mn(1) = '01_'
   mn(2) = '02_'
   mn(3) = '03_'
   mn(4) = '04_'
   mn(5) = '05_'
   mn(6) = '06_'
   mn(7) = '07_'
   mn(8) = '08_'
   mn(9) = '09_'
   mn(10) = '10_'
   mn(11) = '11_'
   mn(12) = '12_'

   plev(1) = 1000.
   plev(2) = 975.
   plev(3) = 950.
   plev(4) = 925.
   plev(5) = 900.
   plev(6) = 875.
   plev(7) = 850.
   plev(8) = 825.
   plev(9) = 800.
   plev(10) = 775.
   plev(11) = 750.
   plev(12) = 700.
   plev(13) = 650.
   plev(14) = 600.
   plev(15) = 550.
   plev(16) = 500.
   plev(17) = 450.
   plev(18) = 400.
   plev(19) = 350.
   plev(20) = 300.
   plev(21) = 250.
   plev(22) = 225.
   plev(23) = 200.
   plev(24) = 175.
   plev(25) = 150.
   plev(26) = 125.
   plev(27) = 100.
   plev(28) = 70.
   plev(29) = 50.
   plev(30) = 30.
   plev(31) = 20.
   plev(32) = 10.
   plev(33) = 7.
   plev(34) = 4.
   plev(35) = 3.
   plev(36) = 2.
   plev(37) = 1.

 do k = 1,37
    zlev(k) = -hh*alog(plev(k)/1000.)
 enddo
 do kk = 2,kmax   ! vertical interpolation 
   ttt = height(kk)
  do k = 1,36
   tt2 = zlev(k+1)
   tt1 = zlev(k)
   if((ttt.ge.tt1).and.(ttt.lt.tt2)) then
     k0(kk) = k
     kp(kk) = k+1
     dd1(kk) = (ttt-tt1)/(tt2-tt1)
     dd2(kk) = 1.-dd1(kk)
   endif
  enddo
 enddo
 
do m = 11,11

   nn = mm(m)*4
   yr = yy//mn(m)

   file1 = '/data2/nnn/ERA5/'//y00//'/'//yr//'U'
   file2 = '/data2/nnn/ERA5/'//y00//'/'//yr//'V'
   file3 = '/data2/nnn/ERA5/'//y00//'/'//yr//'W'
   file4 = '/data2/nnn/ERA5/'//y00//'/'//yr//'T'
   file5 = '/data2/nnn/ERA5/'//y00//'/'//yr//'Z'
   file6 = '/data2/nnn/ERA5/'//y00//'/'//yr//'QVORT'
   file7 = '/data2/nnn/ERA5/'//y00//'/'//yr//'QGPV'
   file8 = '/data2/nnn/ERA5/'//y00//'/'//yr//'QGU'
   file9 = '/data2/nnn/ERA5/'//y00//'/'//yr//'QGV'
   file10 = '/data2/nnn/ERA5/'//y00//'/'//yr//'QGW'
   file11 = '/data2/nnn/ERA5/'//y00//'/'//yr//'QGT'
   file12 = '/data2/nnn/ERA5/'//y00//'/'//yr//'QGZ'

  open(31,file=file1,form='unformatted',status='old')
  open(32,file=file2,form='unformatted',status='old')
  open(33,file=file3,form='unformatted',status='old')
  open(34,file=file4,form='unformatted',status='old')
  open(35,file=file5,form='unformatted',status='old')
  open(36,file=file6,form='unformatted',status='new')
  open(37,file=file7,form='unformatted',status='new')
  open(38,file=file8,form='unformatted',status='new')
  open(39,file=file9,form='unformatted',status='new')
  open(40,file=file10,form='unformatted',status='new')
  open(41,file=file11,form='unformatted',status='new')
  open(42,file=file12,form='unformatted',status='new')

do l = 1,nn

  read(31) uu
  read(32) vv
  read(33) ww
  read(34) tt 
  read(35) zz

! ====  vertical interpolation ====

 do i = 1,360
 do j = 1,181

   st(i,j) = tt(i,j,1)      ! surface pot. temp

 do kk = 2,kmax   ! vertical interpolation 
     uq(i,j,kk) = uu(i,j,k0(kk))*dd2(kk) + uu(i,j,kp(kk))*dd1(kk)
     vq(i,j,kk) = vv(i,j,k0(kk))*dd2(kk) + vv(i,j,kp(kk))*dd1(kk)
     wq(i,j,kk) = ww(i,j,k0(kk))*dd2(kk) + ww(i,j,kp(kk))*dd1(kk)
     tq(i,j,kk) = tt(i,j,k0(kk))*dd2(kk) + tt(i,j,kp(kk))*dd1(kk)
     tq(i,j,kk) = tq(i,j,kk)*pks(kk)  ! potential temperature
     zq(i,j,kk) = zz(i,j,k0(kk))*dd2(kk) + zz(i,j,kp(kk))*dd1(kk)
 enddo

     tq(i,j,1) = tt(i,j,1)
     uq(i,j,1) = uu(i,j,1)
     vq(i,j,1) = vv(i,j,1)
     wq(i,j,1) = ww(i,j,1)
     zq(i,j,1) = zz(i,j,1)
  enddo
  enddo

!  **** compute zonal mean ****

   tzd = 0.

  do j = 1,181
  do k = 1,kmax
  do i = 1,360
   tzd(j,k) = tzd(j,k) + tq(i,j,k)/360.
  enddo
  enddo
  enddo


! reference theta
   do kk = 1,kmax
       ts0(kk) = 0.
       tn0(kk) = 0.
       csm = 0.
       cnm = 0.
    do j = 1,91
       phi0 = -90.+float(j-1)
       phi0 = phi0*pi/180.
       ts0(kk) = ts0(kk) + tzd(j,kk)*cos(phi0)
       csm = csm + cos(phi0)
    enddo
       ts0(kk) = ts0(kk)/csm
    do j = 91,181
       phi0 = -90.+float(j-1)
       phi0 = phi0*pi/180.
       tn0(kk) = tn0(kk) + tzd(j,kk)*cos(phi0)
       cnm = cnm + cos(phi0)
    enddo
       tn0(kk) = tn0(kk)/cnm
   enddo

! static stability
   do kk = 2,kmax-1
     stats(kk) = (ts0(kk+1)-ts0(kk-1))/(height(kk+1)-height(kk-1))
     statn(kk) = (tn0(kk+1)-tn0(kk-1))/(height(kk+1)-height(kk-1))
   enddo
     stats(kmax) = 2.*stats(kmax-1)-stats(kmax-2)
     statn(kmax) = 2.*statn(kmax-1)-statn(kmax-2)
     stats(1) = 2.*stats(2)-stats(3)
     statn(1) = 2.*statn(2)-statn(3)

! surface temp

    do j = 1,181
     zmst(j) = 0.
    do i = 1,360
     zmst(j) = zmst(j) + st(i,j)/360.
    enddo
    enddo

! interior abs. vort

  do kk = 1,kmax
  do j = 2,180
   phi0 = -90.+float(j-1)
   phi0 = phi0*pi/180.
   phim = -90.+float(j-2)
   phim = phim*pi/180.
   phip = -90.+float(j)
   phip = phip*pi/180.

  do i = 2,359
   av1 = 2.*omega*sin(phi0)
   av2 = (vq(i+1,j,kk)-vq(i-1,j,kk))/(2.*aa*cos(phi0)*dphi)
   av3 = -(uq(i,j+1,kk)*cos(phip)-uq(i,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)  
   avort(i,j,kk) = av1+av2+av3
  enddo

   av1 = 2.*omega*sin(phi0)
   av2 = (vq(2,j,kk)-vq(360,j,kk))/(2.*aa*cos(phi0)*dphi)
   av3 = -(uq(1,j+1,kk)*cos(phip)-uq(1,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)  
   avort(1,j,kk) = av1+av2+av3
   av4 = 2.*omega*sin(phi0)
   av5 = (vq(1,j,kk)-vq(359,j,kk))/(2.*aa*cos(phi0)*dphi)
   av6 =   & 
-(uq(360,j+1,kk)*cos(phip)-uq(360,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)
   avort(360,j,kk) = av4+av5+av6
  enddo

    avs = 0.
    avn = 0.
   do i = 1,360
    avs = avs + avort(i,2,kk)/360.
    avn = avn + avort(i,180,kk)/360.
   enddo
    avort(:,1,kk) = avs
    avort(:,181,kk) = avn
  enddo

! zonal mean vort

  do kk = 1,kmax
    do j = 1,181
      zmav(j,kk) = 0.
     do i = 1,360
      zmav(j,kk) = zmav(j,kk)+avort(i,j,kk)/360.
     enddo
    enddo
  enddo

! interior pv

  do kk = 2,kmax-1
   do j = 1,181
      phi0 = -90.+float(j-1)
      phi0 = phi0*pi/180.
      f = 2.*omega*sin(phi0)
      if(j.le.91) then
       statp = stats(kk+1)
       statm = stats(kk-1)
       t00p = ts0(kk+1)
       t00m = ts0(kk-1)
      else
       statp = statn(kk+1)
       statm = statn(kk-1)
       t00p = tn0(kk+1)
       t00m = tn0(kk-1)
      endif
       
    do i = 1,360
      thetap = tq(i,j,kk+1)
      thetam = tq(i,j,kk-1)
      altp = exp(-height(kk+1)/hh)*(thetap-t00p)/statp
      altm = exp(-height(kk-1)/hh)*(thetam-t00m)/statm
      strc = (altp-altm)*f/(height(kk+1)-height(kk-1))
      pv(i,j,kk) = avort(i,j,kk) + exp(height(kk)/hh)*strc
    enddo
   enddo
  enddo

! zonal mean pv

  do kk = 1,kmax
    do j = 1,181
      zmpv(j,kk) = 0.
     do i = 1,360
      zmpv(j,kk) = zmpv(j,kk)+pv(i,j,kk)/360.
     enddo
    enddo
  enddo

 write(36) avort
 write(37) pv
 write(38) uq 
 write(39) vq
 write(40) wq
 write(41) tq,tn0,ts0,statn,stats
 write(42) zq
 !write(40) tt0,ts0,tn0,stats,statn


!hape(1) = 121
!hape(2) = 37
!fn = '/data/nnn/ERA_Interim/'//yr
! ret = dsadata(fn//'uz.df',2,shape,uz)
! ret = dsadata(fn//'vz.df',2,shape,vz)
! ret = dsadata(fn//'tz.df',2,shape,tz)
! ret = dsadata(fn//'zz.df',2,shape,zzz)

 write(6,*) 'month =',m,'   file =',l
enddo
 close(31)
 close(32)
 close(33)
 close(34)
 close(35)
 close(36)
 close(37)
 close(38)
 close(39)
 close(40)
 close(41)
 close(42)

enddo
enddo

!do k = 1,kmax
!  write(6,*) k,height(k),statn(k),stats(k)
!enddo

stop
end
