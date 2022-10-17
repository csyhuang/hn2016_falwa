SUBROUTINE interpolate_fields_direct_inv(nlon, nlat, nlev, kmax, jd, uu, vv, tt, plev, &
        aa, omega, dz, hh, rr, cp, &
        pv, uq, vq, avort, tq, statn, stats, tn0, ts0)


  INTEGER, INTENT(IN) :: nlon, nlat, nlev, kmax, jd
  REAL, INTENT(IN) :: uu(nlon,nlat,nlev), vv(nlon,nlat,nlev), tt(nlon,nlat,nlev), &
            plev(nlev)
  REAL, INTENT(in) :: aa, omega, dz,hh, rr, cp
  REAL, INTENT(out) :: pv(nlon,nlat,kmax), uq(nlon,nlat,kmax), vq(nlon,nlat,kmax), avort(nlon,nlat,kmax)
  REAL, INTENT(out) :: tq(nlon,nlat,kmax), statn(kmax), stats(kmax), tn0(kmax), ts0(kmax)

   real ::  tzd(nlat,kmax)
   real ::  height(kmax)
   real ::  zlev(nlev)
   real ::  st(nlon,nlat),zmst(nlat)
   real ::  zmav(nlat,kmax)
   real ::  zmpv(nlat,kmax)
   integer :: k0(kmax),kp(kmax)
   real :: dd2(kmax),dd1(kmax),pks(kmax)

   rkappa = rr/cp
   pi = acos(-1.)
   dphi = pi/float(nlat-1)

  write(6,*) 'nlon, nlat, nlev, kmax, jd'
  write(6,*) nlon, nlat, nlev, kmax, jd

! ====== Assign pseudoheight =====

  do k = 1,nlev
    zlev(k) = -hh*alog(plev(k)/1000.)
  enddo

  do k = 1,kmax
    height(k) = float(k-1)*dz
    pks(k) = exp(rkappa*height(k)/hh)
  enddo


  do kk = 2,kmax   ! vertical interpolation
    ttt = height(kk)
    do k = 1,nlev-1
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

! ====  vertical interpolation ====

  do i = 1,nlon
    do j = 1,nlat

      st(i,j) = tt(i,j,1)      ! surface pot. temp

      do kk = 2,kmax   ! vertical interpolation
        uq(i,j,kk) = uu(i,j,k0(kk))*dd2(kk) + uu(i,j,kp(kk))*dd1(kk)
        vq(i,j,kk) = vv(i,j,k0(kk))*dd2(kk) + vv(i,j,kp(kk))*dd1(kk)
        !     wq(i,j,kk) = ww(i,j,k0(kk))*dd2(kk) + ww(i,j,kp(kk))*dd1(kk)
        tq(i,j,kk) = tt(i,j,k0(kk))*dd2(kk) + tt(i,j,kp(kk))*dd1(kk)
        tq(i,j,kk) = tq(i,j,kk)*pks(kk)  ! potential temperature
        !     zq(i,j,kk) = zz(i,j,k0(kk))*dd2(kk) + zz(i,j,kp(kk))*dd1(kk)
      enddo

      tq(i,j,1) = tt(i,j,1)
      uq(i,j,1) = uu(i,j,1)
      vq(i,j,1) = vv(i,j,1)
      !     wq(i,j,1) = ww(i,j,1)
      !     zq(i,j,1) = zz(i,j,1)
    enddo
  enddo

!  **** compute zonal mean ****

  tzd = 0.

  do j = 1,nlat
    do k = 1,kmax
      do i = 1,nlon
        tzd(j,k) = tzd(j,k) + tq(i,j,k)/float(nlon)
      enddo
    enddo
  enddo


  ! reference theta
  do kk = 1,kmax
    ts0(kk) = 0.
    tn0(kk) = 0.
    csm = 0.
    cnm = 0.
    do j = 1,jd
      phi0 = -0.5*pi + pi*float(j-1)/float(nlat-1)
      ts0(kk) = ts0(kk) + tzd(j,kk)*cos(phi0)
      csm = csm + cos(phi0)
    enddo
    ts0(kk) = ts0(kk)/csm
    do j = jd,nlat
      phi0 = -0.5*pi + pi*float(j-1)/float(nlat-1)
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

  do j = 1,nlat
    zmst(j) = 0.
    do i = 1,nlon
      zmst(j) = zmst(j) + st(i,j)/float(nlon)
    enddo
  enddo

! interior abs. vort

  do kk = 1,kmax
    do j = 2,nlat-1
      phi0 = -0.5*pi + pi*float(j-1)/float(nlat-1)
      phim = -0.5*pi + pi*float(j-2)/float(nlat-1)
      phip = -0.5*pi + pi*float(j)/float(nlat-1)

      do i = 2,nlon-1
        av1 = 2.*omega*sin(phi0)
        av2 = (vq(i+1,j,kk)-vq(i-1,j,kk))/(2.*aa*cos(phi0)*dphi)
        av3 = -(uq(i,j+1,kk)*cos(phip)-uq(i,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)
        avort(i,j,kk) = av1+av2+av3
      enddo

      av1 = 2.*omega*sin(phi0)
      av2 = (vq(2,j,kk)-vq(nlon,j,kk))/(2.*aa*cos(phi0)*dphi)
      av3 = -(uq(1,j+1,kk)*cos(phip)-uq(1,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)
      avort(1,j,kk) = av1+av2+av3
      av4 = 2.*omega*sin(phi0)
      av5 = (vq(1,j,kk)-vq(nlon-1,j,kk))/(2.*aa*cos(phi0)*dphi)
      av6 =   &
      -(uq(nlon,j+1,kk)*cos(phip)-uq(nlon,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)
      avort(nlon,j,kk) = av4+av5+av6
    enddo

    avs = 0.
    avn = 0.
    do i = 1,nlon
      avs = avs + avort(i,2,kk)/float(nlon)
      avn = avn + avort(i,nlat-1,kk)/float(nlon)
    enddo
    avort(:,1,kk) = avs
    avort(:,nlat,kk) = avn
  enddo

  ! zonal mean vort

  do kk = 1,kmax
    do j = 1,nlat
      zmav(j,kk) = 0.
      do i = 1,nlon
        zmav(j,kk) = zmav(j,kk)+avort(i,j,kk)/float(nlon)
      enddo
    enddo
  enddo

  ! interior pv

  do kk = 2,kmax-1
    do j = 1,nlat
      phi0 = -0.5*pi + pi*float(j-1)/float(nlat-1)
      f = 2.*omega*sin(phi0)
      if (j .le. jd) then
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

      do i = 1,nlon
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
    do j = 1,nlat
      zmpv(j,kk) = 0.
      do i = 1,nlon
        zmpv(j,kk) = zmpv(j,kk)+pv(i,j,kk)/float(nlon)
      enddo
    enddo
  enddo
END SUBROUTINE
