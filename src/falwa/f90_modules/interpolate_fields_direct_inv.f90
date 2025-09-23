SUBROUTINE interpolate_fields_direct_inv(nlon, nlat, nlev, kmax, jd, uu, vv, tt, plev, &
        aa, omega, dz, hh, rr, cp, &
        pv, uq, vq, avort, tq, statn, stats, tn0, ts0)


  INTEGER, INTENT(IN) :: nlon, nlat, nlev, kmax, jd
  REAL, INTENT(IN) :: uu(nlev,nlat,nlon), vv(nlev,nlat,nlon), tt(nlev,nlat,nlon), &
            plev(nlev)
  REAL, INTENT(in) :: aa, omega, dz,hh, rr, cp
  REAL, INTENT(out) :: pv(kmax,nlat,nlon), uq(kmax,nlat,nlon), vq(kmax,nlat,nlon), avort(kmax,nlat,nlon)
  REAL, INTENT(out) :: tq(kmax,nlat,nlon), statn(kmax), stats(kmax), tn0(kmax), ts0(kmax)

   real ::  tzd(kmax,nlat)
   real ::  height(kmax)
   real ::  zlev(nlev)
   real ::  st(nlat,nlon),zmst(nlat)
   real ::  zmav(kmax,nlat)
   real ::  zmpv(kmax,nlat)
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

      st(j,i) = tt(1,j,i)      ! surface pot. temp

      do kk = 2,kmax   ! vertical interpolation
        uq(kk,j,i) = uu(k0(kk),j,i)*dd2(kk) + uu(kp(kk),j,i)*dd1(kk)
        vq(kk,j,i) = vv(k0(kk),j,i)*dd2(kk) + vv(kp(kk),j,i)*dd1(kk)
        !     wq(i,j,kk) = ww(i,j,k0(kk))*dd2(kk) + ww(i,j,kp(kk))*dd1(kk)
        tq(kk,j,i) = tt(k0(kk),j,i)*dd2(kk) + tt(kp(kk),j,i)*dd1(kk)
        tq(kk,j,i) = tq(kk,j,i)*pks(kk)  ! potential temperature
        !     zq(i,j,kk) = zz(i,j,k0(kk))*dd2(kk) + zz(i,j,kp(kk))*dd1(kk)
      enddo

      tq(1,j,i) = tt(1,j,i)
      uq(1,j,i) = uu(1,j,i)
      vq(1,j,i) = vv(1,j,i)
      !     wq(i,j,1) = ww(i,j,1)
      !     zq(i,j,1) = zz(i,j,1)
    enddo
  enddo

!  **** compute zonal mean ****

  tzd = 0.

  do j = 1,nlat
    do k = 1,kmax
      do i = 1,nlon
        tzd(k,j) = tzd(k,j) + tq(k,j,i)/float(nlon)
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
      ts0(kk) = ts0(kk) + tzd(kk,j)*cos(phi0)
      csm = csm + cos(phi0)
    enddo
    ts0(kk) = ts0(kk)/csm
    do j = jd,nlat
      phi0 = -0.5*pi + pi*float(j-1)/float(nlat-1)
      tn0(kk) = tn0(kk) + tzd(kk,j)*cos(phi0)
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
      zmst(j) = zmst(j) + st(j,i)/float(nlon)
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
        av2 = (vq(kk,j,i+1)-vq(kk,j,i-1))/(2.*aa*cos(phi0)*dphi)
        av3 = -(uq(kk,j+1,i)*cos(phip)-uq(kk,j-1,i)*cos(phim))/(2.*aa*cos(phi0)*dphi)
        avort(kk,j,i) = av1+av2+av3
      enddo

      av1 = 2.*omega*sin(phi0)
      av2 = (vq(kk,j,2)-vq(kk,j,nlon))/(2.*aa*cos(phi0)*dphi)
      av3 = -(uq(kk,j+1,1)*cos(phip)-uq(kk,j-1,1)*cos(phim))/(2.*aa*cos(phi0)*dphi)
      avort(kk,j,1) = av1+av2+av3
      av4 = 2.*omega*sin(phi0)
      av5 = (vq(kk,j,1)-vq(kk,j,nlon-1))/(2.*aa*cos(phi0)*dphi)
      av6 =   &
      -(uq(kk,j+1,nlon)*cos(phip)-uq(kk,j-1,nlon)*cos(phim))/(2.*aa*cos(phi0)*dphi)
      avort(kk,j,nlon) = av4+av5+av6
    enddo

    avs = 0.
    avn = 0.
    do i = 1,nlon
      avs = avs + avort(kk,2,i)/float(nlon)
      avn = avn + avort(kk,nlat-1,i)/float(nlon)
    enddo
    avort(kk,1,:) = avs
    avort(kk,nlat,:) = avn
  enddo

  ! zonal mean vort

  do kk = 1,kmax
    do j = 1,nlat
      zmav(kk,j) = 0.
      do i = 1,nlon
        zmav(kk,j) = zmav(kk,j)+avort(kk,j,i)/float(nlon)
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
        thetap = tq(kk+1,j,i)
        thetam = tq(kk-1,j,i)
        altp = exp(-height(kk+1)/hh)*(thetap-t00p)/statp
        altm = exp(-height(kk-1)/hh)*(thetam-t00m)/statm
        strc = (altp-altm)*f/(height(kk+1)-height(kk-1))
        pv(kk,j,i) = avort(kk,j,i) + exp(height(kk)/hh)*strc
      enddo
    enddo
  enddo

! zonal mean pv

  do kk = 1,kmax
    do j = 1,nlat
      zmpv(kk,j) = 0.
      do i = 1,nlon
        zmpv(kk,j) = zmpv(kk,j)+pv(kk,j,i)/float(nlon)
      enddo
    enddo
  enddo
END SUBROUTINE
