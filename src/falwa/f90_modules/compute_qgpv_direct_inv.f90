SUBROUTINE compute_qgpv_direct_inv(nlon, nlat, kmax, jd, uq, vq, tq, height, &
        ts0, tn0, stats, statn, &
        aa, omega, dz, hh, rr, cp, &
        pv, avort)


  INTEGER, INTENT(IN) :: nlon, nlat, kmax, jd
  REAL, INTENT(IN) :: uq(nlon,nlat,kmax), vq(nlon,nlat,kmax), tq(nlon,nlat,kmax), height(kmax)
  REAL, INTENT(in) :: stats(kmax), statn(kmax), ts0(kmax), tn0(kmax)
  REAL, INTENT(in) :: aa, omega, dz,  hh, rr, cp
  REAL, INTENT(out) :: pv(nlon,nlat,kmax), avort(nlon,nlat,kmax)

   real ::  tzd(nlat,kmax)
   !real ::  st(nlon,nlat),zmst(nlat)
   real ::  zmav(nlat,kmax)
   real ::  zmpv(nlat,kmax)
   integer :: k0(kmax),kp(kmax)

   rkappa = rr/cp
   pi = acos(-1.)
   dphi = pi/float(nlat-1)

  write(6,*) 'nlon, nlat, kmax, jd'
  write(6,*) nlon, nlat, kmax, jd

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
