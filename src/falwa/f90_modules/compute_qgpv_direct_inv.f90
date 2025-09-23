SUBROUTINE compute_qgpv_direct_inv(nlon, nlat, kmax, jd, uq, vq, tq, height, &
        ts0, tn0, stats, statn, &
        aa, omega, dz, hh, rr, cp, &
        pv, avort)


  INTEGER, INTENT(IN) :: nlon, nlat, kmax, jd
  REAL, INTENT(IN) :: uq(kmax,nlat,nlon), vq(kmax,nlat,nlon), tq(kmax,nlat,nlon), height(kmax)
  REAL, INTENT(in) :: stats(kmax), statn(kmax), ts0(kmax), tn0(kmax)
  REAL, INTENT(in) :: aa, omega, dz,  hh, rr, cp
  REAL, INTENT(out) :: pv(kmax,nlat,nlon), avort(kmax,nlat,nlon)

   real ::  tzd(kmax,nlat)
   !real ::  st(nlon,nlat),zmst(nlat)
   real ::  zmav(kmax,nlat)
   real ::  zmpv(kmax,nlat)
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
