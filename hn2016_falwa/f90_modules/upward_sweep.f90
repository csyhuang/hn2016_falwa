SUBROUTINE upward_sweep(jmax, kmax, nd, jb, jd, sjk, tjk, ckref, tb, qref_over_cor, tref, qref, u, a, om, dz, h, rr, cp)

  INTEGER, INTENT(IN) :: jmax, kmax, nd, jb, jd
  REAL, INTENT(IN) :: sjk(jd-2,jd-2,kmax-1),tjk(jd-2,kmax-1),ckref(nd,kmax),tb(kmax),qref_over_cor(nd,kmax)
  REAL, INTENT(IN) :: a, om, dz, h, rr, cp
  REAL, INTENT(OUT) :: qref(nd,kmax), tref(jd,kmax), u(jd,kmax)
  real :: tg(kmax)
  real :: pjk(jd-2,kmax)
  real :: tj(jd-2)
  real :: yj(jd-2)
  real :: sjj(jd-2,jd-2)
  real :: pj(jd-2)

  rkappa = rr/cp
  pi = acos(-1.)
  dp = pi/float(jmax-1)


  pjk(:,1) = 0.
  do k = 1,kmax-1
    pj(:) = pjk(:,k)
    sjj(:,:) = sjk(:,:,k)
    tj(:) = tjk(:,k)

    do i = 1,jd-2
      yj(i) = 0.
      do kk = 1,jd-2
        yj(i) = yj(i)+sjj(i,kk)*pj(kk)
      enddo
      pjk(i,k+1) = yj(i)+tj(i)
    enddo
    !        call gemv(sjj,pj,yj)
    !        pjk(:,k+1) = yj(:) + tj(:)
  enddo

  ! **** Recover u *****
  do k = 1,kmax
    do j = 2,jd-1
      u(j,k) = pjk(j-1,k)
    enddo
  enddo

  ! *** Corner boundary conditions ***
  u(1,1) = 0.
  u(jd,1) = 0.
  !        u(1,kmax) = ubar(1+jb,kmax)*cos(dp*float(jb))
  u(1,kmax) = ckref(1+jb,kmax)/(2.*pi*a)-om*a*cos(dp*float(jb))
  u(jd,kmax) = 0.

  ! *** Divide by cos phi to revover Uref ****
  do jj = jb+1,nd-1
    j = jj-jb
    phi0 = dp*float(jj-1)
    u(j,:) = u(j,:)/cos(phi0)
  enddo
  u(jd,:) = 2.*u(jd-1,:)-u(jd-2,:)

  ! ******** compute tref *******
  qref(:, :) = qref_over_cor(:, :)  ! modify for f2py wrapping purpose
  do k = 2,kmax-1
    t00 = 0.
    zz = dz*float(k-1)
    tref(1,k) = t00
    tref(2,k) = t00
    do j = 2,jd-1
      phi0 = dp*float(j-1+jb)
      cor = 2.*om*sin(phi0)
      uz = (u(j,k+1)-u(j,k-1))/(2.*dz)
      ty = -cor*uz*a*h*exp(rkappa*zz/h)
      ty = ty/rr
      tref(j+1,k) = tref(j-1,k)+2.*ty*dp
    enddo
    do j = 1,nd
      phi0 = dp*float(j-1)
      qref(j,k) = qref_over_cor(j,k)*sin(phi0)
    enddo

    tg(k) = 0.
    wt = 0.
    do jj = jb+1,nd
      j = jj-jb
      phi0 = dp*float(jj-1)
      tg(k) = tg(k)+cos(phi0)*tref(j,k)
      wt = wt + cos(phi0)
    enddo
    tg(k) = tg(k)/wt
    tres = tb(k)-tg(k)
    tref(:,k) = tref(:,k)+tres
  enddo
  tref(:,1) = tref(:,2)-tb(2)+tb(1)
  tref(:,kmax) = tref(:,kmax-1)-tb(kmax-1)+tb(kmax)

END SUBROUTINE upward_sweep