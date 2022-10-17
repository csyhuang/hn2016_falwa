SUBROUTINE matrix_b4_inversion(k,jmax,kmax,nd,jb,jd,z,statn,qref,ckref,&
        a, om, dz, h, rr, cp, &
        qjj,djj,cjj,rj,tj,sjk,tjk)

  integer, INTENT(in) :: k, jmax, kmax, nd, jb, jd
  REAL, INTENT(in) :: z(kmax),statn(kmax),qref(nd,kmax),ckref(nd,kmax)
  REAL, INTENT(in) :: a, om, dz, h, rr, cp
  REAL, INTENT(OUT) :: qjj(jd-2,jd-2),djj(jd-2,jd-2),cjj(jd-2,jd-2),rj(jd-2),tj(jd-2)
  REAL, INTENT(INOUT) :: sjk(jd-2,jd-2,kmax-1),tjk(jd-2,kmax-1)
  REAL :: xjj(jd-2,jd-2), u(jd,kmax)
  REAL :: sjj(jd-2,jd-2)

  rkappa = rr/cp
  pi = acos(-1.)
  dp = pi/float(jmax-1)

  zp = 0.5*(z(k+1)+z(k))
  zm = 0.5*(z(k-1)+z(k))
  statp = 0.5*(statn(k+1)+statn(k))
  statm = 0.5*(statn(k-1)+statn(k))
  cjj(:,:) = 0.
  djj(:,:) = 0.
  qjj(:,:) = 0.
  sjj(:,:) = sjk(:,:,k)
  tj(:) = tjk(:,k)
  do jj = jb+2,(nd-1)
    j = jj - jb
    phi0 = float(jj-1)*dp
    phip = (float(jj)-0.5)*dp
    phim = (float(jj)-1.5)*dp
    cos0 = cos(phi0)
    cosp = cos(phip)
    cosm = cos(phim)
    sin0 = sin(phi0)
    sinp = sin(phip)
    sinm = sin(phim)

    fact = 4.*om*om*h*a*a*sin0*dp*dp/(dz*dz*rr*cos0)
    amp = exp(-zp/h)*exp(rkappa*zp/h)/statp
    amp = amp*fact*exp(z(k)/h)
    amm = exp(-zm/h)*exp(rkappa*zm/h)/statm
    amm = amm*fact*exp(z(k)/h)

    ! ***** Specify A, B, C, D, E, F (Eqs. 4-9) *****
    ajk = 1./(sinp*cosp)
    bjk = 1./(sinm*cosm)
    cjk = amp
    djk = amm
    ejk = ajk+bjk+cjk+djk
    fjk = -0.5*a*dp*(qref(jj+1,k)-qref(jj-1,k))

    ! ***** Specify rk (Eq. 15) ****

    ! **** North-south boundary conditions ****
    u(jd,k) = 0.
    phi0 = dp*float(jb)
    !        u(1,k) = ubar(jb+1,k)*cos(phi0)
    u(1,k) = ckref(jb+1,k)/(2.*pi*a)-om*a*cos(phi0)

    rj(j-1) = fjk
    if(j.eq.2) rj(j-1) = fjk - bjk*u(1,k)
    if(j.eq.jd-1) rj(j-1) = fjk - ajk*u(jd,k)

    ! ***** Specify Ck & Dk (Eqs. 18-19) *****
    cjj(j-1,j-1) = cjk
    djj(j-1,j-1) = djk

    ! **** Specify Qk (Eq. 17) *****
    qjj(j-1,j-1) = -ejk
    if(j-1.ge.1.and.j-1.lt.jd-2) then
      qjj(j-1,j) = ajk
    endif
    if(j-1.gt.1.and.j-1.le.jd-2) then
      qjj(j-1,j-2) = bjk
    endif
  enddo

  ! **** Compute Qk + Ck Sk *******
  do i = 1,jd-2
    do j = 1,jd-2
      xjj(i,j) = 0.
      do kk = 1,jd-2
        xjj(i,j) = xjj(i,j)+cjj(i,kk)*sjj(kk,j)
      enddo
      qjj(i,j) = qjj(i,j)+xjj(i,j)
    enddo
  enddo

END SUBROUTINE matrix_b4_inversion
