SUBROUTINE compute_qref_and_fawa_first(pv, uu, vort, pt, tn0, imax, JMAX, kmax, nd, nnd, jb, jd, &
        a, omega, dz, h, rr, cp, &
        qref, ubar, tbar, fawa, ckref, tjk, sjk)


  !USE mkl95_LAPACK, ONLY: GETRF,GETRI

  INTEGER, INTENT(IN) :: imax, JMAX, kmax, nd, nnd, jb, jd
  REAL, INTENT(in) :: a, omega, dz, h, rr, cp
  REAL, INTENT(IN) :: pv(imax,jmax,kmax),uu(imax,jmax,kmax),vort(imax,jmax,kmax),pt(imax,jmax,kmax),tn0(kmax)
  REAL, INTENT(OUT) :: qref(nd,kmax),ubar(nd,kmax),tbar(nd,kmax),fawa(nd,kmax),ckref(nd,kmax),&
          tjk(jd-2,kmax-1),sjk(jd-2,jd-2,kmax-1)

  !   **** take QG analysis and compute Q_ref and invert for U_ref & Theta_ref for NH (Direct solver) ***

  !integer,parameter :: imax = 360, JMAX = 181, KMAX = 97
  !integer,parameter :: nd = 91,nnd=181
  !integer,parameter :: jb = 5   ! lower bounding latitude
  !integer,parameter :: jd = 86  ! nd - lower bounding latitude

  REAL :: pv2(imax,jmax)
  REAL :: vort2(imax,jmax)
  REAL :: qn(nnd),an(nnd),aan(nnd),tb(kmax)
  REAL :: cn(nnd),ccn(nnd),cref(nd,kmax)
  REAL :: alat(nd),phi(nd),z(kmax),cbar(nd,kmax)
  REAL :: qbar(nd,kmax)

  pi = acos(-1.)
  dp = pi/float(jmax-1)
  rkappa = rr/cp

  do nn = 1,nd
    phi(nn) = dp*float(nn-1)
    alat(nn) = 2.*pi*a*a*(1.-sin(phi(nn)))
  enddo

  do k = 1,kmax
    z(k) = dz*float(k-1)
  enddo


  ! **** Zonal-mean field ****
  do j = nd,jmax
    qbar(j-(nd-1),:) = 0.
    tbar(j-(nd-1),:) = 0.
    ubar(j-(nd-1),:) = 0.
    do i = 1,imax
      qbar(j-(nd-1),:) = qbar(j-(nd-1),:)+pv(i,j,:)/float(imax)
      tbar(j-(nd-1),:) = tbar(j-(nd-1),:)+pt(i,j,:)/float(imax)
      ubar(j-(nd-1),:) = ubar(j-(nd-1),:)+uu(i,j,:)/float(imax)
    enddo
  enddo

  ! **** hemispheric-mean potential temperature ****
  tb(:) = tn0(:)

  do k = 2,kmax-1
    pv2(:,:) = pv(:,:,k)
    vort2(:,:) = vort(:,:,k)

    !  **** compute qref via area analysis ****
    qmax = maxval(pv2)
    qmin = minval(pv2)
    dq = (qmax-qmin)/float(nnd-1)
    qn(:) = 0.
    an(:) = 0.
    cn(:) = 0.
    do nn = 1,nnd
      qn(nn) = qmax - dq*float(nn-1)
    enddo
    do j = 1,jmax
      phi0 = -0.5*pi+dp*float(j-1)
      do i = 1,imax
        ind = 1+int((qmax-pv2(i,j))/dq)
        da = a*a*dp*dp*cos(phi0)
        an(ind) = an(ind) + da
        cn(ind) = cn(ind) + da*pv2(i,j)
      enddo
    enddo
    aan(1) = 0.
    ccn(1) = 0.
    do nn = 2,nnd
      aan(nn) = aan(nn-1)+an(nn)
      ccn(nn) = ccn(nn-1)+cn(nn)
    enddo
    do j = 1,nd-1
      do nn = 1,nnd-1
        if(aan(nn).le.alat(j).and.aan(nn+1).gt.alat(j)) then
          dd = (alat(j)-aan(nn))/(aan(nn+1)-aan(nn))
          qref(j,k) = qn(nn)*(1.-dd)+qn(nn+1)*dd
          cref(j,k) = ccn(nn)*(1.-dd)+ccn(nn+1)*dd
        endif
      enddo
    enddo

    qref(nd,k) = qmax

    cbar(nd,k) = 0.
    do j=nd-1,1,-1
      phi0 = dp*(float(j)-0.5)
      cbar(j,k) = cbar(j+1,k)+0.5*(qbar(j+1,k)+qbar(j,k)) &
      *a*dp*2.*pi*a*cos(phi0)
    enddo

    ! **** compute Kelvin's circulation based on absolute vorticity (for b.c.) ****


    qmax = maxval(vort2)
    qmin = minval(vort2)
    dq = (qmax-qmin)/float(nnd-1)
    qn(:) = 0.
    an(:) = 0.
    cn(:) = 0.
    do nn = 1,nnd
      qn(nn) = qmax - dq*float(nn-1)
    enddo
    do j = 1,jmax
      phi0 = -0.5*pi+dp*float(j-1)
      do i = 1,imax
      ind = 1+int((qmax-vort2(i,j))/dq)
      da = a*a*dp*dp*cos(phi0)
      an(ind) = an(ind) + da
      cn(ind) = cn(ind) + da*vort2(i,j)
      enddo
    enddo
    aan(1) = 0.
    ccn(1) = 0.
    do nn = 2,nnd
      aan(nn) = aan(nn-1)+an(nn)
      ccn(nn) = ccn(nn-1)+cn(nn)
    enddo
    do j = 1,nd-1
      do nn = 1,nnd-1
        if(aan(nn).le.alat(j).and.aan(nn+1).gt.alat(j)) then
          dd = (alat(j)-aan(nn))/(aan(nn+1)-aan(nn))
          ckref(j,k) = ccn(nn)*(1.-dd)+ccn(nn+1)*dd
        endif
      enddo
    enddo

  enddo

  ! ***** normalize QGPV by sine (latitude)  ****

  do j = 2,nd
    phi0 = dp*float(j-1)
    cor = sin(phi0)
    qref(j,:) = qref(j,:)/cor
  enddo

  do k = 2,kmax-1
    qref(1,k) = 2.*qref(2,k)-qref(3,k)
  enddo

  ! ***** FAWA *****
  fawa(:,:) = (cref(:,:)-cbar(:,:))/(2.*pi*a)

  !  ***** Direct solver to invert Q_ref *****

  !  *** downward sweep ***

  ! **** top boundary condition (Eqs. 24-25) *****
  tjk(:,:) = 0.
  sjk(:,:,:) = 0.
  do jj = jb+2,(nd-1)
    j = jj-jb
    phi0 = float(jj-1)*dp
    cos0 = cos(phi0)
    sin0 = sin(phi0)
    tjk(j-1,kmax-1) = -dz*rr*cos0*exp(-z(kmax-1)*rkappa/h)
    tjk(j-1,kmax-1) = tjk(j-1,kmax-1)*(tbar(j+1,kmax)-tbar(j-1,kmax))
    tjk(j-1,kmax-1) = tjk(j-1,kmax-1)/(4.*omega*sin0*dp*h*a)
    sjk(j-1,j-1,kmax-1) = 1.
  enddo
END
