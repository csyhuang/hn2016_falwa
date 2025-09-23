SUBROUTINE compute_qref_and_fawa_first(pv, uu, vort, pt, tn0, imax, JMAX, kmax, nd, nnd, jb, jd, &
        a, omega, dz, h, dphi, dlambda, rr, cp, &
        qref, ubar, tbar, fawa, ckref, tjk, sjk)


  !USE mkl95_LAPACK, ONLY: GETRF,GETRI

  REAL, INTENT(IN) :: pv(kmax,jmax,imax),uu(kmax,jmax,imax),vort(kmax,jmax,imax),pt(kmax,jmax,imax),tn0(kmax)
  INTEGER, INTENT(IN) :: imax, JMAX, kmax, nd, nnd, jb, jd
  REAL, INTENT(in) :: a, omega, dz, h, dphi, dlambda, rr, cp
  REAL, INTENT(OUT) :: qref(kmax,nd),ubar(kmax,nd),tbar(kmax,nd),fawa(kmax,nd),ckref(kmax,nd),&
          tjk(kmax-1,jd-2),sjk(kmax-1,jd-2,jd-2)

  !   **** take QG analysis and compute Q_ref and invert for U_ref & Theta_ref for NH (Direct solver) ***

  REAL :: pv2(jmax,imax)
  REAL :: vort2(jmax,imax)
  REAL :: qn(nnd),an(nnd),aan(nnd),tb(kmax)
  REAL :: cn(nnd),ccn(nnd),cref(kmax,nd)
  REAL :: alat(nd),phi(nd),z(kmax),cbar(kmax,nd)
  REAL :: qbar(kmax,nd)

  pi = acos(-1.)
  !dphi = pi/float(jmax-1)  !!!  This is dlat
  !dlambda = 2*pi/float(imax)  !!!! pragallva- correction added on Dec 13, 2023. This is dlon
  rkappa = rr/cp

  do nn = 1,nd
    phi(nn) = dphi*float(nn-1)
    alat(nn) = 2.*pi*a*a*(1.-sin(phi(nn)))
  enddo

  do k = 1,kmax
    z(k) = dz*float(k-1)
  enddo


  ! **** Zonal-mean field ****
  do j = nd,jmax
    do k = 1,kmax
      qbar(k,j-(nd-1)) = 0.
      tbar(k,j-(nd-1)) = 0.
      ubar(k,j-(nd-1)) = 0.
      do i = 1,imax
        qbar(k,j-(nd-1)) = qbar(k,j-(nd-1))+pv(k,j,i)/float(imax)
        tbar(k,j-(nd-1)) = tbar(k,j-(nd-1))+pt(k,j,i)/float(imax)
        ubar(k,j-(nd-1)) = ubar(k,j-(nd-1))+uu(k,j,i)/float(imax)
      enddo
    enddo
  enddo

  ! **** hemispheric-mean potential temperature ****
  tb(:) = tn0(:)

  do k = 2,kmax-1
    pv2(:,:) = pv(k,:,:)
    vort2(:,:) = vort(k,:,:)

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
      phi0 = -0.5*pi+dphi*float(j-1)
      do i = 1,imax
        ind = 1+int((qmax-pv2(j,i))/dq)
        da = a*a*dphi*dlambda*cos(phi0)
        an(ind) = an(ind) + da
        cn(ind) = cn(ind) + da*pv2(j,i)
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
          qref(k,j) = qn(nn)*(1.-dd)+qn(nn+1)*dd
          cref(k,j) = ccn(nn)*(1.-dd)+ccn(nn+1)*dd
        endif
      enddo
    enddo

    qref(k,nd) = qmax

    cbar(k,nd) = 0.
    do j=nd-1,1,-1
      phi0 = dphi*(float(j)-0.5)
      cbar(k,j) = cbar(k,j+1)+0.5*(qbar(k,j+1)+qbar(k,j)) &
      *a*dphi*2.*pi*a*cos(phi0)
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
      phi0 = -0.5*pi+dphi*float(j-1)
      do i = 1,imax
      ind = 1+int((qmax-vort2(j,i))/dq)
      da = a*a*dphi*dlambda*cos(phi0)
      an(ind) = an(ind) + da
      cn(ind) = cn(ind) + da*vort2(j,i)
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
          ckref(k,j) = ccn(nn)*(1.-dd)+ccn(nn+1)*dd
        endif
      enddo
    enddo

  enddo

  ! ***** normalize QGPV by sine (latitude)  ****

  do j = 2,nd
    phi0 = dphi*float(j-1)
    cor = sin(phi0)
    qref(:,j) = qref(:,j)/cor
  enddo

  do k = 2,kmax-1
    qref(k,1) = 2.*qref(k,2)-qref(k,3)
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
    phi0 = float(jj-1)*dphi
    cos0 = cos(phi0)
    sin0 = sin(phi0)
    tjk(kmax-1,j-1) = -dz*rr*cos0*exp(-z(kmax-1)*rkappa/h)
    tjk(kmax-1,j-1) = tjk(kmax-1,j-1)*(tbar(kmax,j+1)-tbar(kmax,j-1))
    tjk(kmax-1,j-1) = tjk(kmax-1,j-1)/(4.*omega*sin0*dphi*h*a)
    sjk(kmax-1,j-1,j-1) = 1.
  enddo
END
