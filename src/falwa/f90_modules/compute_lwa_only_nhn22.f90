SUBROUTINE compute_lwa_only_nhn22(pv,uu,qref, &
        imax, JMAX, kmax, nd, jb, is_nhem, &
        a, om, dz, h, rr, cp, prefac, &
        astarbaro, ubaro, astar1, astar2)
  ! This was from compute_flux_dirinv.f90

  REAL, INTENT(IN) :: pv(kmax,jmax,imax),uu(kmax,jmax,imax),qref(kmax,nd)
  INTEGER, INTENT(IN) :: imax, JMAX, kmax, nd, jb
  LOGICAL, INTENT(IN) :: is_nhem
  REAL, INTENT(IN) :: a, om, dz, h, rr, cp, prefac
  REAL, INTENT(OUT) :: astarbaro(nd,imax), ubaro(nd,imax), astar1(kmax,nd,imax), astar2(kmax,nd,imax)

  REAL :: qe(nd,imax)
  REAL :: z(kmax)
  REAL :: aa, ab
  INTEGER :: jstart, jend

  pi = acos(-1.)
  dp = pi/float(jmax-1)
  rkappa = rr/cp

  ! *** Default values for boundary ***
  !jb = 5
  !jd = 86 ! nd - lower bounding latitude

  do k = 1,kmax
   z(k) = dz*float(k-1)
  enddo


! **** wave activity and nonlinear zonal flux F2 ****

  astarbaro(:,:) = 0.
  ubaro(:,:) = 0.
  dc = dz/prefac

  ! Bounds of y-indices in N/SHem
  if (is_nhem) then  ! 5N and higher latitude
    jstart = jb+1  ! 6
    jend = nd-1
  else
    jstart = 2
    jend = nd-jb   ! nd - 5
  endif

  do k = 2,kmax-1
    zk = dz*float(k-1)
    do i = 1,imax
      do j = jstart, jend
        astar1(k,j,i) = 0.       ! LWA*cos(phi)
        astar2(k,j,i) = 0.       ! LWA*cos(phi)
        if (is_nhem) then        !latitude
          phi0 = dp*float(j-1)
        else
          phi0 = dp*float(j-1)-0.5*pi
        endif
        cor = 2.*om*sin(phi0)          !Coriolis parameter
        ab = a*dp  !constant length element
        do jj = 1,nd
          if (is_nhem) then  ! Northern Hemisphere
            phi1 = dp*float(jj-1)
            qe(jj,i) = pv(k,jj+nd-1,i)-qref(k,j)    !qe; Q = qref
          else  ! Southern Hemisphere
            phi1 = dp*float(jj-1)-0.5*pi
            qe(jj,i) = pv(k,jj,i)-qref(k,j)     !qe; Q = qref
          endif
          aa = a*dp*cos(phi1)   !cosine factor in the meridional integral
          if((qe(jj,i).le.0.).and.(jj.ge.j)) then  !LWA*cos and F2
            if (is_nhem) then  ! Northern Hemisphere
              astar2(k,j,i)=astar2(k,j,i)-qe(jj,i)*aa  !anticyclonic
            else  ! Southern Hemisphere
              astar1(k,j,i)=astar1(k,j,i)-qe(jj,i)*aa  !cyclonic
            endif
          endif
          if((qe(jj,i).gt.0.).and.(jj.lt.j)) then
            if (is_nhem) then  ! Northern Hemisphere
              astar1(k,j,i)=astar1(k,j,i)+qe(jj,i)*aa  !cyclonic
            else  ! Southern Hemisphere
              astar2(k,j,i)=astar2(k,j,i)+qe(jj,i)*aa  !anticyclonic
            endif
          endif
        enddo

      enddo
      phip = dp*jb
      phi0 = dp*(jb-1)
      cosp = cos(phip)          ! cosine for one grid north
      cos0 = cos(phi0)          ! cosine for latitude grid
    enddo

    ! ******** Column average: (25) of SI-HN17 ********

    astarbaro(:,:) = astarbaro(:,:)+(astar1(k,:,:)    &
    + astar2(k,:,:))*exp(-zk/h)*dc

    if (is_nhem) then
      do j = jstart,jend
        ubaro(j,:) = ubaro(j,:)+uu(k,nd-1+j,:)*exp(-zk/h)*dc
      enddo
    else
      do j = jstart,jend
        ubaro(j,:) = ubaro(j,:)+uu(k,j,:)*exp(-zk/h)*dc
      enddo
    endif
  enddo

END SUBROUTINE
