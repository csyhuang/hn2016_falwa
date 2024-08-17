SUBROUTINE compute_lwa_only_nhn22(pv,uu,qref, &
        imax, JMAX, kmax, nd, jb, is_nhem, &
        a, om, dz, h, rr, cp, prefac, &
        astarbaro, ubaro, astar1, astar2)
  ! This was from compute_flux_dirinv.f90

  REAL, INTENT(IN) :: pv(imax,jmax,kmax),uu(imax,jmax,kmax),qref(nd,kmax)
  INTEGER, INTENT(IN) :: imax, JMAX, kmax, nd, jb
  LOGICAL, INTENT(IN) :: is_nhem
  REAL, INTENT(IN) :: a, om, dz, h, rr, cp, prefac
  REAL, INTENT(OUT) :: astarbaro(imax, nd), ubaro(imax, nd), astar1(imax,nd,kmax), astar2(imax,nd,kmax)

  REAL :: qe(imax,nd)
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
        astar1(i,j,k) = 0.       ! LWA*cos(phi)
        astar2(i,j,k) = 0.       ! LWA*cos(phi)
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
            qe(i,jj) = pv(i,jj+nd-1,k)-qref(j,k)    !qe; Q = qref
          else  ! Southern Hemisphere
            phi1 = dp*float(jj-1)-0.5*pi
            qe(i,jj) = pv(i,jj,k)-qref(j,k)     !qe; Q = qref
          endif
          aa = a*dp*cos(phi1)   !cosine factor in the meridional integral
          if((qe(i,jj).le.0.).and.(jj.ge.j)) then  !LWA*cos and F2
            if (is_nhem) then  ! Northern Hemisphere
              astar2(i,j,k)=astar2(i,j,k)-qe(i,jj)*aa  !anticyclonic
            else  ! Southern Hemisphere
              astar1(i,j,k)=astar1(i,j,k)-qe(i,jj)*aa  !cyclonic
            endif
          endif
          if((qe(i,jj).gt.0.).and.(jj.lt.j)) then
            if (is_nhem) then  ! Northern Hemisphere
              astar1(i,j,k)=astar1(i,j,k)+qe(i,jj)*aa  !cyclonic
            else  ! Southern Hemisphere
              astar2(i,j,k)=astar2(i,j,k)+qe(i,jj)*aa  !anticyclonic
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

    astarbaro(:,:) = astarbaro(:,:)+(astar1(:,:,k)    &
    + astar2(:,:,k))*exp(-zk/h)*dc

    if (is_nhem) then
      do j = jstart,jend
        ubaro(:,j) = ubaro(:,j)+uu(:,nd-1+j,k)*exp(-zk/h)*dc
      enddo
    else
      do j = jstart,jend
        ubaro(:,j) = ubaro(:,j)+uu(:,j,k)*exp(-zk/h)*dc
      enddo
    endif
  enddo

END SUBROUTINE
