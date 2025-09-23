SUBROUTINE matrix_after_inversion(k,kmax,jd,qjj,djj,cjj,rj,sjk,tjk)

  INTEGER, INTENT(in) :: k, kmax, jd
  REAL, INTENT(in) :: qjj(jd-2,jd-2),djj(jd-2,jd-2),cjj(jd-2,jd-2),rj(jd-2)
  REAL, INTENT(INOUT) :: sjk(kmax-1,jd-2,jd-2),tjk(kmax-1,jd-2)  ! Note that tj is not used in subsequent modules

  integer :: i, j
  real :: xjj(jd-2,jd-2),yj(jd-2),tj(jd-2)

  tj(:) = tjk(k,:)
  do i = 1,jd-2
    do j = 1,jd-2
    xjj(i,j) = 0.
    do kk = 1,jd-2
      xjj(i,j) = xjj(i,j)+qjj(i,kk)*djj(kk,j)
    enddo
    sjk(k-1,i,j) = -xjj(i,j)
    enddo
  enddo

  !  **** Evaluate rk - Ck Tk ****
  do i = 1,jd-2
    yj(i) = 0.
    do kk = 1,jd-2
      yj(i) = yj(i)+cjj(i,kk)*tj(kk)
    enddo
    yj(i) =  rj(i)-yj(i)
  enddo

  ! ***** Evaluate Eq. 23 *******
  do i = 1,jd-2
    tj(i) = 0.
    do kk = 1,jd-2
      tj(i) = tj(i)+qjj(i,kk)*yj(kk)
    enddo
    tjk(k-1,i) = tj(i)
  enddo

END SUBROUTINE matrix_after_inversion
