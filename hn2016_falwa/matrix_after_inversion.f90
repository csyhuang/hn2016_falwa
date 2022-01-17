! *** Copying from lines 333-368 from era4004n.f90 ***

SUBROUTINE matrix_after_inversion(k,kmax,jb,jd,qjj,djj,cjj,tj,rj,sjk,tjk)

  INTEGER, INTENT(in) :: k, kmax, jb, jd
  REAL, INTENT(in) :: qjj(jd-2,jd-2),djj(jd-2,jd-2),cjj(jd-2,jd-2),rj(jd-2)
  REAL, INTENT(INOUT) :: sjk(jd-2,jd-2,kmax-1),tjk(jd-2,kmax-1),tj(jd-2)  ! Note that tj is not used in subsequent modules

  integer :: i, j
  real :: xjj(jd-2,jd-2),yj(jd-2)


  do i = 1,jd-2
    do j = 1,jd-2
    xjj(i,j) = 0.
    do kk = 1,jd-2
      xjj(i,j) = xjj(i,j)+qjj(i,kk)*djj(kk,j)
    enddo
    sjk(i,j,k-1) = -xjj(i,j)
    enddo
  enddo

  !         call gemm(qjj,djj,xjj)
  !         sjk(:,:,k-1) = -xjj(:,:)

  !  **** Evaluate rk - Ck Tk ****
  do i = 1,jd-2
    yj(i) = 0.
    do kk = 1,jd-2
      yj(i) = yj(i)+cjj(i,kk)*tj(kk)
    enddo
    yj(i) =  rj(i)-yj(i)
  enddo

  !         call gemv(cjj,tj,yj)
  !         yj(:) = rj(:)-yj(:)
  !         call gemv(qjj,yj,tj)
  !         tjk(:,k-1) = tj(:)


  ! ***** Evaluate Eq. 23 *******
  do i = 1,jd-2
    tj(i) = 0.
    do kk = 1,jd-2
      tj(i) = tj(i)+qjj(i,kk)*yj(kk)
    enddo
    tjk(i,k-1) = tj(i)
  enddo

END SUBROUTINE matrix_after_inversion
