! Routine pour lire rapidement les matrices: lancer
!  f2py -c readmat.f90 -m readmat --fcompiler=gnu95
! pour compiler

program test

  implicit none

  integer,parameter :: nnz=10
  integer           :: icoo(nnz),jcoo(nnz)
  complex*16        :: zcoo(nnz)
  character*64      :: fname

  fname = 'results/LNS.dat'

  call readcomplexmat(fname,nnz,icoo,jcoo,zcoo)

end program test

subroutine readcomplexmat(fname,nnz,icoo,jcoo,zcoo)

  implicit none
  character*64,intent(in) :: fname
  integer,intent(in)      :: nnz
  integer,intent(out)     :: icoo(nnz),jcoo(nnz)
  complex*16,intent(out)  :: zcoo(nnz)
  integer                 :: i
  character*64            :: str

  open(unit = 10,file = trim(fname),position = 'rewind')

  do i=1,4
     read(10,*) str
  end do

  do i=1,nnz
     read(10,*) icoo(i),jcoo(i),zcoo(i)
  end do

  close(10)

end subroutine readcomplexmat

subroutine readrealmat(fname,nnz,icoo,jcoo,zcoo)

  implicit none
  character*64,intent(in) :: fname
  integer,intent(in)      :: nnz
  integer,intent(out)     :: icoo(nnz),jcoo(nnz)
  real*8,intent(out)      :: zcoo(nnz)
  integer                 :: i
  character*64            :: str

  open(unit = 10,file = trim(fname),position = 'rewind')

  do i=1,4
     read(10,*) str
  end do

  do i=1,nnz
     read(10,*) icoo(i),jcoo(i),zcoo(i)
  end do

  close(10)

end subroutine readrealmat

