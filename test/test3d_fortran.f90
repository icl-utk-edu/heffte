! /* ----------------------------------------------------------------------
!     -- heffte (version 0.2) --
!        fortran test for 3dfft using heffte
! /* ----------------------------------------------------------------------
! NOTES:
! To use double precision, change functions ending from *_s() to *_d()
! The GPU version uses the same fucntion names, just need to compile with fft=CUFFT
! and make sure a device array is allocated and initialized,
! Compare with test3d_cpu.cpp and test3d_gpu.cpp
! Compilation
! -----------
! cd heffte/test
! make fft=FFTW3 test3d_fortran
! Running example
! ---------------
! mpirun -n 4 ./test3d_fortran
! ------------------------------------------------------------------------- */

program test_fortran

use iso_c_binding
use heffte
implicit none

include 'mpif.h'

! ----------------------------------------
! global parameters
! ----------------------------------------
! MPI parameters
integer fft_comm, me, nprocs
! Geometry
integer fftsize, sendsize, recvsize
integer nfast, nmid, nslow, nfft, precision
! Grid
integer npfast, npmid, npslow, npmidslow, ipfast, ipmid, ipslow
integer i, j, k, n, ierr
integer ilo, ihi, jlo, jhi, klo, khi
! Timing, error, performance
double precision timestart, timestop, timetotal
double precision mydiff, alldiff, gflops
! FFT object pointer
type(c_ptr) :: fft

real(4), allocatable, target :: work(:)
! double precision, allocatable, target :: work(:)

! read size from user
nfast  = 16
nmid   = 16
nslow  = 16
nfft  = nfast * nmid * nslow;

! ----------------------------------------
! MPI initialisation
! ----------------------------------------
call MPI_Init(ierr)
fft_comm = MPI_COMM_WORLD
call MPI_Comm_size(fft_comm, nprocs, ierr)
call MPI_Comm_rank(fft_comm, me, ierr)

! ----------------------------------------
! Create fft plan
! ----------------------------------------
call heffte_create_s(fft_comm, fft)

! Fast algorithm to create a grid of processors, split nprocs into roughly cube roots
npfast = nprocs**(1.0/3.0)
do while (npfast < nprocs)
  if (mod(nprocs,npfast) == 0) exit
  npfast = npfast + 1
enddo

npmidslow = nprocs / npfast
npmid = sqrt(1.0*npmidslow)
do while (npmid < npmidslow)
  if (mod(npmidslow,npmid) == 0) exit
    npmid = npmid + 1
enddo
npslow = nprocs / npfast / npmid


! Partition grid
ipfast = mod(me,npfast)
ipmid  = mod((me/npfast),npmid)
ipslow = me / (npfast*npmid)

ilo = 1.0*ipfast*nfast/npfast + 1
ihi = 1.0*(ipfast+1)*nfast/npfast
jlo = 1.0*ipmid*nmid/npmid + 1
jhi = 1.0*(ipmid+1)*nmid/npmid
klo = 1.0*ipslow*nslow/npslow + 1
khi = 1.0*(ipslow+1)*nslow/npslow

! Setup plan
call heffte_setup_s(fft, nfast, nmid, nslow,  &
        ilo, ihi, jlo, jhi, klo, khi, ilo, ihi, jlo, jhi, klo, khi,  &
        0, fftsize, sendsize, recvsize)

! ----------------------------------------
! Data initialize
! ----------------------------------------
allocate(work(2*fftsize))

! Warmup calls
call heffte_compute_s(fft, c_loc(work), c_loc(work), 1)        ! forward fft
call heffte_compute_s(fft, c_loc(work), c_loc(work), -1)       ! inverse fft

n = 1
do k = klo,khi
  do j = jlo,jhi
    do i = ilo,ihi
      work(n) = n
      n = n + 1
      work(n) = n
      n = n + 1
    enddo
  enddo
enddo


! ----------------------------------------
! Compute ffts
! ----------------------------------------
timestart = mpi_wtime()
call heffte_compute_s(fft, c_loc(work), c_loc(work), 1)        ! forward fft
call heffte_compute_s(fft, c_loc(work), c_loc(work), -1)       ! inverse fft
timestop = mpi_wtime()
timetotal = timestop - timestart

if (me == 0) then
  print *, "------------------------------------------------"
  print *, "        Testing heFFTe library  - Fortran       "
  print *, "------------------------------------------------"
  write(6,*) ' nx =', nfast, ' ny =', nmid, ' nz =', nslow
  write(6,*) ' px =', npfast, ' py =', npmid, ' pz =', npslow
  write(6,*) 'time   =', timetotal, 'secs'
  gflops = 5 * nfft * log(nfft * 1.0)/ timetotal / 1.0D9 / log(2.0)
  write(6,*) 'gflops =', gflops
endif

! ----------------------------------------
! Find max difference between initial/final values
! ----------------------------------------

n = 1
mydiff = 0.0
do k = klo, khi
  do j = jlo, jhi
    do i = ilo, ihi
      if (abs(work(n)-n) > mydiff) mydiff = abs(work(n)-n)
      n = n + 1
      if (abs(work(n)-n) > mydiff) mydiff = abs(work(n)-n)
      n = n + 1
    enddo
  enddo
enddo


call MPI_Allreduce(mydiff, alldiff, 1, MPI_DOUBLE, MPI_MAX, fft_comm, ierr)
if (me == 0)   write(6,*) 'error  =', alldiff


! ----------------------------------------
! Cleaning up
! ----------------------------------------
deallocate(work)
call heffte_destroy_s(fft)
call MPI_Finalize(ierr)

end program test_fortran
