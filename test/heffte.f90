! /* ----------------------------------------------------------------------
!     -- HEFFTE (version 0.2) --
!        Fortran wrapper for heFFTe
! ------------------------------------------------------------------------- */

module heffte

interface

! Creator/Destructor function
! DOUBLE precision

  subroutine heffte_create_d(comm, ptr) &
    bind(c, name='heffte_create_fortran_d')
    use iso_c_binding
    integer(c_int), value :: comm
    type(c_ptr) :: ptr
  end subroutine heffte_create_d

  subroutine heffte_destroy_d(ptr) &
    bind(c, name='heffte_destroy_d')
    use iso_c_binding
    type(c_ptr), value :: ptr
  end subroutine heffte_destroy_d

  subroutine heffte_setup_d(ptr, nfast, nmid, nslow,  &
    in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi,  &
    out_ilo, out_ihi, out_jlo, out_jhi, out_klo, out_khi,  &
    permute, fftsize, sendsize, recvsize) &
    bind(c, name='heffte_setup_fortran_d')
    use iso_c_binding
    type(c_ptr), value :: ptr
    integer(c_int), value :: nfast,nmid,nslow
    integer(c_int), value :: in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi
    integer(c_int), value :: out_ilo,out_ihi,out_jlo,out_jhi,out_klo,out_khi
    integer(c_int), value :: permute
    integer(c_int) :: fftsize,sendsize,recvsize
  end subroutine heffte_setup_d

  subroutine heffte_compute_d(ptr, in, out, flag) &
    bind(c, name='heffte_compute_d')
    use iso_c_binding
    type(c_ptr), value :: ptr
    type(c_ptr), value :: in, out
    integer(c_int), value :: flag
  end subroutine heffte_compute_d

!  SINGLE precision

  subroutine heffte_create_s(comm, ptr) &
    bind(c, name='heffte_create_fortran_s')
    use iso_c_binding
    integer(c_int), value :: comm
    type(c_ptr) :: ptr
  end subroutine heffte_create_s

  subroutine heffte_destroy_s(ptr) &
    bind(c, name='heffte_destroy_s')
    use iso_c_binding
    type(c_ptr), value :: ptr
  end subroutine heffte_destroy_s

  subroutine heffte_setup_s(ptr, nfast, nmid, nslow,  &
    in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi,  &
    out_ilo, out_ihi, out_jlo, out_jhi, out_klo, out_khi,  &
    permute, fftsize, sendsize, recvsize) &
    bind(c, name='heffte_setup_fortran_s')
    use iso_c_binding
    type(c_ptr), value :: ptr
    integer(c_int), value :: nfast,nmid,nslow
    integer(c_int), value :: in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi
    integer(c_int), value :: out_ilo,out_ihi,out_jlo,out_jhi,out_klo,out_khi
    integer(c_int), value :: permute
    integer(c_int) :: fftsize,sendsize,recvsize
  end subroutine heffte_setup_s

  subroutine heffte_compute_s(ptr, in, out, flag) &
    bind(c, name='heffte_compute_s')
    use iso_c_binding
    type(c_ptr), value :: ptr
    type(c_ptr), value :: in, out
    integer(c_int), value :: flag
  end subroutine heffte_compute_s

  subroutine alloc_device_d(ptr, size) bind(c)
    use iso_c_binding
    type(c_ptr), value :: ptr
    integer(c_int), value :: size
  end subroutine alloc_device_d

  subroutine alloc_device_s(ptr, size) bind(c)
    use iso_c_binding
    type(c_ptr), value :: ptr
    integer(c_int), value :: size
  end subroutine alloc_device_s

  subroutine dealloc_device_d(ptr) bind(c)
    use iso_c_binding
    type(c_ptr), value :: ptr
  end subroutine dealloc_device_d

  subroutine dealloc_device_s(ptr) bind(c)
    use iso_c_binding
    type(c_ptr), value :: ptr
  end subroutine dealloc_device_s

end interface

end module heffte
