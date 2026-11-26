! This file is compiled only when HOLO_ENABLE_OPENMP=ON.
subroutine holo_set_threads(requested, actual) bind(C, name="holo_set_threads")
  use iso_c_binding, only: c_int
  use omp_lib
  implicit none
  integer(c_int), intent(in), value :: requested
  integer(c_int), intent(out) :: actual
  integer :: count, status
  character(len=64) :: env
  count = requested
  if (count == 0) then
    call get_environment_variable("OMP_NUM_THREADS", env, status=status)
    if (status == 0 .and. len_trim(env) > 0) then
      count = omp_get_max_threads()
    else
      count = min(8, omp_get_max_threads())
    end if
  end if
  call omp_set_dynamic(.false.)
  call omp_set_num_threads(count)
  !$omp parallel default(none) shared(actual)
  !$omp single
  actual = omp_get_num_threads()
  !$omp end single
  !$omp end parallel
end subroutine
