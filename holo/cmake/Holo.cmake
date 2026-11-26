# Shared CMake implementation for CPU and CUDA. Do not edit the NASA sources.
get_filename_component(HOLO_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(MISH_ROOT "${HOLO_DIR}/../mishenko" ABSOLUTE)
# The mode is the single public selector. Migrate older caches on first use.
if(NOT DEFINED HOLO_MODE)
  if(HOLO_ENABLE_CUDA)
    set(_holo_default_mode CPU_OPENMP_CUDA)
  elseif(DEFINED HOLO_ENABLE_OPENMP AND NOT HOLO_ENABLE_OPENMP)
    set(_holo_default_mode CPU)
  elseif(DEFINED HOLO_DEFAULT_MODE)
    set(_holo_default_mode "${HOLO_DEFAULT_MODE}")
  else()
    set(_holo_default_mode CPU_OPENMP_CUDA)
  endif()
else()
  set(_holo_default_mode "${HOLO_MODE}")
endif()
set(HOLO_MODE "${_holo_default_mode}" CACHE STRING "HOLO execution mode: CPU, CPU_OPENMP, CPU_OPENMP_CUDA")
set_property(CACHE HOLO_MODE PROPERTY STRINGS CPU CPU_OPENMP CPU_OPENMP_CUDA)
foreach(_legacy HOLO_ENABLE_CPU HOLO_ENABLE_OPENMP HOLO_ENABLE_CUDA)
  unset(${_legacy} CACHE)
endforeach()
set(HOLO_ENABLE_CPU ON)
set(HOLO_ENABLE_OPENMP OFF)
set(HOLO_ENABLE_CUDA OFF)
if(HOLO_MODE STREQUAL "CPU")
  # Serial CPU reference.
elseif(HOLO_MODE STREQUAL "CPU_OPENMP")
  set(HOLO_ENABLE_OPENMP ON)
elseif(HOLO_MODE STREQUAL "CPU_OPENMP_CUDA")
  set(HOLO_ENABLE_OPENMP ON)
  set(HOLO_ENABLE_CUDA ON)
else()
  message(FATAL_ERROR "Invalid HOLO_MODE='${HOLO_MODE}'. Choose CPU, CPU_OPENMP or CPU_OPENMP_CUDA.")
endif()
if(HOLO_ENABLE_CUDA AND NOT WIN32)
  set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA compute capability (70 = TITAN V)")
  # CLion's WSL toolchain may not inherit the interactive shell's CUDA PATH.
  # Preserve explicit compiler choices and CUDACXX; otherwise search standard
  # toolkit locations before enable_language, including a failed cache entry.
  if(NOT CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "")
    find_program(HOLO_NVCC_EXECUTABLE NAMES nvcc
      HINTS "${CUDAToolkit_ROOT}/bin" "$ENV{CUDAToolkit_ROOT}/bin"
        "$ENV{CUDA_HOME}/bin" "$ENV{CUDA_PATH}/bin" /usr/local/cuda/bin)
    if(NOT HOLO_NVCC_EXECUTABLE)
      message(FATAL_ERROR "CUDA mode requires nvcc. Install the Linux CUDA toolkit or set -DCMAKE_CUDA_COMPILER=/path/to/nvcc.")
    endif()
    set(CMAKE_CUDA_COMPILER "${HOLO_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(HOLO_ENABLE_CUDA AND WIN32 AND NOT MINGW)
  message(FATAL_ERROR "Use the MinGW/GNU C++ and Fortran toolchain; the CUDA DLL is built separately with nvcc/MSVC.")
endif()
message(STATUS "HOLO mode: ${HOLO_MODE} (CPU=${HOLO_ENABLE_CPU}, OpenMP=${HOLO_ENABLE_OPENMP}, CUDA=${HOLO_ENABLE_CUDA})")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(CORE "${GENERATED}/holo_core.f")
set(QCORE "${GENERATED}/holo_qp.f")
set(QPAR "${GENERATED}/build_src/fixed_qp/amplq.par.f")
set(LPD "${GENERATED}/build_src/fixed_dp/lpd.f")
set(PAR "${GENERATED}/build_src/fixed_dp/ampld.par.f")
add_custom_command(
  OUTPUT "${CORE}" "${LPD}" "${PAR}" "${QCORE}" "${QPAR}"
  COMMAND "${Python3_EXECUTABLE}" "${HOLO_DIR}/prepare_cmake.py" "${GENERATED}"
  DEPENDS "${HOLO_DIR}/prepare_cmake.py" "${HOLO_DIR}/prepare_holo.py" "${HOLO_DIR}/prepare_qp.py"
    "${HOLO_DIR}/tmatrix_openmp.py"
    "${MISH_ROOT}/tools/prepare_sources.py" "${MISH_ROOT}/tools/prepare_capsule.py"
    "${MISH_ROOT}/src/fixed_dp/ampld.lp.f"
    "${MISH_ROOT}/src/fixed_dp/lpd.f"
    "${MISH_ROOT}/src/fixed_dp/ampld.par.f"
    "${MISH_ROOT}/src/fixed_qp/amplq.lp.f"
    "${MISH_ROOT}/src/fixed_qp/lpq.f"
    "${MISH_ROOT}/src/fixed_qp/amplq.par.f"
  VERBATIM COMMENT "Preparing private Fortran source copies")
add_custom_target(holo_generated DEPENDS "${CORE}" "${LPD}" "${PAR}" "${QCORE}" "${QPAR}")

if(HOLO_ENABLE_CPU)
if(MINGW)
  set(FFTW_ROOT "${HOLO_DIR}/fftw" CACHE PATH "Local FFTW Windows bundle")
  find_path(FFTW_INCLUDE_DIR NAMES fftw3.h HINTS "${FFTW_ROOT}" "${FFTW_ROOT}/include" REQUIRED)
  find_library(FFTW_LIBRARY NAMES fftw3.dll.a fftw3-3.dll.a fftw3
    HINTS "${FFTW_ROOT}" "${FFTW_ROOT}/lib" REQUIRED)
else()
  set(FFTW_ROOT "" CACHE PATH "Optional FFTW installation prefix")
  find_path(FFTW_INCLUDE_DIR NAMES fftw3.h HINTS "${FFTW_ROOT}/include" REQUIRED)
  find_library(FFTW_LIBRARY NAMES fftw3 HINTS "${FFTW_ROOT}/lib" REQUIRED)
endif()
endif()

# Compile separate serial and OpenMP libraries from the same generated sources.
# Never link both variants into one executable (they export the same symbols).
function(holo_add_fortran target)
  add_library(${target} STATIC "${CORE}" "${LPD}" "${QCORE}")
  add_dependencies(${target} holo_generated)
  set_target_properties(${target} PROPERTIES Fortran_FORMAT FIXED)
  target_include_directories(${target} PRIVATE "${GENERATED}/build_src/fixed_dp" "${GENERATED}/build_src/fixed_qp")
  target_compile_options(${target} PRIVATE -std=legacy -ffixed-line-length-none -fallow-argument-mismatch -w)
endfunction()
holo_add_fortran(holo_fortran_cpu)
if(HOLO_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS Fortran)
  holo_add_fortran(holo_fortran_openmp)
  target_sources(holo_fortran_openmp PRIVATE "${HOLO_DIR}/omp_config.f90")
  set_source_files_properties("${HOLO_DIR}/omp_config.f90" PROPERTIES Fortran_FORMAT FREE)
  target_link_libraries(holo_fortran_openmp PUBLIC OpenMP::OpenMP_Fortran)
  # Large work arrays stay off the Windows thread stack. No concurrent
  # calls to the legacy routines: only their independent integrals run in parallel.
  target_compile_options(holo_fortran_openmp PRIVATE -fmax-stack-var-size=32768)
  target_compile_definitions(holo_fortran_openmp INTERFACE HOLO_OPENMP=1)
  message(STATUS "HOLO OpenMP Fortran: ${OpenMP_Fortran_FLAGS}; ${OpenMP_Fortran_LIBRARIES}")
endif()

enable_testing()
function(holo_add_executable target test_prefix fortran_lib)
  add_executable(${target} "${HOLO_DIR}/holo.cpp")
  target_link_libraries(${target} PRIVATE ${fortran_lib})
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO LINKER_LANGUAGE CXX)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
  add_test(NAME ${test_prefix}_help COMMAND ${target} --help)
  add_test(NAME ${test_prefix}_numerical
    COMMAND "${Python3_EXECUTABLE}" "${HOLO_DIR}/test_cmake.py" "$<TARGET_FILE:${target}>")
  set_tests_properties(${test_prefix}_numerical PROPERTIES TIMEOUT 120)
endfunction()

function(holo_add_cpu target test_prefix fortran_lib)
  holo_add_executable(${target} ${test_prefix} ${fortran_lib})
  target_include_directories(${target} PRIVATE "${FFTW_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${FFTW_LIBRARY}")
endfunction()
holo_add_cpu(mish_holo_cpu hologram_cpu holo_fortran_cpu)
if(HOLO_ENABLE_OPENMP)
  holo_add_cpu(mish_holo_openmp hologram_openmp holo_fortran_openmp)
endif()

if(HOLO_ENABLE_CUDA)
  holo_add_executable(mish_holo_cuda hologram_cuda holo_fortran_openmp)
  target_compile_definitions(mish_holo_cuda PRIVATE HOLO_CUDA=1)
  if(WIN32)
    # nvcc uses MSVC in a separate process. MinGW only sees the C ABI.
    set(HOLO_CUDA_DLL_DIR "${HOLO_DIR}/cuda/windows/build" CACHE PATH "Separate nvcc DLL build directory")
    set(cuda_dll "${HOLO_CUDA_DLL_DIR}/holo_cuda.dll")
    add_custom_command(OUTPUT "${cuda_dll}"
      COMMAND cmd /c call "${HOLO_DIR}/cuda/windows/build_cuda.bat" "${HOLO_CUDA_DLL_DIR}"
      DEPENDS "${HOLO_DIR}/cuda/windows/build_cuda.bat"
        "${HOLO_DIR}/cuda/gpu_holo.cu" "${HOLO_DIR}/cuda/gpu_holo.hpp"
        "${HOLO_DIR}/cuda/gpu_holo_dll.cpp" "${HOLO_DIR}/cuda/gpu_holo_abi.h"
      VERBATIM COMMENT "Building separate CUDA DLL using nvcc/MSVC")
    add_custom_target(holo_cuda_dll DEPENDS "${cuda_dll}")
    add_custom_target(holo_cuda_runtime
      COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${HOLO_CUDA_DLL_DIR}"
        "-DDEST_DIR=$<TARGET_FILE_DIR:mish_holo_cuda>" -P "${HOLO_DIR}/cmake/DeployCuda.cmake"
      VERBATIM COMMENT "Deploying CUDA DLLs next to mish_holo_cuda.exe")
    add_dependencies(holo_cuda_runtime holo_cuda_dll)
    add_dependencies(mish_holo_cuda holo_cuda_runtime)
    target_sources(mish_holo_cuda PRIVATE "${HOLO_DIR}/cuda/gpu_holo_win.cpp")
  else()
    target_sources(mish_holo_cuda PRIVATE "${HOLO_DIR}/cuda/gpu_holo.cu")
    target_link_libraries(mish_holo_cuda PRIVATE CUDA::cudart CUDA::cufft)
    target_compile_options(mish_holo_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    set_target_properties(mish_holo_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
  endif()
endif()

if(MINGW)
  get_filename_component(compiler_bin "${CMAKE_Fortran_COMPILER}" DIRECTORY)
  get_filename_component(cxx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
  # Deploy DLLs beside the executable for use outside the compiler environment.
  set(holo_runtime_dlls libgfortran-5.dll libquadmath-0.dll libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
  if(HOLO_ENABLE_OPENMP)
    list(APPEND holo_runtime_dlls libgomp-1.dll)
  endif()
  foreach(dll IN LISTS holo_runtime_dlls)
    unset(runtime_dll)
    unset(runtime_dll CACHE)
    find_file(runtime_dll NAMES "${dll}" HINTS "${compiler_bin}" "${cxx_bin}" NO_DEFAULT_PATH)
    if(runtime_dll)
      foreach(target mish_holo_cpu mish_holo_openmp mish_holo_cuda)
        if(TARGET ${target})
          if(dll STREQUAL "libgomp-1.dll" AND target STREQUAL "mish_holo_cpu")
            continue()
          endif()
          add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${runtime_dll}" "$<TARGET_FILE_DIR:${target}>" VERBATIM)
        endif()
      endforeach()
    elseif(dll STREQUAL "libgomp-1.dll")
      message(FATAL_ERROR "Cannot find libgomp-1.dll beside the selected GNU toolchain.")
    endif()
  endforeach()
  if(HOLO_ENABLE_CPU AND FFTW_LIBRARY MATCHES "\\.dll\\.a$")
    find_file(FFTW_RUNTIME NAMES libfftw3-3.dll libfftw3.dll
      HINTS "${FFTW_ROOT}" "${FFTW_ROOT}/bin" REQUIRED)
    foreach(target mish_holo_cpu mish_holo_openmp)
      if(TARGET ${target})
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${FFTW_RUNTIME}" "$<TARGET_FILE_DIR:${target}>" VERBATIM)
      endif()
    endforeach()
  endif()
endif()

