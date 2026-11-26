# HOLO — T-matrix simulation of in-line holograms

HOLO computes electromagnetic extinction, scattering and absorption cross sections and synthetic in-line holograms of homogeneous, axisymmetric particles illuminated by a monochromatic plane wave. Its main application is the generation of training data for machine learning: estimate the three-dimensional position and orientation of bacteria from experimental holograms. Spheroids, finite circular cylinders and capsules are supported.

The solver runs on a CPU, with optional OpenMP parallelization of T-matrix construction and optional CUDA FP64 acceleration of the scattered field and hologram formation. Windows and Linux/WSL use the same physical model. See [the technical documentation](doc/INDEX.md) for the equations, complete CLI reference, output formats, validation and benchmarks.

The original TMatrix-Holo material is licensed for noncommercial
scientific research and education under the [project license](../LICENSE).
Commercial use requires prior written permission. Third-party components
retain their own terms; see the [repository README](../README.md#third-party-terms)
before distributing any combined executable.

## 1. Mishchenko T-matrix code: provenance and integration

HOLO uses Michael I. Mishchenko's fixed-orientation T-matrix programs from NASA Goddard Institute for Space Studies (GISS), specifically the LAPACK-based release whose two source headers state **Last update 08/06/2005** (the date is reproduced exactly as printed in the original files).

Official distribution: https://www.giss.nasa.gov/staff/mmishchenko/tmatrix/

The original sources and preparation scripts must be available in the sibling directory named **mishenko** (the project folder uses this spelling). HOLO does not require the separate NASA random-orientation or Mie programs.

| Precision | Original source files, relative to ../mishenko |
|---|---|
| DP (double, REAL*8) | src/fixed_dp/ampld.lp.f, lpd.f, ampld.par.f |
| QP (extended, REAL*16 with GNU Fortran) | src/fixed_qp/amplq.lp.f, lpq.f, amplq.par.f |

The original programs calculate amplitude and phase matrices of axisymmetric particles at fixed orientations. They implement the extended boundary condition method (EBCM, also called the T-matrix method). Their LAPACK inversion routines replace the historical proprietary NAG routines.

During compilation, prepare_cmake.py stages private copies of these six files, uses ../mishenko/tools/prepare_sources.py to remove the sample output and adapt legacy quadruple-precision intrinsics, and uses prepare_capsule.py to add the capsule profile. HOLO's prepare_holo.py and prepare_qp.py convert the original main programs into callable DP/QP solvers; QP symbols are isolated from DP, while both share the final FP64 T-matrix. The generated sources and bundled LAPACK/BLAS routines are compiled into libholo_fortran_cpu.a and, when enabled, libholo_fortran_openmp.a. tmatrix_openmp.py inserts OpenMP directives in the independent surface-integral loops. The original NASA source files are not changed.

QP uses binary128 arithmetic for the extended-precision parts of the original solver, including the LAPACK inversion; the final T-matrix and certain operations retain binary64 precision. QP therefore does not mean that every operation is performed in binary128. CUDA uses FP64, not binary128. The output intensities are IEEE Float32. The original header permits unrestricted non-profit scientific research and requests acknowledgment of the source and citation of the relevant publications. Do not assume broader redistribution or commercial-use rights.

Reference: M. I. Mishchenko, *Calculation of the amplitude matrix for a nonspherical particle in a fixed orientation*, Applied Optics **39**, 1026–1031 (2000), https://doi.org/10.1364/AO.39.001026.

## 2. Source tree

~~~text
workspace/
  mishenko/
    src/fixed_dp/                 # Original NASA DP sources
    src/fixed_qp/                 # Original NASA QP sources
    tools/prepare_sources.py
    tools/prepare_capsule.py
  holo/
    README.md
    CMakeLists.txt
    cmake/Holo.cmake
    holo.cpp
    prepare_cmake.py
    prepare_holo.py
    prepare_qp.py
    tmatrix_openmp.py
    omp_config.f90
    fftw/                         # Local Windows FFTW installation
    mingw/CMakeLists.txt
    mingw/build_all.bat
    wsl/CMakeLists.txt
    wsl/build_all.sh
    cuda/gpu_holo.cu
    cuda/gpu_holo_abi.h
    cuda/gpu_holo_dll.cpp
    cuda/gpu_holo_win.cpp
    cuda/windows/build_cuda.bat
    cuda/wsl/build_all.sh
    doc/INDEX.md
~~~

Keep holo and mishenko as siblings when extracting a source archive. Generated Fortran copies are written into the chosen build directory; do not edit these generated files directly. The original NASA files are supplied separately; see the [repository dependency instructions](../README.md#dependencies).

## 3. Build modes

Use the single CMake cache variable HOLO_MODE. The default is CPU_OPENMP_CUDA for a fresh root, MinGW or WSL configuration. The compatibility entry point cuda/wsl defaults to CPU_OPENMP_CUDA.

| HOLO_MODE | CPU T-matrix | Hologram backend | Targets | Script argument / build directory |
|---|---|---|---|---|
| CPU | Serial DP/QP | FFTW CPU | mish_holo_cpu | cpu / build_cpu |
| CPU_OPENMP | Serial and OpenMP DP/QP | FFTW CPU | mish_holo_cpu, mish_holo_openmp | openmp / build_openmp |
| CPU_OPENMP_CUDA | Serial and OpenMP DP/QP | FFTW CPU or CUDA FP64/cuFFT | mish_holo_cpu, mish_holo_openmp, mish_holo_cuda | all or cuda / build |

The full mode builds all three independent executable targets in the same CMake project. Select Build All in CLion to compile all three. The serial target links holo_fortran_cpu, without OpenMP; the other two share holo_fortran_openmp. Both libraries use the same generated Fortran sources, compiled with different options. The old HOLO_ENABLE_CPU, HOLO_ENABLE_OPENMP and HOLO_ENABLE_CUDA switches are derived internally; set HOLO_MODE instead. Existing legacy caches are migrated on first configuration when HOLO_MODE is absent. The script arguments all, cuda and the older both all select the full three-target build. Separate directories prevent confusion between previously built serial and OpenMP executables and preserve earlier simulation results.

## 4. Dependencies

| Dependency | CPU | CPU_OPENMP | CPU_OPENMP_CUDA |
|---|---|---|---|
| GNU C++17 and GNU Fortran with REAL*16 / libquadmath | Required | Required | Required |
| CMake >= 3.20, Ninja, Python 3 | Required | Required | Required |
| Double-precision FFTW3 headers and library | Required | Required | Required for CPU reference |
| Mishchenko LAPACK/BLAS DP/QP | Required | Required | Required |
| GNU OpenMP (libgomp) | No | Required | Required |
| NVIDIA driver, CUDA toolkit and cuFFT | No | No | Required |
| MSVC x64 toolchain (native Windows only) | No | No | Required for separate CUDA DLL |

The LAPACK/BLAS implementation distributed with the original Mishchenko sources is sufficient; an external OpenBLAS installation is not needed. Python uses only its standard library for source preparation and automatic validation. NumPy and Matplotlib are optional for post-processing.

Tested environment: MinGW-w64 GCC/GFortran 11.2, Visual Studio 2022 x64, CUDA 12.9; Ubuntu WSL GCC/GFortran 13.3 and CUDA 12.9; NVIDIA TITAN V (compute capability 7.0) and OpenMP 4.5. Native Linux uses the same Linux build system as WSL; validation was performed in WSL.

## 5. Native Windows compilation (MinGW)

Install a compatible MinGW-w64 GNU C++/Fortran toolchain, CMake, Ninja and Python 3 and make the latter tools available on PATH. Provide compatible FFTW Windows libraries under holo/fftw, or configure FFTW_ROOT. Install CUDA and MSVC only for the hybrid mode. The tested toolchain is F:\mingw64_11.2\bin; override MINGW_BIN if necessary.

From the holo root, in Command Prompt:

~~~bat
mingw\build_all.bat cpu
mingw\build_all.bat openmp
mingw\build_all.bat all
~~~

Run the command for the desired mode; omitting the argument selects all. Each script configures Release with Ninja, builds all targets and runs CTest, stopping on failure. Output directories are mingw\build_cpu, mingw\build_openmp and mingw\build respectively.

The dedicated Windows CUDA build shortcut is:

~~~bat
mingw\build_all_cuda.bat
~~~

It selects the full CPU_OPENMP_CUDA configuration, builds the separate
nvcc/MSVC CUDA DLL when needed, builds all three MinGW executables in
mingw/build, and runs all six CTest tests. It is equivalent to
mingw\build_all.bat cuda and preserves existing run directories.

For CUDA, the device code is built **separately as a DLL using nvcc and MSVC**, not by MinGW. The independent build command is:

~~~bat
cuda\windows\build_cuda.bat
~~~

It produces cuda\windows\build\holo_cuda.dll and deploys CUDA runtime dependencies. The default CUDA root is %ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9, with Visual Studio 2022 Community x64 vcvars64.bat (2019 Community fallback). Set HOLO_CUDA_ROOT and HOLO_VCVARS64 for other installed toolchains. The batch targets sm_70 for the TITAN V and uses the dynamic CUDA runtime. Choose a toolkit that supports the target GPU architecture; the Windows batch currently specifies sm_70 explicitly.

CMake automatically calls the batch if the CUDA DLL needs building and copies the DLLs beside mish_holo_cuda.exe. The executable loads the DLL through a versioned C ABI, with caller-owned buffers and error codes; no C++ object or exception crosses the MinGW/MSVC boundary. The Fortran solver stays in the MinGW executable. The GNU runtime DLLs are copied from the active compiler's bin directory (including libgfortran-5.dll, libquadmath-0.dll, libgcc_s_seh-1.dll, libstdc++-6.dll, libwinpthread-1.dll and libgomp-1.dll for OpenMP). The latter is from F:\mingw64_11.2\bin on the tested PC, rather than IF-DDA's MSYS2 deployment directory.

Manual CMake example (hybrid):

~~~bat
cmake -S mingw -B mingw/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHOLO_MODE=CPU_OPENMP_CUDA -DCMAKE_CXX_COMPILER=F:/mingw64_11.2/bin/g++.exe -DCMAKE_Fortran_COMPILER=F:/mingw64_11.2/bin/gfortran.exe
cmake --build mingw/build --parallel
ctest --test-dir mingw/build --output-on-failure
mingw\build\mish_holo_cuda.exe --help
~~~

For the serial or OpenMP-only mode, use CPU or CPU_OPENMP and choose the matching separate build directory.

## 6. Linux / Ubuntu WSL compilation

For Debian/Ubuntu, install the CPU build dependencies:

~~~bash
sudo apt install build-essential gfortran cmake ninja-build python3 libfftw3-dev
~~~

For CUDA, install a compatible NVIDIA driver and CUDA toolkit (nvcc, CUDA runtime, cuFFT). On WSL, use the Windows NVIDIA driver with WSL GPU support and the Linux CUDA toolkit inside WSL. Ensure nvcc is accessible; the script adds /usr/local/cuda/bin to PATH. Check the installed compiler with nvcc --version and the visible GPU with nvidia-smi.

From the holo root:

~~~bash
bash wsl/build_all.sh cpu
bash wsl/build_all.sh openmp
bash wsl/build_all.sh all
~~~

These commands build and test in wsl/build_cpu, wsl/build_openmp and wsl/build. The default argument is all. For Linux, the CUDA code is compiled directly by CMake/nvcc and linked with CUDA::cudart and CUDA::cufft; a Windows DLL is neither built nor used.

Manual CMake example (hybrid):

~~~bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S wsl -B wsl/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHOLO_MODE=CPU_OPENMP_CUDA -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build wsl/build --parallel
ctest --test-dir wsl/build --output-on-failure
wsl/build/mish_holo_cuda --help
~~~

The root CMakeLists.txt is also a valid entry point on both platforms. In CLion select the corresponding GNU C++ and Fortran toolchain and add -DHOLO_MODE=CPU, CPU_OPENMP or CPU_OPENMP_CUDA to the CMake profile options (one mode per profile). Use Release for performance. On Linux, CMAKE_CUDA_ARCHITECTURES selects the GPU architecture (default 70). FFTW_ROOT can override the default FFTW search path; HOLO_CUDA_DLL_DIR selects the Windows DLL build directory. Use a fresh build directory after moving the source tree or changing the compiler. The older Linux build.sh remains a serial CPU-only compatibility script producing the historical executable name mish_holo.

## 7. Example: 8 × 3 µm bacterium, 60 orientations

From mingw\build on Windows (choose a new run name if it already exists):

~~~bat
mish_holo_cuda.exe --name bacterie_80x30_cuda_60 --length-um 8 --diameter-um 3 --tmatrix-precision qp --z-um 25 --nx 512 --ny 512 --pad 3 --theta-steps 59 --theta-end 177 --threads 16
~~~

| Argument | Meaning |
|---|---|
| mish_holo_cuda.exe | Native Windows hybrid executable using the CUDA DLL |
| --name bacterie_80x30_cuda_60 | Output directory name suffix |
| --length-um 8 --diameter-um 3 | Total axial length and transverse diameter, in micrometres |
| --tmatrix-precision qp | Build the T-matrix once in the extended-precision CPU solver |
| --z-um 25 | Observation plane 25 µm downstream of the particle centre |
| --nx 512 --ny 512 | Output image dimensions |
| --pad 3 | Internal FFT size 1536 × 1536 |
| --theta-steps 59 --theta-end 177 | 59 intervals, 60 images from 0 to 177 degrees, 3-degree increment, phi=0 |
| --threads 16 | Request 16 OpenMP threads for T-matrix construction |

Defaults: spheroid, vacuum wavelength 660 nm, particle index 1.38+0i, medium 1.33, NA=0.8, X polarization, centred particle, 100 nm pixels, transverse detection. To retain the same 51.2 µm field of view with 200 nm pixels, use --nx 256 --ny 256 --dx-um 0.2 --dy-um 0.2. The T-matrix is reused within a sweep. Exactly 60 images are generated by 59 intervals, not by --theta-steps 60 (which gives 61 images at the default end angle of 180 degrees).

On Linux use wsl/build/mish_holo_cuda with the same arguments. For CPU use mish_holo_cpu (serial) or mish_holo_openmp (parallel T-matrix). For mish_holo_cpu omit --threads or set it to 1; OpenMP modes accept 1..256, defaulting to OMP_NUM_THREADS if set, otherwise up to 8. Use --tmatrix-precision auto (default) to try DP then QP, or dp to require DP.

## 8. Output and viewing

Results are written under run/run_<name> beside the executable, independent of the working directory. Existing run names are refused, not overwritten. The full sweep produces four files:

~~~text
log.txt
orientations.csv
cross_sections.txt
holograms.raw
~~~

The RAW stack stores normalized intensities (incident background = 1), IEEE Float32 little-endian, no header or gaps, x fastest, then y, then frame. For the example it contains 62,914,560 bytes (60 MiB). In ImageJ choose File > Import > Raw, 32-bit Real, width=512, height=512, number of images=60, little-endian, offset=0, gap=0. A single-image run produces hologram.raw and log.txt. Only use completed runs (last status=complete in log.txt). cross_sections.txt contains theta, Cext, Csca and Cabs in square micrometres; orientations.csv includes frame offsets, intensity statistics and timings. See [doc/INDEX.md](doc/INDEX.md) for Python and Gnuplot examples.

## 9. Validation and limitations

All acceleration methods preserve the same discretized physical calculation. CUDA precontracts the incident multipoles once per orientation on the CPU, reducing the per-direction evaluation from O(NMAX^3) to O(NMAX^2), then evaluates amplitudes, sections and the angular spectrum on the GPU with cuFFT. OpenMP parallelizes surface-integral rows, including the m=0 convergence search, without changing each quadrature sum's order. LAPACK inversions and the sequence of azimuthal blocks are still serial.

Measured on the tested PC, with QP:

| Spheroid (µm) | Images | T-matrix without OpenMP (s) | T-matrix, 16 threads (s) | Total hybrid, 16 threads (s) |
|---|---:|---:|---:|---:|
| 6 × 2.5 | 11 | 15.7803 | 4.4197 | 4.8899 |
| 8 × 3 | 60 | 45.5660 | 12.3987 | 14.8196 |

The RAW stacks and cross sections match the historical non-OpenMP reference exactly in these tests. For the 2 × 0.8 µm QP case with 11 images of 512² pixels, the historical serial CPU time was 117.3364 s and native CUDA time 0.9757 s; all 2,883,584 Float32 pixels match exactly. These are measured results for specific configurations, not universal speedups. The [technical documentation](doc/INDEX.md) retains the full benchmark tables and reference run names.

Independent Mie tests check the sphere cross sections and five hologram pixels. The solver assumes a homogeneous non-absorbing background and an ideal circular pupil. It does not model a calibrated microscope objective, interfaces, detector noise, aberrations, pixel-area integration or apodization. Agreement between CPU, OpenMP and CUDA verifies the implementation, not absolute T-matrix accuracy for every geometry. Check convergence in multipole order, surface quadrature, FFT padding and pixel pitch, especially for large particles. The T-matrix order and quadrature are limited by the original parameter files (NPN1=100 and NPNG1=500). Persistent T-matrix caching between runs has not yet been implemented.

Respect the original NASA source conditions and cite Mishchenko's publications in scientific work.

## Three-target build validation

The full build has been compiled and tested on native Windows and Ubuntu WSL: six CTest tests passed on each platform. Windows PE imports confirm that mish_holo_cpu.exe does not depend on libgomp-1.dll, while mish_holo_openmp.exe and mish_holo_cuda.exe do. The serial executable rejects --threads 2. Older binaries named mish_holo may remain in existing build directories; they are not current CMake targets. Existing simulation results are preserved.

Supported build workflows validated on 26 November 2026:

| Build procedure | Output directory | Status |
|---|---|---|
| Windows build_all (Release), including build_all_cuda.bat | mingw/build/ | Operational; 6/6 CTest passed |
| Windows CLion (Debug) | mingw/cmake-build-debug/ | Build and execution validated |
| WSL build_all (Release) | wsl/build/ | Operational; 6/6 CTest passed |
| WSL CLion (Debug) | wsl/cmake-build-debug/ | Build and execution validated |

Validation was performed on native Windows and Ubuntu WSL. The Linux
build procedure also applies to native Linux with the required toolchain.

## CLion WSL CUDA discovery

CMake searches for nvcc in standard toolkit locations. To select a specific CUDA installation, set -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc in the WSL CMake profile, adapting the path to the installed toolkit. See doc/INDEX.md, Sections 5.6 and 5.7, for executable names, output directories and CLion configuration.
