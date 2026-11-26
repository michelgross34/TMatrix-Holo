# HOLO — T-matrix calculation of in-line holograms

**Technical documentation — 26 November 2026**

## 1. Purpose

HOLO computes electromagnetic extinction, scattering and absorption cross sections and in-line holograms of homogeneous axisymmetric dielectric particles illuminated by a monochromatic plane wave. Its main scientific objective is to generate synthetic holograms of bacteria for training a neural network to recover their three-dimensional position and orientation from experimental holograms.

The CPU implementation uses Michael I. Mishchenko's T-matrix solver, C++17 and double-precision FFTW. The hybrid implementation builds the same Fortran DP/QP T-matrix on the CPU and uses CUDA FP64 and cuFFT to calculate scattering amplitudes and holograms. OpenMP optionally accelerates surface integrals on the CPU. The final intensities use IEEE Float32. Windows MinGW and Linux/WSL share the same sources; on Windows, CUDA is compiled separately as a DLL using nvcc/MSVC.

## 2. Source provenance and project organization

### 2.1. Mishchenko version

The upstream sources are the NASA GISS fixed-orientation amplitude and phase matrix programs for axisymmetric particles, LAPACK-based release. Both original headers state **Last update 08/06/2005**; this notation is reproduced verbatim.

| Solver | Files under ../mishenko |
|---|---|
| Double precision (DP) | src/fixed_dp/ampld.lp.f, lpd.f, ampld.par.f |
| Extended precision (QP) | src/fixed_qp/amplq.lp.f, lpq.f, amplq.par.f |

The original programs use the extended boundary condition method (EBCM). The bundled LAPACK routines replace the proprietary NAG inversion scheme. These are the fixed-orientation programs, not the orientation-averaged or separate Mie programs.

At build time, prepare_cmake.py stages the six files and uses the neighbouring project's prepare_sources.py to remove the sample textual output and modernize legacy QP intrinsics. prepare_capsule.py adds a capsule surface profile and modifies convergence failure handling. HOLO converts the resulting DP and QP programs to callable subroutines, isolates QP symbols except for the common final T-matrix, and adds a packing routine for CUDA. The generated DP/QP sources and LAPACK/BLAS routines are compiled into separate serial and OpenMP static Fortran libraries as required by the selected mode. OpenMP directives are applied only to these generated copies; the original NASA sources remain unmodified.

QP uses REAL*16 (GNU binary128) for extended-precision operations, including the complex LAPACK inversion. The final T-matrix and certain operations remain REAL*8. CUDA evaluates the exported matrix in FP64. QP does not imply binary128 GPU arithmetic or binary128 accuracy for the exported matrix.

Upstream distribution: https://www.giss.nasa.gov/staff/mmishchenko/tmatrix/. The original headers allow unrestricted non-profit scientific research and request acknowledgment and citation of relevant publications. Review those conditions before other use or redistribution.

TMatrix-Holo's original material is available under the
[Noncommercial Scientific Research License, version 1.0](../../LICENSE).
Use, modification and sharing for noncommercial scientific research and
education are permitted; commercial use requires prior written permission.
Third-party terms remain independent. See the [repository README](../../README.md#third-party-terms)
for the FFTW licensing conditions relevant to redistribution.

### 2.2. Project files

| File or directory | Purpose |
|---|---|
| [holo.cpp](../holo.cpp) | CLI, Fortran calls, cross sections, angular spectrum, FFT and RAW output |
| [prepare_holo.py](../prepare_holo.py) | DP solver and amplitude bridge |
| [prepare_qp.py](../prepare_qp.py) | Isolated QP solver sharing the final FP64 T-matrix |
| [prepare_cmake.py](../prepare_cmake.py) | Stage private Fortran sources |
| [tmatrix_openmp.py](../tmatrix_openmp.py) | Insert parallel surface-integral loops in DP/QP |
| [omp_config.f90](../omp_config.f90) | Configure and report OpenMP thread counts |
| [CMakeLists.txt](../CMakeLists.txt) | Unified CMake entry point |
| [cmake/Holo.cmake](../cmake/Holo.cmake) | Common build configuration and mode selection |
| [cmake/DeployCuda.cmake](../cmake/DeployCuda.cmake) | Deploy Windows CUDA DLL and dependencies |
| [mingw/CMakeLists.txt](../mingw/CMakeLists.txt) | Windows MinGW project |
| [wsl/CMakeLists.txt](../wsl/CMakeLists.txt) | Linux/WSL project |
| [mingw/build_all.bat](../mingw/build_all.bat) | Configure, build and test native Windows |
| [wsl/build_all.sh](../wsl/build_all.sh) | Configure, build and test Linux/WSL |
| [cuda/gpu_holo.cu](../cuda/gpu_holo.cu) | FP64 precontraction, amplitudes, sections, spectrum and cuFFT |
| [cuda/gpu_holo_abi.h](../cuda/gpu_holo_abi.h) | Versioned C ABI for the CUDA DLL |
| [cuda/gpu_holo_dll.cpp](../cuda/gpu_holo_dll.cpp) | C ABI adapter on nvcc/MSVC side |
| [cuda/gpu_holo_win.cpp](../cuda/gpu_holo_win.cpp) | DLL loading on MinGW side |
| [cuda/windows/build_cuda.bat](../cuda/windows/build_cuda.bat) | Separate compilation of holo_cuda.dll |
| [cuda/compare_sweep.py](../cuda/compare_sweep.py) | Compare cross sections and every pixel in two sweeps |
| [validate.py](../validate.py) | Independent Mie validation |
| [test_cmake.py](../test_cmake.py) | Automated CTest numerical tests |
| [README.md](../README.md) | Build, dependency and quick-start guide |
| [Repository README](../../README.md) | External dependencies and build overview |

Keep holo/ and mishenko/ as sibling directories. Obtain the original NASA source files separately, as described in the [repository README](../../README.md#dependencies). The fixed-orientation DP/QP source folders and the two preparation scripts must be present before configuring CMake.

## 3. Geometry, coordinate system and units

The surrounding medium is homogeneous, isotropic and non-absorbing, with real index n_medium. Input particle indices are absolute, not relative to the surrounding medium.

- spheroid: prolate or oblate spheroid with total axial length L and transverse diameter D.
- cylinder: finite circular cylinder with length L and diameter D.
- capsule: cylinder terminated by two hemispheres, total length L including both caps, straight cylindrical length L-D (L >= D).

The incident plane wave propagates along +Z. Linear polarization makes an angle pol with X. The particle axis is

$$
\mathbf u=(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta).
$$

The centre has lateral coordinates x0 and y0. The variable z is the positive axial distance from the particle centre to the downstream observation plane, not its absolute microscope coordinate. All CLI lengths are in micrometres, angles in degrees and cross sections in square micrometres.

## 4. Physical and numerical algorithm

### 4.1. T-matrix construction

The program converts absolute input parameters to the surrounding-medium convention:

$$
\lambda_m=\frac{\lambda_0}{n_m},\qquad
m=\frac{n_r+i n_i}{n_m},\qquad k=\frac{2\pi}{\lambda_m}.
$$

The Fortran kernel constructs the T-matrix in the particle frame. It relates incident and scattered multipole coefficients. The solver determines NMAX and NGAUSS using the requested tolerance and NDGS. The original parameter files set NPN1=100 and NPNG1=500.

HOLO_AMPL, based on Mishchenko's AMPL, returns four complex scattering amplitudes for a specified observation direction and particle orientation. Their units are length. CUDA uses an algebraically precontracted representation of the same T-matrix.

### 4.2. Cross sections

For the selected incident polarization, write the scattered spherical amplitudes as A_theta and A_phi. Scattering is integrated over the entire solid angle by Gauss–Legendre quadrature in cos(theta_s) and uniform azimuthal sampling:

$$
C_{\rm sca}=\int_{4\pi} (|A_\theta|^2+|A_\phi|^2)\,d\Omega.
$$

The optical theorem gives extinction from the complex forward-scattering amplitude projected onto the incident polarization:

$$
C_{\rm ext}=\frac{4\pi}{k}\operatorname{Im}(A_{\parallel}(0)),
\qquad C_{\rm abs}=C_{\rm ext}-C_{\rm sca}.
$$

Cross sections are independent of numerical aperture and observation distance. Absorption is obtained by subtraction, not a volume integral of material losses; a small numerical residual can remain for a real index.

### 4.3. Angular spectrum and numerical aperture

For an internal FFT grid of size Fx by Fy and sampling pitches dx and dy, its periodic physical dimensions are Wx=Fx*dx and Wy=Fy*dy. The program constructs transverse frequencies kx, ky within an ideal circular pupil:

$$
k_x^2+k_y^2\le(k_0\,\mathrm{NA})^2,\qquad
k_0=2\pi/\lambda_0.
$$

$$
k_z=\sqrt{k^2-k_x^2-k_y^2},\qquad
\theta_s=\operatorname{atan2}(\sqrt{k_x^2+k_y^2},k_z).
$$

The maximum collected angle is asin(NA/n_medium), approximately 37 degrees for NA=0.8 in water with n=1.33. No paraxial Fresnel approximation is used.

For every frequency within the pupil, the complex scattering amplitudes are evaluated and converted from the spherical basis to Cartesian components Ex, Ey and optionally Ez. The FFT does not replace the per-direction scattering calculation.

### 4.4. Propagation and particle position

With the time convention exp(-i*omega*t), the Cartesian scattered amplitude A(kx,ky) is multiplied by the discrete Weyl factor:

$$
\frac{2\pi i}{k_z W_x W_y}
\exp\left[i(k_z-k)z-i k_x x_0-i k_y y_0\right].
$$

The term -k*z removes the common incident-wave phase at the observation plane. Lateral translations are applied as phase factors. The exact longitudinal wavevector kz is retained.

The unnormalized two-dimensional inverse FFT (FFTW_BACKWARD or cuFFT inverse) uses the 1/(Wx*Wy) factor already included in the spectrum for the discrete quadrature normalization.

### 4.5. Hologram formation

After the inverse FFT, the unit incident field with polarization (cos(pol), sin(pol), 0) is added to the scattered field. The default transverse detector computes

$$
I=|E_{x,\rm diff}+\cos(\mathrm{pol})|^2
  +|E_{y,\rm diff}+\sin(\mathrm{pol})|^2.
$$

The total detector additionally includes |Ez_diff|². Both are intensity models, not a complete simulation of a real vectorial objective. The central Nx by Ny window is saved as normalized intensity (incident background=1).

### 4.6. Sampling and padding

The internal FFT grid has dimensions Fx=Nx*pad and Fy=Ny*pad. Increasing pad enlarges the periodic FFT field of view and refines the transverse frequency step without changing the saved image dimensions.

The complex field requires dx,dy <= lambda0/(2*NA). To sample the full intensity bandwidth, the recommended pitch is dx,dy <= lambda0/(4*NA), or 206.25 nm at lambda0=660 nm and NA=0.8. The 200 nm pitch meets this recommendation.

The approximate geometric cone radius is z*tan(theta_max). The program warns if the FFT window may be too small; this warning does not guarantee convergence. Increase padding and compare holograms on the same saved window. A larger pixel pitch alone does not reduce the number of pupil directions if the physical FFT window remains unchanged.


## 5. Compilation and CLion

### 5.1. Three build modes

HOLO_MODE is the single public CMake selector:

| Mode | CPU target | OpenMP | CUDA target | Script argument | Directory |
|---|---|---|---|---|---|
| CPU | mish_holo_cpu | Off | None | cpu | build_cpu |
| CPU_OPENMP | mish_holo_cpu, mish_holo_openmp | On for OpenMP target | None | openmp | build_openmp |
| CPU_OPENMP_CUDA | mish_holo_cpu, mish_holo_openmp | On for OpenMP and CUDA targets | mish_holo_cuda | all or cuda | build |

The full mode builds all three executable targets in one CMake project. The serial target links holo_fortran_cpu, without OpenMP; the OpenMP and CUDA targets share holo_fortran_openmp. Both libraries use the same generated Fortran sources with distinct compiler flags. CLion Build All builds all three in the full mode. The default on a fresh root, MinGW or WSL configuration is CPU_OPENMP_CUDA. The compatibility entry point cuda/wsl defaults to CPU_OPENMP_CUDA. Old cache switches HOLO_ENABLE_CPU, HOLO_ENABLE_OPENMP and HOLO_ENABLE_CUDA are now internal; old caches are migrated if HOLO_MODE has not been set. The script arguments all, cuda and the older both all select the full three-target build.

Use separate build directories to avoid mixing serial and parallel executables. In CLion select the corresponding GNU C++ and GNU Fortran toolchain, choose Release, and enter -DHOLO_MODE=CPU, -DHOLO_MODE=CPU_OPENMP or -DHOLO_MODE=CPU_OPENMP_CUDA in the CMake profile. Open the root or the mingw/wsl entry point.

### 5.2. Dependencies

- GNU C++17, GNU Fortran supporting REAL*16, libgfortran and libquadmath.
- CMake >= 3.20, Ninja and Python 3 (standard library only for build/tests).
- FFTW3 double-precision headers and library for the CPU target.
- Original Mishchenko DP/QP sources and bundled LAPACK/BLAS routines; no external OpenBLAS is necessary.
- GNU OpenMP/libgomp in the OpenMP modes.
- NVIDIA driver, CUDA toolkit and cuFFT in hybrid mode.
- Native Windows hybrid mode also needs a supported MSVC x64 host compiler for the standalone CUDA DLL.
- Optional post-processing: NumPy, Matplotlib, ImageJ and Gnuplot.

Provide compatible Windows FFTW headers, import library and runtime DLL under holo/fftw. FFTW_ROOT can select another installation. Windows FFTW binaries are not usable on Linux. Under Linux install the native FFTW development package.

### 5.3. Native Windows

From the holo root:

~~~bat
mingw\build_all.bat cpu
mingw\build_all.bat openmp
mingw\build_all.bat all
~~~

Select the desired command. The default argument is all. The script determines its own directory, uses Ninja and stops on a configuration, build or test error. Existing run results are preserved. The MinGW default is F:\mingw64_11.2\bin; set MINGW_BIN to override it.

Dedicated full CUDA build:

~~~bat
mingw\build_all_cuda.bat
~~~

This shortcut calls build_all.bat cuda, selects CPU_OPENMP_CUDA in Release
mode, builds the separate nvcc/MSVC DLL when necessary, and compiles
mish_holo_cpu.exe, mish_holo_openmp.exe and mish_holo_cuda.exe in
mingw/build. All six CTest tests are run; failures propagate to the caller.
Existing run directories are preserved.

Manual hybrid configuration:

~~~bat
cmake -S mingw -B mingw/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHOLO_MODE=CPU_OPENMP_CUDA -DCMAKE_CXX_COMPILER=F:/mingw64_11.2/bin/g++.exe -DCMAKE_Fortran_COMPILER=F:/mingw64_11.2/bin/gfortran.exe
cmake --build mingw/build --parallel
ctest --test-dir mingw/build --output-on-failure
mingw\build\mish_holo_cuda.exe --help
~~~

The CPU and CPU_OPENMP modes require no CUDA or MSVC. Use their respective build directories and mode selectors.

### 5.4. Native Windows CUDA: separate DLL

~~~bat
cuda\windows\build_cuda.bat
mingw\build_all.bat all
~~~

The first batch compiles the CUDA kernel with nvcc 12.9 and Visual Studio 2022 x64, targeting sm_70 (TITAN V), and produces cuda/windows/build/holo_cuda.dll and its dependency DLLs. HOLO_CUDA_ROOT and HOLO_VCVARS64 override the toolkit and MSVC environment script. CMake calls the batch if necessary and deploys the DLLs beside mingw/build/mish_holo_cuda.exe. HOLO_CUDA_DLL_DIR overrides the DLL output directory.

The executable dynamically loads a versioned C ABI. Buffer ownership stays with the caller, errors are returned through codes and message buffers, and no C++ object or exception crosses the MSVC/MinGW boundary. The Fortran computation remains in the MinGW-built executable.

The DLL uses the dynamic CUDA runtime and the MSVC dynamic C runtime. The tested CUDA 12.9 toolkit supports the sm_70 architecture. Select a toolkit compatible with the target GPU.

GNU runtime DLLs are deployed from the active compiler's bin directory. The OpenMP runtime is libgomp-1.dll, not a library embedded in the CUDA DLL. See Section 8.5 for the exact library location on the tested PC.

### 5.5. Linux / Ubuntu WSL

Install the native CPU dependencies, for example on Debian/Ubuntu:

~~~bash
sudo apt install build-essential gfortran cmake ninja-build python3 libfftw3-dev
~~~

For hybrid mode, install a compatible NVIDIA driver and CUDA toolkit with nvcc, CUDA runtime and cuFFT. On WSL, install the Linux toolkit and use a WSL-capable Windows NVIDIA driver. Configure, build and test the selected mode:

~~~bash
bash wsl/build_all.sh cpu
bash wsl/build_all.sh openmp
bash wsl/build_all.sh all
~~~

The default is all. The script adds /usr/local/cuda/bin to PATH in hybrid mode. The CUDA source is compiled directly by CMake/nvcc and linked with CUDA::cudart and CUDA::cufft; no Windows DLL is involved.

Manual hybrid configuration:

~~~bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S wsl -B wsl/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHOLO_MODE=CPU_OPENMP_CUDA -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build wsl/build --parallel
ctest --test-dir wsl/build --output-on-failure
wsl/build/mish_holo_cuda --help
~~~

CMAKE_CUDA_ARCHITECTURES selects the Linux CUDA target (70 by default). The legacy cuda/wsl/build_all.sh builds hybrid mode at its old entry point. The legacy build.sh remains a serial CPU-only Linux script producing the historical executable name mish_holo. Reconfigure in a fresh directory after changing compiler or relocating the project. The same Linux configuration is intended for native Linux; tests reported here were run under Ubuntu WSL.

### 5.6. Executable names and locations: build_all versus CLion

The three CMake targets have distinct executable names:

| Target | Windows executable | Linux/WSL executable |
|---|---|---|
| Serial CPU | mish_holo_cpu.exe | mish_holo_cpu |
| CPU + OpenMP | mish_holo_openmp.exe | mish_holo_openmp |
| CPU + OpenMP + CUDA | mish_holo_cuda.exe | mish_holo_cuda |

With the full build (HOLO_MODE=CPU_OPENMP_CUDA), all three are in the
same binary directory. Paths below are relative to the holo project root.

| Build procedure | Windows MinGW | Linux/WSL |
|---|---|---|
| build_all, default or all | mingw/build/ | wsl/build/ |
| build_all cpu | mingw/build_cpu/ | wsl/build_cpu/ |
| build_all openmp | mingw/build_openmp/ | wsl/build_openmp/ |
| CLion, current Debug profile | mingw/cmake-build-debug/ | wsl/cmake-build-debug/ |
| CLion, conventional Release profile | mingw/cmake-build-release/ | wsl/cmake-build-release/ |

The cpu script builds only the serial executable. The openmp script builds
the serial and OpenMP executables. The all script builds all three.
For example, the full Windows script build produces:

~~~text
mingw/build/mish_holo_cpu.exe
mingw/build/mish_holo_openmp.exe
mingw/build/mish_holo_cuda.exe
mingw/build/holo_cuda.dll
~~~

The current MinGW CLion Debug profile produces:

~~~text
mingw/cmake-build-debug/mish_holo_cpu.exe
mingw/cmake-build-debug/mish_holo_openmp.exe
mingw/cmake-build-debug/mish_holo_cuda.exe
mingw/cmake-build-debug/holo_cuda.dll
~~~

The corresponding Linux/WSL script and CLion Debug locations are:

~~~text
wsl/build/mish_holo_cpu
wsl/build/mish_holo_openmp
wsl/build/mish_holo_cuda
wsl/cmake-build-debug/mish_holo_cpu
wsl/cmake-build-debug/mish_holo_openmp
wsl/cmake-build-debug/mish_holo_cuda
~~~

Paths in this guide are relative to the holo/ source directory.
The Debug build directories above correspond to the existing CLion
profiles for the mingw and wsl subprojects. Release directories are
examples: CLion's actual output location is its CMake profile's Build
directory setting. If opening the root CMake project instead, the default
profile directory is relative to that root, not to mingw or wsl.
Select Release for meaningful performance benchmarks.

Results always go into run/run_<name> beside the executable that was
launched. Thus CLion Debug results and build_all results remain separate.
Use a different run name for CPU, OpenMP and CUDA comparisons within
the same build directory. Older binaries named mish_holo may still exist
in a reused build directory but are not current CMake targets.

### 5.7. CLion WSL: CUDA compiler discovery

The shared Holo.cmake searches for nvcc before enabling the CUDA
language, including /usr/local/cuda/bin and toolkit-root hints.
An explicit CMAKE_CUDA_COMPILER or CUDACXX setting takes precedence.

For the tested WSL installation, nvcc is at
/usr/local/cuda/bin/nvcc (CUDA 12.9.86).
In CLion, select the WSL toolchain in Settings > Build, Execution,
Deployment > CMake, and use the following CMake options if an explicit
compiler path is needed:

~~~text
-DHOLO_MODE=CPU_OPENMP_CUDA -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=70
~~~

Reload the CMake project and select Build All. The architecture 70 is for
the tested TITAN V. The explicit compiler path is a Linux path, not
the Windows nvcc.exe path. Native Windows continues to compile the
CUDA DLL separately with nvcc/MSVC.

### 5.8. Build validation, 26 November 2026

| HOLO_MODE | MinGW Windows | Ubuntu WSL |
|---|---:|---:|
| CPU | 2/2 CTest passed | 2/2 CTest passed |
| CPU_OPENMP | Validated | Validated |
| CPU_OPENMP_CUDA | 6/6 CTest passed | 6/6 CTest passed |

The full three-target build passed all six CTest tests on each platform (12 tests in total). Tested GNU versions are 11.2 on Windows and 13.3 on WSL; OpenMP 4.5 and CUDA 12.9 are used. The CTest numerical checks include a transparent object, independent Mie comparison and angular sweep consistency.

#### Final integration validation (26 November 2026)

All three executables were built and executed using both the build_all
scripts and CLion on the Windows MinGW and Ubuntu WSL toolchains.

| Build procedure | Build directory | CPU | OpenMP | CUDA | Validation |
|---|---|---|---|---|---|
| Windows build_all / build_all_cuda.bat (Release) | mingw/build/ | OK | OK | OK | 6/6 CTest passed |
| Windows CLion (Debug) | mingw/cmake-build-debug/ | OK | OK | OK | Build and execution |
| WSL build_all (Release) | wsl/build/ | OK | OK | OK | 6/6 CTest passed |
| WSL CLion (Debug) | wsl/cmake-build-debug/ | OK | OK | OK | Build and execution; Debug CTest 6/6 |

Both command-line and IDE workflows were validated on native Windows
and Ubuntu WSL. The test results apply to the stated toolchains.

## 6. Command-line options

Unspecified parameters retain the defaults below. Use --help for the executable's own option list.

| Option | Default | Meaning |
|---|---|---|
| --name | Timestamp | Run name suffix |
| --shape | spheroid | spheroid, capsule or cylinder |
| --lambda-um | 0.660 | Vacuum wavelength (660 nm) |
| --length-um | 8 | Total axial length |
| --diameter-um | 2.5 | Transverse diameter |
| --n-real | 1.38 | Real part of absolute particle index |
| --n-imag | 0 | Imaginary part of absolute particle index |
| --n-medium | 1.33 | Real surrounding-medium index (water) |
| --theta | 27 | Particle-axis inclination (degrees) |
| --phi | 48 | Particle-axis azimuth (degrees) |
| --theta-steps | Disabled | Number N of equal intervals from 0 to --theta-end; N+1 holograms, phi=0 |
| --theta-end | 180 | Final sweep angle (degrees) |
| --pol-deg | 0 | Incident linear polarization (0=X, 90=Y) |
| --x-um | 0 | Lateral position X |
| --y-um | 0 | Lateral position Y |
| --z-um | 100 | Positive centre-to-observation-plane distance |
| --na | 0.8 | Numerical aperture, strictly below n_medium |
| --nx | 1024 | RAW image columns |
| --ny | 1024 | RAW image rows |
| --dx-um | 0.1 | X sampling pitch |
| --dy-um | 0.1 | Y sampling pitch |
| --pad | 3 | FFT enlargement factor |
| --tol | 1e-5 | T-matrix convergence tolerance |
| --ndgs | 3 | Surface quadrature parameter |
| --tmatrix-precision | auto | Try DP, then QP if DP fails; dp or qp forces the selected precision |
| --threads | Auto | OpenMP thread request (1..256); OMP_NUM_THREADS or up to 8 if omitted |
| --detector | transverse | transverse or total |
| --help | — | Display help |

Constraints include positive physical lengths, nonnegative imaginary index, 0 < NA < n_medium, tol <= 1e-3, NDGS=2..10, pad <= 16, and at most 16,777,216 internal FFT cells. The observation plane must lie outside the bounding sphere used by the program: z > hypot(L,D)/2. Capsule length must be at least its diameter. For mish_holo_cpu use one thread or omit --threads.

### 6.1. Single-orientation bacterial spheroid

From the holo root on Windows in hybrid mode:

~~~bat
mingw\build\mish_holo_cpu.exe --name bacterie001 --shape spheroid --lambda-um 0.660 --length-um 2 --diameter-um 0.8 --n-real 1.38 --n-imag 0 --n-medium 1.33 --theta 27 --phi 48 --x-um 0 --y-um 0 --z-um 50 --na 0.8 --pol-deg 0 --nx 128 --ny 128 --dx-um 0.15 --dy-um 0.15 --pad 6
~~~

Replace the executable with mish_holo_cuda.exe to use the GPU, or use the appropriate Linux executable under wsl/build. Choose a new run name each time; an existing result directory is not overwritten.

### 6.2. Angular sweep at phi=0

The option --theta-steps N, for positive integer N, replaces the single orientation with N+1 holograms at theta_j=theta_end*j/N degrees, j=0,...,N. Both endpoints are included. The default final angle is 180 degrees. The angular step is theta_end/N degrees. The sweep overrides --theta and --phi (phi=0). Geometry, indices, illumination, distance and grid remain unchanged.

~~~bat
mingw\build\mish_holo_cpu.exe --name serie001 --theta-steps 36 --length-um 2 --diameter-um 0.8 --nx 128 --ny 128 --dx-um 0.15 --dy-um 0.15 --pad 6 --z-um 50 --na 0.8
~~~

This example produces 37 holograms from 0 to 180 degrees in 5-degree steps. The Fortran kernel constructs one T-matrix reused for every orientation, with amplitudes, sections and FFT recomputed for each. A transparent particle does not require T-matrix construction.

Native CUDA example for a 6 × 2.5 µm spheroid, from mingw/build:

~~~bat
mish_holo_cuda.exe --name bacterie_60x25_cuda_60 --length-um 6 --diameter-um 2.5 --tmatrix-precision qp --z-um 50 --nx 512 --ny 512 --pad 3 --theta-steps 60
~~~

This generates 61 images, including 0 and 180 degrees. For exactly 60 images from 0 to 177 degrees use --theta-steps 59 --theta-end 177 and a new run name.

### 6.3. Detailed example: 8 × 3 µm spheroid, 60 orientations

From holo/mingw/build in Windows Command Prompt:

~~~bat
mish_holo_cuda.exe --name bacterie_80x30_cuda_60 --length-um 8 --diameter-um 3 --tmatrix-precision qp --z-um 25 --nx 512 --ny 512 --pad 3 --theta-steps 59 --theta-end 177 --threads 16
~~~

| Argument | Meaning |
|---|---|
| mish_holo_cuda.exe | Native Windows executable using CUDA DLL for FP64 amplitudes, cross sections and cuFFT propagation |
| --name bacterie_80x30_cuda_60 | Run name; choose another if run_bacterie_80x30_cuda_60 already exists |
| --length-um 8 | Total axial length, 8 µm |
| --diameter-um 3 | Transverse diameter, 3 µm |
| --tmatrix-precision qp | Construct the extended-precision T-matrix once on the CPU |
| --z-um 25 | Observation plane 25 µm downstream of the particle centre |
| --nx 512 --ny 512 | Save each hologram as a 512 × 512 image |
| --pad 3 | Internal FFT grid 1536 × 1536 |
| --theta-steps 59 | 59 intervals, 60 images including both bounds |
| --theta-end 177 | Final inclination 177 degrees, 3-degree step, phi=0 |
| --threads 16 | Request 16 OpenMP threads |

The unspecified parameters use defaults: spheroid, 660 nm vacuum wavelength, particle index 1.38+0i, medium index 1.33, NA=0.8, X polarization, centred particle, 100 nm pixels, and transverse detection. The saved field of view is 51.2 × 51.2 µm, and the FFT window is 153.6 × 153.6 µm. For this grid there are 108,925 pupil directions. To retain the same saved field of view with 200 nm pixels, use --nx 256 --ny 256 --dx-um 0.2 --dy-um 0.2.

After completion, the RAW stack contains 60 × 512 × 512 Float32 values (62,914,560 bytes, 60 MiB). This geometry has converged in QP (NMAX=59, NGAUSS=178). The 60 images and cross sections were validated against the earlier non-OpenMP reference; see Section 8.5.


### 6.4. ImageJ stack and Gnuplot cross sections

In sweep mode, holograms.raw concatenates little-endian Float32 images in increasing theta order without header or gaps. X varies fastest, then Y, then frame index. Its byte size is 4*nx*ny*(N+1). There are no redundant per-frame RAW files or theta_* subdirectories.

The overall log reports stack_file, stack_frame_bytes, stack_bytes, image_count, nx, ny and angular step. Both aggregated files are first written with a .part suffix, then renamed when the sweep is complete.

In ImageJ select File > Import > Raw, choose holograms.raw, set 32-bit Real, Width=nx, Height=ny, Offset=0, Number of images=N+1, Gap between images=0, and check Little-endian byte order. Image > Stacks > Start Animation displays the sequence. Keep the same display range for all images. To export AVI use File > Save As > AVI (optionally duplicate and convert to 8-bit with one common intensity range). The original RAW retains Float32 precision.

cross_sections.txt is space-separated ASCII text, with a header comment starting with # and four columns: theta (degrees), Cext, Csca and Cabs (µm²). One line corresponds to each image, including the endpoints. From the run folder, use Gnuplot:

~~~gnuplot
set xlabel "Theta (degrees)"
set ylabel "Cross section (um^2)"
set xrange [0:180]
plot "cross_sections.txt" using 1:2 with lines title "Cext", \
     "" using 1:3 with lines title "Csca", \
     "" using 1:4 with lines title "Cabs"
~~~

Read the entire stack into a NumPy array (angle, y, x):

~~~python
import csv
from pathlib import Path
import numpy as np
p = Path("run/run_serie001")
d = dict(s.split("=", 1) for s in (p / "log.txt").read_text().splitlines() if "=" in s)
assert d["status"] == "complete"
with (p / "orientations.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == int(d["image_count"])
theta = np.array([float(r["theta_deg"]) for r in rows])
I = np.fromfile(p / "holograms.raw", dtype="<f4").reshape(
    len(rows), int(d["ny"]), int(d["nx"]))
~~~

Historical sweep validation at lambda0=0.532 µm (the previous default): test_theta_sweep uses four intervals, 24 × 24 pixels, pad=3 and z=3 µm with the earlier default bacterial spheroid. Five images took 3.15 s under WSL, including 0.0304 s for T-matrix construction. Cext values at 0, 45, 90, 135 and 180 degrees were respectively 0.3129028202, 0.1829293770, 0.1416181208, 0.1829293516 and 0.3129028202 µm². Head-tail symmetry explains the identical endpoints; 45 and 135 degrees yield X-mirrored images for the centred X-polarized test.

## 7. Result directories and files

Results are written under run/run_<name>, next to the executable regardless of the current working directory. For example, in the Windows hybrid configuration:

~~~text
holo/mingw/build/
  mish_holo_cpu.exe
  mish_holo_openmp.exe
  mish_holo_cuda.exe
  holo_cuda.dll
  run/
    run_bacterie001/
      log.txt
      hologram.raw
    run_serie001/
      log.txt
      orientations.csv
      cross_sections.txt
      holograms.raw
~~~

For serial and OpenMP-only builds, the corresponding output roots are mingw/build_cpu/run and mingw/build_openmp/run. Linux uses the analogous wsl directories. The legacy build.sh writes under holo/run. The historical results in mishenko/run remain in place. In the Python examples, run from the executable directory or adjust p.

log.txt contains key=value pairs, including the complete command, physical parameters, backend, effective precision, OpenMP settings, T-matrix timing, Cext_um2, Csca_um2, Cabs_um2, NMAX, NGAUSS, image and FFT sizes, sampling pitch, intensity statistics, file size and total time. Only process runs whose last status is complete.

orientations.csv contains one row per completed frame: index, theta_deg, phi_deg, Cext_um2, Csca_um2, Cabs_um2, nx, ny, raw_file, offset_bytes, raw_min, raw_max, raw_mean, elapsed_seconds. All images are stored in the shared holograms.raw file.

| RAW property | Format |
|---|---|
| Data type | IEEE Float32, 4 bytes per sample |
| Byte order | Little-endian |
| Header | None |
| Layout | Row-major, X varies fastest |
| Dimensions | nx columns, ny rows |
| Single-frame byte size | 4*nx*ny |
| Origin | Pixel [ny//2, nx//2] |
| Coordinates | X increases with columns, Y with rows |
| Quantity | Normalized intensity, incident background=1 |

For a single hologram, the temporary hologram.raw.part is renamed after writing is complete.

### Reading and plotting a single hologram in Python

~~~python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p = Path("run/run_bacterie001")
d = dict(s.split("=", 1) for s in
         (p / "log.txt").read_text().splitlines() if "=" in s)
assert d["status"] == "complete"
nx, ny = int(d["nx"]), int(d["ny"])
assert (p / "hologram.raw").stat().st_size == 4 * nx * ny
I = np.fromfile(p / "hologram.raw", dtype="<f4").reshape(ny, nx)
plt.imshow(I, cmap="gray", origin="lower")
plt.colorbar(label="Normalized intensity")
plt.show()
~~~

## 8. Numerical validation and performance

### 8.1. Transparent object

For an object whose refractive index equals that of the surrounding medium, without absorption, every pixel is exactly 1 and all three cross sections are zero. The initial test_no_contrast used 32 × 24 pixels.

### 8.2. Independent Mie reference

Sphere radius 0.336 µm, vacuum wavelength 0.63 µm, index 3+0.1i in air. The historical MinGW GCC 11.2 and WSL GCC 13.3 CMake builds gave:

| Cross section (µm²) | mish_holo | Independent Mie | Relative error |
|---|---:|---:|---:|
| Cext | 0.883775460214 | 0.883775460214 | 2.51e-16 |
| Csca | 0.509378509004 | 0.509378630252 | 2.38e-7 |
| Cabs | 0.374396951210 | 0.374396829962 | 3.24e-7 |

An independent Mie calculation followed by direct angular-spectrum summation reproduces five hologram pixels with maximum absolute error 7.59e-8. This verifies phase, normalization and amplitude conversion on the chosen discrete grid; it does not by itself establish convergence in padding or validate a real experimental microscope.

The standalone validate.py uses only Python's standard library. Historical results can be checked with python validate.py ../mishenko; for new results give the directory containing the executable and its run subdirectory. CTest generates its own test runs.

### 8.3. Historical bacterial spheroid

Length 2 µm, diameter 0.8 µm, water index 1.33, particle index 1.38, vacuum wavelength 0.532 µm (the earlier default), orientation (27°,48°), NA=0.8, z=50 µm, 128 × 128 output pixels, 150 nm pitch and pad=6. Results of the original WSL CPU run_test_bacterium:

| Quantity | Value |
|---|---:|
| NMAX | 22 |
| Cext (µm²) | 0.235751511459 |
| Csca (µm²) | 0.235751380392 |
| Cabs (µm²) | 1.31067e-7 |
| RAW dimensions | 128 × 128 |
| FFT dimensions | 768 × 768 |
| RAW size | 65,536 bytes |
| Minimum intensity | 0.96080184 |
| Maximum intensity | 1.03971708 |
| Mean intensity | 0.99984708 |
| Measured CPU time | 22.28 s |

For a real refractive index Cabs is a numerical residual. Independent IF-DDA comparison and convergence studies remain necessary for full validation of this nonspherical case.

### 8.4. CPU/CUDA comparison and large bacteria (26 November 2026)

Both versions use the same Fortran DP/QP solver. The reference CPU evaluates AMPL using a triple multipole sum of complexity O(NMAX³) per scattering direction. The CUDA version precontracts the incident multipoles once per orientation on the CPU in FP64, then evaluates a double sum of complexity O(NMAX²) per direction on the GPU. The GPU computes the cross-section quadrature contributions, followed by final summation on the CPU. Spectral propagation and cuFFT inverse transforms use FP64. No angular interpolation is introduced. CUDA does not accelerate the original T-matrix construction itself.

Common settings: vacuum wavelength 0.660 µm, medium index 1.33, particle index 1.38, NA=0.8, QP T-matrix. Historical times below are before introducing OpenMP.

| Dimensions (µm) | Images | RAW / pitch | z (µm) | NMAX | CPU (s) | CUDA (s) | Platform |
|---|---:|---|---:|---:|---:|---:|---|
| 1.6 × 0.8 | 10 | 256² / 200 nm | 20 | 15 | 34.7696 | 1.0070 | WSL |
| 2 × 0.8 | 11 | 512² / 100 nm | 50 | 18 | 117.3364 | 0.9757 | Native Windows |
| 6 × 2 | 1 | 128² / 200 nm | 10 | 45 | 53.3747 | 16.3474 | Native Windows |
| 6 × 2.5 | 11 | 512² / 100 nm | 50 | 45 | — | 16.3036 | Native Windows |
| 6 × 2.5 | 61 | 512² / 100 nm | 50 | 45 | — | 17.9298 | Native Windows |

In the first three comparisons, all Float32 intensities match exactly: 655,360, 2,883,584 and 16,384 pixels, respectively. Maximum absolute cross-section discrepancies are 2.50e-16, 5.83e-16 and 6.84e-14 µm². The first two sweeps use phi=0, X polarization and transverse detection. The 6 × 2 µm case uses theta=27°, phi=48°, polarization=31°, total detection and pad=2; Cext=6.878861177714 and Csca=6.878860691583 µm².

For the 6 × 2.5 µm 61-image sweep, NMAX=45, NGAUSS=136, FFT=1536² and 108,925 pupil directions. A single QP T-matrix takes 15.8546704 s; total time is 17.9298065 s, leaving 2.0751361 s for holograms, cross sections, initialization and output (about 34 ms per image). For the 11-image sweep the corresponding construction time is 15.7802516 s. The 61-image RAW stack is complete (63,963,136 bytes, 61 MiB) with no theta_* directories. There is no matching CPU-only benchmark for these two 6 × 2.5 µm sweeps.

Historical native Windows result directories under mingw/build/run: run_bacterie6x2 (despite its name, the CPU reference uses a 2 × 0.8 µm spheroid), run_bacterie2x08_qp_cuda_native, run_compare_large_6x2_cpu, run_compare_large_6x2_cuda, run_bacterie_60x25_cuda and run_bacterie_60x25_cuda_60. WSL QP references are under build/cmake_dual/run/run_cmake_dual_cpu and run_cmake_dual_cuda.

An earlier WSL DP benchmark of 1.6 × 0.8 µm with ten images, 256² pixels at 200 nm, z=20 µm and pad=2 measured 34.6189 s CPU and 1.70454 s CUDA (20.31-fold total speedup). All 655,360 Float32 pixels match, with maximum cross-section error 3.34e-16 µm². The CPU timing included the old redundant individual RAW output. Reference directories: wsl/build/run/run_cpu_ref_16x08_10angles and cuda/wsl/build/run/run_cuda_ref_16x08_10angles.

The comparison script can be run on two matching sweeps:

~~~bash
python3 cuda/compare_sweep.py wsl/build/run/run_cpu_ref_16x08_10angles cuda/wsl/build/run/run_cuda_ref_16x08_10angles
~~~

Agreement between CPU and CUDA validates the port, not the absolute accuracy of the T-matrix. For a real index the physical absorption is zero; for example, Cabs=-1.1519e-4 µm² at theta=0 for the 6 × 2.5 µm spheroid is a numerical residual, not negative physical absorption.


### 8.5. OpenMP acceleration of the DP/QP T-matrix

The common CMake configuration follows the IF-DDA master Fortran projects:

~~~cmake
find_package(OpenMP REQUIRED COMPONENTS Fortran)
target_link_libraries(holo_fortran_openmp PUBLIC OpenMP::OpenMP_Fortran)
~~~

The CPU_OPENMP and CPU_OPENMP_CUDA modes include the OpenMP target; mish_holo_cpu is always the serial reference, even in a full build. The tested Windows GNU 11.2 toolchain uses OpenMP 4.5 and:

~~~text
F:/mingw64_11.2/lib/gcc/x86_64-w64-mingw32/11.2.0/libgomp.dll.a
F:/mingw64_11.2/bin/libgomp-1.dll
~~~

IF-DDA's deployment directory uses E:/msys64/mingw64/bin/libgomp-1.dll; HOLO uses the DLL matching its own GNU compiler and copies it beside the executables. The separate nvcc/MSVC CUDA DLL is unaffected. WSL uses /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so.

tmatrix_openmp.py inserts two PARALLEL DO regions into the generated DP and QP sources, in TMATR0 (m=0) and TMATR (m>=1). Each thread computes independent rows of the integral matrices. The order of quadrature accumulation in I is unchanged; input arrays are shared read-only, scalar temporaries are private. The convergence search in m=0 benefits from the same parallelization. COMMON blocks and calls to the legacy routines remain sequential across azimuthal blocks; LAPACK inversions are serial. Large work arrays are kept off worker stacks. The original NASA sources are not modified.

The parallel-loop threshold is (NMAX-MM1+1)^2*NGSS >= 10000. Smaller loops may run serially. No change is made to arithmetic precision or the convergence criterion.

The CLI option --threads N requests 1..256 threads. If omitted, OMP_NUM_THREADS is respected when set; otherwise the default is up to eight, limited by the runtime. For mish_holo_cpu only one thread is available, regardless of build mode. The log reports openmp_enabled, openmp_threads_requested (0=automatic), openmp_threads (size of initial test team), and tmatrix_elapsed_seconds (solver construction including CUDA export if applicable, excluding initial OpenMP team setup).

**Windows benchmark: Xeon E5-2673 v2, eight physical cores / 16 logical threads, NVIDIA TITAN V.** QP T-matrix, lambda0=0.660 µm, n_particle=1.38, n_medium=1.33, NA=0.8.

Case 6 × 2.5 µm, NMAX=45, NGAUSS=136, 11 holograms at 512², 100 nm pitch, z=50 µm, pad=3, theta=0..180 degrees in 18-degree steps:

| Threads | T-matrix (s) | Total (s) | T-matrix speedup vs 1 thread |
|---:|---:|---:|---:|
| 1 | 15.4876 | 15.9891 | 1.00 |
| 2 | 9.2039 | 9.6797 | 1.68 |
| 4 | 6.2115 | 6.6872 | 2.49 |
| 8 | 4.9506 | 5.4485 | 3.13 |
| 16 | 4.4197 | 4.8899 | 3.50 |

Case 8 × 3 µm, NMAX=59, NGAUSS=178, 60 holograms at 512², 100 nm pitch, z=25 µm, pad=3, theta=0..177 degrees in 3-degree steps:

| Configuration | T-matrix (s) | Total (s) | Speedup vs historical T-matrix |
|---|---:|---:|---:|
| Historical non-OpenMP | 45.5660 | 48.1438 | 1.00 |
| OpenMP 8 threads | 14.6518 | 17.4005 | 3.11 |
| OpenMP 16 threads | 12.3987 | 14.8196 | 3.68 |

For both cases, the RAW stack SHA-256 matches the historical non-OpenMP reference, and all cross sections read from orientations.csv are exactly equal. These comparisons cover 2,883,584 and 15,728,640 pixels respectively. The measured speedups include the serial portion of the solver. Sixteen threads gave the best measured time among the tested configurations on this machine; this does not establish optimal thread counts on other architectures or geometries.

Result directories under mingw/build/run: run_omp_qp_6x25_t1, run_omp_qp_6x25_t2, run_omp_qp_6x25_t4, run_omp_qp_6x25_t8, run_omp_qp_6x25_t16 (reference run_bacterie_60x25_cuda), and run_omp_qp_8x3_t8, run_omp_qp_8x3_t16 (reference run_bacterie_80x30_cuda_60).

A separate DP test of a 2 × 0.8 µm spheroid, ten images at 256² and 200 nm, z=20 µm, pad=2, compares one and 16 OpenMP threads. All 655,360 pixels and cross sections are exactly equal (run_omp_dp_2x08_t1 and run_omp_dp_2x08_t16). All CTest configurations passed as listed in Section 5.8.

## 9. Current limitations and machine-learning development

The model assumes a homogeneous surrounding medium and an ideal circular pupil. It does not include water/glass interfaces, aberrations, apodization, polarization transfer of a real objective, pixel-area integration or detector noise. CPU/GPU agreement alone does not validate the experimental forward model.

Each invocation currently rebuilds the T-matrix once; it is reused for every angle of a --theta-steps series. The pupil amplitudes are recomputed for each orientation. Even with OpenMP, QP construction still dominates the runtime for large bacteria.

Development priorities:

1. Save and reload a T-matrix between runs with fixed shape, size, refractive indices and wavelength. Persistent binary caching has not yet been implemented.
2. Generalize the existing angular sweep to other orientation and polarization sweeps.
3. Reuse an angular spectrum for multiple lateral positions and axial distances.
4. Further optimize CUDA/cuFFT, including batched processing.
5. Introduce measured microscope properties and variations in bacterial morphology.
6. Train a neural network to estimate x0, y0, z and the axis vector u, optionally followed by refinement against the electromagnetic forward model.

For head-tail symmetric particles, u and -u are physically equivalent; an orientation loss must respect that symmetry.

## 10. References

- M. I. Mishchenko, *Calculation of the amplitude matrix for a nonspherical particle in a fixed orientation*, Applied Optics 39, 1026–1031 (2000), [DOI: 10.1364/AO.39.001026](https://doi.org/10.1364/AO.39.001026).
- M. I. Mishchenko and L. D. Travis, *T-matrix computations of light scattering by large spheroidal particles*, Optics Communications 109, 16–21 (1994).
- M. I. Mishchenko, L. D. Travis and D. W. Mackowski, *T-matrix computations of light scattering by nonspherical particles: a review*, JQSRT 55, 535–575 (1996).
- [NASA GISS T-matrix codes](https://www.giss.nasa.gov/staff/mmishchenko/tmatrix/).
- [Repository dependency instructions](../../README.md#dependencies).

Respect the original NASA source conditions and citation requirements.
