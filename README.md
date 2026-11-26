# HOLO

A scientific electromagnetic simulation package for generating synthetic in-line holograms of axisymmetric particles using the Mishchenko fixed-orientation T-matrix method, optional OpenMP and optional CUDA FP64.

- [Build instructions and quick start](holo/README.md)
- [Technical documentation and numerical benchmarks](holo/doc/INDEX.md)

## Repository structure

The source layout preserves the sibling directories holo/ and mishenko/ expected by the CMake and Python preparation scripts. The two Python adaptation scripts are tracked in mishenko/tools/. The six original NASA Fortran files and the Windows FFTW library are external dependencies, not tracked in Git.

## Dependencies

Obtain the fixed-orientation DP and QP LAPACK-based NASA GISS T-matrix distribution (source headers: Last update 08/06/2005) from https://www.giss.nasa.gov/staff/mmishchenko/tmatrix/ and place the unmodified files as follows:

    mishenko/src/fixed_dp/ampld.lp.f
    mishenko/src/fixed_dp/ampld.par.f
    mishenko/src/fixed_dp/lpd.f
    mishenko/src/fixed_qp/amplq.lp.f
    mishenko/src/fixed_qp/amplq.par.f
    mishenko/src/fixed_qp/lpq.f

On Windows, install a compatible double-precision FFTW3 distribution from https://www.fftw.org/ and place fftw3.h, the MinGW import library and its runtime DLL in holo/fftw/, or set FFTW_ROOT to an appropriate installation in CMake. Linux/WSL uses the system FFTW development package. DEPENDENCIES.sha256 records the exact dependency files used for the development configuration; alternative compatible FFTW builds may have different hashes.

## Build

From the repository root, enter holo/ first. Windows Command Prompt:

    cd holo
    mingw\build_all.bat cpu
    mingw\build_all.bat openmp
    mingw\build_all_cuda.bat

Linux or WSL:

    cd holo
    bash wsl/build_all.sh cpu
    bash wsl/build_all.sh openmp
    bash wsl/build_all.sh all

Select the desired mode. In CLion, open holo/mingw or holo/wsl, select the appropriate toolchain and set HOLO_MODE=CPU_OPENMP_CUDA in the CMake profile to build all three targets. See the linked guides for dependencies, options and examples.

## License and scientific citation

TMatrix-Holo is available under the [Noncommercial Scientific Research License, version 1.0](LICENSE).
Researchers may use, study, modify and share the original project code
free of charge for noncommercial scientific research and scientific
education. Commercial use requires prior written permission from the
relevant copyright holder(s). Please acknowledge TMatrix-Holo, cite this
repository and the relevant Mishchenko publications in scientific work.
This is a source-available research license, not an OSI-approved
open-source license.

## Third-party terms

The original Mishchenko headers specify use in non-profit scientific
research and request acknowledgment and citation. Third-party components
are not relicensed by TMatrix-Holo. Obtain dependencies under their own
applicable terms.

FFTW is available under GPL-2.0-or-later or separately negotiated
alternative terms (see https://fftw.org/doc/License-and-Copyright.html).
The GPL and the TMatrix-Holo noncommercial restriction are not compatible
for distribution of a combined work under both sets of terms. Build for
local research use; do not redistribute FFTW-linked executables without
establishing compatible licensing arrangements. The repository therefore
does not distribute third-party libraries or compiled executables.
