@echo off
setlocal
rem Configure, compile and test HOLO (MinGW, Release).
rem Override MINGW_BIN in the environment to select another GNU toolchain.
if not defined MINGW_BIN set "MINGW_BIN=F:\mingw64_11.2\bin"
pushd "%~dp0" || exit /b 1
if not exist "%MINGW_BIN%\g++.exe" goto :missing
if not exist "%MINGW_BIN%\gfortran.exe" goto :missing
set "PATH=%MINGW_BIN%;%PATH%"
set "MODE=CPU_OPENMP_CUDA"
set "BUILD=build"
if /I "%~1"=="cpu" (
  set "MODE=CPU"
  set "BUILD=build_cpu"
) else if /I "%~1"=="cuda" (
  set "MODE=CPU_OPENMP_CUDA"
  set "BUILD=build"
) else if /I "%~1"=="both" (
  set "MODE=CPU_OPENMP_CUDA"
  set "BUILD=build"
) else if /I "%~1"=="openmp" (
  set "MODE=CPU_OPENMP"
  set "BUILD=build_openmp"
) else if /I not "%~1"=="all" if not "%~1"=="" (
  echo Usage: %~nx0 [cpu^|openmp^|cuda^|all] 1>&2
  goto :failed
)
echo === Configuring HOLO MinGW ===
cmake -S . -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=%MINGW_BIN%\g++.exe" "-DCMAKE_Fortran_COMPILER=%MINGW_BIN%\gfortran.exe" -DHOLO_MODE=%MODE%
if errorlevel 1 goto :failed
echo === Building all targets ===
cmake --build "%BUILD%" --parallel
if errorlevel 1 goto :failed
echo === Running tests ===
ctest --test-dir "%BUILD%" --output-on-failure
if errorlevel 1 goto :failed
echo Build and tests successful.
echo CPU:    %CD%\%BUILD%\mish_holo_cpu.exe
if not "%MODE%"=="CPU" echo OpenMP: %CD%\%BUILD%\mish_holo_openmp.exe
if "%MODE%"=="CPU_OPENMP_CUDA" echo CUDA: %CD%\%BUILD%\mish_holo_cuda.exe
echo Results:    %CD%\%BUILD%\run
popd
exit /b 0
:missing
echo ERROR: GNU C++ or Fortran compiler not found in "%MINGW_BIN%". 1>&2
:failed
echo ERROR: HOLO build or tests failed. 1>&2
popd
exit /b 1
