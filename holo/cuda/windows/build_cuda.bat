@echo off
setlocal
rem Compile the CUDA DLL separately with nvcc and MSVC, never MinGW.
rem Optional argument: output directory. HOLO_CUDA_ROOT and HOLO_VCVARS64 override defaults.
if "%~1"=="" (set "OUT=%~dp0build") else (set "OUT=%~f1")
pushd "%~dp0" || exit /b 1
set "SRC=%~dp0.."
if not defined HOLO_CUDA_ROOT set "HOLO_CUDA_ROOT=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not defined HOLO_VCVARS64 set "HOLO_VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%HOLO_VCVARS64%" set "HOLO_VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%HOLO_VCVARS64%" goto :missing
if not exist "%HOLO_CUDA_ROOT%\bin\nvcc.exe" goto :missing
call "%HOLO_VCVARS64%"
if errorlevel 1 goto :failed
if not exist "%OUT%" mkdir "%OUT%"
if errorlevel 1 goto :failed
echo === CUDA DLL: %OUT%\holo_cuda.dll ===
"%HOLO_CUDA_ROOT%\bin\nvcc.exe" -shared -O3 -std=c++17 -arch=sm_70 --fmad=false -Xcompiler=/EHsc,/MD -cudart shared "%SRC%\gpu_holo.cu" "%SRC%\gpu_holo_dll.cpp" -I"%SRC%" -L"%HOLO_CUDA_ROOT%\lib\x64" -lcufft -o "%OUT%\holo_cuda.dll"
if errorlevel 1 goto :failed
rem Deploy the runtime DLLs from the SAME CUDA toolkit. The driver is installed separately.
for %%F in (cufft64_11.dll cufft64_10.dll cudart64_12.dll cudart64_110.dll nvJitLink_120_0.dll nvrtc64_120_0.dll nvrtc-builtins64_129.dll) do (
  if exist "%HOLO_CUDA_ROOT%\bin\%%F" (
    copy /Y "%HOLO_CUDA_ROOT%\bin\%%F" "%OUT%\%%F" >nul
    if errorlevel 1 goto :failed
  )
)
echo CUDA DLL built successfully.
popd
exit /b 0
:missing
echo ERROR: nvcc or MSVC vcvars64.bat not found. Set HOLO_CUDA_ROOT and HOLO_VCVARS64. 1>&2
:failed
popd
exit /b 1
