@echo off
setlocal
rem Build the complete native Windows CPU/OpenMP/CUDA configuration.
rem CMake builds the CUDA DLL separately using nvcc/MSVC when required.
rem Reuse build_all.bat to keep toolchain settings and CTest in one place.
call "%~dp0build_all.bat" cuda
exit /b %ERRORLEVEL%
