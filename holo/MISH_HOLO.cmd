@echo off
wsl.exe --cd "%~dp0." -- ./mish_holo %*
exit /b %ERRORLEVEL%
