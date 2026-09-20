@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Builds the standard Windows installer (NSIS) from an existing portable build.
rem The installer contains the program and the Worker runtime; models and caches
rem are downloaded from 设置 -> 模型配置 inside the app.
rem
rem   build_installer.bat              CPU installer from dist_windows
rem   build_installer.bat --gpu        GPU installer from dist_windows_gpu
rem   build_installer.bat --version 0.2.0
rem   build_installer.bat --no-sign    skip updater artifacts even with a key
rem
rem Run build.bat / build_gpu.bat first so the portable folder exists.

cd /d "%~dp0"
set "GPU=0"
set "NO_PAUSE=0"
set "EXTRA_ARGS="
set "VERSION="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--gpu" set "GPU=1"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
if /I "%~1"=="--no-sign" set "EXTRA_ARGS=%EXTRA_ARGS% -NoSign"
if /I "%~1"=="--version" (
    set "VERSION=%~2"
    shift
)
if /I "%~1"=="--help" goto help
if /I "%~1"=="-h" goto help
shift
goto parse_args

:args_done
where pnpm >nul 2>&1
if errorlevel 1 goto missing_pnpm
where cargo >nul 2>&1
if errorlevel 1 goto missing_cargo

if "%GPU%"=="1" goto build_gpu
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_installer_windows.ps1" -ProjectRoot "%~dp0." %EXTRA_ARGS%
goto after_build

:build_gpu
if "%VERSION%"=="" goto build_gpu_no_version
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_installer_windows.ps1" -ProjectRoot "%~dp0." -Cuda -Version "%VERSION%" %EXTRA_ARGS%
goto after_build

:build_gpu_no_version
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_installer_windows.ps1" -ProjectRoot "%~dp0." -Cuda %EXTRA_ARGS%
goto after_build

:after_build
if errorlevel 1 goto failed
echo.
echo INSTALLER BUILD SUCCEEDED. See the "installer" folder inside the portable build.
if "%NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0

:missing_pnpm
echo ERROR: pnpm was not found on PATH.
goto failed
:missing_cargo
echo ERROR: cargo was not found on PATH. Install the Rust toolchain first.
goto failed
:failed
set "BUILD_EXIT=%errorlevel%"
if "%BUILD_EXIT%"=="0" set "BUILD_EXIT=1"
echo.
echo INSTALLER BUILD FAILED with exit code %BUILD_EXIT%.
if "%NO_PAUSE%"=="1" exit /b %BUILD_EXIT%
pause
exit /b %BUILD_EXIT%

:help
echo Usage: build_installer.bat [--gpu] [--version X.Y.Z] [--no-sign] [--no-pause]
echo   Builds the standard Windows installer (NSIS) from the portable build.
echo   --gpu       use dist_windows_gpu (CUDA Worker runtime)
echo   --version   override the version written into the installer
echo   --no-sign   skip updater artifacts even when a signing key is set
echo   --no-pause  do not pause before returning to the caller
exit /b 0
