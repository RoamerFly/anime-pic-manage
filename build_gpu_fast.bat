@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Fast GPU rebuild: reuses the GPU Worker runtime and portable Python env that
rem build_gpu.bat already produced in dist_windows_gpu, and only refreshes the
rem desktop executable, Worker exe, Worker source, models and resources.
rem Requires a previous successful build_gpu.bat run.
cd /d "%~dp0"
set "SKIP_TESTS=0"
set "NO_PAUSE=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--skip-tests" set "SKIP_TESTS=1"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
if /I "%~1"=="--help" goto help
if /I "%~1"=="-h" goto help
if /I not "%~1"=="--skip-tests" if /I not "%~1"=="--no-pause" if /I not "%~1"=="--help" if /I not "%~1"=="-h" goto unknown_arg
shift
goto parse_args

:args_done
if not exist "dist_windows_gpu\runtime\ai-worker.exe" goto missing_gpu_build
if not exist "dist_windows_gpu\env\Scripts\python.exe" goto missing_gpu_build

echo [1/5] Checking local toolchain...
where node >nul 2>&1
if errorlevel 1 goto missing_node
where pnpm >nul 2>&1
if errorlevel 1 goto missing_pnpm
where cargo >nul 2>&1
if errorlevel 1 goto missing_cargo
node --version
call pnpm --version

echo [2/5] Installing workspace dependencies with the existing lockfile...
call pnpm install --frozen-lockfile
if errorlevel 1 goto failed

if "%SKIP_TESTS%"=="1" goto build_web
echo [3/5] Running desktop typecheck and tests...
call pnpm --filter @anime-pic-manage/desktop typecheck
if errorlevel 1 goto failed
call pnpm --filter @anime-pic-manage/desktop test
if errorlevel 1 goto failed

:build_web
echo [4/5] Building desktop frontend and Tauri executable...
call pnpm --filter @anime-pic-manage/desktop build
if errorlevel 1 goto failed
call pnpm --filter @anime-pic-manage/desktop exec tauri build --no-bundle
if errorlevel 1 goto failed

echo [5/5] Repackaging into dist_windows_gpu (reusing the GPU environment)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_windows.ps1" -ProjectRoot "%~dp0." -Cuda -DistName dist_windows_gpu -ReuseEnvironment
if errorlevel 1 goto failed
goto success

:success
echo.
echo FAST GPU BUILD SUCCEEDED. Output: dist_windows_gpu
if "%NO_PAUSE%"=="1" exit /b 0
if not "%~1"=="" exit /b 0
pause
exit /b 0

:missing_gpu_build
echo ERROR: no previous GPU build was found in dist_windows_gpu.
echo        Run build_gpu.bat once to create the CUDA runtime and Python env.
set "BUILD_EXIT=1"
goto failed
:missing_node
echo ERROR: node was not found on PATH.
goto failed
:missing_pnpm
echo ERROR: pnpm was not found on PATH.
goto failed
:missing_cargo
echo ERROR: cargo was not found on PATH. Install the Rust toolchain first.
goto failed
:unknown_arg
echo ERROR: unknown argument "%~1". Use --help to see supported options.
set "BUILD_EXIT=2"
goto failed
:failed
if not defined BUILD_EXIT set "BUILD_EXIT=!errorlevel!"
if "!BUILD_EXIT!"=="0" set "BUILD_EXIT=1"
echo.
echo FAST GPU BUILD FAILED with exit code !BUILD_EXIT!.
if "%NO_PAUSE%"=="1" exit /b !BUILD_EXIT!
pause
exit /b !BUILD_EXIT!

:help
echo Usage: build_gpu_fast.bat [--skip-tests] [--no-pause]
echo   Reuses the CUDA runtime and Python env from a previous build_gpu.bat run
echo   and refreshes the desktop exe, Worker exe, Worker source and resources.
echo   --skip-tests  Skip desktop typecheck and tests.
echo   --no-pause    Do not pause before returning to the caller.
exit /b 0
