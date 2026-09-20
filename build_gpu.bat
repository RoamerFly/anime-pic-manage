@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Full GPU build: builds the CUDA Worker runtime, the portable Python env and
rem the desktop executable, then packages everything into dist_windows_gpu.
rem Use build_gpu_fast.bat when only the UI or Worker logic changed.
cd /d "%~dp0"
set "SKIP_TESTS=0"
set "NO_PAUSE=0"
set "RUNTIME_MODE=exe"

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
echo [1/6] Checking local toolchain...
where node >nul 2>&1
if errorlevel 1 goto missing_node
where pnpm >nul 2>&1
if errorlevel 1 goto missing_pnpm
where cargo >nul 2>&1
if errorlevel 1 goto missing_cargo
where uv >nul 2>&1
if errorlevel 1 goto missing_uv
node --version
call pnpm --version
cargo --version
uv --version

echo [2/6] Installing workspace dependencies with the existing lockfile...
call pnpm install --frozen-lockfile
if errorlevel 1 goto failed

if "%SKIP_TESTS%"=="1" goto build_web
echo [3/6] Running desktop typecheck and tests...
call pnpm --filter @anime-pic-manage/desktop typecheck
if errorlevel 1 goto failed
call pnpm --filter @anime-pic-manage/desktop test
if errorlevel 1 goto failed

:build_web
echo [4/6] Building desktop frontend...
call pnpm --filter @anime-pic-manage/desktop build
if errorlevel 1 goto failed

echo [5/6] Building the Tauri executable (installer bundling disabled)...
call pnpm --filter @anime-pic-manage/desktop exec tauri build --no-bundle
if errorlevel 1 goto failed

echo [6/6] Packaging the CUDA portable build into dist_windows_gpu...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_windows.ps1" -ProjectRoot "%~dp0." -Cuda -DistName dist_windows_gpu
if errorlevel 1 goto failed
goto success

:success
echo.
echo GPU BUILD SUCCEEDED. Output: dist_windows_gpu
if "%NO_PAUSE%"=="1" exit /b 0
if not "%~1"=="" exit /b 0
pause
exit /b 0

:missing_node
echo ERROR: node was not found on PATH.
goto failed
:missing_pnpm
echo ERROR: pnpm was not found on PATH.
goto failed
:missing_cargo
echo ERROR: cargo was not found on PATH. Install the Rust toolchain first.
goto failed
:missing_uv
echo ERROR: uv was not found on PATH. It is required for the GPU Worker runtime.
goto failed
:unknown_arg
echo ERROR: unknown argument "%~1". Use --help to see supported options.
set "BUILD_EXIT=2"
goto failed
:failed
if not defined BUILD_EXIT set "BUILD_EXIT=!errorlevel!"
if "!BUILD_EXIT!"=="0" set "BUILD_EXIT=1"
echo.
echo GPU BUILD FAILED with exit code !BUILD_EXIT!.
if "%NO_PAUSE%"=="1" exit /b !BUILD_EXIT!
pause
exit /b !BUILD_EXIT!

:help
echo Usage: build_gpu.bat [--skip-tests] [--no-pause]
echo   Builds the CUDA Worker runtime, portable Python env and desktop app,
echo   then writes the result to dist_windows_gpu.
echo   --skip-tests  Skip desktop typecheck and tests.
echo   --no-pause    Do not pause before returning to the caller.
exit /b 0
