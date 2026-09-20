@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Local-first Windows build helper. Run from any directory or double-click.
cd /d "%~dp0"
set "SKIP_TESTS=0"
set "WEB_ONLY=0"
set "BUNDLE=0"
set "NO_PAUSE=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--skip-tests" set "SKIP_TESTS=1"
if /I "%~1"=="--web-only" set "WEB_ONLY=1"
if /I "%~1"=="--bundle" set "BUNDLE=1"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
if /I "%~1"=="--help" goto help
if /I "%~1"=="-h" goto help
if /I not "%~1"=="--skip-tests" if /I not "%~1"=="--web-only" if /I not "%~1"=="--bundle" if /I not "%~1"=="--no-pause" if /I not "%~1"=="--help" if /I not "%~1"=="-h" goto unknown_arg
shift
goto parse_args

:args_done
echo [1/5] Checking local toolchain...
where node >nul 2>&1
if errorlevel 1 goto missing_node
where pnpm >nul 2>&1
if errorlevel 1 goto missing_pnpm
node --version
call pnpm --version
if "%WEB_ONLY%"=="1" goto install
where cargo >nul 2>&1
if errorlevel 1 goto missing_cargo
cargo --version
where uv >nul 2>&1
if errorlevel 1 goto missing_uv
uv --version

:install
echo [2/5] Installing workspace dependencies with the existing lockfile...
call pnpm install --frozen-lockfile
if errorlevel 1 goto failed

if "%SKIP_TESTS%"=="1" goto build_web
echo [3/5] Running desktop typecheck...
call pnpm --filter @anime-pic-manage/desktop typecheck
if errorlevel 1 goto failed
echo [4/5] Running desktop tests...
call pnpm --filter @anime-pic-manage/desktop test
if errorlevel 1 goto failed

:build_web
echo [5/5] Building desktop frontend...
call pnpm --filter @anime-pic-manage/desktop build
if errorlevel 1 goto failed
if "%WEB_ONLY%"=="1" goto success
if "%BUNDLE%"=="1" goto bundle_build
echo Building the local Tauri executable (installer bundling disabled)...
call pnpm --filter @anime-pic-manage/desktop exec tauri build --no-bundle
if errorlevel 1 goto failed
goto package_portable

:bundle_build
echo Building the Tauri installer bundle (may require WiX or platform bundlers)...
call pnpm --filter @anime-pic-manage/desktop tauri:build
if errorlevel 1 goto failed
goto package_portable

:package_portable
echo [6/6] Creating the portable dist_windows directory...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_windows.ps1" -ProjectRoot "%~dp0."
if errorlevel 1 goto failed
goto success

:success
echo.
echo BUILD SUCCEEDED.
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
echo ERROR: cargo was not found on PATH. Use --web-only for a frontend-only build.
goto failed
:missing_uv
echo ERROR: uv was not found on PATH. It is required for the portable Worker and ENV runtime.
goto failed
:unknown_arg
echo ERROR: unknown argument "%~1". Use --help to see supported options.
set "BUILD_EXIT=2"
goto failed
:failed
if not defined BUILD_EXIT set "BUILD_EXIT=!errorlevel!"
if "!BUILD_EXIT!"=="0" set "BUILD_EXIT=1"
echo.
echo BUILD FAILED with exit code !BUILD_EXIT!.
if "%NO_PAUSE%"=="1" exit /b !BUILD_EXIT!
pause
exit /b !BUILD_EXIT!

:help
echo Usage: build.bat [--skip-tests] [--web-only] [--bundle] [--no-pause]
echo   --skip-tests  Skip desktop typecheck and tests.
echo   --web-only    Build the frontend without requiring cargo or Tauri.
echo   --bundle      Build installer bundles. Omit for a local executable without WiX downloads.
echo                 Full desktop builds also create the portable dist_windows folder.
echo   --no-pause    Do not pause before returning to the caller.
exit /b 0
