param(
    [string]$ProjectRoot,
    # Build the portable runtime with the CUDA-enabled ONNX Runtime build.
    # The target machine still needs a matching NVIDIA driver plus CUDA/cuDNN
    # runtime libraries; CPU execution stays available as a fallback.
    [switch]$Cuda,
    # Output folder name directly under the project root.
    [string]$DistName = "dist_windows",
    # Reuse the Worker runtime and Python env from an earlier build of the same
    # dist folder. Only the desktop executable, Worker source, models and
    # resources are refreshed, which keeps UI/logic rebuilds fast.
    [switch]$ReuseEnvironment
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($DistName) -or
    $DistName -ne [IO.Path]::GetFileName($DistName) -or
    $DistName -match '^\.+$') {
    throw "DistName must be a single folder name directly under the project root."
}
$Dist = [IO.Path]::GetFullPath((Join-Path $Root $DistName))

$distParent = [IO.Directory]::GetParent($Dist).FullName
if ([IO.Path]::GetFileName($Dist.TrimEnd('\')) -ne $DistName -or
    $distParent.TrimEnd('\') -ne $Root.TrimEnd('\')) {
    throw "Refusing to package outside $Root\$DistName."
}

$WorkerRoot = Join-Path $Root "apps\ai-worker"
# A dedicated environment keeps the CUDA runtime from leaking into the normal
# development venv and gives the fast rebuild something stable to reuse.
$WorkerEnvRoot = if ($Cuda) {
    Join-Path $WorkerRoot ".venv-gpu"
} else {
    Join-Path $WorkerRoot ".venv"
}
$WorkerEnvPython = Join-Path $WorkerEnvRoot "Scripts\python.exe"

if ($ReuseEnvironment) {
    $reusedWorkerExe = Join-Path $Dist "app\runtime\ai-worker.exe"
    $reusedEnvPython = Join-Path $Dist "app\env\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $reusedWorkerExe -PathType Leaf) -or
        -not (Test-Path -LiteralPath $reusedEnvPython -PathType Leaf)) {
        throw "ReuseEnvironment requires an existing build in $DistName. Run the full build first."
    }
    Write-Host "[portable] Reusing Worker runtime and Python env from $DistName."
} elseif (Test-Path -LiteralPath $Dist) {
    $existing = Get-Item -LiteralPath $Dist
    if (-not $existing.PSIsContainer) {
        throw "Refusing to remove $DistName because it is not a directory."
    }
    Remove-Item -LiteralPath $Dist -Recurse -Force
}
if (-not (Test-Path -LiteralPath $Dist)) {
    New-Item -ItemType Directory -Path $Dist | Out-Null
}

$DesktopExe = Join-Path $Root "apps\desktop\src-tauri\target\release\anime-pic-manage.exe"
$WorkerExe = Join-Path $WorkerRoot "dist\ai-worker.exe"
$Models = Join-Path $Root "models"
$Resources = Join-Path $Root "resources"
$AppRoot = Join-Path $Dist "app"
$RuntimeRoot = Join-Path $AppRoot "runtime"
$PackagedWorkerExe = Join-Path $RuntimeRoot "ai-worker.exe"
$EnvRoot = Join-Path $AppRoot "env"
$CudaRoot = Join-Path $AppRoot "cuda"
$PythonInstall = Join-Path $Dist ".python-staging"
$Requirements = Join-Path $Dist ".requirements-portable.txt"
$PythonVersion = "3.12.11"
$OnnxRuntimePackage = if ($Cuda) { "onnxruntime-gpu>=1.20,<1.21" } else { $null }
$OnnxDistribution = if ($Cuda) { "onnxruntime-gpu" } else { "onnxruntime" }

if (-not (Test-Path -LiteralPath $DesktopExe -PathType Leaf)) {
    throw "Release desktop executable was not found: $DesktopExe"
}
if (-not (Test-Path -LiteralPath $Models -PathType Container)) {
    throw "Models directory was not found: $Models"
}

if (-not $ReuseEnvironment) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is required to build the portable Worker and ENV runtime."
    }
    Write-Host "[portable] Preparing the Worker build environment ($WorkerEnvRoot)..."
    $env:UV_PROJECT_ENVIRONMENT = $WorkerEnvRoot
    & uv sync --project $WorkerRoot --locked --extra model --extra detector --extra package
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed ($LASTEXITCODE)." }
    if ($Cuda) {
        if (-not (Test-Path -LiteralPath $WorkerEnvPython -PathType Leaf)) {
            throw "uv sync did not provide a build Python for the CUDA runtime: $WorkerEnvPython"
        }
        Write-Host "[portable] Swapping in the CUDA ONNX Runtime for the packaged Worker..."
        & uv pip install --python $WorkerEnvPython --reinstall $OnnxRuntimePackage
        if ($LASTEXITCODE -ne 0) { throw "uv pip install onnxruntime-gpu failed ($LASTEXITCODE)." }
    }
} else {
    Write-Host "[portable] Reusing the existing Worker build environment ($WorkerEnvRoot)."
}

if (-not (Test-Path -LiteralPath $WorkerEnvPython -PathType Leaf)) {
    throw "A Worker build environment is required at $WorkerEnvPython. Run the full build first."
}
# PyInstaller runs with the environment's own interpreter instead of `uv run`,
# because `uv run` would resync the lockfile and silently reinstall the CPU
# ONNX Runtime over the CUDA build.
Write-Host "[portable] Building independent ai-worker.exe..."
& $WorkerEnvPython -m PyInstaller `
    --noconfirm --clean --onefile --console --name ai-worker `
    --hidden-import onnxruntime `
    --hidden-import numpy `
    --hidden-import imgutils.preprocess.pillow `
    --hidden-import imgutils.generic.yolo `
    --hidden-import imgutils.data `
    --hidden-import imgutils.tagging `
    --hidden-import pandas `
    --hidden-import huggingface_hub `
    --copy-metadata dghs-imgutils `
    --copy-metadata $OnnxDistribution `
    --copy-metadata pandas `
    --copy-metadata huggingface_hub `
    --distpath (Join-Path $WorkerRoot "dist") `
    --workpath (Join-Path $WorkerRoot "build") `
    --specpath (Join-Path $WorkerRoot "build") `
    (Join-Path $WorkerRoot "run_worker.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)." }
if (-not (Test-Path -LiteralPath $WorkerExe -PathType Leaf)) {
    throw "PyInstaller did not produce $WorkerExe"
}

if (-not $ReuseEnvironment) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is required to build the portable ENV runtime."
    }
    Write-Host "[portable] Installing relocatable Python $PythonVersion..."
    New-Item -ItemType Directory -Path $PythonInstall | Out-Null
    & uv python install $PythonVersion --install-dir $PythonInstall --force --no-progress
    if ($LASTEXITCODE -ne 0) { throw "uv python install failed ($LASTEXITCODE)." }
    $PythonRoot = Get-ChildItem -LiteralPath $PythonInstall -Directory |
        Where-Object { $_.Name -like "cpython-$PythonVersion-*" } |
        Select-Object -First 1
    if (-not $PythonRoot) {
        throw "uv did not create a managed Python $PythonVersion directory."
    }

    New-Item -ItemType Directory -Path $EnvRoot | Out-Null
    Copy-Item -Path (Join-Path $PythonRoot.FullName "*") -Destination $EnvRoot -Recurse -Force
    $EnvScripts = Join-Path $EnvRoot "Scripts"
    New-Item -ItemType Directory -Path $EnvScripts -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $EnvRoot "python.exe") -Destination (Join-Path $EnvScripts "python.exe") -Force
    foreach ($dll in @("python312.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll")) {
        $source = Join-Path $EnvRoot $dll
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Copy-Item -LiteralPath $source -Destination $EnvScripts -Force
        }
    }
    Set-Content -LiteralPath (Join-Path $EnvScripts "python._pth") -Encoding ascii -Value @(
        "..\python312.zip",
        "..\DLLs",
        "..\Lib",
        "..\Lib\site-packages",
        "..\worker\src",
        "import site"
    )
}

if (-not $ReuseEnvironment) {
    Write-Host "[portable] Installing Worker dependencies into env..."
    $SitePackages = Join-Path $EnvRoot "Lib\site-packages"
    New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null
    $BuildPython = $WorkerEnvPython
    if (-not (Test-Path -LiteralPath $BuildPython -PathType Leaf)) {
        throw "uv sync did not provide a usable build Python: $BuildPython"
    }
    try {
    & uv export --project $WorkerRoot --locked --format requirements.txt --no-dev `
        --extra model --extra detector --no-emit-project --output-file $Requirements | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "uv export failed ($LASTEXITCODE)." }
    # Install from the lock export with the build machine's known-good Python.
    # The portable embedded interpreter is intentionally not used as uv's
    # resolver environment: its python._pth isolation is for runtime imports,
    # and uv may otherwise try to create a temporary virtualenv for --target.
    & uv pip install --python $BuildPython --python-version 3.12 --python-platform windows `
        --only-binary ":all:" --require-hashes --target $SitePackages --requirements $Requirements
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed ($LASTEXITCODE)." }
    if ($Cuda) {
        Write-Host "[portable] Replacing the CPU ONNX Runtime in the compatibility env..."
        foreach ($pattern in @(
            "onnxruntime",
            "onnxruntime.libs",
            "onnxruntime-*.dist-info",
            "onnxruntime_gpu*.dist-info",
            "onnxruntime_gpu.libs"
        )) {
            Get-ChildItem -LiteralPath $SitePackages -Filter $pattern -ErrorAction SilentlyContinue |
                ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
        }
        & uv pip install --python $BuildPython --python-version 3.12 --python-platform windows `
            --only-binary ":all:" --target $SitePackages $OnnxRuntimePackage
        if ($LASTEXITCODE -ne 0) { throw "uv pip install onnxruntime-gpu into env failed ($LASTEXITCODE)." }
    }
    }
    finally {
        if (Test-Path -LiteralPath $Requirements -PathType Leaf) {
            Remove-Item -LiteralPath $Requirements -Force
        }
    }
}

Write-Host "[portable] Copying desktop, Worker source, models, and resources..."
Copy-Item -LiteralPath $DesktopExe -Destination (Join-Path $Dist "anime-pic-manage.exe") -Force
New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
New-Item -ItemType Directory -Path $CudaRoot -Force | Out-Null
# `app\cuda` is where the in-app "download CUDA runtime" action lands. The
# user-facing text lives in resources\portable because Windows PowerShell 5.1
# mis-decodes UTF-8 without BOM; keeping this script ASCII avoids that.
# Names below are built from code points so this file stays pure ASCII.
$cudaNoteName = ([char]0x8BF4) + ([char]0x660E) + ".txt"                    # 说明.txt
Copy-Item -LiteralPath (Join-Path $Root "resources\portable\cuda-$cudaNoteName") `
    -Destination (Join-Path $CudaRoot $cudaNoteName) -Force
Copy-Item -LiteralPath $WorkerExe -Destination $PackagedWorkerExe -Force
New-Item -ItemType Directory -Path (Join-Path $EnvRoot "worker") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $WorkerRoot "run_worker.py") -Destination (Join-Path $EnvRoot "worker\run_worker.py") -Force
$WorkerSourceTarget = Join-Path $EnvRoot "worker\src"
if (Test-Path -LiteralPath $WorkerSourceTarget) {
    Remove-Item -LiteralPath $WorkerSourceTarget -Recurse -Force
}
Copy-Item -LiteralPath (Join-Path $WorkerRoot "src") -Destination $WorkerSourceTarget -Recurse -Force
foreach ($payload in @(
    @{ Source = $Models; Target = (Join-Path $Dist "models") },
    @{ Source = $Resources; Target = (Join-Path $Dist "resources") }
)) {
    # Merge instead of replacing: a rebuild must not delete models the user
    # downloaded in the app (they live under the same models\ tree).
    New-Item -ItemType Directory -Path $payload.Target -Force | Out-Null
    Copy-Item -Path (Join-Path $payload.Source "*") -Destination $payload.Target -Recurse -Force
}

function Invoke-WorkerJsonProbe {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string]$Request,
        [Parameter(Mandatory = $true)][string]$ExpectedRequestId,
        [Parameter(Mandatory = $true)][string]$ExpectedTaskId,
        [string[]]$Arguments = @()
    )
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Executable
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    # NOTE: Windows PowerShell 5.1 (.NET Framework) still writes a UTF-8 BOM
    # preamble to the child's stdin, and ProcessStartInfo has no
    # StandardInputEncoding there. The Worker accepts a leading BOM instead.
    $startInfo.WorkingDirectory = (Get-Item -LiteralPath $Executable).DirectoryName
    # Use the legacy Arguments property for compatibility with both PowerShell
    # 5 and pwsh hosts. Worker paths cannot contain quotes, so this is safe for
    # the portable launcher path and still handles spaces in its parent dirs.
    $startInfo.Arguments = ($Arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' '
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Worker probe could not start $Executable."
    }
    $requestBytes = [Text.Encoding]::UTF8.GetBytes($Request + [Environment]::NewLine)
    $process.StandardInput.BaseStream.Write($requestBytes, 0, $requestBytes.Length)
    $process.StandardInput.Close()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    $process.Dispose()
    if ($exitCode -ne 0) {
        throw "Worker probe failed for $Executable (exit code $exitCode): $stderr"
    }
    $lines = @($stdout -split '\r?\n' | Where-Object { $_.Trim().Length -gt 0 })
    if ($lines.Count -ne 1) {
        throw "Worker probe for $Executable returned $($lines.Count) JSON lines instead of one."
    }
    try {
        $response = $lines[0] | ConvertFrom-Json
    }
    catch {
        throw "Worker probe for $Executable returned invalid JSON: $($_.Exception.Message)"
    }
    if ($response.request_id -ne $ExpectedRequestId -or $response.task_id -ne $ExpectedTaskId) {
        throw "Worker probe for $Executable did not preserve UTF-8 request/task IDs."
    }
    if ($null -ne $response.error) {
        throw "Worker probe for $Executable returned $($response.error.code): $($response.error.message)"
    }
    if ($null -eq $response.payload) {
        throw "Worker probe for $Executable returned no payload."
    }
    return $response.payload
}

function Assert-WorkerCapabilities {
    param(
        [Parameter(Mandatory = $true)]$Payload,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ($Payload.status -ne "ok" -or $Payload.ready -ne $true) {
        $firstError = $Payload.errors | Select-Object -First 1
        $detail = if ($null -ne $firstError) { "$($firstError.code): $($firstError.message)" } else { "unknown capability error" }
        throw "$Label runtime capabilities are not ready: $detail"
    }
    foreach ($featureName in @(
        "imgutils.preprocess.pillow",
        "imgutils.generic.yolo",
        "imgutils.data",
        "imgutils.tagging"
    )) {
        $feature = $Payload.features.PSObject.Properties[$featureName]
        if ($null -eq $feature -or $feature.Value.status -ne "ready") {
            throw "$Label missing required runtime feature: $featureName"
        }
    }
    foreach ($dependencyName in @("dghs_imgutils", "onnxruntime")) {
        $dependency = $Payload.$dependencyName
        if ($null -eq $dependency -or $dependency.status -ne "ready" -or [string]::IsNullOrWhiteSpace($dependency.version)) {
            throw "$Label missing runtime dependency or version: $dependencyName"
        }
    }
}

$healthRequestId = "package-health"
$healthTaskId = "package-health"
$healthRequest = '{"schema_version":"1.0","request_id":"package-health","task_id":"package-health","message_type":"health","payload":{},"error":null}'
$capabilitiesRequestId = "capability-" + [char]0x4e2d + [char]0x6587
$capabilitiesTaskId = [char]0x4efb + [char]0x52a1 + "-" + [char]0x4e2d + [char]0x6587
$capabilitiesRequest = '{"schema_version":"1.0","request_id":"' + $capabilitiesRequestId + '","task_id":"' + $capabilitiesTaskId + '","message_type":"runtime.capabilities","payload":{},"error":null}'
Write-Host "[portable] Verifying independent Worker health and runtime capabilities..."
$exeHealth = Invoke-WorkerJsonProbe -Executable $PackagedWorkerExe -Request $healthRequest -ExpectedRequestId $healthRequestId -ExpectedTaskId $healthTaskId
if ($exeHealth.status -ne "ok" -and $exeHealth.status -ne "stopping") {
    throw "Independent Worker health check failed: $($exeHealth.status)"
}
$exeCapabilities = Invoke-WorkerJsonProbe -Executable $PackagedWorkerExe -Request $capabilitiesRequest -ExpectedRequestId $capabilitiesRequestId -ExpectedTaskId $capabilitiesTaskId
Assert-WorkerCapabilities -Payload $exeCapabilities -Label "Independent Worker"
if ($Cuda -and -not $exeCapabilities.compute.cuda_available) {
    throw "CUDA build did not expose the CUDA execution provider: $($exeCapabilities.compute.available_providers -join ', ')"
}

Write-Host "[portable] Verifying ENV Worker health and runtime capabilities..."
$envPython = Join-Path $EnvRoot "Scripts\python.exe"
$envLauncher = Join-Path $EnvRoot "worker\run_worker.py"
$envHealth = Invoke-WorkerJsonProbe -Executable $envPython -Arguments @($envLauncher) -Request $healthRequest -ExpectedRequestId $healthRequestId -ExpectedTaskId $healthTaskId
if ($envHealth.status -ne "ok" -and $envHealth.status -ne "stopping") {
    throw "ENV Worker health check failed: $($envHealth.status)"
}
$envCapabilities = Invoke-WorkerJsonProbe -Executable $envPython -Arguments @($envLauncher) -Request $capabilitiesRequest -ExpectedRequestId $capabilitiesRequestId -ExpectedTaskId $capabilitiesTaskId
Assert-WorkerCapabilities -Payload $envCapabilities -Label "ENV Worker"
if ($Cuda -and -not $envCapabilities.compute.cuda_available) {
    throw "CUDA compatibility env did not expose the CUDA execution provider."
}

if (Test-Path -LiteralPath $PythonInstall) {
    Remove-Item -LiteralPath $PythonInstall -Recurse -Force
}

$rootExecutables = @(Get-ChildItem -LiteralPath $Dist -File -Filter "*.exe")
if ($rootExecutables.Count -ne 1 -or $rootExecutables[0].Name -ne "anime-pic-manage.exe") {
    throw "Portable output root must contain only anime-pic-manage.exe. Found: $($rootExecutables.Name -join ', ')"
}
if (-not (Test-Path -LiteralPath $PackagedWorkerExe -PathType Leaf)) {
    throw "Portable output is missing the internal inference engine: $PackagedWorkerExe"
}

# Novice-facing entry points. The text lives in resources\portable because
# Windows PowerShell 5.1 mis-decodes UTF-8 scripts without a BOM.
New-Item -ItemType Directory -Path (Join-Path $Dist "output\generated") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Dist "output\loras") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Dist "output\datasets") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Dist "temp") -Force | Out-Null
$launcherName = ([char]0x542F) + ([char]0x52A8) + ".bat"                   # 启动.bat
$guideName = ([char]0x4F7F) + ([char]0x7528) + ([char]0x8BF4) + ([char]0x660E) + ".txt"  # 使用说明.txt
foreach ($file in @($launcherName, $guideName)) {
    Copy-Item -LiteralPath (Join-Path $Root "resources\portable\$file") `
        -Destination (Join-Path $Dist $file) -Force
}
if (Test-Path -LiteralPath (Join-Path $Dist "README-portable.txt")) {
    Remove-Item -LiteralPath (Join-Path $Dist "README-portable.txt") -Force
}

Write-Host "[portable] Output: $Dist"
