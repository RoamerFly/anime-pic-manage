param(
    [string]$ProjectRoot,
    # Package the CUDA Worker runtime instead of the CPU one.
    [switch]$Cuda,
    # Portable folder to stage from; defaults to dist_windows(_gpu).
    [string]$DistName,
    # Version written into the installer. Defaults to tauri.conf.json.
    [string]$Version,
    # Skip updater artifacts even when a signing key is present.
    [switch]$NoSign,
    # Keep the staged payload for inspection.
    [switch]$KeepPayload
)

# Builds the standard Windows installer (Tauri's NSIS bundle) from a portable
# build. The installer contains the program and its Worker runtime, not the
# models: 设置 → 模型配置 downloads those, and can import an existing cache.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($DistName)) {
    $DistName = if ($Cuda) { "dist_windows_gpu" } else { "dist_windows" }
}
if ($DistName -ne [IO.Path]::GetFileName($DistName) -or $DistName -match '^\.+$') {
    throw "DistName must be a single folder name directly under the project root."
}
$Dist = [IO.Path]::GetFullPath((Join-Path $Root $DistName))
if ([IO.Path]::GetFileName($Dist.TrimEnd('\')) -ne $DistName -or
    [IO.Directory]::GetParent($Dist).FullName.TrimEnd('\') -ne $Root.TrimEnd('\')) {
    throw "Refusing to package outside $Root\$DistName."
}

$WorkerExe = Join-Path $Dist "app\runtime\ai-worker.exe"
$DistResources = Join-Path $Dist "resources"
if (-not (Test-Path -LiteralPath $WorkerExe -PathType Leaf)) {
    throw "Portable build not found at $Dist. Run build.bat or build_gpu.bat first."
}
if (-not (Test-Path -LiteralPath $DistResources -PathType Container)) {
    throw "Resources not found at $DistResources. Run build.bat or build_gpu.bat first."
}

$DesktopRoot = Join-Path $Root "apps\desktop"
$PayloadRoot = Join-Path $Root "apps\desktop\src-tauri\installer-payload"
$BundleRoot = Join-Path $Root "apps\desktop\src-tauri\target\release\bundle\nsis"
$OutputRoot = Join-Path $Dist "installer"

if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    throw "pnpm is required to build the installer."
}

Write-Host "[installer] Staging the payload (Worker runtime + resources)..."
if (Test-Path -LiteralPath $PayloadRoot) {
    Remove-Item -LiteralPath $PayloadRoot -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $PayloadRoot "app") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $Dist "app\runtime") `
    -Destination (Join-Path $PayloadRoot "app\runtime") -Recurse -Force
Copy-Item -LiteralPath $DistResources -Destination (Join-Path $PayloadRoot "resources") -Recurse -Force

# Updater artifacts are only produced when a signing key is available; a local
# build without the key still produces a normal, working installer.
$sign = -not $NoSign -and -not [string]::IsNullOrWhiteSpace($env:TAURI_SIGNING_PRIVATE_KEY)
if ($sign) {
    Write-Host "[installer] Updater artifacts enabled (signing key detected)."
} else {
    Write-Host "[installer] Building without updater artifacts (no signing key)."
}

# The portable payload is declared here instead of tauri.conf.json so that
# `cargo test`, `tauri dev` and the portable builds never depend on a staged
# folder that only the installer build creates. Absolute source paths keep the
# overlay file's own location out of the picture.
$overrides = @{
    bundle = @{
        createUpdaterArtifacts = $sign
        resources = @{
            (Join-Path $PayloadRoot "app") = "app"
            (Join-Path $PayloadRoot "resources") = "resources"
        }
    }
}
if (-not [string]::IsNullOrWhiteSpace($Version)) {
    $overrides.version = $Version
    Write-Host "[installer] Version override: $Version"
}
# Tauri accepts either inline JSON or a path. Windows PowerShell 5.1 mangles
# quotes when handing inline JSON to a native command, so write a temp file.
$overrideFile = Join-Path ([IO.Path]::GetTempPath()) ("anime-pic-manage-installer-{0}.json" -f [guid]::NewGuid())
[IO.File]::WriteAllText($overrideFile, ($overrides | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))

try {
    Write-Host "[installer] Running the Tauri NSIS bundle..."
    Push-Location $DesktopRoot
    try {
        & pnpm --filter "@anime-pic-manage/desktop" exec tauri build --config $overrideFile
        if ($LASTEXITCODE -ne 0) {
            throw "tauri build failed ($LASTEXITCODE)."
        }
    } finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $BundleRoot -PathType Container)) {
        throw "Tauri did not produce an NSIS bundle at $BundleRoot."
    }
    $installers = @(Get-ChildItem -LiteralPath $BundleRoot -Filter "*-setup.exe" -File)
    if ($installers.Count -eq 0) {
        throw "No *-setup.exe was produced in $BundleRoot."
    }

    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    foreach ($installer in $installers) {
        $target = Join-Path $OutputRoot $installer.Name
        Copy-Item -LiteralPath $installer.FullName -Destination $target -Force
        Write-Host ("[installer] {0} ({1:N1} MB)" -f $target, ((Get-Item -LiteralPath $target).Length / 1MB))
    }
    Write-Host "[installer] Output: $OutputRoot"
} finally {
    if (Test-Path -LiteralPath $overrideFile) {
        Remove-Item -LiteralPath $overrideFile -Force
    }
    if (-not $KeepPayload -and (Test-Path -LiteralPath $PayloadRoot)) {
        Remove-Item -LiteralPath $PayloadRoot -Recurse -Force
    }
}
