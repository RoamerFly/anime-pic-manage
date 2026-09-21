param(
    [Parameter(Mandatory = $true)][string]$TargetDirectory,
    [switch]$ReuseExisting
)

# Download the official NVIDIA Windows runtime wheels from PyPI and flatten
# their runtime DLLs into app\cuda. Keeping the files beside the application
# makes the GPU package self-contained without modifying the system CUDA setup.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.IO.Compression.FileSystem

$target = [IO.Path]::GetFullPath($TargetDirectory)
$targetRoot = [IO.Path]::GetPathRoot($target)
if ([string]::IsNullOrWhiteSpace($target) -or $target.TrimEnd('\') -eq $targetRoot.TrimEnd('\')) {
    throw "Refusing to prepare CUDA in a filesystem root: $target"
}

$required = @(
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll"
)
$packages = @(
    @{ Name = "nvidia-cuda-runtime-cu12"; Label = "CUDA Runtime 12"; Dlls = @("cudart64_12.dll") },
    @{ Name = "nvidia-cublas-cu12"; Label = "cuBLAS 12"; Dlls = @("cublas64_12.dll", "cublasLt64_12.dll") },
    @{ Name = "nvidia-cufft-cu12"; Label = "cuFFT 12"; Dlls = @("cufft64_11.dll") },
    @{ Name = "nvidia-cudnn-cu12"; Label = "cuDNN 9"; Dlls = @("cudnn64_9.dll") }
)
$licenseRoot = Join-Path $target "licenses"
$licensesComplete = ($packages | Where-Object {
    @(Get-ChildItem -LiteralPath (Join-Path $licenseRoot $_.Name) -File -ErrorAction SilentlyContinue).Count -eq 0
}).Count -eq 0
$complete = ($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $target $_) -PathType Leaf) }).Count -eq 0 -and
    $licensesComplete
if ($ReuseExisting -and $complete) {
    Write-Host "[cuda] Reusing the complete runtime in $target."
    return
}

New-Item -ItemType Directory -Path $target -Force | Out-Null
New-Item -ItemType Directory -Path $licenseRoot -Force | Out-Null
$staging = Join-Path $target "_download"
if ((Test-Path -LiteralPath $staging) -and -not $ReuseExisting) {
    Remove-Item -LiteralPath $staging -Recurse -Force
}
New-Item -ItemType Directory -Path $staging -Force | Out-Null

$installed = @()
try {
    foreach ($package in $packages) {
        $packageLicenseRoot = Join-Path $licenseRoot $package.Name
        $packageComplete = @($package.Dlls | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $target $_) -PathType Leaf)
        }).Count -eq 0 -and
            @(Get-ChildItem -LiteralPath $packageLicenseRoot -File -ErrorAction SilentlyContinue).Count -gt 0
        if ($ReuseExisting -and $packageComplete) {
            $installed += [ordered]@{
                package = $package.Name
                label = $package.Label
                version = "existing"
                dll_count = @($package.Dlls).Count
                notice_count = @(Get-ChildItem -LiteralPath $packageLicenseRoot -File).Count
            }
            Write-Host "[cuda] Reusing $($package.Label) and its license files."
            continue
        }
        Write-Host "[cuda] Resolving $($package.Label)..."
        $metadata = Invoke-RestMethod -Uri "https://pypi.org/pypi/$($package.Name)/json" -UseBasicParsing
        $version = [string]$metadata.info.version
        $files = @($metadata.releases.$version)
        $wheelInfo = $files | Where-Object {
            $_.filename -like "*-win_amd64.whl"
        } | Select-Object -First 1
        if ($null -eq $wheelInfo) {
            throw "$($package.Name) $version has no Windows x64 wheel."
        }

        $wheel = Join-Path $staging ([string]$wheelInfo.filename)
        Write-Host "[cuda] Downloading $($package.Label) $version..."
        if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
            throw "curl.exe is required to download the CUDA runtime wheels."
        }
        & curl.exe --fail --location --silent --show-error --retry 3 --retry-delay 5 --continue-at - `
            --connect-timeout 30 --max-time 1800 --output $wheel ([string]$wheelInfo.url)
        if ($LASTEXITCODE -ne 0) {
            throw "Downloading $($package.Name) failed (curl exit code $LASTEXITCODE)."
        }
        if (-not (Test-Path -LiteralPath $wheel -PathType Leaf) -or (Get-Item -LiteralPath $wheel).Length -eq 0) {
            throw "Downloading $($package.Name) produced an empty wheel."
        }

        $archive = [IO.Compression.ZipFile]::OpenRead($wheel)
        $count = 0
        $licenseCount = 0
        New-Item -ItemType Directory -Path $packageLicenseRoot -Force | Out-Null
        try {
            foreach ($entry in $archive.Entries) {
                $name = $entry.FullName.Replace('\', '/')
                $isDll = $name -match '^nvidia/.+/bin/[^/]+\.dll$'
                $isNotice = $entry.Length -gt 0 -and $entry.Length -lt 1MB -and
                    $entry.Name -match '^(?i:license|notice|eula|third.party)'
                if (-not $isDll -and -not $isNotice) { continue }
                $destination = if ($isDll) {
                    Join-Path $target $entry.Name
                } else {
                    Join-Path $packageLicenseRoot $entry.Name
                }
                $input = $entry.Open()
                $output = [IO.File]::Open($destination, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
                try {
                    $input.CopyTo($output)
                } finally {
                    $output.Dispose()
                    $input.Dispose()
                }
                if ($isDll) { $count += 1 } else { $licenseCount += 1 }
            }
        } finally {
            $archive.Dispose()
        }
        if ($count -eq 0) {
            throw "$($package.Name) $version did not contain runtime DLLs."
        }
        $installed += [ordered]@{
            package = $package.Name
            label = $package.Label
            version = $version
            dll_count = $count
            notice_count = $licenseCount
        }
        Remove-Item -LiteralPath $wheel -Force
        Write-Host "[cuda] Extracted $count DLLs for $($package.Label)."
    }
} finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}

$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $target $_) -PathType Leaf) })
if ($missing.Count -gt 0) {
    throw "CUDA runtime is incomplete; missing: $($missing -join ', ')"
}

$manifest = [ordered]@{
    created_at = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
    source = "Official NVIDIA runtime wheels from PyPI"
    packages = $installed
}
[IO.File]::WriteAllText(
    (Join-Path $target "cuda-runtime.json"),
    ($manifest | ConvertTo-Json -Depth 5),
    (New-Object System.Text.UTF8Encoding($false))
)
$size = (Get-ChildItem -LiteralPath $target -File | Measure-Object Length -Sum).Sum
Write-Host ("[cuda] Runtime ready: {0} DLLs, {1:N1} MB in {2}" -f `
    (Get-ChildItem -LiteralPath $target -Filter "*.dll" -File).Count, ($size / 1MB), $target)
