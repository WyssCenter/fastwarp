# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0
#
# Installs the fastwarp command line from a GitHub release (Windows). See -Help.

param(
    [string]$Version,
    [Alias('InstallDir')][string]$Dir,
    [switch]$NoModifyPath,
    [switch]$Uninstall,
    [switch]$Help
)

$FastwarpInstallerVersion = '@FASTWARP_VERSION@'
$FastwarpInstallerSha256 = '@FASTWARP_SHA256_x86_64-pc-windows-msvc@'

function Invoke-FastwarpInstaller {
    param(
        [string]$Version,
        [string]$Dir,
        [switch]$NoModifyPath,
        [switch]$Uninstall,
        [switch]$Help,
        [string]$InstallerVersion,
        [string]$InstallerSha256
    )

    Set-StrictMode -Version 3.0
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'

    $repoUrl = 'https://github.com/WyssCenter/fastwarp'
    $target = 'x86_64-pc-windows-msvc'
    $defaultKey = 'HKCU:\Environment'

    if ($Help) {
        Write-Output @"
fastwarp installer $InstallerVersion

Downloads the fastwarp command line from $repoUrl/releases, checks its
SHA-256 and installs it without administrator rights. The installer of a
release holds the SHA-256 of that release's archive and checks the download
against it; for another version it uses the release's .sha256 file.

Usage: fastwarp-installer.ps1 [options]

Options:
  -Version VERSION  release to install, for example 0.2.0
                    (default: $InstallerVersion, the release this script belongs to)
  -Dir DIR          install into DIR (default: %LOCALAPPDATA%\Programs\fastwarp\bin)
  -NoModifyPath     do not add DIR to the user PATH
  -Uninstall        remove fastwarp from DIR, and DIR from the user PATH
  -Help             show this help

Environment:
  FASTWARP_VERSION         same as -Version
  FASTWARP_INSTALL_DIR     same as -Dir
  FASTWARP_NO_MODIFY_PATH  set to 1 for -NoModifyPath
  FASTWARP_DOWNLOAD_URL    download from this folder instead of the GitHub
                           release (https:// or file://; http:// only with the
                           checksum held in the installer, or from localhost)

Unless DIR is on PATH already or -NoModifyPath is given, the installer adds
DIR to the user PATH (HKCU\Environment, not the machine PATH); -Uninstall
removes it again once the folder is empty.
"@
        return
    }

    function Get-Text([string]$name) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($null -eq $value) { return '' }
        return $value
    }

    function Resolve-InstallDir([string]$dir) {
        if (-not $dir) { $dir = Get-Text 'FASTWARP_INSTALL_DIR' }
        if (-not $dir) {
            if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA is not set; give the folder with -Dir' }
            $dir = Join-Path $env:LOCALAPPDATA 'Programs\fastwarp\bin'
        }
        if (-not [IO.Path]::IsPathRooted($dir)) {
            $dir = Join-Path (Get-Location).ProviderPath $dir
        }
        $full = [IO.Path]::GetFullPath($dir)
        if ($full.Length -gt 3) { $full = $full.TrimEnd('\', '/') }
        if ($full.Contains(';')) { throw "the install folder must not contain ';': $full" }
        return $full
    }

    function Get-PathKey {
        $key = Get-Text 'FASTWARP_INSTALLER_PATH_KEY'
        if (-not $key) { return $defaultKey }
        if ($key -notmatch '^HKCU:\\Software\\fastwarp-installer-test\\') {
            throw 'FASTWARP_INSTALLER_PATH_KEY must name a key under HKCU:\Software\fastwarp-installer-test\ (tests only)'
        }
        return $key
    }

    function Get-UserPath([string]$key) {
        if (-not (Test-Path -LiteralPath $key)) { return '' }
        $item = Get-Item -LiteralPath $key
        return [string]$item.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    }

    function Write-UserPath([string]$key, [string]$value) {
        if (-not (Test-Path -LiteralPath $key)) { New-Item -Path $key -Force | Out-Null }
        if ($value) {
            New-ItemProperty -LiteralPath $key -Name 'Path' -Value $value -PropertyType ExpandString -Force | Out-Null
        } else {
            Remove-ItemProperty -LiteralPath $key -Name 'Path' -ErrorAction SilentlyContinue
        }
        if ($key -eq $defaultKey) {
            [Environment]::SetEnvironmentVariable('FASTWARP_INSTALLER_REFRESH', '1', 'User')
            [Environment]::SetEnvironmentVariable('FASTWARP_INSTALLER_REFRESH', $null, 'User')
        }
    }

    function Test-SamePath([string]$entry, [string]$dir) {
        if (-not $entry) { return $false }
        $expanded = [Environment]::ExpandEnvironmentVariables($entry)
        if ($expanded.Length -gt 3) { $expanded = $expanded.TrimEnd('\', '/') }
        return [string]::Equals($expanded, $dir, [StringComparison]::OrdinalIgnoreCase)
    }

    function Split-PathList([string]$value) {
        return @($value -split ';' | Where-Object { $_ })
    }

    function Test-InUserPath([string]$dir) {
        return @(Split-PathList (Get-UserPath (Get-PathKey)) | Where-Object { Test-SamePath $_ $dir }).Count -gt 0
    }

    function Test-InProcessPath([string]$dir) {
        return @(Split-PathList $env:Path | Where-Object { Test-SamePath $_ $dir }).Count -gt 0
    }

    function Add-ToUserPath([string]$dir) {
        $key = Get-PathKey
        if ((Test-InUserPath $dir) -or (($key -eq $defaultKey) -and (Test-InProcessPath $dir))) {
            Write-Output "$dir is already on PATH; the user PATH was not changed."
            return
        }
        $entries = @($dir) + @(Split-PathList (Get-UserPath $key))
        Write-UserPath $key ($entries -join ';')
        if ($key -eq $defaultKey) { $env:Path = "$dir;$env:Path" }
        Write-Output "Added $dir to the start of the user PATH ($key, value Path)."
        Write-Output 'Open a new terminal to use fastwarp.'
        Write-Output 'To undo: run the installer with -Uninstall, or remove the entry under Settings > System > About > Advanced system settings > Environment Variables.'
    }

    function Undo-UserPathEntry([string]$dir) {
        $key = Get-PathKey
        $entries = @(Split-PathList (Get-UserPath $key))
        $kept = @($entries | Where-Object { -not (Test-SamePath $_ $dir) })
        if ($kept.Count -eq $entries.Count) { return }
        Write-UserPath $key ($kept -join ';')
        Write-Output "Removed $dir from the user PATH ($key)."
    }

    function Get-Download([string]$url, [string]$out) {
        if ($url -match '^file://') {
            Copy-Item -LiteralPath ([Uri]$url).LocalPath -Destination $out
        } elseif ($url -match '^https?://') {
            Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
        } else {
            throw "unsupported download URL: $url"
        }
    }

    $dir = Resolve-InstallDir $Dir
    $bin = Join-Path $dir 'fastwarp.exe'
    $staged = Join-Path $dir 'fastwarp-new.exe'

    if ($Uninstall) {
        foreach ($f in @($bin, $staged)) {
            if (Test-Path -LiteralPath $f) {
                Remove-Item -LiteralPath $f -Force
                Write-Output "Removed $f"
            }
        }
        if ((Test-Path -LiteralPath $dir) -and -not (Get-ChildItem -LiteralPath $dir -Force | Select-Object -First 1)) {
            Remove-Item -LiteralPath $dir -Force
            $parent = Split-Path -Parent $dir
            if ((Split-Path -Leaf $dir) -eq 'bin' -and (Split-Path -Leaf $parent) -eq 'fastwarp' -and
                -not (Get-ChildItem -LiteralPath $parent -Force | Select-Object -First 1)) {
                Remove-Item -LiteralPath $parent -Force
            }
        }
        if (-not (Test-Path -LiteralPath $dir)) {
            Undo-UserPathEntry $dir
        } elseif (Test-InUserPath $dir) {
            Write-Output "Kept $dir on the user PATH, because the folder holds other programs."
        }
        return
    }

    if ($PSVersionTable.PSEdition -eq 'Core' -and -not $IsWindows) {
        throw 'this installer is for Windows; on Linux and macOS use fastwarp-installer.sh'
    }
    $arch = $env:PROCESSOR_ARCHITEW6432
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    switch ($arch) {
        'AMD64' { }
        'ARM64' { throw 'Windows on ARM64 is not supported; the release has an x86_64 build only' }
        default { throw "unsupported CPU architecture: $arch (supported: AMD64)" }
    }

    $ver = $Version
    if (-not $ver) { $ver = Get-Text 'FASTWARP_VERSION' }
    if (-not $ver) { $ver = $InstallerVersion }
    if ($ver.StartsWith('v')) { $ver = $ver.Substring(1) }
    if ($ver.Contains('@')) {
        throw 'no version given, and this copy of the installer is not from a release; pass -Version or set FASTWARP_VERSION'
    }
    if ($ver -notmatch '\A[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.+-]+)?\z') { throw "invalid version: $ver" }
    $pinned = ''
    if ($ver -eq $InstallerVersion -and -not $InstallerSha256.Contains('@')) { $pinned = $InstallerSha256.ToLowerInvariant() }

    $noModify = $NoModifyPath -or ((Get-Text 'FASTWARP_NO_MODIFY_PATH') -notin @('', '0'))

    $archive = "fastwarp-$target.zip"
    $base = Get-Text 'FASTWARP_DOWNLOAD_URL'
    if (-not $base) { $base = "$repoUrl/releases/download/v$ver" }
    $base = $base.TrimEnd('/')
    if ($base -match '^http://' -and $base -notmatch '^http://(127\.0\.0\.1|localhost)(:[0-9]+)?(/|$)') {
        if (-not $pinned) { throw "refusing to download over plain HTTP without a checksum held in the installer: $base" }
        Write-Warning "downloading over plain HTTP from $base"
    }

    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ('fastwarp-install-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        Write-Output "Downloading fastwarp $ver for $target from $base/$archive"
        $zip = Join-Path $tmp $archive
        $sumFile = "$zip.sha256"
        try { Get-Download "$base/$archive" $zip } catch { throw "download failed: $base/$archive ($($_.Exception.Message))" }
        if ($pinned) {
            $expected = $pinned
            $sumSource = 'the checksum held in this installer'
        } else {
            try { Get-Download "$base/$archive.sha256" $sumFile } catch { throw "download failed: $base/$archive.sha256 ($($_.Exception.Message))" }
            $expected = ([IO.File]::ReadAllText($sumFile).Trim() -split '\s+')[0].ToLowerInvariant()
            $sumSource = "$archive.sha256"
        }
        if ($expected -notmatch '\A[0-9a-f]{64}\z') { throw "$sumSource is not a SHA-256 checksum" }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "SHA-256 checksum mismatch for $archive (expected $expected from $sumSource, got $actual); nothing was installed"
        }
        Write-Output "Checksum OK ($sumSource): $actual"

        $x = Join-Path $tmp 'x'
        Expand-Archive -LiteralPath $zip -DestinationPath $x
        $src = Join-Path $x "fastwarp-$target\fastwarp.exe"
        if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { throw "$archive does not contain fastwarp-$target\fastwarp.exe" }

        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Copy-Item -LiteralPath $src -Destination $staged -Force
        $out = $null
        $code = 1
        try {
            $out = & $staged --version
            $code = $LASTEXITCODE
        } catch {
            $out = $null
        }
        if ($code -ne 0 -or -not $out) {
            Remove-Item -LiteralPath $staged -Force -ErrorAction SilentlyContinue
            throw 'the downloaded fastwarp.exe does not run on this system'
        }
        try {
            Move-Item -LiteralPath $staged -Destination $bin -Force
        } catch {
            Remove-Item -LiteralPath $staged -Force -ErrorAction SilentlyContinue
            throw "could not replace $bin; close any running fastwarp and try again"
        }
        Write-Output "Installed $(@($out)[0]) to $bin"
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ($noModify) {
        if (-not (Test-InUserPath $dir) -and -not (Test-InProcessPath $dir)) {
            Write-Output "$dir is not on PATH, and the user PATH was left unchanged (-NoModifyPath)."
            Write-Output "To use fastwarp, add $dir to Path under Settings > System > About > Advanced system settings > Environment Variables > User variables,"
            Write-Output 'or run the installer again without -NoModifyPath.'
        }
    } else {
        Add-ToUserPath $dir
    }
}

Invoke-FastwarpInstaller -Version $Version -Dir $Dir -NoModifyPath:$NoModifyPath -Uninstall:$Uninstall -Help:$Help -InstallerVersion $FastwarpInstallerVersion -InstallerSha256 $FastwarpInstallerSha256
