# Run MotherBrain as a bulletin board on Windows 10 or 11.
#
#     powershell -ExecutionPolicy Bypass -File scripts\bbs.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\bbs.ps1 -Port 23
#
# Port 23 is telnet's own port and Windows will not let an ordinary process
# bind it, so this defaults to 2323 and says how to get 23 if you want it.
# Nothing here needs administrator by default.

param(
    [int]$Port = 2323,
    [string]$BindHost = "127.0.0.1",
    [string]$Password = "",
    [string]$SysopPassword = ""
)

Set-Location (Split-Path -Parent $PSScriptRoot)
$mb  = ".venv\Scripts\mb.exe"
$vpy = ".venv\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "MotherBrain is not installed yet.`n"
    Write-Host "Install it first:"
    Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\install.ps1"
    exit 1
}

$runner = if (Test-Path $mb) { $mb } else { $vpy }
$prefix = if (Test-Path $mb) { @() } else { @("-m", "motherbrain.cli") }

# ---- is the port free, and are we allowed it? -------------------------------

$admin = ([Security.Principal.WindowsPrincipal] `
          [Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($Port -lt 1024 -and -not $admin) {
    Write-Host "Port $Port needs an administrator prompt on Windows.`n"
    Write-Host "Either right-click PowerShell, choose 'Run as administrator',"
    Write-Host "and run this again - or use a high port, which needs nothing:"
    Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\bbs.ps1 -Port 2323"
    exit 1
}

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Something is already listening on port $Port."
    Write-Host "Pick another:  -Port 2424"
    exit 1
}

# ---- Windows has no telnet client installed by default ----------------------

$telnet = Get-Command telnet.exe -ErrorAction SilentlyContinue
if (-not $telnet) {
    Write-Host "Note: Windows ships without a telnet client. To call the board"
    Write-Host "from this machine, turn one on (administrator, once):"
    Write-Host "    dism /online /Enable-Feature /FeatureName:TelnetClient"
    Write-Host "or use PuTTY, or SyncTERM, which draws the ANSI properly.`n"
}

$args = @("bbs", "--host", $BindHost, "--port", "$Port")
if ($Password)      { $args += @("--password", $Password) }
if ($SysopPassword) { $args += @("--sysop-password", $SysopPassword) }

Write-Host "Starting the board. Call it with:"
Write-Host "    telnet $BindHost $Port`n"

& $runner @prefix @args
exit $LASTEXITCODE
