# Install MotherBrain on Windows 10 or 11.
#
#     powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#
# Windows needs less special-casing than Linux does: python.org's installer
# includes Tkinter, so the window works with nothing extra, and PyPI's torch
# wheel for Windows does not drag in the CUDA runtime the way the Linux one
# does. What does go wrong is Python not being on PATH, and the execution
# policy blocking the script before it starts - hence the line above.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
Write-Host "MotherBrain: $(Get-Location)`n"

# ---- python ----------------------------------------------------------------

# Existing is not the same as working, and on Windows 11 that distinction
# is the whole ballgame. A fresh Windows 11 ships a Microsoft Store
# placeholder at WindowsApps\python.exe: Get-Command finds it happily, and
# running it opens the Store rather than executing anything. `py -3` with no
# Python installed likewise exists and prints "Python 3 not found".
#
# This used to accept either on sight, and the version parse that followed
# then threw a raw PowerShell error partway through - which is what "it did
# not complete" looks like from the outside. So each candidate is now made
# to prove itself by printing its own version.
$py = $null
$ver = $null
foreach ($candidate in @("py -3", "python", "python3")) {
    $exe = $candidate.Split(" ")[0]
    $found = Get-Command $exe -ErrorAction SilentlyContinue
    if (-not $found) { continue }

    $probe = $null
    try {
        $probe = & cmd /c "$candidate -c ""import sys; print('%d.%d' % sys.version_info[:2])"" 2>nul"
    } catch { $probe = $null }

    if ($probe -and ($probe -match '^\s*(\d+)\.(\d+)\s*$')) {
        $py = $candidate
        $ver = $probe.Trim()
        break
    }

    if ($found.Source -like "*WindowsApps*") {
        Write-Host "  ignoring '$exe' - it is the Microsoft Store placeholder, not Python"
    } else {
        Write-Host "  ignoring '$exe' - present but it does not run"
    }
}

if (-not $py) {
    Write-Host "`nno working Python found."
    Write-Host ""
    Write-Host "Install it from https://www.python.org/downloads/ and tick"
    Write-Host "'Add python.exe to PATH' during setup, then run this again."
    Write-Host ""
    Write-Host "If typing 'python' opens the Microsoft Store, that is the"
    Write-Host "placeholder, not Python. Turn it off under Settings >"
    Write-Host "Apps > Advanced app settings > App execution aliases, or just"
    Write-Host "install real Python from the link above - either works."
    exit 1
}

Write-Host "python $ver ($py)"
$parts = $ver.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    Write-Host "MotherBrain needs Python 3.10 or newer; this is $ver."
    exit 1
}

# ---- room ------------------------------------------------------------------

$free = (Get-PSDrive -Name (Get-Location).Drive.Name).Free / 1GB
Write-Host ("free disk: {0:N1} GB" -f $free)
if ($free -lt 3) {
    Write-Host "`nnot enough disk: PyTorch needs roughly 3GB. Free some space first."
    exit 1
}

# ---- environment -----------------------------------------------------------

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "`ncreating the virtual environment"
    Invoke-Expression "$py -m venv .venv"
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        Write-Host "could not create a virtual environment."
        exit 1
    }
}
$vpy = ".venv\Scripts\python.exe"
$pip = ".venv\Scripts\pip.exe"

& $vpy -c "import torch" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "torch is already installed; leaving it alone"
} else {
    Write-Host "`ninstalling (PyTorch is a few hundred MB; this takes a while)"
    & $pip install --quiet --upgrade pip
}

& $pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nthe install did not finish. The pip error above is the reason."
    exit 1
}

# ---- check it ---------------------------------------------------------------

Write-Host ""
& $vpy -c @"
import importlib.util
missing = [m for m in ('torch', 'numpy', 'fastapi', 'uvicorn')
           if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('missing after install: ' + ', '.join(missing))
import torch
from motherbrain import __version__
print(f'MotherBrain {__version__} installed - torch {torch.__version__}')
try:
    import tkinter
    print('Tkinter is present, so the window will open.')
except ImportError:
    print('No Tkinter: mb gui will use your browser instead, which is fine.')
"@
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "`ndone. start it with:"
Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\gui.ps1"
Write-Host "or:"
Write-Host "    .venv\Scripts\mb.exe gui"
Write-Host "    .venv\Scripts\mb.exe console"
