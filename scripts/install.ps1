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

$py = $null
foreach ($candidate in @("py -3", "python", "python3")) {
    $exe = $candidate.Split(" ")[0]
    if (Get-Command $exe -ErrorAction SilentlyContinue) { $py = $candidate; break }
}
if (-not $py) {
    Write-Host "no Python found."
    Write-Host "Install it from https://www.python.org/downloads/ and tick"
    Write-Host "'Add python.exe to PATH' during setup, then run this again."
    exit 1
}

$ver = Invoke-Expression "$py -c ""import sys; print('%d.%d' % sys.version_info[:2])"""
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
