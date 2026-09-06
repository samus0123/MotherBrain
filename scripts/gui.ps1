# Open MotherBrain's window on Windows 10 or 11.
#
#     powershell -ExecutionPolicy Bypass -File scripts\gui.ps1
#
# Checks the things that stop it, in the order they stop it, and says which
# one applies rather than leaving you to work it out.

Set-Location (Split-Path -Parent $PSScriptRoot)
$mb  = ".venv\Scripts\mb.exe"
$vpy = ".venv\Scripts\python.exe"

# ---- 1. installed at all? ---------------------------------------------------

if (-not (Test-Path $vpy)) {
    Write-Host "MotherBrain is not installed yet.`n"
    Write-Host "Install it first:"
    Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\install.ps1"
    exit 1
}

# ---- 2. does this checkout have the gui command? ----------------------------

$runner = if (Test-Path $mb) { $mb } else { $null }
$hasGui = $false
if ($runner) {
    & $runner gui --help *>$null
    $hasGui = ($LASTEXITCODE -eq 0)
} else {
    & $vpy -m motherbrain.cli gui --help *>$null
    $hasGui = ($LASTEXITCODE -eq 0)
}

if (-not $hasGui) {
    Write-Host "this copy of MotherBrain has no 'gui' command - it predates the window."
    Write-Host "updating ...`n"
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`ncould not update automatically. Do it by hand:"
        Write-Host "    git pull"
        Write-Host "    .venv\Scripts\pip.exe install -e ."
        exit 1
    }
    & ".venv\Scripts\pip.exe" install -q -e .
    Write-Host "updated.`n"
}

# ---- 3. is there a model? ---------------------------------------------------

if (-not (Test-Path "models\motherbrain-base.pt")) {
    Write-Host "warning: models\motherbrain-base.pt is missing, so there may be"
    Write-Host "         nothing to run. The window will say so. 'mb status'"
    Write-Host "         explains what is absent.`n"
}

# ---- 4. go ------------------------------------------------------------------
#
# No Tkinter check here on purpose: `mb gui` serves the same four options to
# your browser when it cannot open a window, so there is nothing to stop for.

Write-Host "starting MotherBrain ..."
if ($runner) { & $runner gui @args } else { & $vpy -m motherbrain.cli gui @args }
