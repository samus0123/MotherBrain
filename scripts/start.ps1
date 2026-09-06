# Start MotherBrain on Windows. Run from the MotherBrain directory:
#
#     powershell -ExecutionPolicy Bypass -File scripts\start.ps1
#
# Safe to re-run: each step checks whether it is already done, and stops at the
# step that fails rather than letting a later one fail for a reason that looks
# unrelated.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
Write-Host "MotherBrain: $(Get-Location)"

# 1. The code. main carries everything now - the weights included - so a plain
#    clone is enough and there is no branch to switch to.
if (-not (Test-Path "motherbrain\cli.py")) {
    Write-Host "`nstep 1 failed: this directory does not hold MotherBrain."
    Write-Host "Clone it and run this from inside:"
    Write-Host "  git clone https://github.com/samus0123/MotherBrain"
    Write-Host "  cd MotherBrain"
    exit 1
}
Write-Host "  ok: code present"

# 2. Python and a place to install to.
$py = $null
foreach ($candidate in @("py -3", "python", "python3")) {
    $exe, $arg = $candidate.Split(" ", 2)
    if (Get-Command $exe -ErrorAction SilentlyContinue) { $py = $candidate; break }
}
if (-not $py) {
    Write-Host "`nstep 2 failed: no Python found."
    Write-Host "Install it from https://www.python.org/downloads/ and tick"
    Write-Host "'Add python.exe to PATH' during setup."
    exit 1
}
Write-Host "  ok: python ($py)"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "`nstep 2: creating the virtual environment"
    Invoke-Expression "$py -m venv .venv"
}

$pip = ".venv\Scripts\pip.exe"
$vpy = ".venv\Scripts\python.exe"
& $vpy -c "import torch, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nstep 2: installing (downloads PyTorch, several minutes)"
    & $pip install --quiet --upgrade pip
    & $pip install -e .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nstep 2 failed: the install did not finish."
        exit 1
    }
}
Write-Host "  ok: installed"

# 3. Something to run. A clone ships models\motherbrain-base.pt and the
#    patches; models\motherbrain.pt is a local artifact and is not committed,
#    so looking for that one failed on a perfectly good checkout.
if (-not (Test-Path "models\motherbrain-base.pt") -and
    -not (Test-Path "models\motherbrain.pt") -and
    -not (Test-Path "runs\default\checkpoint.pt")) {
    Write-Host "`nstep 3 failed: no model found — the checkout is incomplete."
    Write-Host "Try a fresh clone:"
    Write-Host "  git clone https://github.com/samus0123/MotherBrain"
    exit 1
}
Write-Host "  ok: model present`n"

if (Test-Path ".venv\Scripts\mb.exe") {
    & ".venv\Scripts\mb.exe" gui @args
} else {
    & $vpy -m motherbrain.cli gui @args
}
