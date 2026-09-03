# Report everything needed to diagnose a MotherBrain that will not start.
# Run it from the repository directory and paste the whole output:
#
#     powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1

Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "--- where ---"
Get-Location | ForEach-Object { $_.Path }

Write-Host "`n--- branch ---"
try { git rev-parse --abbrev-ref HEAD; git log --oneline -1 }
catch { Write-Host "not a git repository (or git not installed)" }

Write-Host "`n--- files that must exist ---"
foreach ($f in @("motherbrain\cli.py", "models\motherbrain.pt",
                 "scripts\start.ps1", "pyproject.toml")) {
    if (Test-Path $f) { Write-Host "  ok      $f" } else { Write-Host "  MISSING $f" }
}

Write-Host "`n--- python ---"
foreach ($c in @("py", "python", "python3")) {
    $exe = Get-Command $c -ErrorAction SilentlyContinue
    if ($exe) {
        $v = & $c -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        Write-Host "  $c -> $($exe.Source) $v"
    }
}

Write-Host "`n--- virtual environment ---"
if (Test-Path ".venv\Scripts\mb.exe") { Write-Host "  ok      .venv\Scripts\mb.exe" }
elseif (Test-Path ".venv") { Write-Host "  .venv exists but has no mb - install did not finish" }
else { Write-Host "  no .venv - run scripts\start.ps1" }

Write-Host "`n--- dependencies ---"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
foreach ($m in @("torch", "numpy", "fastapi")) {
    & $py -c "import $m" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "  ok      $m" } else { Write-Host "  MISSING $m" }
}

Write-Host "`n--- can it start? ---"
if (Test-Path ".venv\Scripts\mb.exe") { & ".venv\Scripts\mb.exe" status }
else { & $py -m motherbrain.cli status }
