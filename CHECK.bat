@echo off
rem CHECK.bat - gather facts about this machine and write them to a file.
rem
rem This exists because "it does not complete" cannot be acted on, and
rem START.bat's own diagnosis needs the install to get far enough to run.
rem This one needs nothing: no Python, no PowerShell, no virtual
rem environment. It cannot fail, it never exits early, and it always leaves
rem CHECK-log.txt behind.
rem
rem Double-click it, then send CHECK-log.txt.

set "LOG=%~dp0CHECK-log.txt"

echo MotherBrain CHECK> "%LOG%"
echo ================>> "%LOG%"
echo.>> "%LOG%"

echo --- where --->> "%LOG%"
echo folder:   %~dp0>> "%LOG%"
echo cwd:      %CD%>> "%LOG%"
echo user:     %USERNAME%>> "%LOG%"
echo computer: %COMPUTERNAME%>> "%LOG%"
echo arch:     %PROCESSOR_ARCHITECTURE%>> "%LOG%"
ver >> "%LOG%" 2>&1
echo.>> "%LOG%"

echo --- files that must be here --->> "%LOG%"
for %%F in (START.bat pyproject.toml motherbrain\cli.py scripts\install.ps1 scripts\doctor.ps1 models\motherbrain-base.pt runs\default\versions.json) do call :one "%%F"
echo.>> "%LOG%"

echo --- patches on disk --->> "%LOG%"
dir /b runs\default\patches\*.pt >> "%LOG%" 2>&1
echo.>> "%LOG%"

echo --- python candidates --->> "%LOG%"
where py >> "%LOG%" 2>&1
where python >> "%LOG%" 2>&1
where python3 >> "%LOG%" 2>&1
echo.>> "%LOG%"
echo py -3 version:>> "%LOG%"
py -3 -c "import sys; print(sys.version)" >> "%LOG%" 2>&1
echo exit code %ERRORLEVEL%>> "%LOG%"
echo.>> "%LOG%"
echo python version:>> "%LOG%"
python -c "import sys; print(sys.version)" >> "%LOG%" 2>&1
echo exit code %ERRORLEVEL%>> "%LOG%"
echo.>> "%LOG%"

echo --- powershell --->> "%LOG%"
where powershell >> "%LOG%" 2>&1
where pwsh >> "%LOG%" 2>&1
powershell -NoProfile -Command "$PSVersionTable.PSVersion.ToString()" >> "%LOG%" 2>&1
echo exit code %ERRORLEVEL%>> "%LOG%"
echo.>> "%LOG%"

echo --- the virtual environment --->> "%LOG%"
if exist "%~dp0.venv" echo .venv exists>> "%LOG%"
if not exist "%~dp0.venv" echo .venv does NOT exist - nothing is installed yet>> "%LOG%"
if exist "%~dp0.venv\Scripts\python.exe" echo .venv\Scripts\python.exe present>> "%LOG%"
if not exist "%~dp0.venv\Scripts\python.exe" echo .venv\Scripts\python.exe MISSING>> "%LOG%"
if exist "%~dp0.venv\Scripts\mb.exe" echo .venv\Scripts\mb.exe present>> "%LOG%"
if not exist "%~dp0.venv\Scripts\mb.exe" echo .venv\Scripts\mb.exe MISSING>> "%LOG%"
echo.>> "%LOG%"

echo --- can the package import? --->> "%LOG%"
if exist "%~dp0.venv\Scripts\python.exe" "%~dp0.venv\Scripts\python.exe" -c "import torch, motherbrain; print('torch', torch.__version__)" >> "%LOG%" 2>&1
if exist "%~dp0.venv\Scripts\python.exe" echo exit code %ERRORLEVEL%>> "%LOG%"
echo.>> "%LOG%"

echo --- free space --->> "%LOG%"
dir /-c "%~d0\" | find "bytes free" >> "%LOG%" 2>&1
echo.>> "%LOG%"

echo --- a previous START log, if any --->> "%LOG%"
if exist "%~dp0START-log.txt" type "%~dp0START-log.txt" >> "%LOG%" 2>&1
if not exist "%~dp0START-log.txt" echo no START-log.txt>> "%LOG%"

echo.
echo ==================================================================
echo Wrote %LOG%
echo.
echo Send that file. It says what is on this machine, which Python
echo actually runs, and how far the install got - which is everything
echo needed to fix this properly.
echo ==================================================================
echo.
type "%LOG%"
echo.
pause
goto :eof

:one
if exist "%~dp0%~1" echo   ok      %~1>> "%LOG%"
if not exist "%~dp0%~1" echo   MISSING %~1>> "%LOG%"
goto :eof
