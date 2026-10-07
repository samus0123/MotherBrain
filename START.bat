@echo off
rem Start MotherBrain on Windows. Double-click this file, or run START from
rem a command prompt. The first run installs; after that it just starts.
rem
rem Anything you pass goes to `mb start`, so START --help lists it all.

setlocal
cd /d "%~dp0"

set "VENV=%~dp0.venv"
set "MB=%VENV%\Scripts\mb.exe"

rem --- make sure there is something to run ---------------------------------
rem The installing lives in scripts\install.ps1, which already handles
rem Python not being on PATH and the execution policy. -ExecutionPolicy
rem Bypass is how that script is meant to be called; it applies to this one
rem invocation only and changes nothing about the machine's policy.
if exist "%MB%" goto :run

rem Everything below runs only on a first install. It is deliberately NOT
rem inside a parenthesised block: cmd expands a whole block before running
rem any of it, so a variable set in one line cannot be read on the next -
rem which is how the PowerShell choice below used to come out empty.

rem A virtual environment cannot be moved - Python writes absolute paths
rem into it - so one that exists without a working python is cleared out
rem rather than reused. It cannot be repaired.
if exist "%VENV%" if not exist "%VENV%\Scripts\python.exe" (
  echo The environment in %VENV% is broken.
  echo ^(This is what happens when the folder is moved or copied.^)
  echo Rebuilding it.
  rmdir /s /q "%VENV%"
)

echo First run: installing. This happens once.
echo.
rem Windows PowerShell 5.1 ships as powershell.exe and is present on every
rem supported Windows. PowerShell 7 installs as pwsh.exe alongside it, and
rem some trimmed images have only that, so try both before giving up.
set "PS="
where powershell >nul 2>&1 && set "PS=powershell"
if not defined PS where pwsh >nul 2>&1 && set "PS=pwsh"
if not defined PS (
  echo Could not find PowerShell, which the installer needs.
  echo Install Python from https://www.python.org/downloads/ and then run:
  echo   .venv\Scripts\python.exe -m pip install -e .
  pause
  exit /b 1
)
%PS% -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\install.ps1"
if errorlevel 1 (
  echo.
  echo The install failed. The output above says why.
  pause
  exit /b 1
)
echo.

:run
if not exist "%MB%" (
  echo The install finished but %MB% is missing.
  pause
  exit /b 1
)

rem --- start it ------------------------------------------------------------
"%MB%" start %*

rem A double-clicked window vanishes on exit and takes the error with it.
if errorlevel 1 pause
endlocal
