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
if not exist "%MB%" (
  rem A virtual environment cannot be moved - Python writes absolute paths
  rem into it - so one that exists without a working python is cleared out
  rem rather than reused. It cannot be repaired.
  if exist "%VENV%" (
    if not exist "%VENV%\Scripts\python.exe" (
      echo The environment in %VENV% is broken.
      echo ^(This is what happens when the folder is moved or copied.^)
      echo Rebuilding it.
      rmdir /s /q "%VENV%"
    )
  )

  echo First run: installing. This happens once.
  echo.
  where powershell >nul 2>&1
  if errorlevel 1 (
    echo Could not find PowerShell, which the installer needs.
    echo Install Python from https://www.python.org/downloads/ and then run:
    echo   .venv\Scripts\python.exe -m pip install -e .
    pause
    exit /b 1
  )
  powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\install.ps1"
  if errorlevel 1 (
    echo.
    echo The install failed. The output above says why.
    pause
    exit /b 1
  )
  echo.
)

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
