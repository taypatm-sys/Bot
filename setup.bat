@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD=py -3"
where py >nul 2>nul
if errorlevel 1 (
  where python >nul 2>nul
  if errorlevel 1 goto :python_missing
  set "PYTHON_CMD=python"
)

echo Creating the Python environment...
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto :install_error

echo Installing dependencies...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :install_error
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :install_error

if not exist ".env" copy ".env.example" ".env" >nul
echo.
echo Setup completed successfully.
echo Open the .env file, enter your data, save it, and run run.bat.
start "" notepad ".env"
pause
exit /b 0

:python_missing
echo.
echo Python was not found.
echo Install Python 3.11 or newer from https://www.python.org/downloads/windows/
echo During installation, enable "Add python.exe to PATH".
pause
exit /b 1

:install_error
echo.
echo Setup failed. Check your internet connection and Python installation.
pause
exit /b 1
