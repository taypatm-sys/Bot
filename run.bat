@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo The bot is not installed yet. Run setup.bat first.
  pause
  exit /b 1
)
if not exist ".env" (
  echo The .env file is missing. Run setup.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" bot.py
echo.
echo The bot has stopped. Read the message above to find the cause.
pause
