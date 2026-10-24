@echo off
title D&D Bot
cd /d "%~dp0"

echo ============================================
echo   Starting the D^&D bot
echo   Close this window to stop it.
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: virtualenv not found at .venv\Scripts\python.exe
    echo Run this from the project folder, or recreate the venv:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo ERROR: no .env file found.
    echo Copy .env.example to .env and fill in your tokens.
    echo.
    pause
    exit /b 1
)

REM -u keeps output unbuffered so the console updates live. The bot writes
REM bot.log itself, so there's no redirection here -- you get both.
REM run_bot.py restarts the bot after a crash, but not after !shutdown.
".venv\Scripts\python.exe" -u run_bot.py

REM Only reached if the bot exits or crashes. Hold the window open so the
REM error is readable instead of vanishing with the console.
echo.
echo ============================================
echo   Bot stopped (exit code %errorlevel%).
echo   Scroll up for the reason, or check bot.log
echo ============================================
pause
