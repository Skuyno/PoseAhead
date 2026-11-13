@echo off
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo Local Python environment is missing. Install the project first.
    pause
    exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py" gui --repository "%~dp0."
