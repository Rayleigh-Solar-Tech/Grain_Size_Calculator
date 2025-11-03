@echo off
echo Starting Grain Size Calculator...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "grain_calculator_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call grain_calculator_env\Scripts\activate.bat
) else (
    echo Virtual environment not found. Running with system Python...
)

REM Check if main.py exists
if not exist "main.py" (
    echo Error: main.py not found
    echo Please make sure you're running this from the correct directory
    pause
    exit /b 1
)

REM Run the application
echo Starting Grain Size Calculator GUI...
python main.py

if errorlevel 1 (
    echo.
    echo Application exited with error. Check the error messages above.
    pause
)