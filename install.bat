@echo off
echo ============================================================
echo Grain Size Calculator - Windows Installation Script
echo ============================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found. Checking version...
python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"
if errorlevel 1 (
    echo Error: Python 3.8 or higher is required
    python --version
    pause
    exit /b 1
)

echo ✓ Python version is compatible

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv grain_calculator_env
if errorlevel 1 (
    echo Error creating virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment created

REM Activate virtual environment
echo.
echo Activating virtual environment...
call grain_calculator_env\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing requirements
    pause
    exit /b 1
)

echo ✓ Requirements installed

REM Run setup script
echo.
echo Running setup script...
python setup.py
if errorlevel 1 (
    echo Warning: Setup script encountered issues
)

echo.
echo ============================================================
echo Installation completed!
echo ============================================================
echo.
echo To run the application:
echo   1. Double-click "run_gui.bat" 
echo   2. Or run: python main.py
echo.
echo For help: python main.py --help
echo.
pause