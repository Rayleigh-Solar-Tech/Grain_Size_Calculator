@echo off
REM Build script for Grain Size Calculator
REM This script builds the application into a standalone executable

echo ========================================
echo Grain Size Calculator - Build Script
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python is available
echo.

REM Check if virtual environment exists
if not exist "grain_calculator_env" (
    echo ERROR: Virtual environment 'grain_calculator_env' not found
    echo Please run install.bat first to set up the environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call grain_calculator_env\Scripts\activate.bat

REM Check if PyInstaller is installed
python -c "import PyInstaller" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller>=5.0.0
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo PyInstaller is available
echo.

REM Clean previous build
echo Cleaning previous build...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "__pycache__" rmdir /s /q "__pycache__"

REM Find and remove all __pycache__ directories
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

echo.

REM Ensure required directories exist
if not exist "outputs" mkdir "outputs"
if not exist "logs" mkdir "logs"
if not exist "temp" mkdir "temp"
if not exist "configs" mkdir "configs"

REM Check if SAM model exists
if not exist "sam_l.pt" (
    if not exist "src\sam_l.pt" (
        echo WARNING: SAM model file (sam_l.pt) not found
        echo The application will try to download it on first run
        echo.
    )
)

REM Check if config file exists
if not exist "configs\default_config.json" (
    echo Creating default config file...
    python -c "from src.core.config import create_default_config_file; create_default_config_file()"
    if %ERRORLEVEL% neq 0 (
        echo WARNING: Could not create default config file
        echo.
    )
)

echo Building application...
echo This may take several minutes...
echo.

REM Run PyInstaller with our spec file
pyinstaller --clean --noconfirm grain_calculator.spec

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Build failed!
    echo Check the output above for error details.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.

REM Check if the executable was created
if exist "dist\GrainSizeCalculator\GrainSizeCalculator.exe" (
    echo Executable created at: dist\GrainSizeCalculator\GrainSizeCalculator.exe
    echo.
    
    REM Get the size of the distribution
    for /f %%i in ('dir "dist\GrainSizeCalculator" /s /-c ^| find "bytes"') do set size=%%i
    echo Distribution size: %size%
    echo.
    
    echo Testing the executable...
    cd dist\GrainSizeCalculator
    GrainSizeCalculator.exe --help >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo Executable test: PASSED
    ) else (
        echo Executable test: WARNING - May have issues
    )
    cd ..\..
    
    echo.
    echo To distribute the application:
    echo 1. Copy the entire 'dist\GrainSizeCalculator' folder
    echo 2. Users can run 'GrainSizeCalculator.exe' directly
    echo 3. No Python installation required on target machines
    echo.
    
    set /p run_now="Do you want to run the application now? (y/n): "
    if /i "%run_now%"=="y" (
        echo Starting application...
        start "" "dist\GrainSizeCalculator\GrainSizeCalculator.exe"
    )
    
) else (
    echo ERROR: Executable not found in expected location
    echo Build may have failed or files were placed elsewhere
    pause
    exit /b 1
)

echo.
echo Build process completed!
pause