@echo off
echo Building Grain Size Calculator in Clean Virtual Environment...

REM Activate virtual environment
call .\grain_env\Scripts\activate.bat

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build with PyInstaller - simple and clean
echo Building executable...
pyinstaller --clean --noconfirm ^
    --onedir ^
    --windowed ^
    --name="GrainSizeCalculator" ^
    --add-data="src;src" ^
    --add-data="configs;configs" ^
    --add-data="sam_l.pt;." ^
    --add-data="tesseract;tesseract" ^
    main.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Build completed successfully!
    echo Executable location: dist\GrainSizeCalculator\GrainSizeCalculator.exe
    echo.
    dir dist\GrainSizeCalculator\GrainSizeCalculator.exe
    echo.
    echo 🚀 Testing the executable...
    .\dist\GrainSizeCalculator\GrainSizeCalculator.exe
) else (
    echo.
    echo ❌ Build failed!
    echo Check the output above for errors.
)

pause