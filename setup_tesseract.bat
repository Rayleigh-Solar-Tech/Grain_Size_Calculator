@echo off
echo Setting up portable Tesseract for bundling...

REM Check if we already have tesseract
if exist "tesseract\tesseract.exe" (
    echo Tesseract already exists!
    goto :test
)

echo.
echo Option 1: Download portable Tesseract automatically
echo Option 2: Manual setup instructions
echo.
choice /c 12 /m "Choose setup method"

if errorlevel 2 goto :manual
if errorlevel 1 goto :auto

:auto
echo.
echo Downloading portable Tesseract...
echo This will download from UB Mannheim (official Windows builds)
echo.

REM Create temp directory
if not exist temp mkdir temp

REM Download Tesseract installer (we'll extract manually)
echo Downloading Tesseract installer...
powershell -Command "Invoke-WebRequest -Uri 'https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe' -OutFile 'temp\tesseract-installer.exe'"

if not exist "temp\tesseract-installer.exe" (
    echo Download failed! Trying manual setup...
    goto :manual
)

echo.
echo Please run the installer in temp\tesseract-installer.exe
echo Install to: %CD%\tesseract
echo Then press any key to continue...
pause

goto :test

:manual
echo.
echo Manual setup instructions:
echo.
echo 1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
echo 2. Install or extract to: %CD%\tesseract
echo 3. Make sure tesseract.exe is at: %CD%\tesseract\tesseract.exe
echo.
echo Alternative - copy from existing installation:
echo If you have Tesseract installed, copy from:
echo "C:\Program Files\Tesseract-OCR\*" to "%CD%\tesseract\"
echo.
pause

:test
echo.
echo Testing Tesseract setup...
if exist "tesseract\tesseract.exe" (
    echo ✅ Found tesseract.exe
    tesseract\tesseract.exe --version
    echo.
    echo ✅ Tesseract is ready for bundling!
) else (
    echo ❌ tesseract.exe not found!
    echo Make sure it's at: %CD%\tesseract\tesseract.exe
)

echo.
pause