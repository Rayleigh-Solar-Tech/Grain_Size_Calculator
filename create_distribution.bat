@echo off
REM Distribution packaging script for Grain Size Calculator
REM Creates a complete distribution package ready for sharing

echo ========================================
echo Grain Size Calculator - Distribution Package
echo ========================================
echo.

REM Check if build exists
if not exist "dist\GrainSizeCalculator" (
    echo ERROR: Build not found. Please run build.bat first.
    pause
    exit /b 1
)

REM Create distribution directory
set DIST_NAME=GrainSizeCalculator_v1.0_Windows
set DIST_DIR=distribution\%DIST_NAME%

echo Creating distribution package: %DIST_NAME%
echo.

REM Clean previous distribution
if exist "distribution" rmdir /s /q "distribution"
mkdir "distribution"
mkdir "%DIST_DIR%"

REM Copy the built application
echo Copying application files...
xcopy "dist\GrainSizeCalculator\*" "%DIST_DIR%\" /E /I /H /Y
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to copy application files
    pause
    exit /b 1
)

REM Copy documentation
echo Copying documentation...
if exist "README.md" copy "README.md" "%DIST_DIR%\"
if exist "requirements.txt" copy "requirements.txt" "%DIST_DIR%\"

REM Create user guide
echo Creating user guide...
(
echo # Grain Size Calculator - User Guide
echo.
echo ## Quick Start
echo 1. Double-click GrainSizeCalculator.exe to start the application
echo 2. No Python installation required
echo 3. All dependencies are included
echo.
echo ## System Requirements
echo - Windows 10 or higher
echo - 4GB RAM minimum ^(8GB recommended^)
echo - 2GB free disk space
echo - CPU with AVX support for optimal performance
echo.
echo ## Features
echo - SEM image analysis for grain size calculation
echo - Automated pinhole detection using SAM model
echo - OCR for Frame Width extraction from image footers
echo - Batch processing capabilities
echo - Semi-automated workflow with user guidance
echo.
echo ## Usage
echo 1. Click "Select Images" to choose SEM images for analysis
echo 2. The application will automatically detect frame width from image footers
echo 3. If frame width cannot be detected, you will be prompted to enter it manually
echo 4. Pinhole detection runs automatically on validated SEM images
echo 5. Results are saved in the outputs directory
echo.
echo ## Troubleshooting
echo - If the application doesn't start, ensure you have Windows 10+ with latest updates
echo - For large batch processing, ensure sufficient RAM and disk space
echo - Check Windows Defender exclusions if antivirus blocks the executable
echo.
echo ## Support
echo - For issues, check the logs directory for error details
echo - Ensure input images are valid SEM images with proper footers
echo.
echo Generated on %DATE% at %TIME%
) > "%DIST_DIR%\USER_GUIDE.txt"

REM Create launcher script for convenience
echo Creating launcher script...
(
echo @echo off
echo REM Launcher for Grain Size Calculator
echo REM This script sets up the environment and starts the application
echo.
echo echo Starting Grain Size Calculator...
echo echo.
echo if not exist "outputs" mkdir "outputs"
echo if not exist "logs" mkdir "logs" 
echo if not exist "temp" mkdir "temp"
echo.
echo start "" "GrainSizeCalculator.exe"
echo.
echo REM Optional: Keep window open for debugging
echo REM pause
) > "%DIST_DIR%\Launch_GrainSizeCalculator.bat"

REM Create system info script
echo Creating system info script...
(
echo @echo off
echo REM System Information for Grain Size Calculator
echo echo System Information for Grain Size Calculator
echo echo =============================================
echo echo.
echo echo Windows Version:
echo ver
echo echo.
echo echo System Information:
echo systeminfo ^| findstr /C:"OS Name" /C:"OS Version" /C:"System Type" /C:"Total Physical Memory"
echo echo.
echo echo Available Disk Space:
echo dir c:\ ^| find "bytes free"
echo echo.
echo echo CPU Information:
echo wmic cpu get name,numberofcores,numberoflogicalprocessors /format:table
echo echo.
echo pause
) > "%DIST_DIR%\SystemInfo.bat"

REM Get distribution size
for /f "tokens=3" %%a in ('dir "%DIST_DIR%" /s ^| find "File(s)"') do set file_count=%%a
for /f "tokens=3" %%a in ('dir "%DIST_DIR%" /s ^| find "bytes"') do set total_size=%%a

echo.
echo Creating ZIP archive...
REM Use PowerShell to create ZIP if available
powershell -command "Compress-Archive -Path 'distribution\%DIST_NAME%' -DestinationPath 'distribution\%DIST_NAME%.zip' -Force" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ZIP archive created: distribution\%DIST_NAME%.zip
    for %%A in ("distribution\%DIST_NAME%.zip") do set zip_size=%%~zA
    echo ZIP size: %zip_size% bytes
) else (
    echo ZIP creation failed or PowerShell not available
    echo You can manually compress the folder: distribution\%DIST_NAME%
)

echo.
echo ========================================
echo Distribution package created successfully!
echo ========================================
echo.
echo Package location: distribution\%DIST_NAME%
echo Package contains:
echo - GrainSizeCalculator.exe ^(main application^)
echo - All required libraries and dependencies
echo - User guide and documentation
echo - Launcher scripts
echo - System information tool
echo.
echo Files: %file_count%
echo Total size: %total_size% bytes
echo.
echo To distribute:
echo 1. Share the entire folder: distribution\%DIST_NAME%
echo 2. Or share the ZIP file: distribution\%DIST_NAME%.zip
echo 3. Users can run Launch_GrainSizeCalculator.bat or GrainSizeCalculator.exe directly
echo.
echo No Python installation required on target machines!
echo.

set /p open_folder="Do you want to open the distribution folder? (y/n): "
if /i "%open_folder%"=="y" (
    explorer "distribution"
)

pause