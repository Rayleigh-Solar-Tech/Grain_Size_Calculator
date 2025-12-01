echo Building with minimal dependencies...

REM Activate virtual environment
call .\grain_env\Scripts\activate.bat

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Simple build - exclude problematic modules
pyinstaller ^
    --onedir ^
    --windowed ^
    --name="GrainSizeCalculator" ^
    --add-data="src;src" ^
    --add-data="configs;configs" ^
    --exclude-module=matplotlib ^
    --exclude-module=pandas ^
    --exclude-module=scipy ^
    --exclude-module=ultralytics ^
    main.py

echo Build complete!
pause