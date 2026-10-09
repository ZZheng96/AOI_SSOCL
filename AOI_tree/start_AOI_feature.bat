@echo off
REM Launch AOI_feature (feature analysis / management GUI)
setlocal
set "PYTHON=E:\CPIPC\CGAIC\.env\Scripts\python.exe"
set "PROJ=E:\CPIPC\CGAIC\AOI_tree\AOI_feature"

if not exist "%PYTHON%" (
    echo [ERROR] venv python not found: %PYTHON%
    exit /b 1
)

cd /d "%PROJ%"
"%PYTHON%" app.py %*
endlocal
