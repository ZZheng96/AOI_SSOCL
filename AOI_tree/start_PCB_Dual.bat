@echo off
REM Launch PCB_Dual (desktop: embedded service + PySide6 UI)
setlocal
set "PYTHON=E:\CPIPC\CGAIC\.env\Scripts\python.exe"
set "PROJ=E:\CPIPC\CGAIC\AOI_tree\PCB_Dual"

if not exist "%PYTHON%" (
    echo [ERROR] venv python not found: %PYTHON%
    exit /b 1
)

cd /d "%PROJ%"
"%PYTHON%" main.py %*
endlocal
