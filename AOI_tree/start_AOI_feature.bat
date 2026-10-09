@echo off
REM 启动 AOI_feature（特征分析与管理 GUI）
setlocal
set "PYTHON=E:\CPIPC\CGAIC\.env\Scripts\python.exe"
set "PROJ=E:\CPIPC\CGAIC\AOI_tree\AOI_feature"

if not exist "%PYTHON%" (
    echo [ERROR] 未找到虚拟环境 Python: %PYTHON%
    exit /b 1
)

cd /d "%PROJ%"
"%PYTHON%" app.py %*
endlocal
