@echo off
REM 启动 PCB_Dual（双引擎桌面端：内嵌服务 + PySide6 UI）
setlocal
set "PYTHON=E:\CPIPC\CGAIC\.env\Scripts\python.exe"
set "PROJ=E:\CPIPC\CGAIC\AOI_tree\PCB_Dual"

if not exist "%PYTHON%" (
    echo [ERROR] 未找到虚拟环境 Python: %PYTHON%
    exit /b 1
)

cd /d "%PROJ%"
"%PYTHON%" main.py %*
endlocal
