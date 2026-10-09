@echo off
REM 启动 AOI_Core（主干桌面端：内嵌 FastAPI 后端 + PySide6 UI）
setlocal
set "PYTHON=E:\CPIPC\CGAIC\.env\Scripts\python.exe"
set "PROJ=E:\CPIPC\CGAIC\AOI_tree\AOI_Core"

if not exist "%PYTHON%" (
    echo [ERROR] 未找到虚拟环境 Python: %PYTHON%
    exit /b 1
)

cd /d "%PROJ%"
"%PYTHON%" main.py %*
endlocal
