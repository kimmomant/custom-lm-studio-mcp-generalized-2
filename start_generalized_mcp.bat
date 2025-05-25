@echo off
cd /d "%~dp0"
set LM_STUDIO_BASE_URL=http://localhost:1234
set PYTHONPATH=%~dp0

REM Try to find Python in common locations
set PYTHON_EXE=
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set PYTHON_EXE="%USERPROFILE%\anaconda3\python.exe"
) else if exist "%USERPROFILE%\miniconda3\python.exe" (
    set PYTHON_EXE="%USERPROFILE%\miniconda3\python.exe"
) else if exist "C:\Python3\python.exe" (
    set PYTHON_EXE="C:\Python3\python.exe"
) else (
    REM Fall back to system python
    set PYTHON_EXE=python
)

echo Using Python: %PYTHON_EXE%

REM Check if required packages are installed and install if missing
echo Checking dependencies...
%PYTHON_EXE% -c "import mcp, httpx, PIL" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    %PYTHON_EXE% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements. Please install manually:
        echo %PYTHON_EXE% -m pip install mcp httpx pillow
        pause
        exit /b 1
    )
)

echo Starting MCP server...
%PYTHON_EXE% "generalized_lm_studio_mcp_server.py" 