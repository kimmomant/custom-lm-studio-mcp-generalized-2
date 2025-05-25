@echo off
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
python "test_mcp_stdio.py" 2>&1 