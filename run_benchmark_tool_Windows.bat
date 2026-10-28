@echo off
rem ──────────────────────────────────────────────
rem  LLM Benchmark Tool - Windows launcher
rem ──────────────────────────────────────────────
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON=%PYTHON%"

if "%PYTHON%"=="" set "PYTHON=python"

where %PYTHON% >nul 2>nul
if errorlevel 1 (
    echo Error: '%PYTHON%' not found. Install Python 3.8+ first.
    pause
    exit /b 1
)

%PYTHON% "%SCRIPT_DIR%llm_benchmark.py" %*
endlocal
