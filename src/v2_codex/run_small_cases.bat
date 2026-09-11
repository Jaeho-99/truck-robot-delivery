@echo off
setlocal
cd /d "%~dp0..\.."
if "%~1"=="" (
    echo Usage: src\v2_codex\run_small_cases.bat RUN_LABEL
    exit /b 2
)
for %%N in (5 10 20) do (
    python src/v2_codex/alns/solve.py --size %%N --iterations 100 --limit 3 --seeds 3 --workers 1 --run-label "%~1"
    if errorlevel 1 exit /b 1
)
endlocal
