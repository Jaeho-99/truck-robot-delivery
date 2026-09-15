@echo off
setlocal
cd /d "%~dp0"
if not defined PYTHON set "PYTHON=python"
if not defined DEVICE set "DEVICE=cpu"
if not defined WORKERS set "WORKERS=10"
if not defined RUN_LABEL for /f %%L in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_LABEL=%%L"
set "SIZES=%*"
if not defined SIZES set "SIZES=50 100"
"%PYTHON%" -c "import os,sys; sys.path.insert(0,'src'); from common.sizes import SUPPORTED_SIZES; from common.artifacts import validate_run_label; validate_run_label(os.environ['RUN_LABEL']); assert all(int(n) in SUPPORTED_SIZES for n in sys.argv[1:]); assert 1 <= int(os.environ['WORKERS']) <= 30; assert os.environ['DEVICE'] in ('cpu','cuda')" %SIZES%
if errorlevel 1 exit /b 1
set "PYTHONUNBUFFERED=1"
for %%N in (%SIZES%) do (
    call :size %%N
    if errorlevel 1 exit /b 1
)
exit /b 0

:size
call :run src/alns/solve.py --size %1 --workers %WORKERS% --run-label "%RUN_LABEL%"
if errorlevel 1 exit /b 1
for %%M in (ppo_alns gnn_ppo_alns) do for %%R in (alns_5310 new_best_5 magnitude) do (
    call :train_test %1 %%M %%R
    if errorlevel 1 exit /b 1
)
call :run scripts/summarize_results.py --size %1
exit /b %errorlevel%

:train_test
call :run src/%2/train.py --size %1 --reward-mode %3 --device %DEVICE% --env-backend process --observation-codec numpy --run-label "%RUN_LABEL%"
if errorlevel 1 exit /b 1
call :run src/%2/test.py --size %1 --reward-mode %3 --workers %WORKERS% --run-label "%RUN_LABEL%" --checkpoint "models/%2_n%1_reward_%3_run-%RUN_LABEL%.pt"
exit /b %errorlevel%

:run
echo [run] "%PYTHON%" %*
if "%DRY_RUN%"=="1" exit /b 0
"%PYTHON%" %*
exit /b %errorlevel%
