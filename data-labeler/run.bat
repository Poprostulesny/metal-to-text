@echo off
setlocal
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo [labeler] Creating isolated venv...
  python -m venv "%VENV%" || goto :fail
  "%PY%" -m pip install --upgrade pip || goto :fail
  "%PY%" -m pip install -r requirements-labeler.txt || goto :fail
)

echo [labeler] Starting on http://127.0.0.1:8765
start "" http://127.0.0.1:8765
"%PY%" -m server
goto :eof

:fail
echo [labeler] Setup failed.
exit /b 1
