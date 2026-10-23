@echo off
rem Detached launcher for the FOLoRA experiment queue (survives session exit).
rem The log is appended to reports\full_queue.log.
rem
rem Paths are resolved relative to this script and the interpreter from PATH, so
rem the repository works on any machine. The previous version hardcoded both the
rem author's project directory (which no longer exists) and the absolute path of
rem one Python install.
cd /d "%~dp0.."
set "PY="
for %%P in (python.exe) do set "PY=%%~$PATH:P"
if not defined PY (
  echo python.exe not found on PATH >> reports\full_queue.log
  exit /b 1
)
echo. >> reports\full_queue.log
echo ===== LAUNCHER START %date% %time% ===== >> reports\full_queue.log
"%PY%" -u -m scripts.supervise_queue >> reports\full_queue.log 2>&1
echo ===== LAUNCHER EXIT  %date% %time% ===== >> reports\full_queue.log
