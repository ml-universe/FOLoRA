# Restart the master supervisor after a power loss / reboot (idempotent).
#
# Why PowerShell instead of .cmd: the project path contains non-ASCII characters,
# and invoking a .cmd from Git Bash gets mangled by MSYS argument/path conversion
# (observed: the .cmd body was executed as a shell script). Start-Process takes an
# explicit interpreter and working directory and bypasses shell parsing entirely.
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads .ps1 as ANSI when
# there is no BOM, so non-ASCII comments get mis-decoded into parse errors.
#
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_supervisor.ps1
#    or:  double-click scripts\resume_all.cmd  (no MSYS escaping involved there)

$ErrorActionPreference = 'Stop'

$Proj = Split-Path -Parent $PSScriptRoot
$Log  = Join-Path $Proj 'reports\supervise_all.log'

# Resolve the interpreter from PATH rather than hardcoding one machine's install
# path: the hardcoded path both leaked the author's user directory into the public
# repository and made the script unusable on any other machine.
$Py = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if (-not $Py) { $Py = (Get-Command py.exe -ErrorAction SilentlyContinue).Source }
if (-not $Py) { Write-Error 'python not found on PATH (tried python.exe, py.exe)'; exit 1 }

if (-not (Test-Path $Py))   { Write-Error "python not found: $Py"; exit 1 }
if (-not (Test-Path $Proj)) { Write-Error "project dir not found: $Proj"; exit 1 }

$reportsDir = Join-Path $Proj 'reports'
if (-not (Test-Path $reportsDir)) { New-Item -ItemType Directory -Path $reportsDir | Out-Null }

function Get-SupervisorCount {
    # Wrap the whole pipeline in @() so .Count is valid even for 0 or 1 results.
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
    $hit = $procs | Where-Object { $_.CommandLine -like '*supervise_all*' }
    return @($hit).Count
}

if ((Get-SupervisorCount) -gt 0) {
    Write-Host "A supervisor is already running; not starting a second one. Log: $Log"
    exit 0
}

Add-Content -Path $Log -Value "`n===== LAUNCHER START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="

# cmd carries the redirection: we need append (>>), which Start-Process's
# -RedirectStandardOutput cannot do (it truncates).
$inner = "set PYTHONIOENCODING=utf-8&& `"$Py`" -u -m scripts.supervise_all >> `"$Log`" 2>&1"
Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $inner -WorkingDirectory $Proj -WindowStyle Hidden

Start-Sleep -Seconds 5
if ((Get-SupervisorCount) -gt 0) {
    Write-Host "Supervisor started. Log: $Log"
} else {
    Write-Host "!! No supervisor process detected 5s after launch. Check the log: $Log"
    exit 1
}
