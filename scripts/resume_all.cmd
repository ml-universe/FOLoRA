@echo off
rem ===================================================================
rem 断电 / 重启后恢复全部未完成实验（幂等，可反复双击）。
rem
rem 真正的实现在 start_supervisor.ps1 —— 这里只是一层双击入口。
rem 之所以不把逻辑写在本文件里：从 Git Bash 调 .cmd 会被 MSYS 的参数/路径
rem 转换搞坏（实测把 .cmd 正文当 shell 脚本执行了）。PowerShell 脚本用
rem -File 调用没有这个问题，两边只维护一份实现。
rem
rem 总守护本身幂等：已完成的 run 跳过、未完成的 --resume 从最后一个已完成
rem 任务继续，重复运行本脚本安全。单例锁会挡住第二个实例。
rem ===================================================================
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_supervisor.ps1"
set RC=%ERRORLEVEL%

echo.
if %RC% NEQ 0 (
  echo [ERROR] 启动失败（退出码 %RC%），请查看 reports\supervise_all.log
  pause
) else (
  echo 查看进度：  type reports\supervise_all.log
  timeout /t 5 >nul
)
exit /b %RC%
