@echo off
rem 独立启动 FOLoRA 实验队列守护（detached，会话结束后仍存活）。
rem 日志追加到 reports\full_queue.log。
cd /d C:\Users\34608\Desktop\peft-cl
echo. >> reports\full_queue.log
echo ===== LAUNCHER START %date% %time% ===== >> reports\full_queue.log
C:\Users\34608\AppData\Local\Programs\Python\Python311\python.exe -u -m scripts.supervise_queue >> reports\full_queue.log 2>&1
echo ===== LAUNCHER EXIT  %date% %time% ===== >> reports\full_queue.log
