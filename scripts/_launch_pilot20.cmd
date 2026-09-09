@echo off
rem 独立启动 P0 试点守护（20 epoch 验证，断电自动续训），detached，会话结束后仍存活。
cd /d C:\Users\34608\Desktop\peft-cl
echo ===== PILOT20 SUPERVISOR START %date% %time% ===== >> reports\pilot20.log
C:\Users\34608\AppData\Local\Programs\Python\Python311\python.exe -u -m scripts.supervise_pilot20 >> reports\pilot20.log 2>&1
echo ===== PILOT20 SUPERVISOR EXIT %date% %time% ===== >> reports\pilot20.log
