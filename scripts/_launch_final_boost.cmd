@echo off
rem 独立启动最终强化实验守护（l2p/coda 20epoch + CIFAR-100 补 seed），detached，会话结束仍存活。
cd /d C:\Users\34608\Desktop\peft-cl
echo ===== FINAL_BOOST SUPERVISOR START %date% %time% ===== >> reports\final_boost.log
C:\Users\34608\AppData\Local\Programs\Python\Python311\python.exe -u -m scripts.supervise_final_boost >> reports\final_boost.log 2>&1
echo ===== FINAL_BOOST SUPERVISOR EXIT %date% %time% ===== >> reports\final_boost.log
