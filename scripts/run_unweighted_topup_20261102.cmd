@echo off
REM 2026-11-02 等权消融补种子（Neurocomputing 大修，R1-1/R7-5/R8-2）。
REM 13 个新 run：v2f_eq_l3_k64 seeds 5-9 以外全部；其余 λ 补到与加权臂同 n。
REM 幂等：is_finished 会跳过已完成的 12 个 run，中断后直接重跑本文件即可续跑。
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0.."
"C:\Users\34608\AppData\Local\Programs\Python\Python311\python.exe" -u -m scripts.run_lambda_grid --batch unweighted >> "logs\unweighted_topup_20261102.log" 2>&1
