# 显著性检验（paired t-test）

对照 = 论文 Table 1 的主表；`*` 表示 p<0.05。FGT 越小越好（diff 为负表示本文方法遗忘更低），avgACC 越大越好。

```

=== cifar100：FOLoRA vs 基线（配对 t 检验，* = p<0.05）===
baseline              proto           n metric               ours     base     diff        t         p
EWC-LoRA (lam=100)    20t/5e/r16     10 avgACC(%)           77.87    77.22    +0.65   +2.026   0.0734 
EWC-LoRA (lam=100)    20t/5e/r16     10 FGT(%)               7.42     7.73    -0.30   -2.049   0.0707 
ewc/ewc_lam1          20t/5e/r16      5 avgACC(%)           78.32    69.47    +8.85   +9.681   0.0006*
ewc/ewc_lam1          20t/5e/r16      5 FGT(%)               7.07     9.63    -2.56   -8.662   0.0010*
ewc/ewc_lam10         20t/5e/r16      5 avgACC(%)           78.32    74.09    +4.23   +5.601   0.0050*
ewc/ewc_lam10         20t/5e/r16      5 FGT(%)               7.07     8.59    -1.52   -4.390   0.0118*
ewc/ewc_lam100        20t/5e/r16      1 avgACC(%)           77.83    75.97    +1.86      n/a       n/a 
ewc/ewc_lam100        20t/5e/r16      1 FGT(%)               7.31     8.07    -0.76      n/a       n/a 
EWC-LoRA (lam=1000)   20t/5e/r16     10 avgACC(%)           77.87    77.65    +0.21   +0.709   0.4961 
EWC-LoRA (lam=1000)   20t/5e/r16     10 FGT(%)               7.42     7.10    +0.32   +2.399   0.0400*
ewc/ewc_lam3          20t/5e/r16      5 avgACC(%)           78.32    71.87    +6.45   +6.960   0.0022*
ewc/ewc_lam3          20t/5e/r16      5 FGT(%)               7.07     9.09    -2.02   -5.724   0.0046*
ewc/ewc_lam30         20t/5e/r16      5 avgACC(%)           78.32    75.67    +2.64   +4.910   0.0080*
ewc/ewc_lam30         20t/5e/r16      5 FGT(%)               7.07     8.17    -1.10   -4.170   0.0140*
EWC-LoRA (lam=300)    20t/5e/r16     10 avgACC(%)           77.87    78.00    -0.14   -0.478   0.6440 
EWC-LoRA (lam=300)    20t/5e/r16     10 FGT(%)               7.42     7.35    +0.07   +0.592   0.5681 
EWC-LoRA (lam=3000)   20t/5e/r16     10 avgACC(%)           77.87    77.40    +0.47   +1.228   0.2507 
EWC-LoRA (lam=3000)   20t/5e/r16     10 FGT(%)               7.42     7.03    +0.39   +2.396   0.0401*
folora/lam100         20t/5e/r16      1 avgACC(%)           77.83    70.36    +7.47      n/a       n/a 
folora/lam100         20t/5e/r16      1 FGT(%)               7.31     9.24    -1.93      n/a       n/a 
folora/lam1000        20t/5e/r16      1 avgACC(%)           77.83    75.68    +2.15      n/a       n/a 
folora/lam1000        20t/5e/r16      1 FGT(%)               7.31     7.67    -0.36      n/a       n/a 
folora/lam30          20t/5e/r16      1 avgACC(%)           77.83    70.23    +7.60      n/a       n/a 
folora/lam30          20t/5e/r16      1 FGT(%)               7.31     9.70    -2.39      n/a       n/a 
folora/lam300         20t/5e/r16      1 avgACC(%)           77.83    70.80    +7.03      n/a       n/a 
folora/lam300         20t/5e/r16      1 FGT(%)               7.31     9.76    -2.45      n/a       n/a 
folora_v2/probe_l10_k1620t/5e/r16      1 avgACC(%)           77.83    76.97    +0.86      n/a       n/a 
folora_v2/probe_l10_k1620t/5e/r16      1 FGT(%)               7.31     6.99    +0.32      n/a       n/a 
folora_v2/v2f_eq_l10_k6420t/5e/r16      3 avgACC(%)           78.37    77.91    +0.46   +0.570   0.6262 
folora_v2/v2f_eq_l10_k6420t/5e/r16      3 FGT(%)               7.05     7.29    -0.24   -0.439   0.7036 
folora_v2/v2f_eq_l1_k6420t/5e/r16      3 avgACC(%)           78.37    72.65    +5.72   +4.394   0.0481*
folora_v2/v2f_eq_l1_k6420t/5e/r16      3 FGT(%)               7.05     8.80    -1.75   -5.032   0.0373*
folora_v2/v2f_eq_l30_k6420t/5e/r16      3 avgACC(%)           78.37    77.81    +0.56   +0.811   0.5026 
folora_v2/v2f_eq_l30_k6420t/5e/r16      3 FGT(%)               7.05     6.91    +0.14   +0.577   0.6222 
folora_v2/v2f_eq_l3_k6420t/5e/r16      3 avgACC(%)           78.37    74.79    +3.58   +7.143   0.0190*
folora_v2/v2f_eq_l3_k6420t/5e/r16      3 FGT(%)               7.05     8.27    -1.22   -4.351   0.0490*
folora_v2/v2f_l0_k64  20t/5e/r16      5 avgACC(%)           78.32    69.59    +8.72  +10.459   0.0005*
folora_v2/v2f_l0_k64  20t/5e/r16      5 FGT(%)               7.07     9.53    -2.46  -10.893   0.0004*
folora_v2/v2f_l1000_k1620t/5e/r16      3 avgACC(%)           78.37    73.85    +4.52   +4.043   0.0561 
folora_v2/v2f_l1000_k1620t/5e/r16      3 FGT(%)               7.05     7.31    -0.26   -2.597   0.1217 
folora_v2/v2f_l100_k1620t/5e/r16      3 avgACC(%)           78.37    75.35    +3.02   +2.293   0.1489 
folora_v2/v2f_l100_k1620t/5e/r16      3 FGT(%)               7.05     7.13    -0.08   -1.437   0.2873 
folora_v2/v2f_l10_k12820t/5e/r16      5 avgACC(%)           78.32    77.30    +1.02   +2.845   0.0466*
folora_v2/v2f_l10_k12820t/5e/r16      5 FGT(%)               7.07     6.94    +0.13   +1.334   0.2529 
folora_v2/v2f_l10_k16 20t/5e/r16      5 avgACC(%)           78.32    77.90    +0.41   +0.979   0.3832 
folora_v2/v2f_l10_k16 20t/5e/r16      5 FGT(%)               7.07     7.18    -0.11   -0.504   0.6409 
folora_v2/v2f_l10_k64 20t/5e/r16      5 avgACC(%)           78.32    78.24    +0.08   +0.152   0.8868 
folora_v2/v2f_l10_k64 20t/5e/r16      5 FGT(%)               7.07     6.87    +0.20   +1.530   0.2008 
folora_v2/v2f_l1_k64  20t/5e/r16      5 avgACC(%)           78.32    77.33    +0.99   +2.043   0.1106 
folora_v2/v2f_l1_k64  20t/5e/r16      5 FGT(%)               7.07     7.56    -0.49   -1.749   0.1552 
folora_v2/v2f_l300_k1620t/5e/r16     10 avgACC(%)           77.87    73.61    +4.25   +8.035   0.0000*
folora_v2/v2f_l300_k1620t/5e/r16     10 FGT(%)               7.42     7.73    -0.30   -2.160   0.0590 
folora_v2/v2f_l30_k16 20t/5e/r16      1 avgACC(%)           77.83    77.47    +0.36      n/a       n/a 
folora_v2/v2f_l30_k16 20t/5e/r16      1 FGT(%)               7.31     7.18    +0.13      n/a       n/a 
folora_v2/v2f_l3_k128 20t/5e/r16      5 avgACC(%)           78.32    77.66    +0.65   +1.258   0.2768 
folora_v2/v2f_l3_k128 20t/5e/r16      5 FGT(%)               7.07     7.49    -0.42   -1.668   0.1706 
folora_v2/v2f_l3_k16  20t/5e/r16      5 avgACC(%)           78.32    78.14    +0.17   +0.343   0.7487 
folora_v2/v2f_l3_k16  20t/5e/r16      5 FGT(%)               7.07     7.32    -0.25   -1.503   0.2072 
InfLoRA               20t/5e/r16      5 avgACC(%)           78.32    70.14    +8.18  +26.695   0.0000*
InfLoRA               20t/5e/r16      5 FGT(%)               7.07     8.87    -1.80   -7.654   0.0016*
O-LoRA                20t/5e/r16     10 avgACC(%)           77.87    64.79   +13.07  +15.512   0.0000*
O-LoRA                20t/5e/r16     10 FGT(%)               7.42     9.73    -2.30   -8.024   0.0000*
olora/olora_mean      20t/5e/r16      3 avgACC(%)           78.37    64.34   +14.02   +6.725   0.0214*
olora/olora_mean      20t/5e/r16      3 FGT(%)               7.05     9.74    -2.69   -5.626   0.0302*
olora/olora_orth_l0.1 20t/5e/r16      3 avgACC(%)           78.37    68.44    +9.93  +13.557   0.0054*
olora/olora_orth_l0.1 20t/5e/r16      3 FGT(%)               7.05     8.91    -1.86   -3.385   0.0773 
olora/olora_orth_l1   20t/5e/r16      3 avgACC(%)           78.37    68.44    +9.93   +9.049   0.0120*
olora/olora_orth_l1   20t/5e/r16      3 FGT(%)               7.05     8.82    -1.77   -3.116   0.0894 
Seq-LoRA              20t/5e/r16     10 avgACC(%)           77.87    69.31    +8.56  +11.855   0.0000*
Seq-LoRA              20t/5e/r16     10 FGT(%)               7.42     9.92    -2.49  -12.703   0.0000*

=== imagenetr：FOLoRA vs 基线（配对 t 检验，* = p<0.05）===
baseline              proto           n metric               ours     base     diff        t         p
EWC-LoRA (lam=100)    20t/5e/r16     10 avgACC(%)           66.37    63.52    +2.85   +5.960   0.0002*
EWC-LoRA (lam=100)    20t/5e/r16     10 FGT(%)               7.27     8.44    -1.17   -5.470   0.0004*
EWC-LoRA (lam=1000)   20t/5e/r16     10 avgACC(%)           66.37    66.33    +0.04   +0.099   0.9235 
EWC-LoRA (lam=1000)   20t/5e/r16     10 FGT(%)               7.27     7.00    +0.27   +1.372   0.2031 
EWC-LoRA (lam=300)    20t/5e/r16     10 avgACC(%)           66.37    66.32    +0.05   +0.128   0.9010 
EWC-LoRA (lam=300)    20t/5e/r16     10 FGT(%)               7.27     7.67    -0.41   -2.540   0.0317*
folora_v2/v2f_l300_k1620t/5e/r16      5 avgACC(%)           66.92    61.25    +5.67  +16.518   0.0001*
folora_v2/v2f_l300_k1620t/5e/r16      5 FGT(%)               7.11     6.55    +0.56   +2.686   0.0549 
InfLoRA               20t/5e/r16      3 avgACC(%)           66.89    50.00   +16.89  +51.809   0.0004*
InfLoRA               20t/5e/r16      3 FGT(%)               7.11     7.53    -0.42   -1.091   0.3891 
O-LoRA                20t/5e/r16      5 avgACC(%)           66.92    44.79   +22.13  +33.966   0.0000*
O-LoRA                20t/5e/r16      5 FGT(%)               7.11     8.25    -1.14   -3.818   0.0188*
olora/olora_orth_l0.1 20t/5e/r16      3 avgACC(%)           66.89    48.14   +18.75  +20.282   0.0024*
olora/olora_orth_l0.1 20t/5e/r16      3 FGT(%)               7.11     7.75    -0.64   -2.978   0.0967 
olora/olora_orth_l1   20t/5e/r16      3 avgACC(%)           66.89    48.48   +18.41  +35.002   0.0008*
olora/olora_orth_l1   20t/5e/r16      3 FGT(%)               7.11     7.85    -0.74   -2.558   0.1248 
Seq-LoRA              20t/5e/r16      5 avgACC(%)           66.92    56.58   +10.34  +13.477   0.0002*
Seq-LoRA              20t/5e/r16      5 FGT(%)               7.11     8.83    -1.72   -7.315   0.0019*
```