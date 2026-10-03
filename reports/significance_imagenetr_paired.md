# 显著性检验（paired t-test）

对照 = 论文 Table 1 的主表；`*` 表示 p<0.05。FGT 越小越好（diff 为负表示本文方法遗忘更低），avgACC 越大越好。

```

=== imagenetr：FOLoRA vs 基线（配对 t 检验，* = p<0.05）===
baseline              proto           n metric               ours     base     diff        t         p
Seq-LoRA              20t/5e/r16      5 avgACC(%)           66.92    56.58   +10.34  +13.477   0.0002*
Seq-LoRA              20t/5e/r16      5 FGT(%)               7.11     8.83    -1.72   -7.315   0.0019*
EWC-LoRA (lam=100)    20t/5e/r16     10 avgACC(%)           66.37    63.52    +2.85   +5.960   0.0002*
EWC-LoRA (lam=100)    20t/5e/r16     10 FGT(%)               7.27     8.44    -1.17   -5.470   0.0004*
EWC-LoRA (lam=300)    20t/5e/r16     10 avgACC(%)           66.37    66.32    +0.05   +0.128   0.9010 
EWC-LoRA (lam=300)    20t/5e/r16     10 FGT(%)               7.27     7.67    -0.41   -2.540   0.0317*
EWC-LoRA (lam=1000)   20t/5e/r16     10 avgACC(%)           66.37    66.33    +0.04   +0.099   0.9235 
EWC-LoRA (lam=1000)   20t/5e/r16     10 FGT(%)               7.27     7.00    +0.27   +1.372   0.2031 
O-LoRA                20t/5e/r16      5 avgACC(%)           66.92    44.79   +22.13  +33.966   0.0000*
O-LoRA                20t/5e/r16      5 FGT(%)               7.11     8.25    -1.14   -3.818   0.0188*
L2P                   20t/20e/r16     5 avgACC(%)           66.92    59.47    +7.45  +40.202   0.0000*
L2P                   20t/20e/r16     5 FGT(%)               7.11     8.02    -0.91   -3.129   0.0352*
CODA-Prompt           20t/20e/r16     3 avgACC(%)           66.89    59.31    +7.58  +21.229   0.0022*
CODA-Prompt           20t/20e/r16     3 FGT(%)               7.11     7.83    -0.72   -1.513   0.2694 
InfLoRA               20t/5e/r16      3 avgACC(%)           66.89    50.00   +16.89  +51.809   0.0004*
InfLoRA               20t/5e/r16      3 FGT(%)               7.11     7.53    -0.42   -1.091   0.3891 
```