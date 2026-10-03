# 显著性检验（paired t-test）

对照 = 论文 Table 1 的主表；`*` 表示 p<0.05。FGT 越小越好（diff 为负表示本文方法遗忘更低），avgACC 越大越好。

```

=== cifar100：FOLoRA vs 基线（配对 t 检验，* = p<0.05）===
baseline              proto           n metric               ours     base     diff        t         p
Seq-LoRA              20t/5e/r16      4 avgACC(%)           78.11    69.51    +8.60   +9.278   0.0026*
Seq-LoRA              20t/5e/r16      4 FGT(%)               6.98     9.71    -2.73  -14.697   0.0007*
EWC-LoRA              20t/5e/r16      3 avgACC(%)           77.88    77.32    +0.56   +1.936   0.1926 
EWC-LoRA              20t/5e/r16      3 FGT(%)               6.88     6.89    -0.01   -0.084   0.9409 
O-LoRA                20t/5e/r16      4 avgACC(%)           78.11    64.70   +13.42   +9.592   0.0024*
O-LoRA                20t/5e/r16      4 FGT(%)               6.98     9.39    -2.41   -8.073   0.0040*
L2P                   20t/20e/r16     3 avgACC(%)           77.88    73.83    +4.05  +23.282   0.0018*
L2P                   20t/20e/r16     3 FGT(%)               6.88     8.06    -1.17   -3.809   0.0625 
CODA-Prompt           20t/20e/r16     3 avgACC(%)           77.88    75.83    +2.05   +4.646   0.0433*
CODA-Prompt           20t/20e/r16     3 FGT(%)               6.88     7.48    -0.59   -2.669   0.1164 
```