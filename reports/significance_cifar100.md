# 显著性检验（welch t-test）

对照 = 论文 Table 1 的主表；`*` 表示 p<0.05。FGT 越小越好（diff 为负表示本文方法遗忘更低），avgACC 越大越好。

```

=== cifar100：FOLoRA vs 基线（Welch t 检验，* = p<0.05）===
baseline              proto           n metric               ours     base     diff        t         p
Seq-LoRA              20t/5e/r16      5 avgACC(%)           78.24    69.31    +8.93  +13.298   0.0000*
Seq-LoRA              20t/5e/r16      5 FGT(%)               6.87     9.92    -3.05  -13.321   0.0000*
EWC-LoRA              20t/5e/r16      5 avgACC(%)           78.24    77.65    +0.59   +1.860   0.1142 
EWC-LoRA              20t/5e/r16      5 FGT(%)               6.87     7.10    -0.23   -1.349   0.2098 
O-LoRA                20t/5e/r16      5 avgACC(%)           78.24    64.79   +13.45  +18.772   0.0000*
O-LoRA                20t/5e/r16      5 FGT(%)               6.87     9.73    -2.86  -12.926   0.0000*
L2P                   20t/20e/r16     5 avgACC(%)           78.24    73.65    +4.59   +9.950   0.0000*
L2P                   20t/20e/r16     5 FGT(%)               6.87     8.12    -1.25   -5.856   0.0004*
CODA-Prompt           20t/20e/r16     5 avgACC(%)           78.24    75.83    +2.41   +7.269   0.0004*
CODA-Prompt           20t/20e/r16     5 FGT(%)               6.87     7.48    -0.61   -2.238   0.1016 
InfLoRA               20t/5e/r16      5 avgACC(%)           78.24    70.14    +8.10  +26.141   0.0000*
InfLoRA               20t/5e/r16      5 FGT(%)               6.87     8.87    -2.00   -9.146   0.0000*
```