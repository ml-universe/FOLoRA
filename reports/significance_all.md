# 显著性检验（welch t-test）

对照 = 论文 Table 1 的主表；`*` 表示 p<0.05。FGT 越小越好（diff 为负表示本文方法遗忘更低），avgACC 越大越好。

```

=== cifar100：FOLoRA vs 基线（Welch t 检验，* = p<0.05）===
baseline              proto           n metric               ours     base     diff        t         p
Seq-LoRA              20t/5e/r16     10 avgACC(%)           10.07     7.19    +2.88   +4.805   0.0002*
Seq-LoRA              20t/5e/r16     10 FGT(%)              87.65    90.90    -3.25   -5.498   0.0000*
EWC-LoRA              20t/5e/r16     10 avgACC(%)           10.07     9.17    +0.90   +1.517   0.1508 
EWC-LoRA              20t/5e/r16     10 FGT(%)              87.65    88.76    -1.11   -1.909   0.0742 
O-LoRA                20t/5e/r16     10 avgACC(%)           10.07     7.74    +2.32   +4.036   0.0013*
O-LoRA                20t/5e/r16     10 FGT(%)              87.65    90.01    -2.36   -4.104   0.0009*
L2P                   20t/20e/r16    10 avgACC(%)           10.07     8.67    +1.40   +2.167   0.0587 
L2P                   20t/20e/r16    10 FGT(%)              87.65    89.08    -1.43   -2.271   0.0535 
CODA-Prompt           20t/20e/r16    10 avgACC(%)           10.07     9.35    +0.72   +1.141   0.2821 
CODA-Prompt           20t/20e/r16    10 FGT(%)              87.65    88.52    -0.87   -1.319   0.2293 

=== imagenetr：FOLoRA vs 基线（Welch t 检验，* = p<0.05）===
baseline              proto           n metric               ours     base     diff        t         p
Seq-LoRA              20t/5e/r16      5 avgACC(%)            9.56     6.55    +3.01   +6.643   0.0008*
Seq-LoRA              20t/5e/r16      5 FGT(%)              79.44    84.42    -4.98  -14.759   0.0000*
EWC-LoRA              20t/5e/r16      5 avgACC(%)            9.56     8.17    +1.40   +2.500   0.0372*
EWC-LoRA              20t/5e/r16      5 FGT(%)              79.44    82.92    -3.49   -6.113   0.0005*
O-LoRA                20t/5e/r16      5 avgACC(%)            9.56     6.38    +3.18   +6.811   0.0005*
O-LoRA                20t/5e/r16      5 FGT(%)              79.44    83.44    -4.01  -11.803   0.0000*
L2P                   20t/20e/r16     5 avgACC(%)            9.56     7.05    +2.51   +5.324   0.0021*
L2P                   20t/20e/r16     5 FGT(%)              79.44    80.22    -0.78   -1.937   0.1018 
CODA-Prompt           20t/20e/r16     5 avgACC(%)            9.56     6.93    +2.63   +5.772   0.0017*
CODA-Prompt           20t/20e/r16     5 FGT(%)              79.44    79.54    -0.11   -0.202   0.8496 
```