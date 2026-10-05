# 显著性检验（paired t-test，Holm 家族=bench-metric）

对照 = 论文 Table 1 的主表。FGT 越小越好（diff 为负表示本文方法遗忘更低），avgACC 越大越好。

- `p` = 原始配对 p 值（未校正）
- `p_holm` = Holm–Bonferroni 校正后 p；`*` 标在 `p_holm<0.05` 上，即**本文对外声明的显著性口径**
- 家族划分 = `bench-metric`：同一 benchmark、同一指标下的全部对照（默认；主表 m≈6）；本次家族大小 {'cifar100/final_acc_cil': 9, 'cifar100/forgetting_cil': 9, 'imagenetr/final_acc_cil': 8, 'imagenetr/forgetting_cil': 8}
- 不可检验条目（单 seed / 零方差）不计入家族：「没做这个检验」与「做了且不显著」不能混算

```

=== cifar100：FOLoRA vs 基线（配对 t 检验；* = Holm 校正后 p<0.05，家族=bench-metric）===
baseline              proto           n metric               ours     base     diff        t        p   p_holm sig
Seq-LoRA              20t/5e/r16     10 avgACC(%)           77.87    69.31    +8.56  +11.855   0.0000   0.0000*
Seq-LoRA              20t/5e/r16     10 FGT(%)               7.42     9.92    -2.49  -12.703   0.0000   0.0000*
EWC-LoRA (lam=100)    20t/5e/r16     10 avgACC(%)           77.87    77.22    +0.65   +2.026   0.0734   0.2934 
EWC-LoRA (lam=100)    20t/5e/r16     10 FGT(%)               7.42     7.73    -0.30   -2.049   0.0707   0.1599 
EWC-LoRA (lam=300)    20t/5e/r16     10 avgACC(%)           77.87    78.00    -0.14   -0.478   0.6440   0.9921 
EWC-LoRA (lam=300)    20t/5e/r16     10 FGT(%)               7.42     7.35    +0.07   +0.592   0.5681   0.5681 
EWC-LoRA (lam=1000)   20t/5e/r16     10 avgACC(%)           77.87    77.65    +0.21   +0.709   0.4961   0.9921 
EWC-LoRA (lam=1000)   20t/5e/r16     10 FGT(%)               7.42     7.10    +0.32   +2.399   0.0400   0.1599 
EWC-LoRA (lam=3000)   20t/5e/r16     10 avgACC(%)           77.87    77.40    +0.47   +1.228   0.2507   0.7521 
EWC-LoRA (lam=3000)   20t/5e/r16     10 FGT(%)               7.42     7.03    +0.39   +2.396   0.0401   0.1599 
O-LoRA                20t/5e/r16     10 avgACC(%)           77.87    64.94   +12.92  +12.089   0.0000   0.0000*
O-LoRA                20t/5e/r16     10 FGT(%)               7.42    11.03    -3.61  -13.556   0.0000   0.0000*
L2P                   20t/20e/r16     5 avgACC(%)           78.32    73.65    +4.67  +13.225   0.0002   0.0013*
L2P                   20t/20e/r16     5 FGT(%)               7.07     8.12    -1.05   -3.568   0.0234   0.1405 
CODA-Prompt           20t/20e/r16     5 avgACC(%)           78.32    74.71    +3.61   +7.621   0.0016   0.0080*
CODA-Prompt           20t/20e/r16     5 FGT(%)               7.07     7.71    -0.64   -3.283   0.0304   0.1521 
InfLoRA               20t/5e/r16      5 avgACC(%)           78.32    72.95    +5.37  +12.462   0.0002   0.0014*
InfLoRA               20t/5e/r16      5 FGT(%)               7.07     9.60    -2.53  -12.104   0.0003   0.0019*

=== imagenetr：FOLoRA vs 基线（配对 t 检验；* = Holm 校正后 p<0.05，家族=bench-metric）===
baseline              proto           n metric               ours     base     diff        t        p   p_holm sig
Seq-LoRA              20t/5e/r16      5 avgACC(%)           66.92    56.58   +10.34  +13.477   0.0002   0.0011*
Seq-LoRA              20t/5e/r16      5 FGT(%)               7.11     8.83    -1.72   -7.315   0.0019   0.0111*
EWC-LoRA (lam=100)    20t/5e/r16     10 avgACC(%)           66.37    63.52    +2.85   +5.960   0.0002   0.0011*
EWC-LoRA (lam=100)    20t/5e/r16     10 FGT(%)               7.27     8.44    -1.17   -5.470   0.0004   0.0028*
EWC-LoRA (lam=300)    20t/5e/r16     10 avgACC(%)           66.37    66.32    +0.05   +0.128   0.9010   1.0000 
EWC-LoRA (lam=300)    20t/5e/r16     10 FGT(%)               7.27     7.67    -0.41   -2.540   0.0317   0.1268 
EWC-LoRA (lam=1000)   20t/5e/r16     10 avgACC(%)           66.37    66.33    +0.04   +0.099   0.9235   1.0000 
EWC-LoRA (lam=1000)   20t/5e/r16     10 FGT(%)               7.27     7.00    +0.27   +1.372   0.2031   0.4063 
O-LoRA                20t/5e/r16      5 avgACC(%)           66.92    53.72   +13.20  +34.743   0.0000   0.0000*
O-LoRA                20t/5e/r16      5 FGT(%)               7.11     9.72    -2.61  -21.446   0.0000   0.0002*
L2P                   20t/20e/r16     5 avgACC(%)           66.92    59.47    +7.45  +40.202   0.0000   0.0000*
L2P                   20t/20e/r16     5 FGT(%)               7.11     8.02    -0.91   -3.129   0.0352   0.1268 
CODA-Prompt           20t/20e/r16     3 avgACC(%)           66.89    59.31    +7.58  +21.229   0.0022   0.0066*
CODA-Prompt           20t/20e/r16     3 FGT(%)               7.11     7.83    -0.72   -1.513   0.2694   0.4063 
InfLoRA               20t/5e/r16      3 avgACC(%)           66.89    58.93    +7.96  +29.513   0.0011   0.0046*
InfLoRA               20t/5e/r16      3 FGT(%)               7.11     8.86    -1.75  -18.074   0.0030   0.0152*
```