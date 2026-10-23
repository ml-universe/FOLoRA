# imagenetr: run summary (primary metric = NCM)

> `head_acc_LEGACY` is the old head-CIL metric (~0.08). It is not comparable with the other columns and is kept only for provenance.

| method | tag | hyper | proto | n | NCM avgACC(%) | NCM FGT(%) | head_LEGACY(%) |
|---|---|---|---|---|---|---|---|
| coda | default | - | 20t/5e/r16 | 3 | 58.87±0.20 | 7.81±0.11 | 5.96 |
| coda | pilot20 | - | 20t/20e/r16 ⚠ | 3 | 59.16±1.17 | 7.82±0.50 | 6.93 |
| coda | pool100_len8_ep20_inr | - | 20t/20e/r16 ⚠ | 3 | 59.31±0.22 | 7.83±0.42 | 7.02 |
| ewc | default | λ=100.0 | 20t/5e/r16 | 10 | 63.52±1.28 | 8.44±0.42 | 8.06 |
| ewc | ewc_lam1000 | λ=1000.0 | 20t/5e/r16 | 10 | 66.33±0.61 | 7.00±0.30 | 7.41 |
| ewc | ewc_lam300 | λ=300.0 | 20t/5e/r16 | 10 | 66.32±0.62 | 7.67±0.27 | 7.59 |
| folora_v2 | v2f_l300_k16 | λ=300.0 k=16 | 20t/5e/r16 | 5 | 61.25±0.86 | 6.55±0.22 | 9.56 |
| folora_v2 | v2f_l3_k64 | λ=3.0 k=64 | 20t/5e/r16 | 10 | 66.37±0.77 | 7.27±0.49 | 7.15 |
| inflora | default | - | 20t/5e/r16 | 3 | 50.00±0.48 | 7.53±0.44 | 6.20 |
| inflora | default_fix1 | - | 20t/5e/r16 | 3 | 58.93±0.19 | 8.86±0.52 | 6.17 |
| inflora | ep20_inr | - | 20t/20e/r16 ⚠ | 3 | 49.78±0.15 | 7.82±0.58 | 6.19 |
| l2p | default | - | 20t/5e/r16 | 3 | 58.95±0.16 | 7.96±0.21 | 5.74 |
| l2p | pilot20 | - | 20t/20e/r16 ⚠ | 5 | 59.47±0.43 | 8.02±0.36 | 6.98 |
| olora | default | - | 20t/5e/r16 | 5 | 44.79±1.58 | 8.25±0.40 | 6.38 |
| olora | default_fix1 | - | 20t/5e/r16 | 5 | 53.72±0.65 | 9.72±0.35 | 6.64 |
| olora | olora_orth_l0.1 | - | 20t/5e/r16 | 3 | 48.14±1.56 | 7.75±0.42 | 6.41 |
| olora | olora_orth_l0.1_fix1 | - | 20t/5e/r16 | 3 | 56.30±2.24 | 9.26±0.44 | 6.50 |
| olora | olora_orth_l1 | - | 20t/5e/r16 | 3 | 48.48±1.23 | 7.85±0.36 | 6.54 |
| olora | olora_orth_l1_fix1 | - | 20t/5e/r16 | 3 | 61.15±0.92 | 8.86±0.33 | 6.58 |
| olora | vdc_olora | - | 20t/5e/r16 | 1 | 43.38±0.00 | 8.62±0.00 | 6.75 |
| seq | default | - | 20t/5e/r16 | 5 | 56.58±1.49 | 8.83±0.70 | 6.55 |
| seq | vdc_seq | - | 20t/5e/r16 | 1 | 59.38±0.00 | 8.15±0.00 | 7.44 |