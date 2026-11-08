# Revision evidence log

Manuscript **NEUCOM-D-26-24287**, *Neurocomputing*. Major revision, 8 November 2026.

This page exists so that no number in the revision has to be taken on trust. For each
experiment that is new in this round it gives the run tag, the seed budget, and the file
under `reports/` from which the number can be recomputed. All per-seed files are in
`reports.zip`; every run's resolved hyperparameters are in the matching
`experiments/<benchmark>/<method>/<tag>/seed<k>/config.json`, which is in the same archive,
and the per-boundary records of the bound probe are in
`bound_terms/<benchmark>/<method>/<tag>/seed<k>/bound_terms.jsonl`.

## Where each new number comes from

| ID | Question | Run tag(s) under `reports/ncm/<benchmark>/` | Seeds | Recompute from |
|---|---|---|---|---|
| E1 | Fisher weighting vs a tuned equal-weight penalty | `folora_v2/v2f_eq_l{1,3,10,30}_k64` against `folora_v2/v2f_l{1,3,10,30}_k64` | 10 | `eq_contrast.json` |
| E2 | How much is forgotten over time | `folora_v2/ncm_trace_l3_k64`, `ewc/ncm_trace_l1000`, `ewc/ncm_trace_l300`, `seq/ncm_trace` | 3 | `forgetting_true_cifar100.json` |
| E3 | Non-Fisher baselines at a 20-epoch budget | `seq/ep20_fix1`, `olora/ep20_fix1`, `inflora/ep20_fix1` | 3 | `revision_extra_tests.json` |
| E4 | Run-to-run floor and equivalence test | `repro_ewc100_s0_{a..e}`, `repro_folora_s0_{a..c}` | 11 runs | `repro_floor.json`, `repro_spread.json`, `tost_cifar100.json`, `tost_imagenetr.json` |
| E5 | Strictly sequential prototypes | `folora_v2/ncm_strict_l3_k64`, `ewc/ncm_strict_l300`, `seq/ncm_strict` | 3 | `ncm/` |
| E6 | Faithful O-LoRA at both constraint strengths | `olora/olora_orth_l0.1_fix1`, `olora/olora_orth_l1_fix1` | 5 (CIFAR-100) / 3 (ImageNet-R) | `orth_variant_tests.json` |
| E7 | Random protected subspace, same rank and strength | `folora_v2/v2f_rand_l3_k64` | 3 | `ncm/` |
| A4 | Isotropic (L2-SP) displacement penalty | `folora_v2/v2f_l2sp_l3`, `folora_v2/v2f_l2sp_l10` | 3 | `revision_extra_tests.json` |
| A5 | Size of the terms the derivation drops | `folora_v2/bound_l3_k64` (CIFAR-100 seeds 0--2, ImageNet-R seed 0); the earlier $\lambda{=}10$, $k{=}16$ probe of Appendix~B (i)--(iii) is `folora_v2/probe_l10_k16` | 3 + 1 | `bound_terms.jsonl` in each run directory (packed under `bound_terms/`) |
| A10 | Measured peak device memory | one process per method, 20 tasks, 1 epoch per task | 1 | `peak_memory.json` |
| T=50 | Sequence-length generality check | `folora_v2/t50_l{3,10,30}_k64`, `ewc/t50_ewc_l{100,300,1000}`, `seq/t50_seq` | 5 | `ncm/` |

Every paired test quoted in the manuscript and in the response letter is produced by the
same routine from these files, over the seeds the two arms have in common; the seed count is
printed with every p-value. `significance_all_paired_all.json` holds the confirmatory family
(Table A.9) and `summary_cifar100.json` / `summary_imagenetr.json` hold the means that
Table 1 prints.

## Results that do not favour the method

These four are reported in full in the manuscript and the response letter, and are listed
here with their provenance so that a reader can check that nothing was dropped.

1. **A4 — an isotropic displacement penalty ties the method.** 78.27 ± 0.38 against
   78.37 ± 0.84 at λ=3 (Δ = −0.10, p = 0.79) and 77.77 ± 0.56 against 77.88 ± 0.58 at λ=10
   (Δ = −0.11, p = 0.52), three common seeds. Source: `v2f_l2sp_l3`, `v2f_l2sp_l10`.
2. **E7 — the random-subspace control is directional but not significant.** 74.35 ± 1.43
   against 78.37 ± 0.84 (−4.02, p = 0.070, n = 3). Source: `v2f_rand_l3_k64`.
3. **A5 — the terms the derivation drops are not small.** The quadratic term fits well
   (R² ≥ 0.978 at every one of the 76 boundaries), but the first-order term is 40–63 % of
   it at the median boundary and the cubic remainder exceeds 30 % at 6–8 of the 19
   boundaries, and 100 % at 2–4. Source: `bound_terms.jsonl` of `bound_l3_k64`.
4. **T=50 — longer sequences do not separate FOLoRA from tuned EWC-LoRA.** 74.25 ± 1.46
   against 74.94 ± 0.69 (−0.69, p = 0.26, n = 5, uncorrected). Source: `t50_l3_k64`,
   `t50_ewc_l1000`.

Two further comparisons are adverse in one direction only and are reported at both ends:
against EWC-LoRA at λ=1000 the method forgets significantly more (BWT −2.65, p = 0.013,
n = 3), and the importance state at k=64 is 32.5× EWC's. Neither is a control that failed;
both are costs of the design, and both appear in Section 5.6.

## Reproducing the checks

```bash
py -3.11 scripts/make_paper_tables.py --check   # tables agree with reports/ncm/
py -3.11 scripts/significance.py --all-configs  # paired tests and Holm correction
py -3.11 scripts/build_submission_pack.py --out <dir>
```

`git status` at the commit this pack was built from is clean apart from the generator
scripts themselves; `MANIFEST-2026-11-08.json` in `reports.zip` records the byte size and
SHA-256 that each generated step produced.
