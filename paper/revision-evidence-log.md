# Revision evidence log

Manuscript **NEUCOM-D-26-24287R1**, *Neurocomputing*. Second revision, 24 November 2026.

This page carries over from the first revision and is extended for the second. Rows marked
**[R2]** are new or extended in this round; unmarked rows are from the first revision and are
included because their numbers are still quoted. No previously reported number was altered.

This page exists so that no number in the revision has to be taken on trust. For each
experiment it gives the run tag, the seed budget, and the file
under `reports/` from which the number can be recomputed. All per-seed files are in
`reports.zip`; every run's resolved hyperparameters are in the matching
`experiments/<benchmark>/<method>/<tag>/seed<k>/config.json`, which is in the same archive,
and the per-boundary records of the bound probe are in
`bound_terms/<benchmark>/<method>/<tag>/seed<k>/bound_terms.jsonl`.

## What changed in the second revision

Not all of the second round is new computation; the list below says which items are.

- **A4** (isotropic L2-SP penalty) is promoted from the limitations into a row of the main
  comparison, on both benchmarks; the ImageNet-R arm is new in this round.
- **A11** (validation-split re-selection of $\lambda$) is new in this round.
- **A12** (learned-head recency tiers) is a new analysis of runs already in the package; it
  reads each run's own per-task record, so it retrains nothing.
- The main table's O-LoRA row moves to the training-time objective and the initialization-only
  variant becomes an ablation; every Holm-corrected $p$-value in the confirmatory family was
  recomputed from the membership change. Source: `significance_all_paired_holm-bench-metric.json`.
- Five uncited metaheuristic references are removed from the bibliography. No result changed.
- Title, abstract, highlights, Section 3 and the conclusion are reworded to attribute the
  margin to the shared adapter; Section 4 is compressed and its survey of prior bounds moved
  to Appendix C. No result changed.

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
| A4 **[R2]** | Isotropic (L2-SP) displacement penalty, now a row of the main comparison on both benchmarks; on ImageNet-R the comparison is at λ=3 only, since FOLoRA was not run at λ=10 on that benchmark | `folora_v2/v2f_l2sp_l3`, `folora_v2/v2f_l2sp_l10` | 3 (CIFAR-100) / 3 (ImageNet-R) | `revision_extra_tests.json`, `ncm/` |
| A5 | Size of the terms the derivation drops | `folora_v2/bound_l3_k64` (CIFAR-100 seeds 0--2, ImageNet-R seed 0); the earlier $\lambda{=}10$, $k{=}16$ probe of Appendix~B (i)--(iii) is `folora_v2/probe_l10_k16` | 3 + 1 | `bound_terms.jsonl` in each run directory (packed under `bound_terms/`) |
| A10 | Measured peak device memory | one process per method, 20 tasks, 1 epoch per task | 1 | `peak_memory.json` |
| T=50 | Sequence-length generality check | `folora_v2/t50_l{3,10,30}_k64`, `ewc/t50_ewc_l{100,300,1000}`, `seq/t50_seq` | 5 | `ncm/` |
| A11 **[R2]** | Validation-split re-selection of $\lambda$ (R8-6a) | `folora_v2/valsel_l{1,3,10}_k64` (CIFAR-100) | 3 | `val_selection.json`; holdout per (seed, task, class) |
| A12 **[R2]** | Learned-head recency tiers (R8-4): accuracy of the jointly trained head on the last task, the second-last, and the mean over the first eighteen | Table 2 rows (all methods), both benchmarks | as Table 2 (3--10 per row) | `head_recency_tiers.json`, recomputed by `scripts/head_recency_tiers.py` from each run's `results.json` (`acc_cil[-1]`, per-task head accuracy at the final boundary) |

Every paired test quoted in the manuscript and in the response letter is produced by the
same routine from these files, over the seeds the two arms have in common; the seed count is
printed with every p-value. `significance_all_paired_holm-bench-metric.json` holds the
confirmatory family (Table A.9) and `summary_cifar100.json` / `summary_imagenetr.json` hold
the means that Table 1 prints.

## Results that do not favour the method

These four are reported in full in the manuscript and the response letter, and are listed
here with their provenance so that a reader can check that nothing was dropped.

1. **A4 — an isotropic displacement penalty ties the method.** On CIFAR-100, 78.27 ± 0.38
   against 78.37 ± 0.84 at λ=3 (Δ = −0.10, p = 0.79) and 77.77 ± 0.56 against 77.88 ± 0.58 at
   λ=10 (Δ = −0.11, p = 0.52); on ImageNet-R, 66.89 ± 0.29 against 66.89 ± 0.41 at λ=3
   (Δ = +0.00, p = 0.98); three common seeds each. On the gap metric the point estimate is
   nominally in its favour on CIFAR-100 (−0.22, p = 0.49) and nominally in FOLoRA's on
   ImageNet-R (+0.18, p = 0.38). Source: `v2f_l2sp_l3`, `v2f_l2sp_l10`.
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
py -3.11 -m scripts.make_paper_tables --check   # tables agree with reports/ncm/
py -3.11 -m scripts.significance --check-paper  # every p-value quoted in the text
py -3.11 -m scripts.head_recency_tiers --check  # the recency tiers of Section 5.1
py -3.11 scripts/build_submission_pack.py --out <dir>
```

The first two are run from the repository root. Both recompute their output from
`reports/ncm/` instead of trusting a stored artefact, so a disagreement shows up as a
diff rather than as a silent pass; `--check` additionally stops the first one from
writing.

Everything above is in the public repository `github.com/ml-universe/FOLoRA`, at the commit
whose `paper/main.pdf` is the manuscript uploaded with this revision.
`MANIFEST-2026-11-08.json` in `reports.zip` records the byte size and SHA-256 that each
generated step of the evaluation produced.
