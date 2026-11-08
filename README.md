# FOLoRA: Fisher-Orthogonalized Low-Rank Adaptation for Continual Learning

Code, run configurations, and evaluation outputs for the paper *FOLoRA: Fisher-Orthogonalized
Low-Rank Adaptation for Continual Learning* (submitted to Neurocomputing).

> **Status.** The first version was submitted on 23 October 2026; the revised manuscript was
> submitted on 8 November 2026 and is under review (manuscript NEUCOM-D-26-24287).
> `paper/main.pdf` is the **revised** manuscript, 95 pages. The code, the paper source and the
> evaluation outputs in this repository are the post-revision state. The first-submission
> version is preserved in the git history: commit `1df2bd3` is the 23 October tree, and the
> diff `1df2bd3..HEAD` is the revision. See [What the revision changed](#what-the-revision-changed).

FOLoRA adapts a **shared** LoRA adapter across a task sequence while limiting catastrophic
forgetting. After each task it estimates a second-order Fisher kernel of the adapter
parameters from per-sample gradients, keeps the top-$k$ spectral directions of the accumulated
kernel, and penalizes the projection of the adapter *displacement* onto those directions with a
soft, importance-weighted orthogonality term. Directions that matter for earlier tasks are
protected; the remaining capacity stays available for new ones.

## Repository layout

| Path | Contents |
|---|---|
| `src/peft_cl/` | Method and training code: adapters, Fisher estimation, continual-learning methods, evaluation |
| `scripts/` | Training, NCM evaluation, aggregation, significance testing, table generation |
| `tests/` | Unit tests (CPU only, no GPU required) |
| `paper/` | LaTeX source of the manuscript and the compiled `main.pdf` (the revised 95-page version) |
| `paper/tables/` | Generated tables; produced by `scripts/make_paper_tables.py`, do not edit by hand |
| `reports/` | Frozen-feature NCM evaluation outputs, summaries, and significance reports — the data source of every number in the paper |
| `reports/ncm/` | Per-seed frozen-feature NCM evaluation, one JSON per run and seed — the raw input of Tables 1–3, 6 and 8 |
| `experiments/` | Per-run configuration, results, and training logs for the runs reported in the paper (no model weights) |

## What the revision changed

The manuscript is the authority for all of this; this section only maps its claims to files.

**Four statements of the first submission were corrected.**

1. The abstract no longer reports `∼18×` fewer *trainable* parameters. The factor is the ratio
   of **cumulative allocated** parameters (trainable plus frozen) between one shared adapter
   and a per-task adapter bank; the *trainable* count is the same as O-LoRA's at $T=20$. See
   Table 7 (`tab:resources`).
2. The comparison with EWC-LoRA is now stated as **no significant difference** at the
   baseline's best tuned strength on either benchmark, not as a win. The positive result is
   against the parameter-efficient family (L2P, CODA-Prompt, O-LoRA, InfLoRA), which survives
   Holm–Bonferroni correction. See Table 3 (`tab:ewc-lambda`) and Table A.9.
3. The cross-term of the second-order forgetting bound, Eq. (14), is expanded rather than
   left implicit.
4. Table 7 counts the per-task modules of O-LoRA and InfLoRA as **frozen**, not trainable: at
   $T=20$ both train a parameter count comparable to or below FOLoRA's, and what grows is
   allocated-but-frozen memory.

**New experiments.** All of them use the same frozen-feature nearest-class-mean decision rule
as the rest of the paper.

| Paper | Question it answers | Output |
|---|---|---|
| Table 2 (`tab:head-acc`) | How good is each method's own trained classifier head? | `reports/summary_{cifar100,imagenetr}.json` |
| Table 3 (`tab:ewc-lambda`) | Is the EWC-LoRA comparison sensitive to its regularization strength? | `reports/ncm/` (`ewc__*`), `reports/significance_all_paired_holm-bench-metric.json` |
| Table 4 (`tab:posthoc`) | Is the ranking an artifact of the NCM decision rule? Re-scored with RanPAC. | `reports/ranpac_*.json`, `scripts/eval_ranpac.py` |
| Table 5 (`tab:bwt`) | What is true temporal forgetting, measured at every task boundary? | `reports/forgetting_true_cifar100.json`, `reports/bwt_paired_tests.json` |
| Table 7 (`tab:resources`) | Trainable vs frozen vs cumulative parameters, importance state, measured peak memory, seconds per epoch | `reports/resource_table.json`, `reports/peak_memory.json` + `reports/peak_memory/`, `reports/memory_footprint.json`, `reports/timing_cifar100.json` |
| Table 8 (`tab:epoch-budget`) | Do the adapter baselines lose because they get fewer epochs? | `reports/ncm/cifar100/*__ep20*` |
| Sec. 5.6, Sec. 5.5 | Equivalence to EWC-LoRA (TOST), run-to-run spread, floor tests, orthogonality variants, Fisher-weighted vs uniform penalty, $T{=}50$ | `reports/tost_*.json`, `reports/repro_*.json`, `reports/floor_tests.json`, `reports/orth_variant_tests.json`, `reports/eq_contrast.json`, `reports/revision_extra_tests.json` |

The unfavourable results are in the repository as well, and are stated in the paper rather
than omitted: the isotropic (uniformly weighted) penalty ties the Fisher-weighted one;
random-subspace protection is not significantly worse; at $T{=}50$ the two methods do not
separate; and the strict sequential protocol costs FOLoRA 8.5 points. Sec. 5.6 collects them.

The response to the reviewers is not part of this repository.

## Installation

Python 3.11 is what the paper's runs used. `torch` and `torchvision` must come from the CUDA
index used for the experiments; the versions in `requirements.txt` are lower bounds, and the CPU
builds will not complete the runs. Install them first so that the requirements below do not pull
the CPU builds:

```bash
pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

## Reproducing the paper's numbers

The paper's tables can be checked without a GPU, directly from the evaluation outputs in
`reports/`:

```bash
# Tables in paper/tables/ are consistent with reports/
python -m scripts.make_paper_tables --check

# Tables added in the revision: true forgetting (Table 5) and the RanPAC re-scoring (Table 4)
python -m scripts.make_bwt_table --check
python -m scripts.make_posthoc_table --check

# p-values quoted in the text are consistent with reports/
python -m scripts.significance --benchmark all --check-paper

# Regenerate the tables and the significance reports
python -m scripts.make_paper_tables
python -m scripts.make_bwt_table
python -m scripts.make_posthoc_table
python -m scripts.significance --benchmark all

# Tables 7 and 8, the equivalence and reproducibility analyses in Sec. 5.5-5.6
python -m scripts.resource_table
python -m scripts.tost_equivalence --benchmark cifar100 --out reports/tost_cifar100.json
python -m scripts.tost_equivalence --benchmark imagenetr --out reports/tost_imagenetr.json
python -m scripts.repro_spread --refs ""    # the run-to-run spread of the repeated runs only
```
```

Full retraining (GPU required) is driven by `scripts/run_single.py` and the queue in
`scripts/run_full_queue.py`:

```bash
# Paper configuration: lambda = 3, top-k = 64 for folora_v2
python -m scripts.run_single --benchmark cifar100 --num_tasks 20 \
       --method folora_v2 --seed 0 --folora_lambda 3 --folora_topk 64

# Frozen-feature nearest-class-mean evaluation used in the paper
python -m scripts.eval_ncm_sweep --benchmark cifar100
python -m scripts.eval_ncm_sweep --benchmark imagenetr
```

Datasets are downloaded by `scripts/download_cifar.py` (CIFAR-100) and the ImageNet-R loader.

## Two conventions named `final_acc_cil`

`experiments/*/results.json` and `reports/ncm/*.json` both carry a field called `final_acc_cil`,
but they are different quantities:

- `experiments/*/results.json` — accuracy of the linear classification head (an internal
  bookkeeping value, on the order of 0.07);
- `reports/ncm/*.json` — accuracy of a nearest-class-mean classifier on frozen features, the
  protocol used throughout the paper (on the order of 0.76).

All numbers in the manuscript and in `paper/tables/` come from the second convention.

## Methods

| Method | Description |
|---|---|
| `seq` | Sequential LoRA without any protection (lower bound) |
| `ewc` | EWC-LoRA (diagonal Fisher penalty on the LoRA parameters) |
| `olora` | O-LoRA (per-task adapter with orthogonal initialization) |
| `inflora` | InfLoRA (per-task subspace projected against previous subspaces) |
| `l2p` | L2P (prompt pool with cosine top-k selection and key loss) |
| `coda` | CODA-Prompt (prompt components with attention-based weighting; implemented without its orthogonality regularizer, a faithful simplification of the original) |
| `folora` | FOLoRA v1 (historical variant; not used for the reported results) |
| `folora_v2` | FOLoRA, the method of the paper |

The paper's configuration is `folora_v2` with `lambda = 3` and `top-k = 64`, with LoRA rank 16
and scale 16 on a frozen ViT-B/16 backbone. The defaults in `src/peft_cl/utils/config.py`
(`300` and `0`) are historical values from an earlier grid search; pass `--folora_lambda 3
--folora_topk 64` explicitly to reproduce the paper.

## Tests

```bash
python -m pytest tests -q
```

## What this repository does not contain

The CIFAR-100 and ImageNet-R datasets and the trained checkpoints are not redistributed here.
Datasets are public benchmarks and can be downloaded with the scripts above; the run
configurations, per-run results, and training logs that produced every reported number are
included under `experiments/`.

## Citation

Jiang Tao. *FOLoRA: Fisher-Orthogonalized Low-Rank Adaptation for Continual Learning.*
Manuscript NEUCOM-D-26-24287, under review at Neurocomputing, 2026. The LaTeX source and the
compiled manuscript are in `paper/`.

## License

MIT. See `LICENSE`.
