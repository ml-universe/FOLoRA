# FOLoRA: Fisher-Orthogonalized Low-Rank Adaptation for Continual Learning

Code, run configurations, and evaluation outputs for the paper *FOLoRA: Fisher-Orthogonalized
Low-Rank Adaptation for Continual Learning* (submitted to Neurocomputing).

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
| `paper/` | LaTeX source of the manuscript and the compiled `main.pdf` |
| `paper/tables/` | Generated tables; produced by `scripts/make_paper_tables.py`, do not edit by hand |
| `reports/` | Frozen-feature NCM evaluation outputs, summaries, and significance reports — the data source of every number in the paper |
| `experiments/` | Per-run configuration, results, and training logs for the runs reported in the paper (no model weights) |

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

# p-values quoted in the text are consistent with reports/
python -m scripts.significance --benchmark all --check-paper

# Regenerate the tables and the significance reports
python -m scripts.make_paper_tables
python -m scripts.significance --benchmark all
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
Manuscript submitted to Neurocomputing, 2026. The LaTeX source and the compiled manuscript are
in `paper/`.

## License

MIT. See `LICENSE`.
