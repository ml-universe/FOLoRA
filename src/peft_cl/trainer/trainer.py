"""持续学习训练循环（含断点续训）。

断点续训设计（应对学校断电）：
- 每训练完一个任务立即把「模型可训参数 + 方法状态 + RNG 状态 + 类顺序 + 准确率矩阵」
  落盘为 checkpoint.pt；每个任务只有 ~1-2 分钟，断电最多损失当前任务，重启即可续跑。
- 新对话/新进程只需 `python -m scripts.run_single --resume`，trainer 会自动检测
  checkpoint 并从「上一个已完成任务 + 1」继续，产出与一气呵成完全一致的结果。
- 全部完成时写 results.json（含 finished=true），网格调度器据此跳过已完成实验。
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.datasets import load_cifar
from ..data.split import ContinualSplit, make_class_order
from ..backbone.vit import build_vit
from ..methods import build_method
from ..metrics.metrics import (average_incremental_accuracy, evaluate,
                               final_average_accuracy, forgetting)
from ..utils.config import CLConfig
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logger
from ..utils.paths import run_dir
from ..utils.persist import (capture_base_params, missing_persistent_keys,
                             persistent_model_state)
from ..utils.seed import get_rng_state, set_rng_state, set_seed

BENCHMARK_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenetr": 200}


class CLTrainer:
    """一次持续学习实验的训练器。构造后调用 run()。"""

    def __init__(self, config: CLConfig, resume: bool = False):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dir = run_dir(config)
        self.ckpt_path = self.dir / "checkpoint.pt"
        self.results_path = self.dir / "results.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(str(self.dir / "run.log"))

        if config.benchmark not in BENCHMARK_CLASSES:
            raise ValueError(f"unsupported benchmark: {config.benchmark}")
        num_classes = BENCHMARK_CLASSES[config.benchmark]
        self.classes_per_task = num_classes // config.num_tasks

        set_seed(config.seed)
        self._build_data()
        self._build_model()

        self.acc_cil = [[0.0] * config.num_tasks for _ in range(config.num_tasks)]
        self.acc_til = [[0.0] * config.num_tasks for _ in range(config.num_tasks)]
        self.start_task = 0

        if resume and self.ckpt_path.exists():
            self._load_checkpoint()

    # ---------- 构建 ----------

    def _build_data(self) -> None:
        cfg = self.config
        train, test = load_cifar(cfg.benchmark, cfg.data_root, cfg.image_size)
        num_classes = BENCHMARK_CLASSES[cfg.benchmark]
        order = make_class_order(num_classes, cfg.num_tasks, cfg.seed)
        self.split = ContinualSplit(train, test, order)

    def _build_model(self) -> None:
        cfg = self.config
        self.model, self.head = build_vit(cfg.backbone, self.classes_per_task, pretrained=True)
        # 注入方法**之前**记下预训练主干的参数对象，供 checkpoint 筛选排除它们
        # （见 utils/persist.py 与 _save_checkpoint 的注释）。
        self._base_params = capture_base_params(self.model)
        # 先注入 LoRA（此时模型在 CPU），再统一移到 GPU，避免 LoRA 层与主干不同设备
        self.method = build_method(cfg.method, self.model, cfg)
        self.model.to(self.device)
        # `.to()` 正常只改 `.data` 而不换 Parameter 对象，身份判据因此仍然成立。
        # 但换了版本的 torch 若改成替换参数对象，_base_params 会整体失效 ——
        # 后果是主干 344 MB 被当成「新参数」写进每个 checkpoint（不报错、只是暴涨）。
        # 这里做一个廉价的哨兵断言，把这个静默失效变成显式异常。
        n_alive = sum(1 for p in self.model.parameters()
                      if self._base_params.get(id(p)) is p)
        if n_alive != len(self._base_params):
            raise RuntimeError(
                f"build_method / .to(device) 之后预训练主干参数对象身份发生变化"
                f"（{n_alive}/{len(self._base_params)} 仍匹配），checkpoint 的"
                f"「非主干参数」判据会失效。请检查 torch 版本的 module 转换行为。")

    # ---------- 训练 ----------

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _train_task(self, task_id: int) -> None:
        cfg = self.config
        self.head.expand((task_id + 1) * self.classes_per_task)

        train_ds = self.split.task_dataset(task_id, "train")
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers)

        self.method.before_task(task_id, train_loader)
        opt = torch.optim.AdamW(self._trainable_params(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)

        self.model.train()
        amp = cfg.use_amp and self.device.type == "cuda"
        for epoch in range(cfg.epochs):
            running_ce = running_reg = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                # 正则项在 fp32 下算（避免 bf16 精度损失）；CE 走 bf16 autocast 提速
                reg = self.method.regularization_loss()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = self.model(x)
                    ce = F.cross_entropy(logits, y)
                    loss = ce + reg
                    loss.backward()
                opt.step()
                running_ce += ce.item()
                running_reg += reg.item() if isinstance(reg, torch.Tensor) else reg
            if self.logger:
                self.logger.info(
                    f"task {task_id} epoch {epoch}: ce={running_ce / max(1, len(train_loader)):.4f} "
                    f"reg={running_reg / max(1, len(train_loader)):.4f}")

        self._evaluate(task_id)

        # 诊断探针必须在 after_task **之前**：此刻 ref_params 还是 θ_{t-1}、
        # acc_grads 还没并入本任务（= F̄_{<t}）、模型已是 θ_t。after_task 一跑，
        # ref_params 就被覆盖成 θ_t，三元组永远丢失。默认关闭，见 config.bound_probe。
        if getattr(self.config, "bound_probe", False):
            try:
                from peft_cl.utils.bound_probe import probe_bound_terms, append_jsonl
                rec = probe_bound_terms(self.model, self.method, self.split,
                                        self.device, task_id)
                if rec is not None:
                    append_jsonl(self.dir / "bound_terms.jsonl", rec)
                    if self.logger:
                        cq = rec["cubic_over_quad"]
                        self.logger.info(
                            f"bound probe task {task_id}: cubic/quad={cq:+.3f} "
                            f"fisher/fit_b={rec['fisher_over_fitted']:.3f} "
                            f"delta_norm={rec['delta_norm']:.4f}")
            except Exception as e:            # 诊断绝不能把训练搞崩
                if self.logger:
                    self.logger.warning(f"bound probe failed at task {task_id}: {e!r}")

        self.method.after_task(task_id, train_loader, self.device)

    @torch.no_grad()
    def _evaluate(self, task_id: int) -> None:
        cfg = self.config
        self.model.eval()
        for j in range(task_id + 1):
            test_loader = DataLoader(self.split.task_dataset(j, "test"),
                                     batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers)
            lo, hi = j * self.classes_per_task, (j + 1) * self.classes_per_task
            self.acc_cil[task_id][j] = evaluate(self.model, test_loader, self.device, None)
            self.acc_til[task_id][j] = evaluate(self.model, test_loader, self.device, (lo, hi))
        acc_so_far = sum(self.acc_cil[task_id][:task_id + 1]) / (task_id + 1)
        self.logger.info(f"task {task_id} done: avg_acc_so_far(CIL)={acc_so_far:.4f}")

    # ---------- 断点续训 ----------

    def _save_checkpoint(self, task_id: int) -> None:
        # model_state 保存的是「非预训练主干」的全部参数，**不能按 requires_grad 过滤**。
        # O-LoRA / InfLoRA 会把历史任务的 adapter 冻结，按 requires_grad 过滤会把它们
        # 整批丢掉，checkpoint 里只剩最后一个任务的 adapter；评估与续训都会因此拿到
        # 一个「只学了最后一个任务」的模型，且 strict=False 全程不报错。详见
        # utils/persist.py 的模块 docstring。
        state = {
            "config": self.config.to_dict(),
            "task_id": task_id,
            "class_order": self.split.class_order,
            "model_state": persistent_model_state(self.model, self._base_params),
            "method_state": self.method.state_dict(),
            "rng_state": get_rng_state(),
            "acc_cil": self.acc_cil,
            "acc_til": self.acc_til,
            "finished": False,
        }
        # 原子写：先写 .tmp、fsync 落盘、再 os.replace，避免断电/休眠在写盘中途杀死
        # 进程导致 checkpoint.pt 截断损坏（曾踩坑：ImageNet-R folora seed0 在 task8 被杀、
        # 53MB checkpoint 只写了一半，resume 报 "failed finding central directory"）。
        # fsync 不能省：只做 os.replace 而数据还在页缓存里时，断电后文件仍是半截的。
        tmp = self.ckpt_path.with_name(self.ckpt_path.name + ".tmp")
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.ckpt_path)
        self.config.save(str(self.dir / "config.json"))

    def _load_checkpoint(self) -> None:
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.start_task = ckpt["task_id"] + 1
        self.split.set_class_order(ckpt["class_order"])
        self.head.expand((ckpt["task_id"] + 1) * self.classes_per_task)
        # 先重建「逐任务动态创建」的 adapter（O-LoRA/InfLoRA），否则 model_state 里
        # 那些 adapters.N.* 的键在模型中不存在，会被 strict=False 静默丢弃
        self.method.rebuild_for_resume(ckpt["task_id"])
        # 校验 checkpoint 覆盖了全部应在册的参数。缺失 = 它由旧版 trainer 写出
        # （按 requires_grad 过滤，冻结的历史 adapter 未入库），续训会带着随机初始化的
        # 历史 adapter 继续训练、产出「看着正常」的结果。宁可在这里硬失败。
        missing = missing_persistent_keys(self.model, self._base_params,
                                          ckpt["model_state"])
        if missing:
            sample = sorted(missing)[:4]
            raise RuntimeError(
                f"checkpoint 缺少 {len(missing)} 个应在册的参数（例：{sample}）。"
                f"该文件由旧版代码写出（按 requires_grad 过滤存盘，O-LoRA/InfLoRA "
                f"的历史 adapter 从未入库），**不能续训**——续训会让那些任务回到"
                f"随机初始化。请从头重训该 run（不要删除旧产物，留作证据）。")
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.to(self.device)
        self.method.load_state_dict(ckpt["method_state"])
        set_rng_state(ckpt["rng_state"])
        self.acc_cil = ckpt["acc_cil"]
        self.acc_til = ckpt["acc_til"]
        self.logger.info(f"resumed from checkpoint, last completed task={ckpt['task_id']}, "
                         f"next task={self.start_task}")

    # ---------- 主流程 ----------

    def run(self) -> None:
        for task_id in range(self.start_task, self.config.num_tasks):
            self.logger.info(f"=== task {task_id}/{self.config.num_tasks - 1} ===")
            self._train_task(task_id)
            self._save_checkpoint(task_id)
        self._finalize()

    def _finalize(self) -> None:
        results = {
            "config": self.config.to_dict(),
            "acc_cil": self.acc_cil,
            "acc_til": self.acc_til,
            "final_acc_cil": final_average_accuracy(self.acc_cil),
            "final_acc_til": final_average_accuracy(self.acc_til),
            "forgetting_cil": forgetting(self.acc_cil),
            "forgetting_til": forgetting(self.acc_til),
            "incremental_acc_cil": average_incremental_accuracy(self.acc_cil),
            "finished": True,
        }
        # 原子写：结果文件一旦被截断，调度器会把它判成「未完成」而重跑整个 run
        atomic_write_json(self.results_path, results)
        # 标记 checkpoint 完成
        if self.ckpt_path.exists():
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            ckpt["finished"] = True
            tmp = self.ckpt_path.with_name(self.ckpt_path.name + ".tmp")
            with open(tmp, "wb") as f:
                torch.save(ckpt, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.ckpt_path)
        self.logger.info(
            f"finished. final_acc_cil={results['final_acc_cil']:.4f} "
            f"forgetting_cil={results['forgetting_cil']:.4f} "
            f"final_acc_til={results['final_acc_til']:.4f}")
