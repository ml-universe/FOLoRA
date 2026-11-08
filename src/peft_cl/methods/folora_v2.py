"""FOLoRA v2：参数空间 Fisher 低秩投影（修复 v1 输出空间正则失效）。

v1 的问题：正则惩罚「输出方向投影」u_jᵀ δW，u_j 是输出 Fisher 核 E[g_δ g_δᵀ] 的
top 方向。但理论遗忘上界是 δθᵀ F_θ δθ，F_θ 是「参数梯度」的 Fisher E[g_θ g_θᵀ]，
两者差一个输入 x（链式法则 g_θᵀ δθ = g_δᵀ δW x）。v1 把 x 积分掉了，保护方向与
真实遗忘方向不对齐，实测正则对训练几乎 no-op（遗忘 ≈ seq）。

v2：直接估「参数梯度」g_θ = (g_A, g_B) 的逐样本外积，维护累积 Fisher 的低秩
近似 Ĝ（top-k 主方向，按奇异值加权做增量 PCA），正则 Ω = λ ‖Ĝ δθ‖²，
δθ = θ − θ_ref（θ 为展平的 A、B，θ_ref 为上一任务快照）。这正好最小化理论的
δθᵀ F_θ δθ 上界，且比 EWC 的对角 Fisher 多了参数相关性（低秩全协方差）——正是
论文「Fisher 加权低秩投影」卖点的正确落地。
"""

import torch

from ..adapters.lora import inject_lora
from ..fisher.fisher import param_gradients
from .base import CLMethod


def _topk_directions(G: torch.Tensor, k: int) -> torch.Tensor:
    """G (m×d) 的 top-k 主方向（按奇异值加权，等价于低秩近似 G ≈ Ũ Σ Vᵀ）。

    返回 (kk, d)，每行 σ_j v_jᵀ（含能量权重）。用 gram 矩阵 G Gᵀ (m×m) 的谱分解求
    top-k（m << d 时高效），保留 σ_j 权重使后续可与新任务原始梯度拼接做增量 PCA。
    """
    gram = G @ G.t()  # (m, m)
    _, eigvecs = torch.linalg.eigh(gram)  # 升序
    kk = min(k, gram.shape[0])
    return eigvecs[:, -kk:].t() @ G  # (kk, d)


class FOLoRAv2Method(CLMethod):
    name = "folora_v2"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.loras = inject_lora(model, config.lora_rank, config.lora_alpha)
        self.acc_grads = {}   # lora idx -> (k, d) 累积参数 Fisher 低秩方向（CPU）
        self.ref_params = {}  # lora idx -> (d,) 上一任务参数字典快照（device）
        # 2026-10-05（P1-13）：acc_grads 的**设备侧缓存**。原本 regularization_loss()
        # 每个训练步都做一次 `self.acc_grads[i].to(dtheta.device)`，而该张量在**同一任务
        # 内不变**（只在 after_task 末尾被重算），等于每步白拷约 188 MB（k=64 时 12 层 ×
        # 64 × 2 × 768）H2D。缓存后每任务只拷一次；任务边界与载入 state_dict 时失效。
        self._acc_dev = {}
        # 2026-11-04 机制对照（R7 rec4）的两个缓存：
        #   _rand_basis：每个 lora 层的**固定**随机正交基 (k,d)，整个 run 不变；
        #   _rand_dev / _iso_dev：按当前任务的 σ_rms 缩放好的设备侧张量，每任务只算一次
        #     （理由同 _acc_dev：同一任务内不变，逐步重算等于白烧算力）。
        self._rand_basis = {}
        self._rand_dev = {}
        self._iso_dev = {}

    def before_task(self, task_id, train_loader=None):
        dev = self.config.device
        self.ref_params = {k: v.to(dev) for k, v in self.ref_params.items()}
        self._acc_dev = {}   # 设备/cache 可能变（resume 后第一次进任务），一律失效

    def after_task(self, task_id, train_loader, device):
        Gs = param_gradients(self.model, train_loader,
                             self.config.fisher_batches, device)
        k = self.config.folora_topk or self.config.lora_rank
        for i, G in enumerate(Gs):
            if G is None:
                continue
            if i in self.acc_grads:
                # acc_grads 存在 CPU（state_dict 序列化），cat 前移到 G 所在设备
                G = torch.cat([self.acc_grads[i].to(G.device), G], dim=0)  # (k_old + n, d)
            self.acc_grads[i] = _topk_directions(G, k).cpu()
        self._acc_dev = {}   # 保护方向已变，设备侧缓存作废（见 __init__ 的说明）
        self._rand_dev = {}
        self._iso_dev = {}
        # 记录本任务结束时的参数快照，作为下一任务正则的参考点
        for i, lora in enumerate(self.loras):
            self.ref_params[i] = self._flatten(lora).detach().clone()

    def regularization_loss(self):
        if not self.acc_grads:
            return 0.0
        loss = 0.0
        for i, lora in enumerate(self.loras):
            if i not in self.acc_grads:
                continue
            theta = self._flatten(lora)                 # (d,)，可导（含 grad_fn）
            dtheta = theta - self.ref_params[i]         # (d,)，ref 已 detach
            # 设备侧缓存：同一任务内 acc_grads 不变，故每任务只做一次 H2D 拷贝
            G = self._acc_dev.get(i)
            if G is None or G.device != dtheta.device:
                G = self.acc_grads[i].to(dtheta.device)
                self._acc_dev[i] = G
            # ---- 机制对照（2026-11-04，R7 rec4）：投影方向从哪来 ----
            # fisher（默认）：用累积 Fisher 的 top-k 主方向，即论文方法。
            # random：**同样的秩 k、同样的尺度**，但方向是固定随机正交基。它回答
            #   「收益是来自 Fisher 挑对了方向，还是只来自限制了位移的低维自由度」。
            #   ——没有这一条，「保护重要方向」和「随便保护 k 个方向」在数据上无法区分。
            # none：不投影，惩罚全部 d 维各向同性的位移（= ordinary displacement
            #   penalty / L2-SP）。它把「投影」这一操作从「把位移压小」里剥出来。
            # 两种对照的**尺度都按 RMS 对齐**（理由见 _equalize_row_weights 的注释）：
            #   Fisher 惩罚的期望 = (Σσ_j²/d)·‖δθ‖²；random 取 k 个正交方向 × σ_rms² 后
            #   期望同样是 (Σσ_j²/d)·‖δθ‖²；none 直接取系数 Σσ_j²/d，与前两者同期望。
            #   这样同一个 λ 在三种模式下可比，不需要为对照重扫 λ（重扫会引入调参混杂）。
            mode = getattr(self.config, "folora_proj", "fisher")
            if mode == "none":
                scale = self._isotropic_scale(i, G)
                loss = loss + scale * dtheta.pow(2).sum()
                continue
            if mode == "random":
                G = self._random_directions(i, G, dtheta.device)
            elif not getattr(self.config, "folora_weighted", True):
                G = self._equalize_row_weights(G)
            loss = loss + (G @ dtheta).pow(2).sum()
        return self.config.folora_lambda * loss

    @staticmethod
    def _equalize_row_weights(G: torch.Tensor) -> torch.Tensor:
        """等权消融：保持方向不变，把各方向的权重统一成平均奇异值 σ̄。

        G 的每行是 σ_j v_jᵀ，于是 ‖G δθ‖² = Σ_j σ_j² (v_jᵀ δθ)²——正则惩罚天然按
        Fisher 特征值 σ_j² 加权，这正是论文「重要性加权」卖点。等权对照要把这个
        加权去掉，但**不能简单地把行归一化**：那样会连惩罚的整体尺度一起改掉，
        原来的 λ 就不再适用，必须重新扫 λ，而「重扫过的等权 vs 没重扫的加权」
        比出来的差异分不清是加权带来的还是调参带来的。

        这里改成「方向保持、权重齐次化」——每行缩放到共同的 σ̄_rms（保留符号），
        于是两边可以在同一个 λ 下直接比较，消融干净。

        **2026-09-25 修正（此前用算术平均 σ̄，尺度没对齐）**：
        加权版总权重 = Σ_j σ_j²；齐次化到公共模长 m 后总权重 = k·m²。
        要让两者**精确相等**，须取 m = sqrt(mean(σ_j²)) = **RMS**，此时
        k·RMS² = k·mean(σ²) = Σσ_j²，恒等成立（与 σ 的分布无关）。
        曾用 m = mean(σ_j)，那只是**上界**（k·σ̄² ≤ Σσ²，仅当所有 σ_j 相等时取等），
        在真实 σ 分布（1.94–19.03）下实测比值仅 **0.248**——即等权臂的惩罚只有
        加权臂的 1/4，两臂同时差了「分配方式」和「总尺度」两件事，
        效应无法干净归因。改用 RMS 后该混杂消除。
        """
        norms = G.norm(dim=1, keepdim=True).clamp_min(1e-12)   # (k,1) 即 σ_j
        return G / norms * norms.pow(2).mean().sqrt()

    def _isotropic_scale(self, layer_idx: int, G: torch.Tensor) -> torch.Tensor:
        """`--folora_proj none`（ordinary displacement penalty）的系数 Σσ_j²/d。

        为什么不直接用 λ‖δθ‖²：λ 是按 ‖G δθ‖² 的量级标定的。δθ 有 d 维，而 G 只有
        k 行，直接去掉投影会让惩罚尺度跳变几个数量级，同一个 λ 立刻失效，等于逼人
        重扫 λ——那样比出来的差异就分不清是「投影」还是「调参」。

        推导（务必按这个来，别"看起来差不多"就写）：
          Fisher 惩罚对**各向同性**位移的期望
              E‖Gδθ‖² = Σ_j σ_j² · E[(v_jᵀδθ)²] = Σ_j σ_j² · ‖δθ‖²/d
        所以系数是 **Σσ_j² / d = mean(σ²)·k/d**，不是 mean(σ²)。
        2026-11-04 初版写成 mean(σ²)，与 Fisher/random 两臂差了 k/d = 64/61440 即
        约 1000 倍——那样 none 臂等于没加正则，会被读成「投影无效」，是纯粹的尺度错误。
        """
        c = self._iso_dev.get(layer_idx)
        if c is None:
            with torch.no_grad():
                c = G.norm(dim=1).pow(2).sum() / G.shape[1]   # Σσ_j²/d
            self._iso_dev[layer_idx] = c
        return c

    def _random_directions(self, layer_idx: int, G: torch.Tensor,
                           device) -> torch.Tensor:
        """`--folora_proj random`：固定随机正交基，按当前任务 σ_rms 缩放。

        方向在整个 run 内**固定**（由 config.seed 与该层序号决定，可复现），只在任务
        边界按当前累积 Fisher 的 σ_rms 重新缩放，以保持与 Fisher 版同尺度。
        形状/内存与 acc_grads 完全一致（k×d），所以两条臂的资源开销也一致——这正是
        R7 rec4 要的「同样的预算下，换掉方向来源」。
        """
        key = (layer_idx, G.shape[0], G.shape[1])
        Q = self._rand_basis.get(key)
        if Q is None:
            g = torch.Generator(device="cpu")
            # 10_000_019 取一个与主 seed 互质的常数，避免不同层/不同 run 撞同一组方向
            g.manual_seed(10_000_019 * (int(self.config.seed) + 1) + layer_idx)
            z = torch.randn(G.shape[1], G.shape[0], generator=g)   # (d, k)
            Q, _ = torch.linalg.qr(z)                              # 正交列
            Q = Q.t().contiguous()                                 # (k, d)
            self._rand_basis[key] = Q
        R = self._rand_dev.get(layer_idx)
        if R is None or R.device != device:
            with torch.no_grad():
                sigma_rms = G.norm(dim=1).pow(2).mean().sqrt()
            R = (Q.to(device) * sigma_rms)
            self._rand_dev[layer_idx] = R
        return R

    @staticmethod
    def _flatten(lora) -> torch.Tensor:
        return torch.cat([lora.lora_A.weight.reshape(-1),
                          lora.lora_B.weight.reshape(-1)])

    def state_dict(self):
        return {
            "acc_grads": {str(i): v.cpu() for i, v in self.acc_grads.items()},
            "ref_params": {str(i): v.cpu() for i, v in self.ref_params.items()},
        }

    def load_state_dict(self, d):
        self.acc_grads = {int(k): v for k, v in d["acc_grads"].items()}
        self.ref_params = {int(k): v for k, v in d["ref_params"].items()}
        self._acc_dev = {}
        self._rand_dev = {}
        self._iso_dev = {}
