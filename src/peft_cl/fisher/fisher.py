"""经验 Fisher 估计（逐样本梯度）。

关键设计：两个函数都用「逐样本梯度」（batch=1 反向累计）而非「均值梯度」。
原因：均值梯度 ḡ 在接近收敛时退化为 0，其平方/外积 E[ḡ²]、E[ḡ]E[ḡ]ᵀ 会把重要性
低估到接近零；逐样本二阶矩 E_s[g_s²]、E_s[g_s g_sᵀ] 正是 Fisher 的低秩限制，
不随收敛消失，才能真实反映「哪些方向重要」。

- diagonal_fisher      逐样本对角 Fisher（EWC 基线用）；
- output_kernel_fisher 逐样本输出方向二阶核（FOLoRA 用，out×out，out=768 紧凑）。

delta 的梯度能拿到，依赖 adapters.lora 里 forward 对 delta 调了 retain_grad()。
"""

import torch
import torch.nn.functional as F

from ..adapters.lora import iter_lora


def diagonal_fisher(model, loader, n_samples, device):
    """逐样本对角 Fisher：F_diag = E_s[g_s²]，g_s 为逐样本参数梯度。

    返回 dict[参数名 -> 同形状 tensor]。调用方自行过滤（如只留 LoRA 参数）。
    """
    model.eval()
    amp = device.type == "cuda"
    fishers = {}
    counts = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for s in range(x.shape[0]):
            if counts >= n_samples:
                break
            xs, ys = x[s:s + 1], y[s:s + 1]
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(xs)
                F.cross_entropy(logits, ys).backward()
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fishers.setdefault(name, torch.zeros_like(p))
                    fishers[name].add_(p.grad.detach().float() ** 2)
            counts += 1
        if counts >= n_samples:
            break
    if counts == 0:
        return {}
    return {k: v / counts for k, v in fishers.items()}


def output_kernel_fisher(model, loader, n_samples, device):
    """逐样本输出方向二阶核：对每个 LoRA 层返回 C = E_s[g_s·g_sᵀ]（out×out）。

    返回与 iter_lora(model) 顺序对齐的 list[Tensor|None]（数据为空时为 None）。
    """
    model.eval()
    amp = device.type == "cuda"
    loras = list(iter_lora(model))
    kernels = [None] * len(loras)
    counts = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for s in range(x.shape[0]):
            if counts >= n_samples:
                break
            xs, ys = x[s:s + 1], y[s:s + 1]
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(xs)
                F.cross_entropy(logits, ys).backward()
            for i, lora in enumerate(loras):
                delta = lora._last_delta
                if delta is not None and delta.grad is not None:
                    g = delta.grad.detach().float()      # (1, seq, out)
                    g = g.reshape(-1, g.shape[-1])       # (seq, out)，token 当独立样本
                    C = g.t() @ g                        # (out, out)，fp32 保证精度
                    kernels[i] = C if kernels[i] is None else kernels[i] + C
            counts += 1
        if counts >= n_samples:
            break
    if counts == 0:
        return kernels
    return [k / counts if k is not None else None for k in kernels]


def param_gradients(model, loader, n_samples, device):
    """逐样本「参数」梯度：对每个 LoRA 层返回 G (n × d) 的逐样本参数梯度矩阵。

    与 output_kernel_fisher 的关键区别：这里取「参数梯度」g_θ = (g_A, g_B) 而非
    「输出梯度」g_δ。理论遗忘上界是 δθᵀ F_θ δθ，其中 F_θ = E[g_θ g_θᵀ]；而
    g_θᵀ δθ = g_δᵀ δW x（链式法则），天然带输入 x。v1 只用 E[g_δ g_δᵀ] 把 x
    积分掉了，导致保护方向与真实遗忘方向不对齐——这是 v1 正则失效的根因。

    返回 list[Tensor|None]，与 iter_lora(model) 顺序对齐；每项 (n, d)，
    d = numel(lora_A) + numel(lora_B)，fp32。
    """
    model.eval()
    amp = device.type == "cuda"
    loras = list(iter_lora(model))
    grads = [[] for _ in loras]
    counts = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for s in range(x.shape[0]):
            if counts >= n_samples:
                break
            xs, ys = x[s:s + 1], y[s:s + 1]
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(xs)
                F.cross_entropy(logits, ys).backward()
            for i, lora in enumerate(loras):
                gA = lora.lora_A.weight.grad
                gB = lora.lora_B.weight.grad
                if gA is None or gB is None:
                    continue
                g = torch.cat([gA.detach().float().reshape(-1),
                               gB.detach().float().reshape(-1)])
                grads[i].append(g)
            counts += 1
        if counts >= n_samples:
            break
    return [torch.stack(g) if g else None for g in grads]
