# Decision-log entry of 23 September 2026 (verbatim excerpt)

The manuscript states (Sec. 5.1) that the main configuration and the wording thresholds for
the headline verdict were fixed in a dated project record before the remaining seeds of the
grid were run. This file is that record, reproduced without edits. It is supplied in answer
to Reviewer 8, item 6(c).

**Source.** The project's running decision log, `FOLoRA决策日志.md` --- 5,787 lines,
442,863 bytes, written in Chinese, maintained as a log of dated sections.

**Position.** This entry is section 13 and occupies lines 834--886 of that file. The next
entry (section 14, `2026-09-24 08:10`) begins at line 889.

**Timestamp.** The entry's heading records 23 September 2026, 19:45, at which point 13 of
the 23 runs of the lambda grid had finished and FOLoRA's seeds 3 and 4 had not yet been
run; section 13.3 fixes the decision rule for those remaining seeds. The file's own
filesystem modification time is 2026-10-05 13:36 (UTC+08:00) --- the log was written in
after this entry by later sections, so the modification time dates the last append, not
this entry. The timestamp of the entry itself is the one in its heading. We state this
openly rather than let the excerpt imply a stronger form of timestamping than the record
carries.

**Verbatim text.** Lines 834--886 of the source, byte-for-byte; SHA-256 of this block
(UTF-8, trailing blank lines removed) is `721b9da2a66bdc70f98ee52070da5c9baa0035a8866f046efc165dfa1a2ac649`.

---

````text
## 13. 2026-09-23 19:45：λ 网格中途（13/23）——**先冻结判决规则**

### 13.1 n=3 的现状

| 配置 | n | NCM avgACC |
|---|---|---|
| **FOLoRA v2 k=64 λ=3** | 3 | **78.37 ± 0.84** |
| FOLoRA v2 k=64 λ=10 | 3 | 77.88 ± 0.58 |
| FOLoRA v2 k=64 λ=1 | 3 | 77.29 ± 1.60 |
| EWC λ=1000 | 2 | 77.31 ± 0.15 |
| **EWC λ=100（n=10 基线）** | 10 | **77.22 ± 1.08** |
| Seq | 10 | 69.31 ± 1.92 |
| O-LoRA | 10 | 64.79 ± 2.08 |

### 13.2 配对检验（关键：**λ=3 逐个 seed 都赢**）

```
FOLoRA k64 λ=3  vs  EWC λ=100 : 逐 seed 差 = [+0.57, +0.43, +1.06]  配对 t=3.60  p=0.069
FOLoRA k64 λ=10 vs  EWC λ=100 : 逐 seed 差 = [+1.29, +0.02, -0.71]  配对 t=0.34  p=0.765
```

**注意两件事：**
1. **λ=3 的好处是「每个 seed 都赢」（三个差全为正），λ=10 则符号翻转。**
   前者是弱但一致的信号，后者是噪声——n=2 时 λ=10 看起来更好（78.03），到 n=3 就塌了。
   **不要再拿 λ=10 当主配置。**
2. p=0.069 仍 > 0.05，且这是**在 λ∈{1,3,10} 里挑最好的那个**——事后挑参会让 p 偏乐观
   （Bonferroni ×3 后约 0.21）。所以现在的正确说法是「有希望，未显著」。

### 13.3 **冻结判决规则（在 seed 3/4 落地之前写死，防 garden-of-forking-paths）**

剩下的 seed 3/4 落地后，**按下面这条规则判决，不再改口径**：

1. **论文一律报告完整 λ 曲线**（λ∈{1,3,10} 各 5 seeds 的 mean±std + 误差棒），
   **不隐藏 λ=10 不显著这件事**——曲线有拐点（λ=1 偏低、峰在 3、10 回落）
   本身就是「存在内点最优」的证据，比挑一个最高点更有说服力。
2. 主配置 = **k=64, λ=3**（判决前已定，不是看到结果才定）。
3. 检验用**配对 t（共同 seed）**，与 EWC λ=100（n=10）比：
   - 若 p<0.05 且 5 seeds 逐 seed 差全为正 → 可写 "outperforms EWC-LoRA"，
     并同时报效应量（约 +1.1 点）。
   - 若 0.05≤p<0.10 → 只写 "matches or slightly exceeds"，**不用 outperforms**。
   - 若 p≥0.10 → 回到「平价」定位（§12.1 的措辞）。
4. **绝不做的事**：不因为 λ=10 某几个 seed 高就换成 λ=10；不在 λ 上继续试探新档位
   （λ=0.3/0.1 已在 §11 砍掉，理由不变）。

### 13.4 ETA（实测 ~0.9 h/run）

> **⚠️ 已过期（2026-09-25）。** λ 网格已于 9/24 跑完，判决见 §14.2/§25.1（**平价**，
> 且 INR 上落后）。当前 ETA 与阶段顺序见 **§25.7**，不再用下面这条。

λ 网格剩 10 个 → **~9/24 04:45 跑完**，随即出 5-seed 判决。
之后：λ=0（~4h）→ 等权消融（~7h，决定 Highlights 第 2 条）→ O-LoRA 聚合对照 → InfLoRA →
T=50（30 run，~30h）。全部收尾 ~9/26。

````

---

## English translation of section 13.3 (the part the response letter quotes)

Section 13.1 records the three-seed state of the grid; 13.2 gives the paired tests at that
point (FOLoRA k=64, lambda=3 against EWC-LoRA lambda=100: per-seed differences
[+0.57, +0.43, +1.06], paired t = 3.60, p = 0.069) and notes that the differences are
positive at every seed, unlike lambda=10, whose sign flips. Section 13.4 is a wall-clock
estimate for the remaining runs and was later annotated as expired. Section 13.3 reads:

> **Frozen decision rule (written down before seeds 3/4 land, against
> garden-of-forking-paths)**
>
> Once the remaining seeds 3/4 have landed, the decision is made **by the rule below, and
> the criteria are not changed again**:
>
> 1. **The paper always reports the full lambda curve** (mean plus/minus std and error bars
>    over 5 seeds for each lambda in {1, 3, 10}), and **does not hide the fact that
>    lambda=10 is not significant** --- the curve has an interior maximum (low at lambda=1,
>    highest at 3, lower at 10), which is itself evidence that an interior optimum exists
>    and is more persuasive than picking the single highest point.
> 2. Main configuration = **k=64, lambda=3** (fixed before the decision, not chosen after
>    seeing the result).
> 3. The test is a **paired t-test (common seeds)** against EWC lambda=100 (n=10):
>    - if p<0.05 and the per-seed difference is positive at every one of the 5 seeds ->
>      "outperforms EWC-LoRA" may be written, together with the effect size (about +1.1
>      points);
>    - if 0.05 <= p < 0.10 -> write only "matches or slightly exceeds", **not**
>      "outperforms";
>    - if p >= 0.10 -> fall back to the "parity" positioning (the wording of section 12.1).
> 4. **What is never done**: switching to lambda=10 because a few of its seeds are high;
>    trying further lambda values (0.3 and 0.1 were cut in section 11, for the same
>    reason).

**Scope of the record.** The entry fixes the main configuration, the comparison point and
the wording thresholds. It does not fix the TOST margin, and it does not cover the second
benchmark. The manuscript says so in Sec. 5.1 and in the response to R8-6(c).
