# P0-1 重训完成记录（`*_fix1`）

生成时间：2026-10-05 09:25。用时不手抄，由每个 run 的 `run.log` 首末时间戳现算。

- 目标 **35** 个 run，**全部完成**：第一个 10-03 22:20 落盘，最后一个 10-05 09:08。
- Run 时间合计 **2056.9 min = 34.3 h**；墙钟 10-03 22:20 → 10-05 09:08 = **34.8 h**
  （差值来自两次 launcher 重启的间隙与队列重排，非睡眠丢小时；巡检期间无一次触发
  「相邻落盘间隔 > 基准 1.5 倍」或「run.log > 15 min 无新行」的判据）。
- **P0-1 修复核实**：抽检的 fix1 checkpoint 的 adapter 下标集合均为 `(0, 19, 20)`、
  共 480 个 adapter 键；旧 run 为 `(19, 19, 1)`（olora 24 键、inflora 12 键）——
  即「只存末个任务的 adapter」的原始症状。评估侧加载时打印「载入 480 个张量」，一致。
- 旧 run 目录、旧 NCM 缓存、旧派生聚合全部保留：派生聚合移入
  `reports/_archive_pre_fix1/`，`reports/ncm_matrix/*/olora__default*.json` 亦移入该目录
  （画图脚本要求同一方法内 tag 唯一，故必须让位给 `default_fix1`）。
  数据完整性铁律：不删、不覆盖。

| # | run | 用时 (min) | 落盘时刻 |
|---|---|---|---|
| 1 | cifar100/olora/default_fix1/seed0 | 65.2 | 10-03 22:20 |
| 2 | cifar100/olora/default_fix1/seed1 | 66.0 | 10-04 00:28 |
| 3 | cifar100/olora/default_fix1/seed4 | 65.3 | 10-04 01:33 |
| 4 | cifar100/olora/default_fix1/seed5 | 65.3 | 10-04 02:39 |
| 5 | cifar100/olora/default_fix1/seed6 | 65.4 | 10-04 03:44 |
| 6 | cifar100/olora/default_fix1/seed7 | 65.3 | 10-04 04:50 |
| 7 | cifar100/olora/default_fix1/seed8 | 65.4 | 10-04 05:55 |
| 8 | cifar100/olora/default_fix1/seed9 | 65.5 | 10-04 07:01 |
| 9 | imagenetr/olora/default_fix1/seed0 | 44.9 | 10-04 07:46 |
| 10 | imagenetr/olora/default_fix1/seed1 | 43.5 | 10-04 08:30 |
| 11 | imagenetr/olora/default_fix1/seed2 | 43.5 | 10-04 09:13 |
| 12 | imagenetr/olora/default_fix1/seed3 | 44.0 | 10-04 09:57 |
| 13 | imagenetr/olora/default_fix1/seed4 | 43.7 | 10-04 10:41 |
| 14 | cifar100/olora/default_fix1/seed2 | 65.5 | 10-04 12:16 |
| 15 | cifar100/olora/default_fix1/seed3 | 65.9 | 10-04 13:23 |
| 16 | cifar100/olora/olora_orth_l0.1_fix1/seed0 | 65.6 | 10-04 14:28 |
| 17 | cifar100/olora/olora_orth_l0.1_fix1/seed1 | 65.8 | 10-04 15:34 |
| 18 | cifar100/olora/olora_orth_l0.1_fix1/seed2 | 65.8 | 10-04 16:40 |
| 19 | cifar100/olora/olora_orth_l1_fix1/seed0 | 65.8 | 10-04 17:46 |
| 20 | cifar100/olora/olora_orth_l1_fix1/seed1 | 65.9 | 10-04 18:52 |
| 21 | cifar100/olora/olora_orth_l1_fix1/seed2 | 65.7 | 10-04 19:58 |
| 22 | cifar100/inflora/default_fix1/seed0 | 69.3 | 10-04 21:07 |
| 23 | cifar100/inflora/default_fix1/seed1 | 71.0 | 10-04 22:19 |
| 24 | cifar100/inflora/default_fix1/seed2 | 81.4 | 10-04 23:40 |
| 25 | cifar100/inflora/default_fix1/seed3 | 72.4 | 10-05 00:53 |
| 26 | cifar100/inflora/default_fix1/seed4 | 72.1 | 10-05 02:05 |
| 27 | imagenetr/olora/olora_orth_l0.1_fix1/seed0 | 48.0 | 10-05 02:53 |
| 28 | imagenetr/olora/olora_orth_l0.1_fix1/seed1 | 46.0 | 10-05 03:39 |
| 29 | imagenetr/olora/olora_orth_l0.1_fix1/seed2 | 45.9 | 10-05 04:25 |
| 30 | imagenetr/olora/olora_orth_l1_fix1/seed0 | 46.0 | 10-05 05:11 |
| 31 | imagenetr/olora/olora_orth_l1_fix1/seed1 | 46.2 | 10-05 05:58 |
| 32 | imagenetr/olora/olora_orth_l1_fix1/seed2 | 46.0 | 10-05 06:44 |
| 33 | imagenetr/inflora/default_fix1/seed0 | 47.9 | 10-05 07:32 |
| 34 | imagenetr/inflora/default_fix1/seed1 | 48.0 | 10-05 08:20 |
| 35 | imagenetr/inflora/default_fix1/seed2 | 47.7 | 10-05 09:08 |

**一处需要记录的运行事实**：第 24 行（`cifar100/inflora/default_fix1/seed2`）用时 81.4 min，
比同组其它 run 高约 10 min。查其 `run.log` 的**内部最大间隔只有 6.1 min**，
故是真实算力耗时（inflora 的 checkpoint 体积为 olora 的 2–3 倍，落盘更重），
不是睡眠丢小时——两者在日志上的表现不同，判据见 `FOLoRA决策日志.md` §26.53 的巡检口径。
