# FlagQuantum IR Phase 1 Batch F 性能阻塞

状态：**Blocked — 不允许 Phase 1 退出**
日期：2026-09-02

## 结论

Batch F 的差分、梯度和默认路径专项已经通过，但已批准的 importer/verifier 性能预算首次
执行未通过。预算未被修改，Phase 1 不能据此宣布完成。

| Gates | p95 / 预算 | Peak memory / 预算 | 结果 |
| ---: | ---: | ---: | --- |
| 10 | 0.378 / 1.0 ms | 36,351 / 1,048,576 B | PASS |
| 100 | 1.484 / 1.5 ms | 154,471 / 1,048,576 B | PASS |
| 1,000 | 20.638 / 12.5 ms | 1,511,363 / 1,572,864 B | FAIL |
| 10,000 | 132.651 / 120 ms | 14,907,165 / 12,582,912 B | FAIL |

## 定位结果

10K gate import 完成后的 retained memory 约为 9.37 MB，低于 12 MB 预算；峰值达到约
14.93 MB。主要峰值来自 QuantumIR module 已分配后再计算公共 `CircuitIR.content_hash`，
序列化临时对象与 module 同时存活。

候选修复是先计算并保存 source hash，再分配 QuantumIR module。该调整不改变 schema、
identity 值、诊断或公共路径，但会改动 Batch C 已用精确文件 hash 冻结的 importer。

## 为什么暂停

直接修改 importer 会使已接受的 Batch C→D→E 授权哈希链失效。为避免静默改写历史
评审证据，本轮停在这里，等待明确授权建立“性能修订后的 successor evidence”，而不是
偷偷刷新旧 candidate。

建议授权口令：

```text
approve IR-PHASE1-BATCH-F-PERFORMANCE-REMEDIATION
```

该修订只允许调整 source-hash 计算时序、重新执行原预算，并建立非追溯式 successor
证据；不允许放宽预算、改变 importer 语义、修改公共 API、切换默认路径或进入 Phase 2。
