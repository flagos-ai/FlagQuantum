# FlagQuantum IR Phase 1 Batch F 性能修订

状态：**Passed**
日期：2026-09-02

## 修订结果

未修改既定预算，仅调整私有 importer 的对象分配时序和空 metadata/parameter fast path：

- 在构造 QuantumIR module 前计算公共 source content hash；
- 空 instruction metadata 不再重复创建临时集合和字典；
- 复用空 immutable attributes 与已限定的 operation name；
- importer 输出、identity、diagnostic、round-trip 和公共路径语义不变。

| Gates | p95 / 预算 | Peak memory / 预算 | 结果 |
| ---: | ---: | ---: | --- |
| 10 | 0.259 / 1.0 ms | 39,816 / 1,048,576 B | PASS |
| 100 | 1.142 / 1.5 ms | 130,678 / 1,048,576 B | PASS |
| 1,000 | 11.923 / 12.5 ms | 1,055,846 / 1,572,864 B | PASS |
| 10,000 | 93.778 / 120 ms | 9,956,478 / 12,582,912 B | PASS |

10K 峰值内存相对首次失败证据从 14,907,165 B 降至 9,956,478 B；p95 从
132.651 ms 降至 93.778 ms。

## 审计方式

原 Batch C candidate 保持原文件和原 hash，不追溯修改。新 importer hash 通过
`contracts/ir-phase1-batch-f-performance-remediation.json` 作为授权 successor 记录，
同时绑定原 blocker、修订授权和未变化的预算文件。

这不是公共性能 SLA，也不构成 Phase 2、默认 runtime 或生产路径授权。
