# FlagQuantum IR Phase 2 Batch A 性能修复复核

状态：**Awaiting successor-budget approval — 不允许 Batch A 退出或进入 Batch B**

日期：2026-09-02

原 blocker：`IR2A-PERF-001`

## 1. 结论

授权范围内的实现修复、等价性证明和 10/100/1K/10K 测量已经完成。1K gate 完整流水线
p95 从 67.629 ms 降至 35.769 ms（降低 47.11%），峰值内存从 2,726,417 bytes 降至
2,505,077 bytes（降低 8.12%）。

修复有效，但原 15 ms / 2 MiB 的 1K 预算仍未通过。因此：

- 原预算文件保持不变；
- 原 blocker 仍有效；
- 后继预算只是候选，尚未获批；
- 不允许 Batch A 退出、进入 Batch B、接入默认路径或删除 legacy compiler。

## 2. 修复内容

1. immutable `QuantumModule` 与 import artifact 的 program identity 缓存；
2. pass pipeline digest 缓存；
3. immutable opcode registry 单例化，并将 opcode 查询从线性扫描改为 O(1)；
4. production benchmark 使用 fused private pipeline，只在入口和出口做完整验证；
5. 独立 pass 仍保留逐 pass verifier，可单独测试；
6. fused 与独立流水线通过 module、canonical encoding、program identity 和 lowering 结果精确等价测试；
7. 增加 import/pass/lower/identity 分段测量，不改变完整流水线计时边界。

以上均为 `_compiler` 私有实现；没有改变 public API、默认 compiler/runtime 路径或失败语义。

## 3. 实测结果

环境：`flagquantum-dev:pr-check`，Python 3.12.13，Torch 2.13.0+cpu，单线程；每档 3 次
warmup、7 次测量。

| Gates | 完整 p95 | 分段观测 total p95 | Peak bytes | 原预算结果 |
| ---: | ---: | ---: | ---: | --- |
| 10 | 0.554 ms | 0.493 ms | 44,480 | 通过 |
| 100 | 5.366 ms | 3.893 ms | 257,326 | latency 失败 |
| 1,000 | 35.769 ms | 50.915 ms | 2,505,077 | latency、memory 失败 |
| 10,000 | 434.161 ms | 517.262 ms | 25,075,043 | latency、memory 失败 |

分段观测包含 Python 分段计时扰动，仅用于定位和预算抗波动推导；完整流水线结果用于端到端事实判断。
所有档位的 output identity 与 pipeline digest 均保持确定性。

## 4. 后继预算候选

候选采用：

```text
latency = max(原上限, 1.25 × max(完整 p95, 分段 total p95))，向上取审查边界
memory  = max(原上限, 1.25 × tracemalloc peak)，向上取二进制边界
```

| Gates | latency candidate | memory candidate |
| ---: | ---: | ---: |
| 10 | 1.25 ms | 1.25 MiB |
| 100 | 7 ms | 1.25 MiB |
| 1,000 | 65 ms | 3 MiB |
| 10,000 | 650 ms | 30 MiB |

这些值是内部回归预算，不是公开 SLA，也不表示放弃继续优化。

## 5. 需单独批准的决策

若 owner 接受测量口径与候选预算，下一步精确授权口令为：

```text
approve IR-PHASE2-BATCH-A-SUCCESSOR-BUDGET
```

该授权仅允许固化并启用后继预算 gate；仍不自动授权 Batch A exit、Batch B、默认路径变更、
public API 变更或 legacy retirement。
