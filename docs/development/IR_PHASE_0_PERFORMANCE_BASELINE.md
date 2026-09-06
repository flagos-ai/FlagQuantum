# FlagQuantum IR Phase 0 性能基线与 Phase 1 预算候选

状态：Phase 0 本地 CPU 基线已测量；Phase 1 内部预算已获 API owner 批准
Benchmark：`benchmarks/internal/ir_phase0_baseline.py`
机器结果：`tests/fixtures/internal_ir/phase0_performance_baseline.json`
活动回归预算：`tests/fixtures/internal_ir/phase1_performance_budget.json`

## 1. 测量边界

- Docker CPU profile；
- Linux aarch64、Python 3.12.13、Torch 2.13.0+cpu；
- 4 wires、complex64；
- 10、100、1K、10K gate；
- 3 次 warmup，15 次测量；
- timing 前显式收集并在测量区间关闭 Python GC；
- 单线程 Torch；
- 只代表本地开发基线，不代表 GPU、分布式或 QPU 性能。

第一次试跑曾观察到 10-gate plan 的一次性导入开销污染，因此没有保存为基线。最终
结果显式 warmup，并用独立脚本可复现。

## 2. 关键结果

| Gates | Build→IR p95 | JSON p95 | simple compile p95 | plan p95 | run p95 | plan peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.043 ms | 0.067 ms | 0.034 ms | 0.190 ms | 1.229 ms | 11,511 B |
| 100 | 0.327 ms | 0.189 ms | 0.163 ms | 0.850 ms | 7.257 ms | 70,456 B |
| 1,000 | 3.284 ms | 1.846 ms | 1.473 ms | 6.876 ms | 64.468 ms | 628,493 B |
| 10,000 | 36.265 ms | 32.251 ms | 17.378 ms | 67.788 ms | 未运行 | 6,288,342 B |

10K run 未执行，避免把长线路执行吞吐和 IR compiler 基线混为一谈。

## 3. Phase 1 预算候选

Importer + verifier 的 p95 延迟候选使用：

```text
max(1.0 ms, 1.75 × legacy plan p95)
```

并向上取便于 review 的边界。峰值内存候选使用：

```text
max(1 MiB, 2 × legacy plan peak memory)
```

| Gates | Import+verify p95 上限 | Peak memory 上限 |
| ---: | ---: | ---: |
| 10 | 1.0 ms | 1 MiB |
| 100 | 1.5 ms | 1 MiB |
| 1,000 | 12.5 ms | 1.5 MiB |
| 10,000 | 120 ms | 12 MiB |

预算为首次 Python internal IR 留出验证与 typed node 构建空间，同时阻止无界通用编译器
开销。默认 legacy `fq.plan/fq.run` 不得 import `_compiler` 或执行任何新路径工作。

## 4. 使用规则

- 当前预算状态为 `active_private_regression_budget`；
- 预算仅用于内部 import+verify 回归，不是公开 SLA；
- 测试只能比较实现结果，不能自动上调预算；
- 环境变化需保留新旧结果，不覆盖原始基线；
- 超预算必须优化、缩小范围或记录 blocker；
- 不以一次快结果证明性能，也不以 CPU 结果作扩展性声明；
- Phase 1 实现前先增加 benchmark 输出 schema test 和预算 gate test。

## 5. P0-005/P0-006 验收

- [x] 可复现 benchmark 脚本；
- [x] 10/100/1K/10K gate 基线；
- [x] warmup、iterations、环境、dtype 与线程数完整记录；
- [x] 时间、序列化大小和峰值 host memory 已记录；
- [x] 基于实测结果生成预算候选；
- [x] 预算推导关系由测试保护；
- [x] API owner 批准 Phase 1 内部性能预算；
- [ ] Phase 1 实现后通过相同环境的 budget gate。
