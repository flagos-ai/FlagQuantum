# FlagQuantum IR Phase 1 退出评审

状态：**Technical complete — 等待正式 Phase 1 退出审批**
日期：2026-09-02
范围：内部 `circuit_ir_v1_static` profile；不包含 Phase 2 或生产路径切换

## 1. 结论

Phase 1 的 P1-001～P1-009 技术证据已经闭环：静态 `CircuitIR` 可以进入私有、typed、
verified、deterministic 的 QuantumIR；支持范围内可精确 round-trip，并可通过显式测试桥
从 verified QuantumIR operation/value chain 降回现有 executor 进行科学语义差分。

Stable Core、`CircuitIR` 1.0、默认 `fq.run/fq.plan` 和部署合同未改变。

## 2. 已完成能力

- immutable type/value/operation/module 与线性 qubit value；
- schema registry、verifier、source location 和 structured diagnostics；
- 35 个 canonical opcode 的静态 profile importer；
- program 与 execution request 分离；
- Parameter/ParameterExpression 和外部 autograd binding identity；
- sealed restricted exact round-trip；
- DefUse、QubitLifetime、revision-aware analysis cache；
- PassManager、preserved analysis、pipeline digest 与无语义变化示例 Pass；
- 显式、可整体删除的 differential bridge；
- state、expectation、measurement、wire/result ordering、custom matrix、dtype 和 gradient parity；
- 固定 seed property/negative tests；
- 性能预算、公共 API、架构边界和默认路径零影响门。

## 3. 差分桥的真实边界

候选路径不是简单再次执行原 source。它从 verified QuantumIR 的 operation、linear value
chain、wire ownership、parameter/custom matrix attribute 和 runtime binding 生成候选
`CircuitIR`，再交给现有 executor。密封 source envelope 仅保留 Phase 1 尚未独立 codegen
的 classified metadata 与 execution request 表示。

因此本证据验证当前静态 importer 与 test lowering 的科学语义，但不代表：

- 任意变换后 QuantumIR 的通用 exporter；
- Provider/native codegen；
- 新 runtime 或默认路径切换；
- dynamic、control-flow、timing、pulse 或 Phase 2 ProgramIR。

## 4. 性能门

预算没有放宽。首次失败后，通过正式 performance-remediation 授权，仅调整对象分配时序和
空 metadata/parameter fast path，并以 successor evidence 保留原 Batch C candidate。

| Gates | Import+verify p95 | 预算 | Peak memory | 预算 | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 0.259 ms | 1.0 ms | 39,816 B | 1,048,576 B | PASS |
| 100 | 1.142 ms | 1.5 ms | 130,678 B | 1,048,576 B | PASS |
| 1,000 | 11.923 ms | 12.5 ms | 1,055,846 B | 1,572,864 B | PASS |
| 10,000 | 93.778 ms | 120 ms | 9,956,478 B | 12,582,912 B | PASS |

这是指定 Docker CPU 环境的内部门，不是公共 SLA、GPU 或分布式性能声明。

## 5. 验证结果

| 检查 | 结果 |
| --- | --- |
| Internal IR | 220 passed |
| Default CI | 1187 passed, 10 skipped |
| Runtime CI | 168 passed, 31 skipped |
| Performance | 10/100/1K/10K 全部通过 |
| Public API snapshot | passed |
| Architecture / repository hygiene / docs source | passed |
| Ruff / Black | passed |

机器证据与精确文件 hash 见
`contracts/ir-phase1-exit-review-candidate.json`。

## 6. 回滚与下一暂停点

默认路径不依赖 `_compiler`。撤回 Phase 1 只需删除 `_compiler` 及其内部测试/benchmark，
不需要迁移用户代码、公共序列化数据或 legacy compiler/runtime。

如 API/compiler/runtime/training owner 接受本证据并仅批准 Phase 1 退出，应使用：

```text
approve IR-PHASE1-EXIT
```

该口令不授权 Phase 2、公共 IR/Pass API、默认或生产路径切换、Provider codegen 或现有
优化器迁移。
