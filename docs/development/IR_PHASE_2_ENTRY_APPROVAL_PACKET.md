# FlagQuantum IR Phase 2 准入与 Batch A 授权包

状态：**Ready for owner review — 未授权实现**
日期：2026-09-02
前置条件：[IR Phase 1 完成记录](IR_PHASE_1_COMPLETION.md)

## 1. 本轮目标

Phase 2 迁移现有静态编译能力，但继续保持内部、显式 opt-in 和可整体回滚。首批只迁移
`simple_compile` 中最小的 canonicalization 子集，不接入 `fq.run`、`fq.plan`、
`compile_for_backend` 或 deployment 默认路径。

本轮不是重新设计公共 compiler API，也不以删除 legacy compiler 为目标。

## 2. 事实基线

现有静态编译链包含：

```text
CircuitIR
  -> remove identity / zero rotations
  -> cancel adjacent self-inverse gates
  -> merge adjacent rotations
  -> optional topology routing
  -> post-routing optimization
  -> QASM/QCIS emission
  -> DeploymentPackage
```

Phase 1 已提供 immutable QuantumIR、verifier、pass contract、restricted lowering、
differential bridge 和性能门，但当前 canonicalization pass 只是无变化扫描，不包含上述
legacy 优化语义。

## 3. Phase 2 批次

| 批次 | 范围 | 退出证据 |
| --- | --- | --- |
| A | identity/零旋转移除、自逆门抵消、相邻旋转合并 | 独立 golden、幂等性、state/expectation/gradient parity、确定性 identity |
| B | 目标门集 decomposition | 每条 rewrite 的精确 gate-set 合同、参数和梯度差分 |
| C | placement/routing | 全门合法、逻辑—物理映射、非对称位序、独立性能基线 |
| D | OpenQASM 2、受限 OpenQASM 3 static profile、QCIS emitter | parse/golden、字符级确定性、语义 fixture |
| E | pipeline cache、diagnostics、compilation identity | cache key、失效、错误路径和可观测性 |
| F | legacy/new deployment corpus 与 Quafu 静态用例 | 完整差分、性能门、默认路径零影响、Phase 2 退出审查 |

批次不得并行抢跑。Routing 和 emitter 必须在前序 rewrite 语义稳定后分别建立自己的基线。

## 4. Batch A 允许范围

仅允许在以下私有表面实现：

```text
flagquantum/_compiler/passes/
flagquantum/_compiler/testing/
tests/internal_ir/
tests/fixtures/internal_ir/
benchmarks/internal/
docs/development/
contracts/
```

Batch A 必须：

- 使用线性 qubit value 的显式重连，不能通过跳过 verifier 伪造删除；
- 对不可证明安全的参数、tensor、custom unitary、channel 或 measurement fail closed；
- 保留 wire、参数 identity、dtype、batch shape、measurement 和 execution request；
- 数值旋转仅按 legacy `_is_zero` 行为处理，trainable tensor 不得折叠；
- 参数表达式合并必须保持 autograd；
- 每个 pass 具有独立 descriptor、确定性 digest 和统计信息；
- restricted lowering 后与 legacy `simple_compile` 做结构与科学语义差分；
- 默认公共路径不 import 或执行 Phase 2 pass。

## 5. 明确禁止

本准入包不授权：

- 修改 Stable Core、根级导出、`CircuitIR` 1.0 或公共序列化 schema；
- 修改 `fq.run`、`fq.plan`、`compile_for_backend`、`DeploymentPackage` 的默认行为；
- 使用环境变量或 import side effect 静默切换编译器；
- 删除、转发或弃用 legacy compiler；
- 启动 Batch B–F；
- 迁移 routing、QASM/QCIS emitter 或 Provider codegen；
- 引入 TargetIR、ProgramIR、公共 PassManager 或新的稳定 API；
- 将 Quafu/backend ID、credential、queue、job 或价格写入程序 IR；
- 为通过测试而修改公共快照、Phase 0 基线或已批准预算。

## 6. 差分 Oracle

Batch A 至少验证：

| 轴 | Oracle |
| --- | --- |
| 结构 | legacy/new 降低结果 canonical instruction 序列精确一致 |
| state | 继承现有 backend/dtype 容差并显式处理全局相位 |
| expectation | 继承现有 dtype 合同 |
| gradient | forward 和参数梯度同时比较 |
| ordering | wire、measurement、result ordering 精确一致 |
| identity | 相同输入和 pipeline 得到相同 program/pipeline identity |
| failure | unsupported 输入结构化拒绝，不 fallback |
| default path | `fq.run/fq.plan/compile_for_backend` 不加载新 pass |

禁止创建覆盖所有 backend/dtype 的单一宽松容差。

## 7. 性能预算候选

当前机器回归预算：`tests/fixtures/internal_ir/phase2_batch_a_performance_budget.json`。

预算覆盖 import + verify + 已批准 pass + restricted lowering + lowered-result verification：

| Gates | p95 上限 | Peak host memory 上限 |
| ---: | ---: | ---: |
| 10 | 1.25 ms | 1.25 MiB |
| 100 | 2.0 ms | 1.25 MiB |
| 1,000 | 15 ms | 2 MiB |
| 10,000 | 140 ms | 14 MiB |

Routing、emitter 和 deployment construction 不得套用该预算，进入对应批次前必须采集独立
基线。预算超限只能优化、缩小范围或记录 blocker，不能自动放宽。

## 8. 回滚

Batch A 保持 opt-in。任何语义、梯度、identity、性能或默认路径门失败时：

1. 停止进入下一批；
2. 保留 legacy compiler 为唯一默认路径；
3. 删除或禁用 Phase 2 pass 和对应 opt-in pipeline；
4. 保留失败 fixture、诊断和审计记录；
5. 不修改公共合同吸收差异。

## 9. 审批语义

普通 `do`、`continue` 或 Phase 1 授权不等价于 Phase 2 授权。具备权限的 owner 只有在
接受机器 candidate、差分范围、性能预算和回滚条件后，才可使用：

```text
approve IR-PHASE2-ENTRY-BATCH-A
```

该口令只授权 Batch A。Batch B–F、公共路径切换和 legacy 退役均需后续独立证据与授权。
