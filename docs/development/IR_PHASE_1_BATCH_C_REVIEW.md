# FlagQuantum IR Phase 1 Batch C 评审

状态：**Technical complete — 等待 Batch C 退出评审**
日期：2026-09-01
实现范围：CircuitIR importer、request split、constraints、source identity、bindings

## 1. 结论

Batch C 已完成公共 `CircuitIR` 到私有 QuantumIR 的只读、fail-closed 导入路径。全部 Phase 0
manifest fixture 的声明状态均由机器测试匹配，35 个 canonical opcode 均能建立确定性线性
value 链。

本 importer 没有接入 `fq.run/fq.plan`，也没有成为公开 API。

## 2. 已完成能力

- `CircuitIR n_wires/instructions` 导入 immutable QuantumModule；
- 每个 wire 生成初始 linear value，每次 gate 产生确定后继 value；
- observables、measurements、shots、seed 和 ordering 拆入 typed execution request；
- dtype、shape、batch、logical-state shape、runtime config 拆入 constraints；
- public `content_hash` 与 internal program identity 并存，互不冒充；
- routing/source/interop 等 provenance 不污染 internal program identity；
- global phase、diagonal、MPO、Pauli、split 等已分类语义进入 typed annotation；
- `Parameter/ParameterExpression` 保持名称、operation 与依赖结构；
- `requires_grad=True` tensor 只进入稳定 binding slot，原 tensor object 不 detach/copy；
- custom unitary 验证维度、有限性、unitarity、dtype 和 trainable 边界；
- dynamic、未知 metadata、冲突 channel marker、lossy 参数明确拒绝；
- importer 不修改输入 `CircuitIR`。

## 3. Identity 边界

`internal_program_identity` 包含 module semantics、dtype/shape constraints 和 typed instruction
semantics；不包含 execution request、source location、routing provenance 或 late-bound tensor
数值。`CircuitIR.content_hash` 仍完整保留为 source identity。

因此改变 shots 不会改变 internal program identity；改变 dtype、静态常量或语义 annotation
会改变 identity；改变训练 tensor 当前数值不会改变 program identity。

## 4. 明确未做

- 未实现 CircuitIR exporter 或 round-trip；
- 未实现 analysis、PassManager 或 differential execution；
- 未接入默认 compiler/runtime；
- 未修改公共 schema、API snapshot 或 deployment/provider 合同。

## 5. 验证结果

| 检查 | 结果 |
| --- | --- |
| Internal IR | 155 passed |
| Default CI | 1122 passed, 10 skipped |
| Runtime CI | 168 passed, 31 skipped |
| Ruff | passed |
| Black | passed |

Default CI 保留一条既有 PyTorch complex module warning；Runtime 的 31 个 skip 为既有
环境/硬件条件跳过。

机器证据和精确文件 hash 见
`contracts/ir-phase1-batch-c-review-candidate.json`。

## 6. 下一暂停点

如 owner 接受本证据并希望启动 Batch D，应使用：

```text
approve IR-PHASE1-BATCH-C-EXIT-BATCH-D
```

该口令只授权 restricted CircuitIR exporter 与 canonical payload 精确 round-trip；不授权
Provider codegen、公共 API、默认 runtime、Phase 2 或生产路径切换。普通 `do`、`continue`
不等价于该授权。
