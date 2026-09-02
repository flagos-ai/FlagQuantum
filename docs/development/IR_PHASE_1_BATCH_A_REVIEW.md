# FlagQuantum IR Phase 1 Batch A 评审

状态：**Technical complete — 等待 Batch A 退出评审**
日期：2026-09-01
实现范围：内部 `types / values / operations / module / schema`

## 1. 结论

Batch A 已按授权边界完成，具备进入 owner review 的技术条件。实现没有新增公开导出，
没有修改 `CircuitIR` 1.0、`fq.run/fq.plan`、Runtime、Provider 或默认执行路径。

本结论不等于 Phase 1 完成，也不自动授权 Batch B。

## 2. 已完成能力

- 不可变 `IRType`，其中 qubit 类型显式标记为 linear；
- 确定性、带 scope 的 `ValueId` 和 typed `ValueRef`；
- 不可变 `Operation / Block / Region / QuantumModule`；
- 深度冻结 attributes，拒绝无序 set、非有限数值和不可确定对象；
- bool、int、float、complex 分类型 canonical encoding；
- source location 与 program identity 分离，module revision 也不污染 semantic identity；
- versioned、immutable、fail-closed operation schema registry；
- 覆盖 35 个 canonical opcode 及受限 `custom_unitary` schema；
- `_compiler` 与其 `ir` 包均不提供 convenience exports。

## 3. 明确未做

- 未实现 verifier 和 diagnostics；
- 未实现 `CircuitIR` importer/exporter；
- 未实现参数 binding table、analysis、pass 或 differential bridge；
- 未让任何默认 runtime/compiler 路径依赖 `_compiler`；
- 未修改公共 API snapshot 或稳定合同。

这些工作分别属于 Batch B–F，必须逐批授权。

## 4. 验证结果

| 检查 | 结果 |
| --- | --- |
| Internal IR | 75 passed |
| Default CI | 1042 passed, 10 skipped |
| Runtime CI | 168 passed, 31 skipped |
| Ruff | passed |
| Black | passed |

Runtime 的 31 个 skip 均为既有环境/硬件条件跳过。本批没有新增 warning；Default CI 仍有
一条既有 PyTorch complex module warning。

机器可读证据与精确文件 hash 见
`contracts/ir-phase1-batch-a-review-candidate.json`。

## 5. Batch A 退出判断

| 验收项 | 状态 |
| --- | --- |
| 数据模型不可变 | 通过 |
| qubit linear type 可表达 | 通过 |
| Value identity 确定 | 通过 |
| semantic identity 确定且排除 source location | 通过 |
| schema registry immutable/versioned/fail-closed | 通过 |
| 35 个 canonical opcode 有 schema | 通过 |
| 公共 API 和默认路径零影响 | 通过 |
| Owner 接受 Batch A evidence | 待审批 |

## 6. 下一暂停点

如 owner 接受本证据并希望启动 Batch B，应使用精确口令：

```text
approve IR-PHASE1-BATCH-A-EXIT-BATCH-B
```

该口令只授权 verifier 与 structured diagnostics。它不授权 importer、公共 API、默认
runtime、Phase 2 或生产路径切换。普通 `do`、`continue` 不等价于该授权。
