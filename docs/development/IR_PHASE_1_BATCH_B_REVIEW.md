# FlagQuantum IR Phase 1 Batch B 评审

状态：**Technical complete — 等待 Batch B 退出评审**
日期：2026-09-01
实现范围：内部 verifier 与 structured diagnostics

## 1. 结论

Batch B 已完成内部 QuantumIR 的 fail-closed 验证闭环，具备进入 owner review 的技术条件。
验证器仅接收内部 module 与显式 schema registry，不读取公共 Runtime、Provider 或环境状态。

本结论不等于 Phase 1 完成，也不自动授权 Batch C。

## 2. 已完成能力

- 14 个稳定内部诊断码；
- 不可变 `Diagnostic`，包含 code、severity、message、location 和 notes；
- unknown operation、unknown/missing/type-invalid attribute fail closed；
- operand/result/region/variadic arity 验证；
- use-before-definition、duplicate definition 验证；
- linear qubit 重复消费、gate 后旧 value 使用验证；
- release 后使用验证；
- measurement result 类型验证；
- terminator 后继续放置 operation 的验证；
- nested region 独立结构验证；
- verifier 不调用 `print`，正常失败以 `VerificationResult` 返回。

## 3. 正确的阶段边界

下列判断必须读取公共 `CircuitIR` 源信息，因此留在 Batch C importer 边界：

- wire 是否超出 `n_wires`；
- 源 payload 能否无损表达为内部 IR；
- dynamic、condition 和未分类 metadata 的 source diagnostic。

内部 `ValueId` 不是 wire ID。为满足“wire 检查”而把二者混为一谈会破坏 IR-002，故本批
没有伪造 wire validation。

## 4. 明确未做

- 未实现 CircuitIR importer、execution request split 或 binding table；
- 未实现 exporter、round-trip、analysis 或 pass；
- 未修改 Batch A 数据模型与 schema artifact；
- 未新增公开导出或接入默认执行路径。

## 5. 验证结果

| 检查 | 结果 |
| --- | --- |
| Internal IR | 101 passed |
| Default CI | 1068 passed, 10 skipped |
| Runtime CI | 168 passed, 31 skipped |
| Ruff | passed |
| Black | passed |

Default CI 保留一条既有 PyTorch complex module warning；Runtime 的 31 个 skip 为既有
环境/硬件条件跳过。

机器证据和精确文件 hash 见
`contracts/ir-phase1-batch-b-review-candidate.json`。

## 6. 下一暂停点

如 owner 接受本证据并希望启动 Batch C，应使用：

```text
approve IR-PHASE1-BATCH-B-EXIT-BATCH-C
```

该口令仅授权 CircuitIR importer、typed execution-request split、constraints、source
identity 与 binding-table 基础；不授权 exporter、公共 API、默认 runtime、Phase 2 或
生产路径切换。普通 `do`、`continue` 不等价于该授权。
