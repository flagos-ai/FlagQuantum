# FlagQuantum IR Phase 1 Batch D 评审

状态：**Technical complete — 等待 Batch D 退出评审**
日期：2026-09-02
实现范围：restricted CircuitIR exporter 与 canonical exact round-trip

## 1. 结论

Batch D 已为 `circuit_ir_v1_static` 支持子集建立可审计的精确 round-trip。全部 supported
Phase 0 fixture 均满足：

```text
CircuitIR.to_dict() == export(import_and_seal(CircuitIR)).to_dict()
```

exporter 不公开、不进入默认执行路径，也不承担 Provider codegen。

## 2. 精确性机制

密封 artifact 同时绑定：

- Batch C 生成的 `ImportedCircuitProgram`；
- import 时的 `internal_program_identity`；
- 公共 `CircuitIR.content_hash`；
- canonical source JSON；
- 固定 profile `circuit_ir_v1_static`。

export 前逐项复核这些 identity，再通过公共 `CircuitIR.from_json` 恢复并重新计算 canonical
JSON 与 content hash。任何一项不一致都返回 structured diagnostic。

## 3. 能力边界

当前能力是“恢复已密封、已验证支持的源 CircuitIR”，不是“从任意变换后的 QuantumIR
生成 CircuitIR”。采用 source envelope 是因为公共 payload 中存在 metadata presence、
custom matrix 原始编码和 request 字段等精确表示信息；仅凭语义 IR 无法诚实保证字节级
canonical payload 相等。

因此当前明确不支持：

- arbitrarily transformed QuantumIR export；
- best-effort 或 lossy recovery；
- Provider/native instruction codegen；
- dynamic、timing、pulse 或 Phase 2 ProgramIR export。

## 4. Fail-closed 行为

- unsupported source 不生成 artifact；
- 非 artifact 输入直接拒绝；
- profile、program identity、source identity 或 source JSON 被篡改均拒绝；
- exporter 只有一个 `artifact` 参数，没有 `best_effort/provider` 开关；
- 失败不返回半成品 `CircuitIR`。

## 5. 验证结果

| 检查 | 结果 |
| --- | --- |
| Internal IR | 178 passed |
| Default CI | 1145 passed, 10 skipped |
| Runtime CI | 168 passed, 31 skipped |
| Ruff | passed |
| Black | passed |

Default CI 保留一条既有 PyTorch complex module warning；Runtime 的 31 个 skip 为既有
环境/硬件条件跳过。

机器证据和精确文件 hash 见
`contracts/ir-phase1-batch-d-review-candidate.json`。

## 6. 下一暂停点

如 owner 接受本证据并希望启动 Batch E，应使用：

```text
approve IR-PHASE1-BATCH-D-EXIT-BATCH-E
```

该口令只授权 analysis cache、DefUse、QubitLifetime、PassManager、deterministic pipeline
digest 与一个无语义变化示例 Pass；不授权 differential runtime bridge、公共 API、默认
runtime、Phase 2 或生产路径切换。
