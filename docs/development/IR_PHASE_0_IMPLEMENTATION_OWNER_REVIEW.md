# FlagQuantum IR Phase 0 实现 Owner 技术复核

复核日期：2026-09-01
复核范围：compiler、runtime、training 与公共合同保护
复核性质：Codex 技术审计建议，不代替项目 owner 或 API owner 的正式签署
结论：**技术证据满足 Phase 0 退出要求，建议 owner 签署并授权受限 Phase 1**

机器候选记录：`contracts/ir-phase0-exit-phase1-review-candidate.json`

## 1. 复核边界

本复核只判断以下问题：

1. Phase 0 characterization 是否足以保护 Phase 1 importer；
2. compiler、runtime 和 training 的关键 legacy 语义是否有机器 oracle；
3. 性能预算是否来自可复现基线；
4. Phase 1 是否能够在不改变 Stable Core 的前提下安全启动；
5. 是否存在必须继续阻止实现的技术 blocker。

本复核不授权 Phase 2、多云、公共 IR API、默认 Runtime 切换或生产成熟度声明。

## 2. Compiler owner 视角

### 已复核

- 35/35 canonical opcode 由 registry drift test 保护；
- custom 2×2 与 4×4 matrix 均进入 serialization/statevector characterization；
- parameter、ParameterExpression add/mul/nested 结构保持确定序列化；
- persistent-layout routing 在 line topology 上验证所有双比特 gate 合法；
- routing 后状态与源状态一致，mapping 恢复为 logical layout；
- QASM 2 与 QCIS emitter 使用固定 SHA-256 golden；
- 未知 opcode、arity、缺失参数、重复/越界 wire 和非法 matrix 边界 fail closed；
- compiler evidence 与 program identity 的分离已经由 IR-003 和 metadata inventory 约束；
- Phase 1 不迁移现有 Pass，只允许一个无语义变化的示例 Pass。

### 结论

**技术通过。** 当前 corpus 足以作为 Phase 1 importer/verifier 的 compiler 基线。Phase 2
的 Pass、routing 和 emitter 迁移仍需独立授权，不能由本结论推导。

## 3. Runtime owner 视角

### 已复核

- statevector oracle 对所有声明 fixture 实际执行并检查确定性与归一化；
- complex64 与 complex128 expectation 均有端到端 `fq.run` oracle；
- seeded measurement 重复执行精确一致，并保护请求 wire order；
- requested、resolved、decision backend 和 fallback=false 由 ExecutionPlan 机器检查；
- dynamic instruction 在稳定 `fq.run` 路径结构化拒绝；
- 12 个正向与 14 个负向 fixture 全部由 manifest 驱动；
- `CircuitIR(dtype="complex128")` 被默认值降精度的问题已经修复；
- runtime integration tier 为 168 passed、31 skipped；
- 默认层为 1019 passed、10 skipped；
- 默认 `fq.run/fq.plan` 尚未引入 `_compiler` 或新路径开关。

### 结论

**技术通过。** Runtime 语义和精度边界足以允许 opt-in、测试专用的 Phase 1 差分桥。
不得让内部 IR 进入默认执行路径。

## 4. Training owner 视角

### 已复核

- legacy RX analytic gradient oracle 保持 tensor identity；
- complex128 两参数纠缠线路验证 forward 与两个解析梯度；
- 独立 Parameter、重复 Parameter 和组合 ParameterExpression 均进入 corpus；
- gradient 使用 PyTorch autograd，不以仅 forward parity 代替训练语义；
- `pr-default` 覆盖 Module、training、checkpoint 责任边界；
- Phase 1 预算不包含或暗示分布式训练、QPU gradient 或性能认证。

### 结论

**技术通过。** 当前证据足以保护 Phase 1 的参数 identity 与基础 gradient differential。
Phase 1 实现仍必须在 complex64/complex128 上执行完整新旧路径梯度 parity。

## 5. API 与公共合同视角

### 已复核

- Stable Core 仍为 22 个根导出；
- `CircuitIR` schema、version、JSON、content hash 与异常边界未修改；
- `fq.plan/fq.run` 稳定签名未修改；
- CircuitIR 精度修复只恢复已有 dtype program constraint，不增加公共字段或参数；
- 公共 API snapshot、CircuitIR 合同和 Markdown 对齐测试通过；
- Phase 1 候选目录为内部 `_compiler`，不得进入根 namespace 或自动补全；
- Phase 1 授权包明确禁止修改公共 snapshot 来接受漂移。

### 结论

**技术通过。** 未发现需要重新打开 Stable Core 或 CircuitIR 1.0 的理由。

## 6. 性能复核

原始 CPU 基线使用 10、100、1K、10K gate，15 次测量和 3 次 warmup。本轮在相同
Linux aarch64、Python 3.12.13、Torch 2.13.0+cpu 环境复现：

- 100 gate plan p95：0.812 ms；
- 1K gate plan p95：6.777 ms；
- 10K gate plan p95：67.534 ms；
- 四档 plan peak memory 与原记录精确一致。

Phase 1 importer+verifier 预算已经由 API owner 单独批准，只作为内部 gate，不是公开 SLA。

## 7. 风险复核

| 风险 | 当前控制 | 复核结论 |
| --- | --- | --- |
| 新 IR 改变科学语义 | 全 manifest oracle、新旧差分门 | 可控 |
| complex128 被降精度 | CircuitIR program constraint + regression test | 已修复 |
| metadata 被静默丢弃 | 55-key inventory、typed destination、drift test | 可控，Phase 1 必须 fail closed |
| internal API 泄漏 | `_compiler` 边界、root snapshot、不可发现要求 | 可控 |
| 本地快路径回归 | 默认路径零新工作、性能 gate | 可控 |
| 过早进入 Phase 2 | 授权包明确禁止 | 可控 |
| 代理越权实施 | 机器 candidate 保持 authorization=false | 可控 |

## 8. 技术建议

技术审计建议：

1. owner 正式确认 compiler/runtime/training 三项结论；
2. API owner 使用授权包中的精确语义批准 Phase 0 退出和 Phase 1 受限实现；
3. 将批准绑定到机器 candidate 中记录的 SHA-256；
4. 获批后只启动 Phase 1 批次 A，不一次实现 P1-001～P1-009；
5. 批次 A 通过独立 review 后再进入批次 B。

## 9. 尚未完成的正式动作

- [ ] compiler owner 正式签署；
- [ ] runtime owner 正式签署；
- [ ] training owner 正式签署；
- [ ] API owner 明确批准 Phase 0 退出；
- [ ] API owner 明确授权 Phase 1 实现；
- [ ] 机器 candidate 更新为批准状态并绑定批准语义。

在这些动作完成前，技术建议不能被解释为项目 owner 的正式批准。
