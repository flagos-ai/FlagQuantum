# IR-004：自定义 Matrix Operation 的 Phase 1 边界

状态：Approved
日期：2026-09-01
适用阶段：Phase 1 CircuitIR importer、verifier 与受限 round-trip

批准记录：API owner 于 2026-09-01 通过明确指令批准 IR-004～006。批准范围仅限内部
importer 对 concrete custom unitary 的验证与 round-trip 边界；不改变公共 CircuitIR
构造行为，也不授权公开新的 custom-unitary API。

## 背景

公共 `CircuitIR` 当前允许未注册 opcode 携带显式 `matrix`。Phase 1 不能删除这一既有
输入能力，也不能因为 `Instruction` 当前只检查 matrix 是否存在，就把任意 shape、dtype
或不可确定对象声明为合法 QuantumIR operation。

## 决策

Phase 1 使用内部 `quantum.custom_unitary` operation 表示可验证的具体矩阵：

- wire arity 为 `k` 时，matrix 必须是 `2**k × 2**k`；
- matrix 必须是数值、有限、二维方阵；
- dtype 必须能无歧义映射到当前 complex64/complex128 数值合同；
- importer 验证 unitary 性质，容差按 dtype 合同确定；
- 原始 opcode 名作为 typed symbolic name 保留，不作为 provider-native gate；
- wire ordering、matrix element ordering 和 dtype 精确保留；
- concrete custom unitary 可进入 Phase 1 round-trip；
- symbolic、callable、稀疏未知格式或动态生成 matrix 返回结构化 unsupported diagnostic；
- importer 的更严格验证不改变公共 `CircuitIR` 构造行为。

含 `requires_grad=True` 的 matrix 不进入 Phase 1 静态 custom-unitary 范围；未来如需可训练
matrix，必须独立定义 parameterization、gradient 和 identity 合同。

## 否决方案

- **原样接受任意 matrix**：无法保证 shape、unitarity、确定性和目标合法化；
- **Phase 1 全部拒绝 custom matrix**：会缩小当前可表达静态 CircuitIR 的重要子集；
- **自动分解为基础门**：改变 round-trip，并把 target-dependent 优化提前引入 Phase 1。

## 兼容与回滚

验证只发生在内部 importer。legacy compiler/runtime 继续遵守当前行为；删除 importer 即可
回滚，不改变公共 schema 或用户数据。

## 验收

- 1、2 qubit concrete unitary 精确 round-trip；
- 非方阵、错误维度、非有限和非 unitary matrix fail closed；
- wire/matrix ordering differential 通过；
- trainable/symbolic matrix 返回明确 blocker；
- public CircuitIR tests 保持不变；
- [x] API owner 批准架构决策；
- [ ] compiler owner 在实现评审中确认 matrix verifier 与 dtype 容差。
