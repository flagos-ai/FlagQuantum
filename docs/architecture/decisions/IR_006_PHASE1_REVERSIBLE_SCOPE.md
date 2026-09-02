# IR-006：Phase 1 可逆静态范围

状态：Approved
日期：2026-09-01
适用阶段：Phase 1 importer、round-trip 与 differential bridge

批准记录：API owner 于 2026-09-01 通过明确指令批准 IR-004～006。批准范围仅限内部
`circuit_ir_v1_static` profile；不把 dynamic、timing、pulse 或 provider-native 能力
升级为已支持，也不授权替换默认 runtime。

## 背景

“CircuitIR -> QuantumIR -> CircuitIR round-trip”必须有精确支持范围。把所有能构造的
CircuitIR 都描述为可逆，会误纳入 dynamic metadata、provider evidence、未知扩展和当前
没有强类型模型的语义。

## 决策

Phase 1 profile 命名为内部 `circuit_ir_v1_static`，支持：

- 当前 `OPERATOR_SCHEMAS` 注册的 35 个 canonical opcode；
- unitary 与 channel schema，但 channel 必须满足当前 typed semantic marker 规则；
- IR-004 批准范围内的 concrete custom unitary；
- Python/static scalar、Parameter、ParameterExpression 和 IR-005 binding 类型；
- n_wires、instruction ordering、wire ordering、dtype 和 shape constraints；
- observables 与 request 型 measurements，通过 IR-001 组合对象拆分并重组；
- 已分类且具有 typed destination 的 metadata。

Phase 1 明确拒绝：

- `is_dynamic`、condition/conditions、classical bit feedback；
- function、call、loop、branch、region 和 controller program；
- timing、pulse、calibration 和 provider-native instruction；
- 未分类且可能影响语义/identity 的 metadata；
- IR-004 范围外 matrix；
- 无法确定性编码或无法保持梯度的参数；
- 任何需要 lossy conversion 才能降回 CircuitIR schema 1.0 的结构。

Round-trip 成功标准是 canonical `CircuitIR.to_dict()` 精确相等；编译优化后的语义等价
不属于 Phase 1 round-trip 定义。

## 支持状态

Importer 对每个输入返回且仅返回一种状态：

```text
supported_exact
unsupported_with_diagnostics
invalid_input
```

不允许 `best_effort`、隐式忽略 metadata 或默认 lossy 模式。

## 否决方案

- **所有 CircuitIR 均支持**：与当前 metadata/dynamic 事实不符；
- **只支持纯 unitary gate**：无法覆盖当前 channel 和 execution-request 边界；
- **允许 silent lossy**：破坏科学语义与 round-trip 可信度。

## 兼容与回滚

Profile 是内部 importer 能力标签，不进入公共 API。删除 importer/profile 不影响 legacy
path。未来扩大范围必须增加 fixture、diagnostic、differential evidence 和 ADR 修订。

## 验收

- 35-opcode manifest 覆盖测试通过；
- supported fixture canonical payload 精确 round-trip；
- dynamic/unknown/lossy fixture fail closed；
- request 拆分重组不改变 ordering、shots 或 metadata；
- public schema/API 无变化；
- [x] API owner 批准 profile 范围；
- [ ] compiler/runtime owner 在实现评审中确认 importer 与 differential bridge 细节。
