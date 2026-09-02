# IR-002：QuantumIR 线性 qubit value 模型

状态：Approved
日期：2026-09-01
适用阶段：Phase 1 QuantumIR core、verifier 与 importer

批准记录：API owner 于 2026-09-01 通过明确指令批准 IR-001～003。批准范围仅限内部
QuantumIR 的线性 value 模型；不改变公共 `Circuit` 命令式体验或 `CircuitIR` wire
表示。具体实现仍须通过 compiler owner 评审和 Phase 1 verifier 验收。

## 背景

当前 `CircuitIR` 用整数 wire 和有序 instruction 表示命令式线路。内部 QuantumIR 需要
显式数据流，以便验证生命周期、重写、调度和未来动态控制，但不能暗示量子状态可复制。

## 决策

Phase 1 QuantumIR 使用线性 value 模型：

```text
%q1 = quantum.h %q0
%q2, %q3 = quantum.cx %q1, %q_other
%q4, %m = quantum.measure %q2
```

规则：

- 每个逻辑 wire import 时产生唯一初始 qubit value；
- 每个 gate 消费操作数的当前 value，并按 wire 顺序产生新 value；
- 旧 value 在消费后不可再用于量子操作；
- 多 qubit gate 对每个输入产生对应后继 value，operand/result 顺序保持 wire ordering；
- value 是程序点上的资源版本，不是 statevector，也不可复制；
- `ValueId` 在同一 canonical importer 中确定生成；
- Phase 1 无控制流，只验证单 block 线性链；
- measurement 后 qubit 是否继续可用由 operation schema 明确；
- future block argument/phi-like join 必须延续线性所有权，但不在 Phase 1 实现。

## Phase 1 verifier 必须拒绝

- 同一 qubit value 被两个量子 operation 消费；
- gate 后继续使用旧 value；
- operand/result arity 或 qubit 类型错误；
- 两个逻辑资源被错误映射为同一 value；
- release/terminal consumption 后继续使用；
- 把 qubit value 放入不具备线性语义的普通 attribute/container；
- importer 生成不连续或不确定的 wire/value mapping。

## 否决方案

### 继续只使用整数 wire

否决原因：无法直接验证 ownership、def-use 和未来控制流 join，Pass 容易依赖隐式顺序。

### 把 qubit 当成普通 SSA value

否决原因：普通 SSA 允许多次读取，会错误暗示量子资源可复制。

### Phase 1 立即实现完整通用 SSA

否决原因：超出静态线路验证所需范围，增加实现和维护风险。

## 兼容与回滚

线性 value 只存在于内部 importer 结果。公共 `Circuit` 继续使用命令式 builder，公共
`CircuitIR` 继续使用整数 wire。删除 `_compiler` 不影响任何用户代码或持久化对象。

## 验收

- 每个 Phase 0 static fixture 生成确定 value 链；
- negative fixtures 覆盖重复消费、旧 value 使用和错误 arity；
- round-trip 保持原 instruction/wire 顺序；
- state、expectation 和 gradient differential 通过；
- [x] API owner 批准架构决策；
- [ ] compiler owner 在实现评审中确认 value/verifier 细节。
