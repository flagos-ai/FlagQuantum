# API Change Proposal 010：动态线路构建契约

## 状态

**Frozen by API owner — 候选稳定契约已批准并冻结。**

- 候选稳定命名空间：`flagquantum.dynamic`；
- 候选稳定名称：仅 `DynamicCircuit`；
- 根 API 变化：无；
- 机器可读契约：`contracts/dynamic-circuit-v1-candidate.json`；
- `run_dynamic`、provider、dialect 和原生结果继续 experimental；
- API owner 于 2026-09-01 通过 `approve 008-010` 明确批准本契约；
- 不修改已由 review packet 哈希绑定的总 API 候选清单。

## 决策

稳定动态程序的表达方式，不提前稳定执行实现。用户通过独立命名空间构建程序：

```python
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2)
circuit.h(0)
circuit.measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0, equals=1)
```

`measure`、`reset`、单 bit 或多 bit conjunction 条件以及 CircuitIR v1 编码进入候选
契约。静态门沿用已经冻结的 `Circuit` operator schema。

## 为什么不稳定 DynamicExecutionResult

当前本地 trajectory 返回 `final_states`，Qiskit Aer 路径则返回空 tensor；
`provider_metadata` 和 `statistics` 也具有实现专属字段。现在冻结它会把后端差异固化成长期
用户契约。

稳定返回值继续是 `fq.ExecutionResult`。实验执行器暂时返回
`fq.experimental.dynamic.DynamicExecutionResult`，并通过 `to_execution_result()` 显式
投影。未来稳定 `run_dynamic` 前，必须先决定其直接返回 `ExecutionResult` 的测量、runtime、
provenance 和 mid-circuit 数据语义。

## 本轮修正

1. `DynamicCircuit.state` 与父类签名对齐为 `state(*, refresh=False)`；
2. 非法 wire、classical bit 和 condition 统一抛出 `ValidationError`；
3. 对包含动态指令的 `state()` 调用统一抛出 `CapabilityError`；
4. 动态指令写入后完整清理编译与 kernel cache；
5. `DynamicCircuit` 从 experimental 顶层功能清单迁入独立候选稳定命名空间。

## 验收标准

- [x] `flagquantum.dynamic.__all__ == ("DynamicCircuit",)`；
- [x] 构造器和动态方法签名由机器契约保护；
- [x] CircuitIR round trip 保留 measurement、reset 与 condition metadata；
- [x] 错误类型进入稳定错误体系；
- [x] stable root 不增加名称；
- [x] 执行、部署和适配器实现未被误标为稳定；
- [x] API owner 批准冻结 DynamicCircuit 构建契约。
