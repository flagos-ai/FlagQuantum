# FlagQuantum IR Phase 0 事实基线

状态：P0-001 与 P0-002 已完成盘点；IR-001～003 已获 API owner 批准
基线日期：2026-09-01
实施计划：[`MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md`](../architecture/MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)

本文只记录当前仓库事实和 Phase 1 兼容边界，不声明新 IR 能力，不修改任何公共合同。

## 1. P0-001：公共合同保护清单

### 1.1 当前公共入口

以下签名由当前 Docker 开发环境通过 `inspect.signature` 读取：

```text
fq.Circuit(
  n_qubits=None, *, n_wires=None, nqubits=None, bsz=1,
  device=None, dtype=None, inputs=None, config=None
)

fq.CircuitIR(
  n_wires, instructions, version="1.0", dtype="complex64", shape=(),
  observables=(), measurements=(), metadata=<factory>
)

fq.plan(
  program, *, options=None, measurements=None, noise_model=None
) -> ExecutionPlan

fq.run(
  program_or_plan, *, options=None, measurements=None, noise_model=None
) -> ExecutionResult

flagquantum.compiler.compile_for_backend(
  circuit_or_ir, *, coupling_map=None,
  routing_strategy="restore_after_each_gate", optimize=True, config=None
) -> CircuitIR
```

当前稳定根 `__all__` 数量为 22。Phase 0–1 不增加、删除、重排或重命名其中任何符号。

### 1.2 CircuitIR schema 1.0

权威实现：`flagquantum/core/ir.py`。

当前顶层 canonical payload 字段：

```text
kind = "flagquantum.circuit_ir"
version
n_wires
dtype
shape
instructions
observables
measurements
metadata
```

当前保护行为：

- `IR_VERSION == "1.0"`；
- 版本不等于 1.0 时构造失败；
- `n_wires` 必须为正整数；
- instruction、observable、measurement wire 必须存在、非负、不重复且不越界；
- 已注册 opcode 必须满足 arity 和参数要求；
- 未注册 opcode 只有携带显式 matrix、channel 标记或 dynamic 标记时才可构造；
- measurement shots 如存在必须为正整数；
- `to_json()` 使用 ASCII、稳定 key 排序和确定分隔符；
- `content_hash` 是 canonical `to_json()` 的 SHA-256；
- `from_dict()` 对未知顶层字段 fail closed；
- 参数、参数表达式、复数和 tensor 使用当前版本化编码规则；
- 无法确定性编码的对象抛出 `IRSerializationError`；
- `ensure_circuit_ir()` 只接受 `CircuitIR` 或提供 `to_ir()` 的对象。

Phase 0–1 不得改变以上行为，也不得创建公共 `CircuitIRV1/CircuitIRV2` 类。

### 1.3 plan/run 当前语义

当前稳定执行链为：

```text
fq.run(program)
  -> fq.plan(program)
  -> validated ExecutionPlan
  -> execute_plan(plan)
```

受保护边界：

- `fq.plan` 只接受 `Circuit` 或 `CircuitIR`；
- 调用方传入 measurements 时，源 IR 已含 measurements 会失败，而不是覆盖或合并；
- samples target 可由稳定 options 构造明确的 sample measurement；
- expectation/amplitudes target 没有对应 request 时 fail closed；
- 当前动态 instruction 不进入稳定 `fq.run`；
- 执行已有 `ExecutionPlan` 时不得同时再次传 options、measurements 或 noise model；
- `fq.run(plan)` 执行已验证计划，不应静默重新规划；
- requested、selected、actual backend 与 fallback 可见性继续遵守已批准合同。

内部 QuantumIR 不能成为修改这些规则的理由。

### 1.4 当前合同 hash

以下 hash 用于发现 Phase 0–1 的非预期改动，不授权自动更新合同：

| 文件 | SHA-256 |
| --- | --- |
| `docs/public_api_v1.json` | `d211967831ced3947445259acb7e5f6557c8fdc4bc560a0978ee2101123e28d4` |
| `contracts/public-api-v0.2-baseline.json` | `ee8f0b7959cc0f758ae14e73f92f1dbbfd4f66d022beadce2cc37f5c4a843dea` |
| `contracts/public-api-v1-candidate.json` | `72124b6557b09ffee1ecbcc682e958d50aa6a1895e77f96d6e3bb63581ddc256` |
| `contracts/api-convergence-review-packet-008-010.json` | `068e45620f6d6beecf8ad02960c40f20db05be5906bddde6fd46308df747de03` |
| `contracts/experimental-namespace-v1-candidate.json` | `852c74c6441b89764836615ad4aca93eff6b20919e897f1166a1c26635177ba4` |
| `contracts/experimental-surface-v2-candidate.json` | `f400fb5d8de97f4cf51a15e4c8d2634398fbd158b5b0fda0dbfcf0d4afd63ef4` |

注意：Experimental v2 当前状态为 `implemented_pending_review`；记录 hash 不等于批准
或冻结 Proposal 011。

### 1.5 Phase 0–1 禁改文件/表面

没有独立 API change proposal 和 owner 批准时，不得修改：

- `flagquantum/__init__.py` 稳定导出；
- `flagquantum/core/ir.py` 的公共类型、schema 与序列化行为；
- `fq.plan`、`fq.run`、`compile_for_backend` 的签名和稳定语义；
- `docs/public_api_v1.json` 与已批准 contracts；
- ExecutionPlan、ExecutionResult、DeploymentPackage 的受保护 schema；
- 当前 measurement 冲突规则和 plan/run 等价关系。

## 2. P0-002：CircuitIR 消费关系矩阵

### 2.1 字段分类

| 字段 | 当前主要用途 | 内部分类建议 | Phase 1 要求 |
| --- | --- | --- | --- |
| `n_wires` | 构建、校验、内存估算、backend capacity、emit | program semantics | 精确保留 |
| `instructions` | 编译、规划、模拟、训练、绘图、emit、部署 | program semantics | 顺序、opcode、wire、参数和 matrix 精确保留 |
| `version` | schema 读取和 plan serialization | source schema identity | 保持 1.0，不进入新公共版本 |
| `dtype` | state representation、精度和计划 | numerical constraint | 不得隐式升降精度 |
| `shape` | batch/dense state 描述 | execution/representation constraint | 与 program identity 的关系由 IR-003 决定 |
| `observables` | expectation request、interop 边界、执行 | execution request | 按 IR-001 拆出 typed request |
| `measurements` | samples/expectation request、shots、result ordering | execution request；显式 terminal operation 需另行区分 | 保留当前冲突和排序语义 |
| `metadata` | runtime config、routing、dynamic/channel 标记、provenance | mixed；不能整体视为注释 | 分类 allowlist，未知语义项 fail closed |

### 2.2 组件消费矩阵

| 组件 | 读取/变换 | 当前关键假设 | Phase 1 兼容要求 |
| --- | --- | --- | --- |
| `Circuit.to_ir()` | 生成 instructions、dtype、shape、batch/runtime metadata；缓存结果 | 4096 wires 以上不物化指数 shape | importer 不迫使 `Circuit` 改变生成行为 |
| `Circuit.from_ir()` | 当前主要恢复 instructions/n_wires | 不完整恢复 request 和 metadata | 不把它误当作完整 round-trip oracle |
| native compiler | 删除/融合 instruction，以 `replace()` 保留其他字段 | 优化必须保持参数梯度和科学语义 | Phase 1 不迁移现有 Pass；先做差分 |
| routing | 改写 wire/instruction，写 routing metadata | physical mapping 与证据当前位于 metadata | 先分类和验证，不静默丢弃 |
| planner | 分析 instructions、noise、memory、measurements 和 runtime config | measurements 来源唯一；动态 fail closed | importer 不重写公共冲突规则 |
| ExecutionPlan contract | canonicalize instruction layer order，保留 observable/measurement | program hash 与 executable plan identity 相连 | 新 program identity 不替代当前 plan identity |
| stable runtime | 消费 validated plan，并执行 measurement requests | `run(plan)` 不再接收额外 options/request | Phase 1 默认路径零变化 |
| local statevector/density | 消费 opcode、wire、params、dtype、request | backend/dtype 有各自数值合同 | 使用现有 oracle 分别比较 |
| MPS/TN | 消费 interaction/order/observables，可能不需要 dense shape | 不允许静默 fallback 为 statevector | importer 不改变 representation semantics |
| distributed/JAX | 消费 CircuitIR 并建立 ownership、communication、gradient plan | distribution semantics 必须如实记录 | Phase 1 不提出扩展性声明 |
| noise lowering | 读取 instruction/channel metadata 并生成新 IR | `is_channel` 当前影响构造与执行语义 | channel metadata 必须列入语义 allowlist |
| dynamic runtime | 使用 `is_dynamic`、`condition` 等 instruction metadata | 当前与 stable `fq.run` 分离 | Phase 1 静态 importer 应明确拒绝，不得忽略 |
| Qiskit/PennyLane interop | 导入/导出 CircuitIR；将 observable/measurement 视为边界请求 | lossy conversion 必须显式 | 沿用诊断，不扩大支持范围 |
| QASM exporter | 主要消费 n_wires 与 instructions | 不等于完整 OpenQASM 3 importer/compiler | Phase 1 不替换 exporter |
| QCIS exporter | 消费静态 instructions，要求参数已绑定 | 不支持的 gate/参数 fail closed | 保留现有限制和异常 |
| drawer | 将 n_wires/instructions 投影为 drawable history | 仅展示，不是语义 oracle | 新 IR printer 与 drawer 分离 |
| deployment | compile IR、routing、QASM/QCIS、artifact hash | DeploymentPackage 当前绑定 compiled IR 与格式字段 | Phase 1 不替换 schema 或提交链路 |
| Module/training | 使用 topology、parameter、IR hash 和 backend training | forward 与 gradient 都是合同 | Phase 1 必须覆盖 gradient parity |
| correctness/fuzz tools | 生成并最小化 CircuitIR | 固定 seed 与 opcode schema | 复用为 corpus 工具，不作为唯一 golden truth |

### 2.3 Metadata 事实与 blocker

当前 metadata 至少分为四类：

| 类别 | 已发现示例 | 处理原则 |
| --- | --- | --- |
| instruction semantics | `is_channel`、`is_dynamic`、`condition` | Phase 1 静态 importer 支持或结构化拒绝，绝不忽略 |
| execution/planning constraints | circuit `runtime_config`、`batch_size`、`logical_state_shape` | 进入 typed import constraints，不进入任意 attrs |
| compiler evidence | `routing`、`routing_strategy_selection` | 不属于源 program semantics；作为 provenance/evidence 保留 |
| provenance/debug | source、audit、非语义标签 | 不进入 program semantic hash，保留策略须明确 |

Phase 1 blocker：完成 repository-wide metadata key inventory，并为每个会影响执行、合法性、
结果、梯度或 identity 的 key 建立 typed destination。未知 key 默认不能被宣称无害。

### 2.4 已确认的依赖方向

```text
public CircuitIR
  -> compilation / planning / runtime / interop / deployment / drawer

future internal importer
  -> may depend on public CircuitIR

public CircuitIR
  -X-> must not depend on future internal IR
```

Phase 1 `_compiler` 不得反向成为 `core.ir` 的依赖，也不得通过 import side effect 注册根
API。

## 3. P0-001/002 验收记录

- [x] 当前公共签名通过 Docker 环境读取；
- [x] CircuitIR schema、hash 和 fail-closed 行为已记录；
- [x] plan/run measurement 与动态边界已记录；
- [x] 关键合同 hash 已记录；
- [x] 主要消费组件和字段用途已建立矩阵；
- [x] metadata 非纯注释风险已明确；
- [ ] API owner 复核事实基线；
- [ ] compiler/runtime owner 补充遗漏消费者；
- [x] repository-wide metadata consumed-key 静态 inventory 完成；
- [x] metadata typed destination 与人工二次审计完成。

未完成的复核项不阻止继续完成 Phase 0 corpus 设计，但阻止 Phase 1 importer 合入。
