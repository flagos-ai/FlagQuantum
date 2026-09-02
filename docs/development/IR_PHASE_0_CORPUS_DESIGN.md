# FlagQuantum IR Phase 0 Characterization Corpus 设计

状态：P0-003 技术实现完成，等待 compiler/runtime/training owner 最终复核
日期：2026-09-01
依赖：[`IR_PHASE_0_BASELINE.md`](IR_PHASE_0_BASELINE.md)、IR-001～003 Proposed ADR

## 1. 目标

Corpus 用于固定 legacy `CircuitIR` 的真实语义，并成为未来 importer、verifier、
round-trip 和新旧执行差分的共同输入。它不是随机 demo 集合，也不用于扩大公开能力。

完成后的 corpus 必须回答：

- 当前哪些 schema 1.0 输入被接受；
- 每个 canonical opcode 如何编码；
- 哪些输入必须失败以及失败类别；
- parameter、dtype、shape、request、metadata 和 wire ordering 如何影响结果；
- 哪些 fixture 可以精确 round-trip；
- 哪些 fixture 只能导入或必须由 Phase 1 静态 importer 拒绝。

## 2. 建议目录

```text
tests/fixtures/internal_ir/circuit_ir_v1/
  manifest.json
  static/
  parameterized/
  observables/
  measurements/
  custom_unitary/
  channels/
  metadata/
  unsupported_static_import/
  invalid/

tests/internal_ir/
  corpus.py
  test_corpus_coverage.py
  test_legacy_characterization.py
```

Fixture 使用现有 canonical `CircuitIR.to_dict()` payload，不发明第二种源格式。
`manifest.json` 记录 fixture identity、预期分类、oracle、dtype、seed 和支持状态。

## 3. 当前 canonical opcode 覆盖

2026-09-01 从 `OPERATOR_SCHEMAS` 读取到 35 个 canonical opcode：

| 类别 | Opcode |
| --- | --- |
| 单 qubit、无参数 | `h`, `i`, `s`, `sdg`, `sx`, `sxdg`, `t`, `tdg`, `x`, `y`, `z` |
| 单 qubit、参数化 | `phase`, `rx`, `ry`, `rz`, `u1`, `u2`, `u3` |
| 双 qubit、无参数 | `cx`, `cy`, `cz`, `swap` |
| 双 qubit、参数化 | `cphase`, `crx`, `cry`, `crz`, `rxx`, `ryy`, `rzz` |
| 三 qubit | `ccx`, `cswap` |
| channel schema | `amplitude_damping`, `bit_flip`, `depolarizing`, `phase_flip` |

Coverage test 必须从 registry 动态读取当前集合，并与 manifest 的 `covered_opcodes` 比较。
Registry 增加 opcode 时测试必须失败并要求新增 fixture，不能自动把新符号标记为已覆盖。

## 4. 固定 fixture 集

### 4.1 每 opcode 最小 fixture

每个 opcode 至少一个最小合法 CircuitIR：

- wire 数等于或略大于 arity；
- 使用非排序 wire 的 opcode 增加 ordering fixture；
- 参数化 gate 使用非零、非特殊角度，避免优化偶然消除；
- `u2/u3` 覆盖参数名称和顺序；
- channel 带当前实现所需的语义 metadata；
- 每个 fixture 明确 `round_trip_expected` 和 `execution_oracle`。

### 4.2 组合线路

固定组合至少包括：

- Bell、GHZ、非相邻双 qubit gate；
- 多层同 wire 依赖与可并行 layer；
- self-inverse cancellation 候选；
- adjacent rotation merge 候选；
- topology routing 和 wire/result ordering；
- custom 2×2 与 4×4 unitary；
- 1、2、3 qubit 混合线路；
- 10、100、1K、10K gate 结构生成规则，用于性能而非 golden 数值文件。

### 4.3 参数化与梯度

必须区分：

- 单个 `Parameter`；
- 同一 Parameter 多次使用；
- 多个独立 Parameter；
- `ParameterExpression` 加、乘和组合；
- scalar torch tensor 参数；
- `requires_grad=True` 参数；
- bind 前/后 content hash 与执行；
- complex64 与 complex128 forward/gradient。

Gradient fixture 至少包含 RX expectation、两参数纠缠线路和 Module/VQE 风格损失。
Oracle 同时比较 forward 与 gradient。

### 4.4 Observable 与 measurement request

固定覆盖：

- 单 wire observable；
- 多 term Hamiltonian/observable ordering；
- sample 全 wires 与子集 wires；
- shots、seed metadata；
- 多 measurement request 顺序；
- IR 内 request 与调用方 request 冲突；
- target 要求 request 但缺失；
- observable/measurement 无法进入第三方格式时的 fail-closed 诊断。

这些 fixture 按 IR-001 进入 `InternalExecutionRequest`，不得污染 Phase 1 program identity。

### 4.5 dtype、shape 与规模

覆盖：

- `complex64`、`complex128`；
- 默认 shape normalization；
- batch size 1 与大于 1；
- `Circuit.to_ir()` 生成的 dense shape；
- 大 wire 数的 `logical_state_shape` metadata；
- 非法空/负 shape；
- dtype 空字符串；
- importer 不得分配与 `2**n_wires` 成比例的状态内存。

### 4.6 Metadata 分类 fixture

| 分类 | Fixture | Phase 1 预期 |
| --- | --- | --- |
| instruction semantics | `is_channel` | typed channel operation 或明确拒绝 |
| instruction semantics | `is_dynamic`, `condition` | static importer 结构化拒绝 |
| execution constraint | `runtime_config`, `batch_size`, `logical_state_shape` | typed ImportConstraints |
| compiler evidence | `routing`, `routing_strategy_selection` | provenance/evidence，不进入 program semantics |
| debug/provenance | 无语义 source label | 保留或报告，但不改变 program identity |
| unknown | 未分类 key | 不得静默声明无害；按 inventory policy 处理 |

## 5. 负向 corpus

至少固定以下错误输入及预期异常类别：

- 非法/缺失 `kind`；
- 不支持的 version；
- 未知顶层字段；
- `n_wires <= 0`；
- 空、负数、重复或越界 wire；
- opcode arity 不匹配；
- 缺失参数；
- 未知 opcode 且无 matrix/semantic marker；
- matrix shape 或 dtype 不合法；
- shots 非正数；
- shape 含非正维度；
- 无法确定性序列化的参数/metadata；
- dynamic/channel metadata 与 opcode schema 冲突；
- measurement/observable wire 越界；
- request 来源冲突。

Legacy characterization 固定当前异常类型和可依赖的 error category；不把完整英文 message
全部冻结，除非现有公共合同已经要求。

## 6. Oracle 层级

每个 fixture 在 manifest 中选择一个或多个 oracle：

```text
serialization_exact
construction_error
compile_exact_or_semantic
statevector
expectation
measurement_exact_seeded
measurement_statistical
gradient
routing_legality
interop_round_trip
emitter_parse_or_golden
unsupported_diagnostic
```

规则：

- serialization、wire ordering、request ordering 和 deterministic hash 精确比较；
- state/expectation/gradient 使用已有 backend/dtype 容差；
- sampling 优先固定 seed 合同，否则使用批准的统计检验；
- global phase 处理必须由具体 state oracle 明确，不能自动忽略所有差异；
- compile optimization 可比较语义，不要求优化后 instruction payload 与源完全相同；
- QASM/QCIS golden 必须同时有 parse/semantic 或支持边界证据，不能只比字符串外观。

## 7. Manifest 最小字段

```json
{
  "schema_version": "1.0",
  "registry_snapshot": {
    "canonical_opcode_count": 35,
    "covered_opcodes": []
  },
  "fixtures": [
    {
      "id": "static.rx.parameterized.v1",
      "path": "parameterized/rx.json",
      "category": "static_parameterized",
      "expected_import": "supported",
      "round_trip_expected": true,
      "oracles": ["serialization_exact", "statevector", "gradient"],
      "dtype": "complex64",
      "seed": 17
    }
  ]
}
```

`covered_opcodes` 在 fixture 实现时填写并由测试验证；本设计文档不伪造已完成覆盖。

## 8. P0-003 实现验收

- [x] 当前 35 个 canonical opcode 已分类；
- [x] fixture 目录、manifest 和 oracle 设计完成；
- [x] parameter、gradient、request、dtype/shape、metadata 和负向范围已定义；
- [x] manifest 与固定基础 fixtures 已创建；
- [x] registry drift test 已实现；
- [x] legacy serialization characterization tests 已实现；
- [x] corpus hash 已写入 Phase 0 evidence manifest；
- [x] legacy analytic gradient oracle 已实现；
- [x] seeded/statistical measurement oracle 已实现；
- [x] manifest 声明的 semantic oracle 已由统一 dispatcher 逐 fixture 执行；
- [x] 12 个正向 fixture 与 14 个机器可读负向 fixture 已覆盖最小语义范围；
- [x] expectation、routing、QASM/QCIS、backend selection 与 dynamic rejection 已纳入 evidence；
- [ ] compiler/runtime owner 复核 corpus 覆盖。

设计完成不等于 P0-003 工作包整体完成。只有机器 fixture、coverage test 和 oracle 全部
落地后，Phase 0 退出门中的 corpus 项才可勾选。
