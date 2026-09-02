# FlagQuantum 多层 IR Phase 0–1 实施计划

状态：待架构评审的执行计划，不代表 Phase 1 代码已获准进入生产路径
上位设计：[`MULTI_LEVEL_IR_ARCHITECTURE.md`](MULTI_LEVEL_IR_ARCHITECTURE.md)
适用范围：当前语义基线、内部 QuantumIR 骨架、差分验证与性能基线
明确不改变：Stable Core、`CircuitIR` schema 1.0、`fq.plan`、`fq.run`、部署合同

实施进度：P0-001/P0-002 事实盘点已完成；P0-003 已建立 35-opcode 基础 corpus、
metadata inventory、漂移测试和 in-progress evidence manifest；API owner 已批准
IR-001～006 的架构决策范围。Metadata typed destination、legacy gradient/statistical
oracle、Phase 0 CPU 性能基线和 Phase 1 内部性能预算已经完成并获批；实现 owner 复核
尚未完成，因此仍未授权 Phase 1 代码。Phase 0 技术整改已经完成 manifest oracle
dispatcher、12 个正向 fixture、14 个负向 fixture，并修复 CircuitIR complex128 精度
约束被默认值覆盖的问题；剩余退出项仅为 owner 签审和明确授权。详见
[`IR_PHASE_0_BASELINE.md`](../development/IR_PHASE_0_BASELINE.md) 和
[`IR_PHASE_0_CORPUS_DESIGN.md`](../development/IR_PHASE_0_CORPUS_DESIGN.md)，以及
[`decisions/`](decisions/)。2026-09-01 退出审计确认机器基线可复现，但 manifest oracle、
owner 复核和明确授权仍未闭合；当前结论与 blocker 见
[`IR_PHASE_0_EXIT_AUDIT.md`](../development/IR_PHASE_0_EXIT_AUDIT.md)，允许范围候选见
[`IR_PHASE_1_IMPLEMENTATION_APPROVAL_PACKET.md`](../development/IR_PHASE_1_IMPLEMENTATION_APPROVAL_PACKET.md)。

## 1. 实施目标

Phase 0–1 只证明一件事：当前静态 `CircuitIR` 能够无公共 API 变化、无科学语义变化、
无不可接受快路径回归地进入一个可验证的内部 QuantumIR。

本计划不以“创建了一批 IR 类”为完成标准。完成必须同时具备：

- 当前行为和性能基线可复现；
- importer 对支持范围 fail closed；
- 内部 IR 具备确定的类型、value、operation、module 和诊断模型；
- state、expectation、measurement 与 gradient 新旧路径差分通过；
- 公共 API、序列化 schema 和默认执行路径保持不变；
- Phase 1 性能预算由 Phase 0 实测数据产生并机器校验；
- 新路径可整体关闭和删除，不影响 legacy path。

## 2. 实施边界

### 2.1 本轮包含

- 盘点当前 `CircuitIR`、compiler、planner、runtime 和 deployment 的真实消费关系；
- 固定静态线路 characterization corpus；
- 建立 legacy/new-path 差分测试基础设施；
- 建立内部 Python QuantumIR core、schema registry、verifier 和 diagnostics；
- 实现 `CircuitIR -> internal QuantumIR` importer；
- 为可逆静态子集实现受限 `QuantumIR -> CircuitIR` 测试出口；
- 建立最小 PassManager/AnalysisManager 合同；
- 记录构建、验证、hash 和 round-trip 的时间与内存；
- 形成进入 Phase 2 的证据包和架构决策记录。

### 2.2 本轮不包含

- 不新增 `fq.compile` 或任何稳定根导出；
- 不公开 ProgramIR、QuantumIR、TargetIR、PassManager 或 AnalysisManager；
- 不修改 `CircuitIR` 字段、版本、JSON、hash 或异常行为；
- 不让 `fq.run`、`fq.plan`、`compile_for_backend` 默认经过新管线；
- 不实现通用函数、递归、完整 SSA、动态控制流、时序、脉冲或 QIR；
- 不替换现有 routing、provider、deployment 或 distributed runtime；
- 不引入 C++、MLIR 或新的运行时依赖；
- 不把 debug textual IR 作为持久化或兼容合同；
- 不基于 CPU 测试提出 GPU、分布式或 QPU 性能声明。

## 3. 必须先解决的架构决策

以下 ADR 未批准前，可以完成 Phase 0，但不得开始对应的 Phase 1 实现。

| ADR | 必须决定的问题 | 默认建议 | 阻塞范围 |
| --- | --- | --- | --- |
| IR-001 | CircuitIR 中 observables/measurements 如何拆分 | importer 返回内部 `ImportedProgram`，包含 `module` 与 typed execution request | importer、round-trip |
| IR-002 | Phase 1 qubit 模型 | QuantumIR 使用线性 value；每次量子操作消费旧 value 并产生新 value | value、verifier、control-free lowering |
| IR-003 | 内部 identity | program、compilation、execution identity 分层；Phase 1 只实现 program identity | hash、cache、evidence |
| IR-004 | 自定义 matrix operation | 原样进入受限 typed custom-unitary op；无法验证时拒绝 | importer、round-trip |
| IR-005 | 参数和 tensor 常量 | 保留 `Parameter`/`ParameterExpression` 语义，禁止隐式数值绑定或 dtype 降级 | importer、gradient parity |
| IR-006 | Phase 1 可逆范围 | 只承诺当前静态 CircuitIR 子集；动态 metadata 不升级为支持能力 | round-trip、诊断 |

每份 ADR 至少包含：上下文、候选方案、决策、否决方案、兼容影响、测试影响和回滚方式。

## 4. 目标目录边界

Phase 1 建议采用以下内部布局。目录名可在 IR-001 至 IR-006 评审时调整，但依赖方向
不得反转：

```text
flagquantum/
  _compiler/
    __init__.py              # 不导出实现符号
    diagnostics.py
    identity.py
    ir/
      __init__.py
      types.py
      values.py
      operations.py
      modules.py
      schemas.py
      verifier.py
      printer.py             # debug-only canonical printer
    importers/
      circuit_ir.py
    exporters/
      circuit_ir.py          # 仅受限 round-trip 与测试使用
    analyses/
      base.py
      def_use.py
      qubit_lifetime.py
    passes/
      base.py
      manager.py
```

对应测试：

```text
tests/internal_ir/
  test_types_and_values.py
  test_operation_schema.py
  test_verifier_negative.py
  test_circuit_ir_import.py
  test_circuit_ir_round_trip.py
  test_identity_determinism.py
  test_pass_manager.py
  test_semantic_differential.py
  test_gradient_differential.py
  test_performance_budget.py
```

规则：

- `_compiler` 不得被 `flagquantum/__init__.py`、稳定 facade 或 `__all__` 导出；
- runtime 可以在显式试验开关下消费内部 IR，内部 IR 不依赖 runtime；
- importer 可以依赖公共 `core.ir`，公共 `core.ir` 不依赖 `_compiler`；
- provider、credential、queue 和网络客户端不得进入 `_compiler/ir`；
- 测试不得通过修改公共 API snapshot 来接受新符号。

## 5. Phase 0：事实基线

### P0-001 公共合同保护清单

交付：`docs/development/IR_PHASE_0_BASELINE.md` 中记录：

- 当前稳定根导出与受保护候选合同；
- `IR_VERSION`、`CircuitIR.to_dict/to_json/from_dict/from_json` 行为；
- content hash、未知字段、未知 opcode 和非法 wire 的失败行为；
- `fq.plan`、`fq.run`、`compile_for_backend` 当前签名与关键语义；
- 本计划禁止修改的合同文件及其基线 hash。

验收：现有 API contract、snapshot 和 IR tests 全部通过；基线文件只记录事实，不更新
任何受保护快照。

### P0-002 现有消费关系矩阵

交付：记录下列组件对 `CircuitIR` 各字段的读取、写入和假设：

- `Circuit` 与参数系统；
- compiler canonicalization；
- planner、routing 与 execution-plan builder；
- statevector、MPS、TN、noise 和 distributed runtime；
- drawer、QASM/QCIS emitter；
- dynamic importer/exporter；
- deployment package 和 provider。

每项必须标记：`semantic`、`execution_request`、`planning_hint`、`provenance` 或
`legacy_metadata_dependency`。无法分类的 metadata 依赖是 Phase 1 blocker。

### P0-003 Characterization corpus

建立固定、确定的静态线路 corpus，至少覆盖：

- 每个当前受支持 canonical opcode；
- 参数化单/双 qubit gate 与 ParameterExpression；
- custom matrix operation；
- observable、terminal measurement、shots 和 wire ordering；
- batched shape、complex64/complex128；
- 空 metadata、合法 provenance metadata 和拒绝用例；
- topology-sensitive circuit；
- VQE/QML 梯度线路；
- 序列化 round-trip fixture。

Corpus 必须来自机器可读 fixture，不从随机测试临时生成唯一真值。随机属性测试使用固定
seed，并与固定 fixture 分开。

### P0-004 差分 oracle

定义统一比较协议：

| 输出 | 比较方法 |
| --- | --- |
| CircuitIR round-trip | canonical dict/JSON 与 content hash 精确相等 |
| statevector | 按 dtype 合同比较，处理全局相位规则必须显式 |
| expectation | 按现有 backend/dtype 容差比较 |
| samples/counts | 固定 seed 时比较确定性合同；否则使用预先批准的统计检验 |
| gradient | forward value 与参数梯度均比较，不接受仅 forward parity |
| wire/result ordering | 精确相等 |
| fallback/backend | requested、selected、actual 与 blocker 精确比较 |

不得在新测试中发明一个覆盖全部 backend 的全局容差。

### P0-005 性能与资源基线

新增内部 benchmark，分别记录 10、100、1K、10K gate 的：

```text
circuit_to_ir_ms
legacy_compile_ms
legacy_plan_ms
legacy_run_cold_ms
legacy_run_warm_ms
peak_host_memory_bytes
serialized_ir_bytes
```

每个结果携带环境、Python、Torch、FlagQuantum commit、CPU/device、dtype、warmup、
iterations 和 seed。Phase 0 不宣称新 IR 更快，只形成预算依据。

### P0-006 Phase 1 性能预算

由 P0-005 的多次可复现实测结果生成并人工批准内部预算文件。预算必须分别约束：

- 小线路固定开销；
- importer 随 gate 数的增长；
- verifier 随 value/operation 数的增长；
- canonical hash 的确定性与耗时；
- round-trip 峰值内存；
- opt-in 新路径对默认 `fq.run` 的零影响。

没有批准的实测预算，Phase 1 性能 gate 保持未满足，不允许用任意百分比代替。

### Phase 0 退出门

- [ ] P0-001 至 P0-006 全部完成；
- [ ] IR-001 至 IR-006 已批准；
- [ ] corpus 覆盖全部当前静态 canonical opcode；
- [ ] legacy 语义与性能基线在干净 Docker 环境可复现；
- [ ] 公共 API 和序列化合同没有变化；
- [ ] blocker、owner、目标日期和回滚方式均已记录；
- [ ] API owner 明确批准进入 Phase 1。

## 6. Phase 1：内部 QuantumIR 骨架

### P1-001 类型、Value 与 Module

实现最小不可变模型：

- `IRType` 与 Phase 1 所需具体类型；
- 确定性 `ValueId` 和 `ValueRef`；
- `Operation`、`Block`、`Region` 与 `QuantumModule`；
- 冻结 attributes；
- module revision/program identity；
- source location 与 semantic hash 分离。

禁止为 provider 创建特殊基础类，也禁止使用可变全局字典承载语义状态。

### P1-002 Operation schema registry

每个 operation schema 声明：

- operand/result 类型和数量；
- attribute 名称、类型、必需性和默认值；
- region 数量；
- effect/linearity 约束；
- parser/printer；
- verifier；
- 是否可降回 CircuitIR v1。

未知 operation、未知核心 attribute 和版本不兼容必须 fail closed。

### P1-003 Verifier 与诊断

首期 verifier 必须拒绝：

- use-before-definition；
- 同一线性 qubit value 被重复消费；
- gate 后使用旧 value；
- wire 越界、arity 或参数不匹配；
- measurement/result 类型错误；
- 非法 block terminator；
- release 后使用；
- 未注册 operation 或影响语义的未知 attribute；
- importer 声称支持但无法无损表达的 CircuitIR。

所有失败返回结构化 code、message、location 和 notes，不使用 `print`。

### P1-004 CircuitIR importer

Importer 必须：

- 接受 `CircuitIR`，不要求用户构建内部对象；
- 保持 opcode、wire order、参数 identity、dtype 和 batch shape；
- 按 IR-001 拆分 program semantics 与 execution request；
- 为每个逻辑 wire 建立唯一线性 value 链；
- 对 custom matrix、metadata 和动态标记执行明确的支持检查；
- 返回 diagnostics 和 source mapping；
- 不修改输入对象；
- 相同输入产生相同 QuantumIR identity。

### P1-005 受限 round-trip

为 Phase 1 可逆静态子集实现测试出口。要求：

- `CircuitIR -> QuantumIR -> CircuitIR` canonical payload 精确相等；
- 无法回到 schema 1.0 的结构返回诊断，不做 lossy export；
- exporter 不进入公共 namespace；
- exporter 不成为 provider codegen 的提前替代品。

### P1-006 Analysis 与 Pass 最小合同

实现：

- `DefUseAnalysis`；
- `QubitLifetimeAnalysis`；
- analysis cache 与 module revision 绑定；
- transformation 默认失效未声明 preserved 的 analysis；
- PassResult、diagnostics 与 statistics；
- pipeline digest 包含 pass 名称、版本、顺序、options 和 seed。

Phase 1 只需一个无语义变化的 canonicalization 示例 Pass，用于证明基础设施，不迁移
现有优化器。

### P1-007 新旧路径差分桥

新增仅测试/开发可启用的执行桥：

```text
CircuitIR
  +-- legacy execution
  +-- import QuantumIR -> verified test lowering -> existing executor
```

约束：

- 默认 `fq.run` 不读取该开关；
- 不使用未声明环境变量改变公共行为；
- 桥接层不得把 QuantumIR 伪装成新 runtime；
- failure、fallback、dtype、device 和 result ordering 必须进入差分报告。

### P1-008 正确性与梯度门

必须通过：

- Phase 0 corpus 全量 importer/verifier；
- round-trip 精确相等；
- state、expectation、measurement 与 wire ordering 差分；
- Parameter/ParameterExpression identity；
- complex64/complex128；
- PyTorch forward 与 gradient parity；
- 固定 seed 下的确定性；
- property/fuzz negative tests；
- 公共 API snapshot 无变化。

### P1-009 性能门

对 P0-006 已批准预算执行机器检查：

- importer 和 verifier 无超线性意外增长；
- 小线路固定开销在预算内；
- 10K gate 构建、验证和 hash 在时间/内存预算内；
- 相同结构的重复导入可安全命中缓存时，记录 hit/miss；
- 默认 legacy `fq.run` 不因仅安装内部 IR 而出现可测新增路径；
- 任何超预算结果必须记录 blocker，不允许放宽预算使 CI 变绿。

### Phase 1 退出门

- [ ] P1-001 至 P1-009 全部完成；
- [ ] 当前支持的静态 CircuitIR corpus 全量通过；
- [ ] verifier 负向 fixture 全量通过；
- [ ] state、expectation、measurement、gradient 和 ordering 差分通过；
- [ ] identity、printer、pipeline digest 在支持平台上确定；
- [ ] 性能与内存满足 P0-006 的批准预算；
- [ ] Stable Core、CircuitIR schema 1.0、默认 run/plan 路径无变化；
- [ ] internal 模块没有进入 root、稳定 namespace 或用户自动补全；
- [ ] 新路径具备单开关回滚和完整删除方案；
- [ ] API owner 与 compiler owner 批准 Phase 1 evidence；
- [ ] 未经新提案不得自动进入 Phase 2。

## 7. 建议实施顺序

```text
P0-001 ─┐
P0-002 ─┼─> P0-003 -> P0-004 -> P0-005 -> P0-006
ADR 001–006 ┘                         |
                                      v
                              Phase 0 approval
                                      |
                 +--------------------+-------------------+
                 v                                        v
             P1-001                                    P1-002
                 +--------------------+-------------------+
                                      v
                                   P1-003
                                      |
                                      v
                                   P1-004
                                      |
                           +----------+----------+
                           v                     v
                        P1-005                P1-006
                           +----------+----------+
                                      v
                                   P1-007
                                      |
                           +----------+----------+
                           v                     v
                        P1-008                P1-009
                           +----------+----------+
                                      v
                              Phase 1 evidence review
```

Phase 0 可并行收集事实，但 ADR 决策必须在 importer/value 实现前完成。Phase 1 不采用
多条长期并行产品路径；同一里程碑完成后再扩大语义范围。

## 8. 每个工作包的合入要求

每个 PR/提交必须包含：

- 工作包编号与明确范围；
- 行为或架构变化说明；
- public API impact：必须为 `none`，否则停止并走 API proposal；
- focused tests；
- 必要的负向测试；
- 性能影响或“不在热路径”的证据；
- rollback 方法；
- 已知 blocker；
- capability maturity 是否变化；Phase 0–1 默认不升级公开成熟度。

最小检查：

```bash
python tools/public_api_snapshot.py
python tools/check_architecture.py
python tools/check_repository_hygiene.py
python tools/docs_source_of_truth.py --check
python -m pytest tests/internal_ir -q
python tools/ci_tier.py pr-default
```

涉及现有 compiler/runtime 适配时，再运行 `pr-runtime`。涉及 distributed metadata 或
执行语义时，再按 `AGENTS.md` 运行 `pr-distributed`；CPU 结果不能作为真实扩展性证据。

## 9. Evidence manifest

Phase 0 和 Phase 1 各维护一份机器可读 evidence manifest，至少包含：

```text
schema_version
phase
source_commit
environment
public_api_contract_hashes
circuit_ir_schema_version
corpus_hash
supported_opcode/profile
test_commands
test_results
correctness_tolerances_by_backend_and_dtype
performance_budget
performance_results
known_blockers
rollback_path
owners
approval
```

Manifest 只引用可重建 evidence，不提交凭据、账户、内部地址、机器身份或无界 benchmark
原始数据。未获批准的 manifest 不得把状态写成 release-certified。

## 10. 回滚策略

Phase 1 必须保持结构性可回滚：

1. 默认生产路径不依赖 `_compiler`；
2. importer、verifier、差分桥可整体删除；
3. 不迁移或删除 legacy compiler/runtime；
4. 不改变公共序列化数据；
5. cache 使用独立 namespace 和版本；
6. evidence 与 debug textual IR 不成为用户数据依赖；
7. 发现语义、性能或维护成本不可接受时，保留 Phase 0 基线和 ADR，撤回 Phase 1 代码。

## 11. 启动建议

推荐第一个实施批次只做：

1. P0-001 公共合同保护清单；
2. P0-002 消费关系矩阵；
3. IR-001、IR-002、IR-003 三份核心 ADR；
4. P0-003 characterization corpus 设计，不立即创建新 IR 类型。

该批次评审通过后，再采集 P0-005 性能基线并批准 P0-006 预算。只有 Phase 0 退出门
全部满足且 API owner 明确批准，才开始 `_compiler/ir` 的 Phase 1 实现。
