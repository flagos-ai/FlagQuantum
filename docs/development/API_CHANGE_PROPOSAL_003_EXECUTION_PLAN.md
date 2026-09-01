# API Change Proposal 003: ExecutionPlan Identity and Execution Semantics

## 状态

**Root approved — 已完成实现、全量验证和根级清单迁移，等待 contract freeze。**

- 目标版本：首次公开 alpha；
- 影响接口：`fq.ExecutionPlan`、`fq.plan`、`fq.run`、`Circuit.plan`、
  `Circuit.run`、`ExecutionResult.plan`；
- 机器可读候选：`contracts/execution-plan-v1-candidate.json`；
- 前置决策：Proposal 001 将 `ExecutionPlan` 列为 Stable Core 计划新增项，
  Proposal 002 已冻结 `ExecutionOptions` 及当前 program 执行入口。

批准记录：API owner 于 2026-08-31 通过明确用户指令批准进入实现阶段。该授权允许实现
`fq.run(plan)` 及相应的 `fq.run` 输入命名迁移，但不允许把 `ExecutionPlan` 加入稳定
根清单，也不等于 contract freeze。

根级批准记录：API owner 于 2026-09-01 通过明确用户指令批准将 `ExecutionPlan`
加入 Stable Core 根清单。该批准不等于整个 FlagQuantum API 或 Proposal 003
序列化契约的最终 freeze。

## 问题

当前 `fq.plan(program)` 返回的对象主要描述电路分析、layer、内存估算和推荐 mode，
但它还不是执行时的唯一事实来源：

1. plan 不拥有规范化后的 `CircuitIR`，无法独立执行；
2. plan 没有稳定 identity、schema version、options 或环境指纹；
3. `runtime_config`、`routing_plan` 等自由格式 mapping 无法成为长期序列化契约；
4. `fq.run(program)` 会重新规划，用户看到的 plan 不一定就是实际执行的 plan；
5. `ExecutionResult.plan` 仍允许多种内部对象，无法与显式计划核对；
6. 环境或插件变化后没有统一 stale-plan 检测，执行器可能重新选择或 fallback。

因此，目前不能专业地承诺：

```python
plan = fq.plan(circuit, options=options)
result = fq.run(plan)
```

## 决策摘要

`ExecutionPlan` 将成为由 `fq.plan` 创建的、不可变、带身份且可直接执行的本地计划。
`fq.run(program, options=...)` 必须在内部执行同一条 `plan → validate → execute` 路径，
而 `fq.run(plan)` 必须执行传入计划，不得重新规划。

```python
automatic = fq.run(circuit, options=options)

plan = fq.plan(circuit, options=options)
explicit = fq.run(plan)

assert automatic.plan.identity == plan.identity
assert explicit.plan.identity == plan.identity
```

`ExecutionPlan` 可以序列化和跨进程恢复，但只在满足其版本、插件和环境约束时执行。
它不是可提交给云平台或 QPU 的部署包；跨机器长期交付、签名、凭证和 provider job
参数继续由 `DeploymentPackage` 承担。

## 对象边界

### 创建方式

稳定创建入口只有：

```python
fq.plan(program, *, options=None)
fq.ExecutionPlan.from_dict(payload)
fq.ExecutionPlan.from_json(text)
```

`ExecutionPlan(...)` 构造器不是公开契约。这样可以在不允许用户伪造半有效计划的前提下，
继续演进内部 planner 数据结构。`from_dict/from_json` 必须执行完整 schema、identity 和
结构验证；它们不是绕过 planner 验证的后门。

### 本地计划与部署包

```text
CircuitIR + ExecutionOptions + 当前能力环境
                  |
                  v
            ExecutionPlan
       可检查 / 可缓存 / 可恢复 / 可执行
                  |
                  | 显式部署转换
                  v
           DeploymentPackage
       可签名 / 可提交 / provider 生命周期
```

`ExecutionPlan` 不包含：

- provider 凭证、job id、队列或重试策略；
- Python callable、Torch/JAX live object、CUDA graph 或进程组句柄；
- 未版本化的 pickle；
- 绝对文件路径、主机名、PID、rank id 或创建时间；
- 原始环境变量和自由格式 runtime config。

## 稳定公开表面

`ExecutionPlan` 是 factory-only 的不可变值对象。首版稳定公开属性为：

| 属性 | 类型 | 含义 |
| --- | --- | --- |
| `identity` | `str` | 计划规范身份的 64 字符小写 SHA-256 十六进制串 |
| `schema_version` | `str` | ExecutionPlan 序列化 schema 版本，首版为 `"1.0"` |
| `program_fingerprint` | `str` | 完整规范化 `CircuitIR` 的内容哈希 |
| `options_fingerprint` | `str` | resolved execution semantics 的规范哈希 |
| `environment_fingerprint` | `str` | 所需环境约束的规范哈希，不是整台主机指纹 |
| `compiler_fingerprint` | `str` | 编译 pipeline、pass 和相关插件版本的规范哈希 |
| `mode` | `str` | 最终数学状态表示 |
| `backend` | `str` | 最终执行 backend |
| `device` | `str` | 最终逻辑设备约束 |
| `target` | `str` | 最终输出目标 |
| `batch_size` | `int` | 计划采用的批大小 |
| `precision` | `str` | 计划采用的复数精度 |
| `world_size` | `int` | 计划要求的逻辑 rank 数 |
| `state_bytes` | `int` | planner 估算的状态工作集字节数 |
| `is_distributed` | `bool` | `world_size > 1` 的只读派生值 |

稳定方法为：

```python
plan.summary() -> Mapping[str, object]
plan.to_dict() -> dict[str, object]
plan.to_json(*, indent: int | None = None) -> str
ExecutionPlan.from_dict(payload) -> ExecutionPlan
ExecutionPlan.from_json(text) -> ExecutionPlan
```

`analysis`、`layers`、候选 cost、routing 细节、backend kernel 选择和 noisy execution
细节首版不作为稳定属性冻结。它们进入版本化序列化 payload 的受控内部 section，或通过
后续稳定诊断接口提供，不能继续依赖自由格式公开 mapping。

## 身份模型

### 规范 identity

计划 identity 定义为：

```text
sha256(canonical_json(identity_payload))
```

其中 `canonical_json` 必须使用 UTF-8、排序后的对象 key、确定的数值编码和无额外空白。
`identity_payload` 只包含影响执行语义或可执行性的内容：

```text
ExecutionPlan schema name/version
program_fingerprint
resolved execution semantics
compiler fingerprint
required environment constraints
selected execution decision
versioned backend/compiler extension identities
```

下列内容禁止进入 identity：

- 创建时间、运行时间、hostname、PID、当前 rank；
- 可用内存的瞬时采样和性能计数器；
- mapping 插入顺序、Python `repr`、对象地址；
- 日志路径、缓存路径、凭证和 provider transport metadata。

相同 IR、resolved options、编译 pipeline、环境约束和 planner 决策必须产生相同 identity。
若两个不同的用户输入最终解析为完全相同的执行语义，它们可以共享 identity。

### 环境约束

`environment_fingerprint` 是**计划所需能力集合**的哈希，不是当前机器所有属性的哈希。
约束至少包含：

- backend/provider 名称及兼容协议版本；
- device kind 和必需 capability；
- precision、world size 和 distribution semantics；
- 必需的通信、gradient、noise 或 approximation capability；
- 影响可执行性的 compiler/backend extension 版本。

当前环境可以是约束的超集。CPU 型号、GPU 序号或剩余显存变化不应仅凭字符串不同就使
计划失效；但 backend 不可用、world size 不满足、精度不支持或插件协议不兼容必须使
计划在执行前失败。

## 序列化边界

首版 schema：

```json
{
  "schema": "flagquantum.execution_plan",
  "version": "1.0",
  "identity": "<sha256>",
  "program": {"kind": "flagquantum.circuit_ir", "version": "1.0"},
  "requested_options": {"schema": "flagquantum.execution_options", "version": "1.0"},
  "resolved_options": {},
  "fingerprints": {
    "program": "<sha256>",
    "options": "<sha256>",
    "environment": "<sha256>",
    "compiler": "<sha256>"
  },
  "environment_requirements": {},
  "decision": {},
  "extensions": []
}
```

具体字段由机器可读候选约束。规则如下：

- `program` 使用现有版本化 `CircuitIR` schema；
- `requested_options` 保留字段级继承意图；`resolved_options` 保存执行采用的具体值；
- `decision` 只保存执行所需的规范决策，不保存 live runtime object；
- extension 必须具有 namespace、kind、schema version 和 identity；
- unknown 顶层字段、schema/version 不支持、重复 extension 或 identity 不匹配必须拒绝；
- `from_dict` 必须重新计算所有 fingerprint 和 identity；
- 首次公开 alpha 冻结后，同一 major schema 必须能读取已有 fixture；
- 反序列化成功不等于当前环境可执行，环境验证发生在 execution preflight。

禁止使用 pickle 作为稳定序列化格式。

## `plan` 与 `run` 语义

批准并实现后的目标签名：

```python
fq.plan(
    program,
    *,
    options: ExecutionOptions | None = None,
) -> ExecutionPlan

fq.run(
    program_or_plan,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
    noise_model: Any | None = None,
) -> ExecutionResult
```

### program 输入

当输入为 program 时：

1. 规范化为 `CircuitIR`；
2. 使用 Proposal 002 的唯一 resolver 解析 options；
3. 创建 `ExecutionPlan`；
4. 验证计划 identity 与当前环境；
5. 执行该计划，不再调用第二套 planner；
6. `ExecutionResult.plan` 返回实际执行的稳定计划。

### plan 输入

当输入为 `ExecutionPlan` 时：

- `options`、`measurements` 和 `noise_model` 必须全部为 `None`；
- 任一额外执行语义存在时立即抛出 `TypeError`，禁止覆盖或合并；
- 先验证 schema、identity、环境、backend/compiler extension compatibility；
- 直接执行计划中封存的 IR、resolved options 和 decision；
- 禁止调用 planner，禁止重新编译会改变 identity 的 pipeline；
- 禁止静默 fallback、缩小 world size、降低精度或改变 mode；
- `result.plan is plan` 应在同一进程执行中成立；跨序列化恢复至少保证 identity 相等。

### measurement 与 noise 边界

本节原先延后给 Proposal 004，现已由该提案取代：measurement 在 planning 前进入规范
`CircuitIR`，noise model 进入经过 identity 校验的 versioned plan extension；两者均不再
通过 transient execution 绕过 plan。已有 plan 仍禁止附加任何临时调用参数。

## stale plan 与失败阶段

执行 preflight 按固定顺序检查：

1. schema/version；
2. payload 结构与 extension 唯一性；
3. program/options/compiler/environment fingerprint；
4. plan identity；
5. 当前 backend、device、precision、world size 与 capability；
6. executable extension compatibility。

失败必须在启动 kernel、创建分布式 worker 或提交 provider job 前发生。至少提供稳定
reason code：

```text
unsupported_schema
identity_mismatch
program_fingerprint_mismatch
options_fingerprint_mismatch
compiler_incompatible
environment_incompatible
backend_unavailable
world_size_mismatch
extension_incompatible
```

精确公开异常类由统一错误模型提案决定；在该提案完成前，Proposal 003 只冻结失败阶段、
reason code 和“不重新规划、不 fallback”的行为，避免永久承诺偶然的底层异常类型。

## 与当前实现的迁移

当前 `flagquantum.compilation.models.ExecutionPlan` 是内部迁移来源，不直接宣布稳定。
实施时应：

1. 新建规范 plan payload 和 identity helper；
2. 将 `CircuitIR`、requested/resolved options、环境约束和 planner decision 纳入 plan；
3. 用只读属性替代自由格式稳定字段，backend 细节迁入版本化 extension；
4. 增加 `_execute_plan(plan)`，让自动和显式路径共用；
5. 先迁移内部调用和 `ExecutionResult.plan`，再开放根级 `ExecutionPlan`；
6. 在批准后更新稳定 manifest 和签名 snapshot；
7. 保留旧内部 analysis/layer 数据，但不把它们误冻结为永久公开字段。

仓库尚未开源，因此不为未承诺的内部 `ExecutionPlan(...)` 构造形式建立弃用负担。

## 明确排除

本提案不决定：

- `ExecutionResult` 所有字段、accessor 和 metadata schema；
- measurement replacement/composition 的最终规则；
- noise model 的稳定序列化位置；
- deployment package、签名、provider submission 或远程 job 生命周期；
- backend/provider/plugin 注册协议本身；
- plan 跨 FlagQuantum major schema 的永久可执行保证；
- 性能缓存、编译产物二进制和 CUDA graph 的稳定格式。

## 验收标准

- [x] API owner 批准对象边界、稳定属性和 factory-only 构造规则；
- [x] API owner 批准 identity 输入和排除项；
- [x] API owner 批准序列化 schema 与兼容窗口；
- [x] API owner 批准 plan 输入时不接受任何 options/measurement/noise 覆盖；
- [x] `fq.plan` 生成的计划可独立执行且不调用第二次 planner/compiler；
- [x] 自动路径与显式计划路径 identity、输出和 provenance 等价；
- [x] stale/tampered/environment-incompatible plan 在执行前 fail closed；
- [x] 可执行 plan 路径的 `ExecutionResult.plan` 是实际执行的 `ExecutionPlan`；
- [x] round trip、unknown field、identity tamper 和 schema 拒绝测试通过；
- [x] default、runtime、distributed 和文档契约通过；
- [x] API owner 单独批准根级导出；
- [ ] API owner 单独批准 contract freeze。

## 实施记录

2026-08-31 完成实现与验证：

- `ExecutionPlan` 已具备规范 program/options/environment/compiler fingerprint、SHA-256
  identity、严格 JSON round trip、篡改检测和 world-size preflight；
- `fq.run(program)` 的常规路径与 `fq.run(plan)` 共用唯一 exact-execution path；
- plan 输入拒绝 options、measurements 和 noise_model 覆盖，并由契约测试证明不会再次调用
  planner 或 compiler；
- measurement 与 noise program 路径已由 Proposal 004 收敛到同一 exact-execution path；
- default、runtime、distributed、benchmark/release contract、API snapshot、architecture、
  Ruff 和 Black 均通过；`ExecutionPlan` 根级导出已于 2026-09-01 获批，contract
  freeze 仍未启用。

## 请求批准的决策

本提案已经获得实现授权和第二次根清单批准，`ExecutionPlan` 已从 planned addition
移入 Stable Core。该批准**不等于 contract freeze**，也不授权改变
measurement/noise/result 的其他稳定语义；冻结仍需 API owner 独立审批。
