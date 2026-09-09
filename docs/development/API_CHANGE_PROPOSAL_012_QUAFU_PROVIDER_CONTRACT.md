# API Change Proposal 012：Quafu Provider 与门户任务契约收敛

## 状态

**Draft — 核心能力在当前 API 收敛分支实施，受保护合同变更仍需 API-owner 批准。**

- 目标版本：首次公开 alpha；
- 机器契约：`contracts/quafu-provider-contract-v1-candidate.json`；
- 输入依据：`flagquantum-quafu-adapter` Issue #4 与 PR #3；
- 根级稳定 API 变化：无；
- `ExecutionResult`、`ExecutionPlan`、`CircuitIR` 和 `DeploymentPackage` schema：不变；
- FlagQuantum 执行、噪声和结果转换代码只进入当前 FlagQuantum 分支；共享 adapter 仓
  只保留 OpenAPI、Schema、Mock、文档和必要的薄绑定，不复制 FlagQuantum 实现源码；
- 新增稳定类型、字段或异常合同仍需独立批准。

## 问题

API 收敛分支已经区分本地 `ExecutionPlan` 与 provider-facing
`DeploymentPackage`，也建立了扩展协议和部署资产身份链，但 Quafu 对接仍存在五类缝隙：

1. FlagQuantum 本地 counts 按 measurement wire 顺序编码，而门户使用 Qiskit
   classical-string 顺序；
2. provider task 只有字符串状态和通用异常，没有稳定终态、错误分类和 result 失败语义；
3. backend capability 没有表达 shots 有效区间和 QASM 子集版本；
4. result metadata 没有统一要求执行者、耗时、校准快照和噪声模型身份；
5. Quafu adapter 真实路径仍依赖已经从 API 收敛分支移除的
   `fq.experimental.QPUTwin`。

另外，旧实现仍向任务请求写入已废弃的顶层 `compile` 布尔字段，
且未将逻辑线路与有序物理比特映射绑定，会使已编译线路的提交语义不完整。

## 决策

### 0. 已编译线路提交契约

QSteed 等本地编译器输出的 OpenQASM 必须继续使用连续逻辑序号
`q[0]...q[N-1]`，不得将物理比特号写入 QASM。提交时三项信息缺一不可：

- `circuit`：使用逻辑序号的完整 OpenQASM 2.0；
- `options.compiler=None`：明确表示不请求云端再编译；
- `options.target_qubits`：长度为 N 的有序物理比特列表，第 i 项对应逻辑
  `q[i]`。

顶层 `compile` 布尔字段已废弃，FlagQuantum 不再发送、接受或记录该字段。
物理映射缺失、重复、长度不匹配，或 QASM 寄存器宽度与之
不一致时，Provider 必须在网络请求前失败。

### 1. 三层比特序必须分别命名

不得再使用未限定含义的“bit order”。候选合同区分：

| 层 | 规范 |
| --- | --- |
| FlagQuantum sample/counts measurement | `measurement_wires_left_to_right`，字符串位置依次对应请求中的 wires |
| OpenQASM classical register | `classical_msb_left`，字符串从最高 classical bit 到 `c[0]` |
| Quafu portal `result.counts` | `classical_msb_left` |

例如 `x q[0]`、`measure q[0] -> c[0]`、`measure q[1] -> c[1]`：

- FlagQuantum 对 wires `(0, 1)` 的本地 counts 是 `"10"`；
- Quafu portal counts 是 `"01"`。

这不是允许静默不一致，而是要求 provider boundary 做显式转换。Quafu deployment result
必须在 metadata 中携带 `counts_bit_order="classical_msb_left"`。合同测试必须使用不对称
线路，Bell/GHZ 等对称线路不能作为比特序证据。

本提案不改变 Proposal 004 已冻结的本地 `MeasurementResult` 值语义。未来若要将
bit-order 字段提升为稳定 `MeasurementResult` 或 `DeploymentResult` 字段，必须另行批准。

### 2. 私有部署环境迁移到收敛后的执行入口

真实 FlagQuantum 模拟实现不得进入共享 adapter 仓，也不得依赖
`fq.experimental.QPUTwin` 或任何未进入 experimental 可发现面的内部名称。部署环境
安装当前 FlagQuantum 分支后，真实模拟路径为：

```text
normalized Quafu calibration
  -> flagquantum.deployment.quafu_noise_model_from_chip_info(...)
  -> fq.run(circuit, measurements=(probabilities,), noise_model=model)
  -> provider-boundary bit-order normalization
  -> portal counts
```

共享 adapter 只负责把已验证请求交给私有 FlagQuantum 服务，并按双方合同返回结果；
不得复制噪声模型、密度矩阵执行器、GPU 调度或其他 FlagQuantum 源码。噪声模型
identity 作为 `noise_model_version`；校准规范化内容的稳定 hash 作为
`calibration_version`。

### 3. Provider task 状态与错误分类

目标五态为 `Pending / Running / Finished / Failed / Cancelled`。平台原生状态可以更多，
但 adapter 必须保留原始状态并映射到目标五态。`submit` 成功只表示任务已受理，不表示
执行成功。

候选错误分类采用门户现有命名：

```text
precheck.badCircuit
precheck.noMeasurement
precheck.shotsOutOfRange
precheck.tooDeep
precheck.notTranspiled
precheck.qubitUnavailable
precheck.edgeUnusable
precheck.gateUnsupported
precheck.simTooLarge
runtime.deviceUnavailable
runtime.quotaExhausted
runtime.rejected
runtime.internal
```

在稳定异常类型获批前，adapter 可以先在 HTTP schema 中提供 `error_category`；核心
`QuafuProvider` 不得通过改变现有公开异常类型抢跑冻结流程。

### 4. Result 与 capability 最小合同

Quafu portal result metadata 至少应包含：

- `backend`：实际执行者；
- `duration_ms`；
- `calibration_version`；
- `noise_model_version`；
- `counts_bit_order`；
- 实际使用的物理比特集合；
- deployment artifact identity 或 program digest。

Backend capability 后续应显式表达 `min_shots`、`max_shots`、shots 步长、QASM profile、
basis gates、coupling map 和动态线路能力。首次实施可放在 provider-specific metadata，
但不得把自由 metadata 宣称为永久稳定 schema。

### 5. 校准策略只有一个事实来源

Adapter 负责把 Quafu 原始 payload 规范化为带 provenance 的快照，核心转换器消费该快照：

- 无效或冻结 qubit/coupler 不进入可用集合；
- `T2 > 2*T1` 若采用截断，必须记录原值、修正值和规则版本；
- 缺失读出数据若使用估计值，必须标记 `estimated`，不能伪装为测量校准；
- 每次结果必须能追溯到规范化快照 hash；
- 同一快照不得在 adapter 与核心路径采用不同修正规则。

## 分阶段实施

### P0：首次 Mock/真实后端联调前

- [x] 用不对称线路冻结 portal counts 顺序；
- [x] 当前 FlagQuantum 分支支持带 readout noise 的 density probability measurement；
- [ ] 私有部署绑定移除对 `fq.experimental.QPUTwin` 的依赖；
- [ ] 私有服务输出 `counts_bit_order`、`calibration_version`、`noise_model_version`；
- [ ] 保持提交成功与执行终态分离；
- [ ] 在不向 adapter 仓复制源码的环境中运行真实端到端合同测试。

P0 实施期间，验证发现 density-matrix 输出被 measurement 层误识别为
statevector，导致 probability request reshape 失败。API 收敛分支已在不改变公开签名的
前提下修复：density probability 直接读取密度矩阵对角线，应用 NoiseModel readout
confusion，再按 measurement wires 顺序生成边际概率。该行为由 API contract test 保护。

### P1：门户契约冻结前

- [ ] 冻结 `/result` 在 Failed/Cancelled 下的状态码和错误体；
- [ ] 增加 `error_category` schema；
- [ ] 冻结 cancel 状态转换图；
- [ ] 冻结 QASM 子集与寄存器约束；
- [ ] 对齐 shots 能力声明与网关实际范围。

### P2：Provider 产品化

- [ ] 实现公共 provider conformance suite；
- [ ] 归一平台原生状态并保留原始值；
- [ ] 使用快窗口加长尾退避的轮询策略；
- [ ] 记录 provider 二次编译后的实际映射和校准快照；
- [ ] 持久化任务、结果、配额和审计记录。

## 验收边界

- Mock 合同通过不证明噪声精度或硬件能力；
- CPU 合同测试不证明生产吞吐和容量；
- provider 返回的 counts 必须先通过 shot accounting 和 bit-order conformance；
- adapter 与核心包的兼容窗口必须由真实安装测试验证；
- 未经批准，不修改 `docs/public_api_v1.json`、稳定根签名或受保护序列化 schema。
