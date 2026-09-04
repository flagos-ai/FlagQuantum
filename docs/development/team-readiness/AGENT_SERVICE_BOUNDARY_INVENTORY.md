# Agent Services 边界盘点

状态：Agent Services 团队交付盘点，2026-09-03

适用提交：`codex/vnext-team-agent-services` 本轮提交

路径分类：`single_device_fast_path` 的确定性发现、校验与规划入口；本文不形成分布式或性能声明

## 结论

`flagquantum._agent_services.AgentApplicationService` 已经是无协议类型、无 LLM、无数值
Kernel、无具体设备和无厂商 SDK 导入的确定性应用服务。MCP Python SDK 不在主仓库依赖
清单中；在导入钩子明确拒绝 `mcp` 和 `fastmcp` 的独立进程里，Agent Services 校验和
本地 `fq.run` 均可运行。

当前边界尚未完全收敛。应用服务通过同团队的 `flagquantum.agent` 门面间接使用现有
Compiler、Runtime 和 Deployment 能力，其中 `flagquantum.agent.capabilities()` 仍直接
导入 `flagquantum.runtime.backend_registry`。这不是协议耦合，但仍是待 Runtime/Core
提供正式能力发现契约后移除的内部模块耦合。当前应用服务仅公开 capabilities、validate
和 plan；execute 与独立 explain 服务尚未建立，不应描述为已经交付。

## 当前能力清单

| 能力 | 当前权威入口 | AgentApplicationService | 输入/输出与失败语义 | 当前判断 |
| --- | --- | --- | --- | --- |
| capabilities | `flagquantum.agent.capabilities` | `capabilities(refresh=False)` | 返回 `flagquantum_agent_capabilities_v1`；单个后端探测失败降为该后端的不可用记录 | 已有；经 `runtime.backend_registry` 间接耦合 Runtime 内部 |
| validate | `flagquantum.agent.validate` | `validate_program(program)` | 接受序列化 `CircuitIR` 或 circuit `ProgramArtifact`，返回结构化 `ValidationReport`；坏 schema/Artifact 直接拒绝 | 已有，确定性 |
| plan | `flagquantum.runtime.planner.plan_runtime_selection`，由 `flagquantum.agent.preflight_execution` 适配 | `plan_execution(program, options=None)` | 返回 `AgentExecutionPlan` 字典；预期规划异常成为 `EXECUTION_PLANNING_FAILED` blocker | 已有，确定性，不执行 Kernel |
| preflight | `flagquantum.agent.preflight_execution`、`preflight_deployment` | execution preflight 与 plan 合并；deployment preflight 未暴露 | execution 已有；deployment 仅存在于旧 Agent 门面 |
| execute | Stable Core 的 `fq.run` | 无 | 本地 SDK 可执行并返回 `ExecutionResult`；应用服务没有 execute 方法 | 缺口；不得由 MCP 网关绕过服务直接补齐 |
| explain | `RuntimeSelectionPlan.summary()` 和结构化 blocker | 无独立方法 | plan 内容可供解释，但没有版本化 explain 请求/响应 | 缺口；不得把 LLM 文本当作权威解释 |

本轮补上了 `ProgramArtifact.required_capabilities` 的 fail-closed 检查。服务从已安装能力
manifest 中构造稳定集合；缺失项按字典序返回
`REQUIRED_CAPABILITY_UNAVAILABLE`，且不会调用 planner。当前支持的匹配词汇来自：

- 可用 backend 的注册名和规范名；
- 为真的 `supports_*` 字段去掉前缀后的名称；
- backend 的 accelerator、device 和 dtype 值；
- 为真的 Agent contract 名称。

这只是已有 manifest 的适配规则，不是新的跨领域能力契约。正式能力词汇及兼容规则仍应
由 Core/集成团队批准的版本化契约定义。

## 依赖审计

审计范围为 `flagquantum/_agent_services/**/*.py`、`flagquantum/_gateways/**/*.py`，并追踪
了其同团队门面 `flagquantum/agent.py` 的一跳依赖。

| 被检查依赖 | `_agent_services` 直接依赖 | `_gateways` 直接依赖 | 一跳/遗留耦合 | 结论 |
| --- | --- | --- | --- | --- |
| MCP SDK / FastMCP | 无 | 无 Python 实现 | 无 | 通过；主仓库没有 MCP 运行时依赖 |
| LLM SDK/模型调用 | 无 | 无 | 无 | 通过；服务输出只由输入、安装能力和确定性代码决定 |
| Simulation 内部实现 | 无 | 无 | 无 | 通过；未导入 `simulation` 或数值 Kernel |
| Runtime 内部模块 | 无 | 无 | `agent.capabilities -> runtime.backend_registry` | 未完全收敛；属于能力发现适配债务 |
| 具体设备/厂商 SDK | 无 | 无 | Runtime backend 探测可在其自身边界加载可选实现 | 通过直接边界；厂商对象未进入服务输入/输出 |
| Runtime | 无内部 `_compiler` 导入 | 无 | `agent.preflight_execution -> runtime.planner` | 确定性规划门面；最终应由正式公共规划契约替代 |
| Deployment | 无 | 无 | `agent.preflight_deployment -> deployment` | 仅旧 Agent 门面存在，未暴露为当前应用服务方法 |

`tests/team/agent/test_agent_service_boundaries.py` 对以上直接禁止导入做 AST 特征检查，覆盖
协议 SDK、常见 LLM SDK、Simulation、Runtime、backend/platform/device 和厂商 SDK
命名空间。架构检查器另行禁止 `_gateways` 直达 Compiler、Runtime、Simulation 和
Deployment，并禁止 `_agent_services` 直达 backend、platform 与 Simulation。

### 已知协议耦合点

主仓库内为零。现实 MCP 耦合位于外部 Compute Service 的
`backend/src/tovx_control/api/mcp/flagquantum_server.py`；只有该适配器导入官方 MCP SDK，
并把 MCP tools/resources 调用映射到外部服务自己的应用服务。它不应被复制回主仓库。

## 主仓库与外部 Compute Service 所有权

### FlagQuantum 主仓库拥有

- `CircuitIR`、`ProgramArtifact` 及其规范序列化、内容身份和拒绝规则；
- 本地安装环境的 backend/能力事实发现；
- 程序语义校验、确定性规划与 execution/deployment preflight；
- 单次本地或分布式执行语义、`ExecutionResult` 和运行证据；
- 编译、Runtime、Simulation 与 Provider 的领域规则；
- 对外应用服务契约中与量子程序语义相关的错误类别和 conformance fixtures。

### 外部 Compute Service 拥有

- MCP、REST、gRPC 等网络协议及 SDK 生命周期、请求大小和超时控制；
- 租户、principal、认证、授权 scope、凭据处理和审计主体；
- 配额、预算、计量、计费策略和并发限制；
- 持久化 preflight、任务、状态机、幂等提交、取消、重试与结果保留；
- worker 心跳与能力观察、队列和长期任务调度；
- 将主仓库结构化结果映射为传输响应，但不得改写科学判断、能力事实或证据结论。

主仓库本轮未实现、也不应实现 tenant、authentication、billing、quota 或 persistent job。
外部仓库已有这些模块和 MCP transport；它们是服务控制面能力，不是 Agent Services 的
本地应用语义。

## 跨仓库版本化 Application Service Contract 提案

这是待集成团队批准的契约需求，不在本团队分支创建 `contracts/` 文件或新的 Stable Core
类型。建议契约名为 `flagquantum.agent_service.v1`，由 FlagQuantum 集成/Core 维护规范
schema 和 fixture corpus，主仓库与 Compute Service 各自运行同一组 conformance tests。

### 必需操作

| operation | 请求核心 | 响应核心 | 副作用 |
| --- | --- | --- | --- |
| `capabilities.get` | refresh policy | runtime identity、capability snapshot、blockers | 只读 |
| `program.validate` | `ProgramArtifact` | validation issues、required/available capabilities | 无 |
| `execution.plan` | `ProgramArtifact`、typed planning options | plan identity、selection、fallback、blockers | 无 |
| `execution.preflight` | plan identity、capability snapshot identity | executable、resource estimate、blockers、warnings | 无执行 |
| `execution.execute` | Core-owned `ExecutionRequest` 或有效 plan identity | Core-owned `ExecutionResult`/Evidence | 一次执行；长期任务生命周期仍在外部服务 |
| `plan.explain` | plan/result identity | 结构化 decisions、blockers、fallback facts、evidence references | 只读；自然语言可由外部层生成但不具权威性 |

### 必需契约规则

1. 每个请求和响应都有精确 schema id、major/minor 版本和规范 JSON 编码；身份摘要不包含
   transport、tenant、credential 或展示字段。
2. major 不兼容必须在调用前失败；minor 只允许明确标记的可选字段。未知字段默认
   fail closed，不能由 MCP SDK 的宽松解析吞掉。
3. `ProgramArtifact.kind`、envelope version、payload schema 和
   `required_capabilities` 在规划前校验；未知 Artifact 不降级为 CircuitIR 猜测。
4. 错误至少区分 `INVALID_ARTIFACT`、`UNSUPPORTED_ARTIFACT_KIND`、
   `CONTRACT_VERSION_UNSUPPORTED`、`REQUIRED_CAPABILITY_UNAVAILABLE`、
   `VALIDATION_FAILED`、`PLANNING_FAILED`、`PREFLIGHT_FAILED` 和
   `EXECUTION_FAILED`，并携带稳定 code、retryable、stage 与结构化 details。
5. capability snapshot identity 必须绑定 plan/preflight；能力变化导致显式 stale failure，
   不允许静默重规划或替换 backend。
6. 规范 fixtures 至少覆盖字段顺序无关、内容摘要、未知/缺失字段、版本不兼容、能力不足、
   planner fake、executor fake、错误投影和 REST/MCP 结果等价。
7. 契约替换证据必须包含一个真实实现和一个不依赖 Runtime/Simulation 的 fake；消费者在
   两者间切换时不改代码。
8. idempotency key、tenant、authorization、billing 和 persistent run id 属于 Compute
   Service 外层 envelope，不进入主仓库的科学计算身份。

外部服务当前同时存在 `tovx.flagquantum_contract` 与
`tovx_control.domain.flagquantum_service` 两组服务侧类型。正式联调应让它们消费批准后的
schema/fixtures 或通过明确 anti-corruption adapter 投影，不能把这些服务侧类型反向导入
FlagQuantum 主仓库。

## `_gateways/mcp` 过渡代码与退出条件

当前提交中 `flagquantum/_gateways/mcp` 目录不存在，`flagquantum/_gateways` 只有边界
`AGENTS.md`，所以主仓库内的 MCP 过渡代码清单为 **空**。长期架构文档中的
`_gateways/mcp` 是迁移台账对可能兼容适配器的命名，不是已实现的生产网关。

| 过渡项 | 当前状态 | 退出条件 |
| --- | --- | --- |
| 主仓库 MCP server/lifecycle | 不存在 | 保持不存在；生产 transport 始终由外部 Compute Service 拥有 |
| 主仓库 MCP tool/resource schema | 不存在 | 由批准后的协议无关 Application Service Contract 加外部协议投影替代 |
| 主仓库 MCP auth/session/task 状态 | 不存在 | 保持不存在；不得引入 |
| 临时 MCP-to-Agent adapter（若集成阶段确需新增） | 未新增 | 外部 Compute Service 通过共享 fixtures；REST/MCP parity、无 SDK 本地测试和替换测试通过；主仓库调用者归零后删除 |

外部 Compute Service 的 `tovx_control/api/mcp/flagquantum_server.py` 是隔离的生产协议
适配器，不属于主仓库 `_gateways/mcp` 过渡清单。其自身退出条件是卸载 `/mcp` 和 MCP
依赖后，REST、领域服务、持久化 run/receipt 仍保持可用；这已由外部仓库 adoption record
定义。

## 确定性与无 MCP 验证

团队测试位于 `tests/team/agent/`：

- `test_agent_service_boundaries.py`：静态禁止依赖检查；独立子进程阻断 MCP/FastMCP 后
  导入本地 SDK、调用 Agent 校验并执行 `fq.run`；
- `test_agent_service_determinism.py`：ProgramArtifact 重复规划与输入不变性、未知 schema
  和 kind 拒绝、已知非 circuit Artifact 拒绝、能力不足早失败、规划异常结构化与重复性。

这些测试证明本地确定性和边界语义，不证明生产服务认证、安全、持久化、远程 transport、
多 GPU/多节点扩展性或性能。
