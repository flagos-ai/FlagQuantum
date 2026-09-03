# ARCH-006：Platform/Execution Provider 两层模型与准入依赖

状态：Proposed

日期：2026-09-03
依据：细化已批准 ARCH-001；本提案不注册、认证或实现 Provider

## 上下文

PlatformRuntime、ExtensionRegistry、Deployment `QuantumProvider` 与 Runtime/Simulation 执行
路径各自可用，但没有共同的 Core-owned 替换合同。Platform 提供设备/kernel/精度/通信事实；
Execution Provider 接收完整执行请求。两者生命周期、失败域和证据不同，不能合成胖接口。

## 决策候选

### Platform Provider

提供 discovery、identity、device lifecycle、memory、kernel/precision、communication/topology
事实和受控进程内 handle。输出 capability discovery/evidence，不接收完整 execution request，
不拥有 planner、任务轮询或用户结果。

### Execution Provider

消费完整 request 与 target snapshot，最小异步协议候选为 capabilities、submit、status、result；
handle/status 必须可序列化。cancel、calibration、realtime、gradient、checkpoint 是经 capability
声明的窄扩展。Runtime 驱动 polling、deadline、retry/cancel orchestration 和 evidence assembly。

Simulation Execution Provider 组合 Simulation Engine 与一个 Platform Provider；QPU/Remote
Service Provider 不依赖模拟算法。ExtensionRegistry 保持唯一扩展注册与生命周期权威，不再建
第二 registry。

Agent Services 是 **Agent-facing deterministic application services**：它向 Agent/协议适配层
提供确定性的 capabilities、validate、plan、preflight、execute/explain 候选操作。LLM 与
Reasoning Layer 位于外部 Compute Service，可替换、可关闭，也可以组合这些操作，但不得绕过
artifact/schema 校验、capability fail-closed、既定 plan identity 或 result/evidence 组装。自然
语言解释不覆盖结构化事实。

## 禁止事项

- 不把两个层级合并成带大量 optional 方法的通用 Provider。
- 不让 vendor object、live job、credentials 或不可序列化 stream/event 越过 adapter。
- 不因 provider 被发现、contract conformance 通过或 A800 开发材料而宣称国产卡/QPU/生产能力。
- 不允许 Provider 自行重写 request policy、吞掉未知状态或静默 backend/CPU fallback。
- 不把 LLM、MCP、租户状态或长期任务控制面放入主仓库 Agent Services，也不允许外部
  reasoning 直接调用数值实现绕过确定性服务。

## 兼容性

现有 PlatformRuntime、Deployment provider 与 Extension SDK 保持当前权威，先由 adapter 满足
候选合同。稳定扩展协议的 `Any` 收紧、根导出、Deployment schema 或 exception 变化均需
独立兼容/API 提案。native state/error 放入 namespaced extension，但标准终态和 failure
category 必须闭合。

## 准入依赖与迁移顺序

1. 先批准 ARCH-002～005 对应 artifact、capability、request、result/evidence 版本化合同。
2. Core 提供两个窄 protocol、fake、failure taxonomy 和 conformance suite。
3. CPU Platform 与 Local Simulation 先通过；证明同一 Simulation Engine 可替换 Platform。
4. Remote/QPU contract fake 通过；证明 Runtime driver 可替换 Execution Provider。
5. 逐个接入真实 adapter；每个都提交 capability-specific 环境、失败、回退和证据材料。
6. 真实硬件/服务审核通过后才更新 capability maturity；最后退出旧 provider lifecycle/result。

### 各团队进入实现阶段的条件

| 团队 | 后续准入条件 |
| --- | --- |
| Core | 五份 ADR 经评审，逐项确认哪些新增类型仅为内部候选、哪些需要 API Change Proposal；先交付版本化值对象、fake 与 conformance，不改稳定导出。 |
| Compiler | artifact/capability/request 合同已落基线；新实现只产出 Core-owned artifact/decision，并维持当前稳定 plan、失败与 identity 行为。 |
| Runtime | request/result/evidence 及 provider fake 已落基线；attempt driver 不导入 Compiler 实现，不承担 durable task 控制面。 |
| Simulation | Simulation 请求/结果投影和 Platform fake 已批准；首个 local statevector 切片保留 autograd、dtype/device 和 fail-closed 行为。 |
| Platform | capability 四态、JSON-safe identity/handle 和证据上限已批准；真实设备测试按具体硬件、kernel、dtype、通信范围提交。 |
| Execution | handle/status/failure/result/evidence 合同及 Runtime driver fake 已通过；真实 provider 另需 sandbox、位序、幂等、取消与校准证据。 |
| Ecosystem | Core metadata 值代数和 wire/bit/parameter 语义已批准；adapter 不泄漏外部对象，执行代码迁到 Execution Provider 边界。 |
| Agent | capability vocabulary 和 application-service schema 已批准；服务只消费 Core snapshot/request/result，协议、租户和长期任务仍留在外部服务。 |

## 验收测试

- Platform：JSON-safe identity、unknown/unmeasured、生命周期清理、无静默设备替换；
- Execution：五态、not-ready、timeout、幂等、取消竞态、可恢复 handle、identity mismatch；
- 同一 Engine 在 CPU 与 fake platform 间替换，消费者不变；
- Runtime driver 在 Local Simulation 与 remote fake 间替换，消费者不变；
- 非对称 counts/bit order、provider recompilation、calibration snapshot 与 executed artifact 证据；
- 真实 accelerator/multinode/QPU 准入必须运行对应硬件测试和审计，Mock 不计。

## 未决问题

- Core protocol 使用同步还是 async-neutral 端口，以及 streaming/realtime 的独立边界；
- provider handle 的恢复期限、幂等键作用域和 cancel-after-terminal 语义；
- Platform stream/event 受控 handle、topology source 与通信 provider 的责任分界；
- 首个真实 QPU sandbox、国产设备和多节点认证的环境及审批主体。
