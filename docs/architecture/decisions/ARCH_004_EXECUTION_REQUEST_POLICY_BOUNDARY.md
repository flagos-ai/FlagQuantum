# ARCH-004：Execution Request 与 policy 边界

状态：Proposed

日期：2026-09-03
依据：Phase 0 八团队盘点；不改变 `fq.run`、`fq.plan` 或稳定 options

## 上下文

仓库尚无 Core 权威 `ExecutionRequest`，并存在稳定 Runtime `ExecutionOptions`、Compiler 私有
同名 options、`RequestedExecution`、importer request 与自由格式 distributed request。
请求事实、用户授权、planner 决策、资源发现和 provider 凭据目前容易混入同一对象。

## 决策候选

1. Core request 是不可变意图信封：artifact identity、parameter/input bindings、measurement/
   output requirement、target constraints、稳定 options 的兼容投影、precision/approximation/
   fallback 授权、budget/seed/deadline/recovery limits 和 evidence level。
2. Policy 是 Runtime/Compiler/Compute Service 各自拥有的决策行为，不进入 request identity；
   request 只记录调用者明确约束和授权。policy 输出必须成为带 identity 的 plan/decision。
3. Runtime 单次 attempt 接收已验证 request/plan 和有界 resource lease；tenant、auth、quota、
   billing、queue、跨 attempt retry/idempotency 属于外部 Compute Service envelope。
4. provider credentials、process group、device handle 和 vendor SDK object 只在适配器私域解析。

## 禁止事项

- 不复制稳定 `ExecutionOptions` 字段或改变其默认值、优先级、Literal 和异常行为。
- 不把 planner selection、发现到的资源或实际 fallback 伪装成用户请求。
- 不用 `Mapping[str, Any]`、Runtime 私有类型或 Compiler re-export 代替 Core 边界。
- 不允许 Runtime 在执行既定 plan 时静默重规划、重编译或替换目标。

## 兼容性

首版通过 adapter 从现有 `fq.run/plan` 调用形态构造 request，公开签名保持不变。Compiler
私有 `ExecutionOptions` 必须消歧为 binding-specific 概念。任何对稳定 options、plan/result
字段或失败阶段的变化另需 API Change Proposal。

## 迁移顺序

1. 批准 artifact/capability 身份依赖和 request schema 草案。
2. 提供 request fake、strict parser 和稳定 options adapter。
3. Compiler 产出 plan decision；Runtime 只执行已决定内容。
4. 移除 distributed free-form request 和私有同名 options。
5. 外部 Compute Service 通过 anti-corruption adapter 加入 durable-task envelope。

## 验收测试

- canonical identity、未知字段/版本拒绝、binding/seed/deadline 确定性；
- unsupported requirement 在最早可知阶段失败；未授权 fallback 必须失败；
- `fq.run(program)` 与 `fq.run(fq.plan(program))` 保持既有等价，plan 输入不重规划；
- tenant/credential/vendor/process-group 对象不能序列化进 request；
- Runtime fake consumer 可在两个 Compiler/Provider 实现间替换而不改调用代码。

## 未决问题

- deadline 与 timeout 的时钟语义、单 attempt retry 的最小授权范围；
- tensor/initial-state 是进程内 handle 还是 artifact reference；
- measurement request 与稳定 `ExecutionOptions` 的最终组合方式及 request minor-version 策略。
