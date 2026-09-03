# ARCH-003：Capability 的需求、发现与证据三分法

状态：Proposed

日期：2026-09-03
依据：Phase 0 八团队盘点；接口存在不等于能力已实现

## 上下文

当前有 Compiler `TargetCapabilities`、Runtime `BackendCapabilities`、Deployment
`CloudBackendProfile`、Runtime `CapabilityEvidence`、扩展 `CapabilityRequest/Response` 和
Agent `required_capabilities` 匹配词汇。它们分别描述目标约束、动态发现、证明材料与协商，
字段重叠却不能互换。尤其“已发现”和“有接口”不能提升为硬件、生产或扩展性证据。

## 决策候选

Core 将 capability 语义拆成三个互相关联但不继承真值的层次：

1. **Requirement**：artifact/request 对执行所需能力的闭集约束；缺失时在最早可知阶段失败。
2. **Discovery Snapshot**：某 provider/target/environment 在指定时间声明或探测到的事实，含
   identity、来源、四态值（unknown/unmeasured/unsupported/verified）和 blockers。
3. **Evidence Reference**：对具体 workload、revision、环境和时间的可验证材料及 claim 上限；
   原始材料、审核 verdict 与 capability snapshot 通过 digest/reference 关联而不合并。

Compiler 消费 requirement 与 snapshot 做合法化；Runtime 比较请求和当前 snapshot；Platform
提供平台事实；Execution Provider 提供目标事实；Agent 只展示/校验 Core 词汇。

## 禁止事项

- 不从 installed、available、接口方法、环境变量、Mock 或 CPU 语义测试推导 verified。
- 不把用户愿望写入 TargetCapabilities，也不把一次 benchmark 结果写成永久 discovery 事实。
- 不把原生 FP64 与 Double-Single、逻辑 backend 与物理 route、replicated 与 sharded 混称。
- 不允许未知能力默认 true，也不允许自由字符串绕过闭集要求。

## 兼容性

现有类型保留为各域输入，通过显式 adapter 生成候选 snapshot/reference。稳定
`ExecutionOptions`、扩展 `CapabilityRequest/Response` 不原地改义。旧 Agent capability 名称
只能进入带版本的 alias 表；未知名称继续 fail closed。

## 迁移顺序

1. 建立词汇、四态真值、identity 和 claim-level fixture。
2. 先适配 `_compiler.TargetCapabilities`，再适配 Runtime/Platform discovery。
3. 适配 Cloud profile 与 Execution Provider target snapshot。
4. Agent 改读统一 snapshot；最后收缩旧类型和 alias。

## 验收测试

- requirement 满足/缺失/未知、snapshot stale、unknown 不可晋级的负向测试；
- 同一 snapshot 在 Compiler、Runtime、Agent 间 identity 一致；
- CPU、CUDA、FlagOS-on-CUDA 和 fake QPU fixtures 保持各自证据上限；
- sharding claim 必须具有 workload-specific evidence，且记录完整分布字段与 blockers。

## 未决问题

- capability vocabulary 的扩展注册与 minor-version 规则；
- snapshot TTL、校准版本和 stale 判定由谁配置；
- workload predicate 的表达能力，以及 evidence 审核签名是否属于 Core envelope 或 Audit 服务。
