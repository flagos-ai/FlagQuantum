# ARCH-007：Simulation 与 Runtime 的规划信息边界

状态：Proposed

日期：2026-09-03

依据：Phase 0 Simulation、Runtime 与 Platform 盘点；不改变当前 planner 或执行实现

## 上下文

Simulation 拥有数值算法知识，Runtime 拥有执行组织和最终策略，Platform 提供实际设备、
kernel、精度、内存、通信与拓扑事实。如果只禁止跨层调用而不给出信息交换边界，Runtime
会重复数值估算，或 Simulation 会越权读取环境并选择设备。分层是所有权边界，不是信息隔离。

## 决策候选

Simulation 通过 Core-owned、版本化、只读的规划输入/输出向 Runtime 提交：

- workload features：qubits、depth、gate/observable shape、batch、shots、dtype、梯度和动态性；
- algorithm constraints：精确/近似条件、支持门集、布局、数值稳定性、梯度与截断约束；
- resource requirements：状态/workspace/峰值内存估计、通信原语、临时存储和设备能力要求；
- candidate partitionings：local、amplitude/site/bond sharding、slice/task 切分及其适用条件；
- cost estimates：计算、内存、通信、重计算和误差成本，附模型版本、置信度和未知项。

Runtime 将这些候选与 Platform discovery/evidence、用户 request/policy、资源租约和失败/回退
授权结合，作出最终 backend、mode、placement、partition 和 fallback 决策，并将实际选择写入
plan/result/evidence。Simulation 不选择租户资源或物理设备；Runtime 不重写算法约束。

## 禁止事项

- 不让 Simulation 读取集群环境、初始化 process group、决定 provider、release claim 或
  静默 fallback。
- 不让 Runtime 复制算法 cost model、改变误差/梯度约束，或把未知估计当精确容量保证。
- 不把 torch/JAX/vendor handle、credentials 或自由格式字典放入可序列化规划合同。
- 不要求每次局部 kernel 优化都经过完整跨层规划；未改变外部语义时走 ARCH-008 快速路径。

## 兼容性

现有 `ExecutionPlan`、planner 和 Simulation 调用保持不变。首版仅以 adapter/fake 验证信息
形状；任何稳定 plan 字段、identity input、默认选择或失败阶段变化都需 API Change Proposal。
已有 backend-specific plan 可作为 adapter 来源，但不能升格为第二权威合同。

## 迁移顺序

1. 冻结现有 planner 选择和 Simulation characterization fixture。
2. Core 批准 workload/constraint/resource/candidate/cost 的最小值对象与 unknown 语义。
3. Local statevector fake 与现有实现产生候选，Runtime fake policy 做最终选择。
4. 接入 MPS/TN 估计，再接 Platform snapshot；逐项移除 Simulation 环境读取和 Runtime 重复估算。
5. 分布式候选只有通过相应语义/evidence 门禁后才可进入默认或认证路径。

## 验收测试

- 相同输入和模型版本产生确定候选 identity；未知成本保持 unknown；
- 替换 Simulation estimator 不修改 Runtime consumer，替换 Platform snapshot 不修改 Simulation；
- 用户禁止 approximation/fallback 时，Runtime 不得选择相关候选；
- 实际 path/partition 与 plan/result/evidence 一致，估计值不得冒充实测值；
- local fast path 不初始化 distributed，sharded 与 sliced/replicated 分类不可互换。

## 未决问题

- 成本模型版本、置信区间、校准数据和 cache identity；
- 谁拥有跨算法候选排序，Compiler schedule 如何只读进入 workload features；
- 分布式通信需求由 Simulation 表达原语还是逻辑数据移动，以及训练 optimizer ownership 表达。
