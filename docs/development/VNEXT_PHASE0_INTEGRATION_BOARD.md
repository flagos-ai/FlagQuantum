# FlagQuantum vNext Phase 0 集成报告

> 状态：第一轮八团队交付已全部审查、合并并完成统一验证
>
> 集成分支：`codex/flagquantum-vnext-architecture`
>
> 共同治理基线：`d7c56603e363bba95d5e98b9a77a75adb3c52e0d`
>
> 集成完成时间：2026-09-03（Asia/Shanghai）

## 1. 本轮结论

Phase 0 已完成八个团队的事实盘点、边界审查、行为冻结和集成验证。所有团队提交均通过路径所有权检查和架构边界检查，并按以下依赖顺序逐个合入：

```text
Core -> Compiler -> Runtime -> Simulation
     -> Platform -> Execution -> Ecosystem -> Agent Services
```

本轮主要产物是现状清单、契约提案和特征测试，不代表所提议的新 Core 契约、Provider 接口或目录迁移已经实施。唯一的产品实现变更位于 Agent Services：在规划 `ProgramArtifact` 前校验其必需能力，并对未知 artifact schema 采取失败关闭策略。

## 2. 团队交付与合并记录

| 顺序 | 团队 | 团队最终提交 | 集成合并提交 | 结论 |
| ---: | --- | --- | --- | --- |
| 1 | Core | `6cf86e15459b06ec5261ec24d63c53ccd3425756` | `83d45119` | 通过 |
| 2 | Compiler | `c8d7ac3a1f5c5a00ab7fccec5940ccf22ac4d3b7` | `d6f43652` | 通过 |
| 3 | Runtime | `adba20d6f22193ad0a00b1132518ea91205998da` | `991d309a` | 通过 |
| 4 | Simulation | `5ab9c74e34743c7dd768cc179ad584f92bbb2a57` | `2d4b8948` | 通过 |
| 5 | Platform Provider | `42aa894574133c59010d022bf4536573b1bc2e77` | `810f0b5a` | 通过 |
| 6 | Execution Provider | `c3e9129a1244e820985ff36f7ccb6b824b7d9405` | `33d3b6a5` | 通过 |
| 7 | Ecosystem | `0fca0f02705a7f8f9fdfb5b9719009f6d9f42354` | `c2f66391` | 通过 |
| 8 | Agent Services | `669adfa7ddcb9c2c7e2aca8a3c5e7f703b1807f8` | `75693ca1` | 通过 |

所有团队 worktree 在交付核验时均为干净状态。八个分支都从同一轮治理基线派生，改动路径互不冲突；Integration 在每次合并后重新运行所有已合入团队的测试，没有发现顺序相关回归。

## 3. 统一验证结果

### 3.1 集成分支本地验证

- 团队范围规则：通过；
- 架构边界规则：通过；
- 八团队特征测试：`41 passed, 2 skipped`；跳过项依赖本机未安装的可选 Qiskit/PennyLane；
- Agent、Runtime、Platform、Deployment、Provider 与 Interop 交叉测试：`118 passed, 1 skipped`。

### 3.2 Linux Docker 标准门禁

- `pr-runtime`：`175 passed, 33 skipped`，零失败；
- `pr-default`：`1831 passed, 12 skipped, 1 failed`；
- 唯一失败为既有 `test_approved_import_verify_budget_is_machine_enforced` 编译导入/校验时延预算。该失败已在共同基线和多个独立团队分支的同口径环境中复现，不是本轮合并引入的回归；本轮未修改阈值、快照或相关编译实现。

### 3.3 专项环境证据

- Ecosystem 在 Docker 可选依赖环境中完成 Qiskit 2.5.2、Aer 0.17.2、PennyLane 0.45.1 的 `51 passed` 互操作验证；
- Platform 在 `a800-node-0`、`a800-node-1` 上分别完成平台测试 `15 passed`，并完成两节点、每节点 1 GPU 的 NCCL 分片 statevector 正确性验证；
- 远程结果只证明 NVIDIA A800 + CUDA/NCCL 开发路径，不证明 FlagOS、国产芯片适配或生产级扩展效率；
- Execution 与 Agent 团队没有把 A800 验证写成真实 QPU 证据。

## 4. 第一轮冻结的模块边界

### Core

拥有跨领域稳定数据语义、身份、序列化和证据的候选权威。当前同名类型只完成分类，禁止未经决策直接合并或删除。

### Compiler

拥有程序导入、规范化、分析、优化、目标合法化、lowering 和可执行产物生成。`flagquantum.compilation` 仍是当前稳定入口，`flagquantum._compiler` 是内部收敛候选；本轮未切换默认编译路径。

### Runtime

拥有一次执行尝试的验证、编排、生命周期、分布式组织、训练、检查点、恢复、观测和结果汇总。长期租户任务、鉴权、计费与持久化队列不属于主仓库 Runtime。

### Simulation

拥有状态演化、MPS 分解/截断、张量收缩、噪声轨迹和前后向数值 Kernel。设备选择、rank/拓扑、通信生命周期、检查点策略和执行证据属于 Runtime/Provider。

### Platform Provider

拥有设备发现、生命周期、内存、stream/event、Kernel、dtype、精度、拓扑和通信能力事实。Platform 只报告硬件/软件平台能力，不提交远程量子任务。

### Execution Provider

拥有完整执行目标的提交、状态、取消、结果获取、错误映射、校准身份和执行证据。Simulation Provider、QPU Provider 和 Remote Service Provider 共享生命周期语义，但不能强行共享所有操作。

### Ecosystem

拥有外部格式、机器学习前端、插件和第三方 SDK 的边界转换。Qiskit、PennyLane、CUDA-Q/QX 等对象不得进入 Core、Compiler、Runtime 或 Simulation 的稳定契约。

### Agent Services

拥有协议无关、确定性的 capabilities、validate、plan 和 preflight 应用服务。MCP/REST 只作为外部网关；租户、鉴权、配额、计费和长期任务属于外部 Compute Service。

## 5. 已确认的主要架构债务

1. `ProgramArtifact`、capability、execution request/plan/result/evidence 存在多组相近类型，必须先完成字段、生命周期、身份和消费者对账；
2. Runtime 仍有十条登记在案的 Runtime→Compiler 依赖，其中既有共享契约依赖，也有内部实现调用和历史噪声/路由耦合；
3. 数值算法与编排仍同时分布于 `simulation` 和 `runtime/backends`；首个迁移候选是单设备 PyTorch dense statevector engine；
4. Platform 与 Execution Provider 尚无经批准的最小 Core 契约；现有多个 Provider/Extension 协议不能直接合成万能接口；
5. `ExecutionResult`、`TargetExecutionResult`、`DeploymentResult` 等结果模型并存，状态、错误、取消、位序和校准证据尚未统一；
6. OpenQASM/QCIS、Qiskit Aer 执行和 Braket IQM 动态方言仍有跨层或双实现；
7. `flagquantum.agent.capabilities()` 仍间接访问 Runtime backend registry；
8. `CUDAPlatformRuntime.event()` 默认事件不启用 timing，不能直接用于 elapsed-time 证据；
9. 当前标准门禁存在一个稳定复现的 Compiler import/verify 性能预算失败，需要 Compiler 单独修复，不能通过放宽阈值消除。

## 6. 第二轮准入顺序

第二轮不应直接进行大规模目录搬迁。建议按以下顺序推进最小可审查切片：

1. Integration/Core 完成 `ProgramArtifact` 与元数据值域对账；
2. 收敛最小 `TargetCapabilities`，严格区分需求、发现事实与执行证据；
3. 定义最小 Execution Request 与 Result/Evidence 边界；
4. 在上述类型稳定后，再批准 Platform Provider 与 Execution Provider 方法集；
5. Runtime 移除第一条纯类型 Runtime→Compiler 依赖；
6. Simulation 抽取首个可替换单设备 statevector engine；
7. Ecosystem 将 Qiskit Aer 执行迁往 Execution Provider，并收敛格式双实现；
8. Agent Services 通过公共能力快照替代对 Runtime registry 的间接依赖；
9. Compiler 单独修复 import/verify 性能预算并恢复默认门禁全绿。

每个切片仍须遵守：单一责任团队、最小差异、先有契约/特征测试、团队范围门禁、架构门禁、集成分支逐项合并和合并后复测。
