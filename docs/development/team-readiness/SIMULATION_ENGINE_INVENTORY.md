# Simulation Engine 现状盘点与首个替换切片

状态：Simulation 团队交付候选（事实盘点与测试草案，不是公共契约）

集成进展（2026-09-04）：本地状态向量执行循环已迁入
`flagquantum/simulation/statevector.py`，布局、门作用和融合 helper 已迁入
`flagquantum/simulation/statevector_ops.py`，`Circuit.state()` 仅保留稳定门面。初态与生命周期
缓存仍暂由 `Circuit` 持有；后续迁移不得改变 `fq.Circuit`、Runtime 或结果契约。

盘点日期：2026-09-03

共同基线：`d7c56603e363bba95d5e98b9a77a75adb3c52e0d`

工作分支：`codex/vnext-team-simulation`

## 1. 结论

当前数值实现尚未全部以可替换 Simulation Engine 为边界收敛：稳定的本地 PyTorch
状态向量执行循环已位于 `flagquantum/simulation/statevector.py`，底层门作用与融合位于
`flagquantum/simulation/statevector_ops.py`；本地 MPS/TN 主要位于
`flagquantum/simulation/`；密度矩阵及大量分布式数值实现位于过渡目录
`flagquantum/runtime/backends/`。两个过渡目录都混合了数值算法、Kernel 调用、执行适配、
资源/通信编排和结果转换。

首个候选应是现有本地 PyTorch 状态向量的 `single_device_fast_path`：执行已经校验的
`CircuitIR`，从调用方提供或规范零态出发，返回保持 PyTorch autograd 图的完整批量状态。
首轮测试没有移动实现、修改 `fq.Circuit`/`run`、增加导出或在 Simulation 内私建契约。
新增测试先冻结数值行为，并利用 `run_native(..., mode="statevector")` 已有的结构性接缝证明
测试假实现可以在消费者调用逻辑不变时被使用。该接缝不是获批契约，只是 Core 提案落地前
的替换可行性证据。

## 2. 盘点方法与边界

盘点覆盖 `flagquantum/simulation/**/*.py`、
`flagquantum/runtime/backends/**/*.py`，并追踪到实际消费者和外置权威实现。分类定义如下：

| 分类 | 判定 |
| --- | --- |
| 纯数值算法 | 状态表示、门作用、分解/截断、收缩、噪声演化、前后向/VJP、数值稳定算法 |
| Kernel 调用 | Triton/JAX/编译 PyTorch kernel 的选择后调用、张量布局封装和 autograd kernel bridge |
| 执行适配 | 把 IR/计划/参数转换为算法调用，循环执行指令，连接 Runtime 或兼容入口 |
| 资源或通信编排 | 设备/进程组/拓扑/所有权/传输/内存/检查点/恢复/训练生命周期/后端策略 |
| 结果转换 | 状态到测量、局部到汇总结果、框架数组转换、记录与证据投影、兼容门面 |

一个文件可以有次要职责；表中“主分类”用于确定未来权威所有者，“混合点”用于决定拆分
顺序。算法正确性所需的局部 workspace 或张量布局仍可由 Simulation 拥有；集群资源、
通信生命周期和用户策略不可因与算法紧邻而留在 Simulation。

路径语义方面，首个切片是 `single_device_fast_path`。分布式状态向量/MPS 是
`sharded_across_ranks`；本地 TN 切片属于 `manual_sliced_tensor_contraction`；轨迹跨 rank
分配是 trajectory task parallel，不是单一状态容量扩展。不得把这些语义互相替代。

## 3. 实际实现位置

| 能力 | 当前数值权威位置 | 编排/消费者位置 | 现状 |
| --- | --- | --- | --- |
| 本地状态向量 | `simulation/statevector.py` 的执行循环；`simulation/statevector_ops.py` 的布局、门作用和融合；`simulation/triton_kernels/statevector_gates.py` 等 CUDA kernel | `runtime/execution.py` 的 statevector 分支、`Circuit.state()`/`Circuit.run()` | 生产支持；数值实现已归 Simulation，初态与生命周期缓存仍暂由 `Circuit` 持有 |
| 小规模专用状态向量 | `simulation/small_statevector.py` | 特定模型/基准调用方 | 2--4 qubit 数据重上传专用核，不是通用 Engine |
| 分布式状态向量 | `runtime/backends/statevector/forward.py`、`reverse_adjoint.py`、`triton.py`、`local_execution.py` 的数值部分 | 同目录 planning/models/environment/forward_executor/training/checkpointing/gradient_reduction | 真正 amplitude/qubit-address sharding 与 Runtime 生命周期高度混合 |
| Split real/imag 与 Double-Single | `runtime/backends/statevector/split_real_imag*.py`、`double_single_device_gates.py` | 同文件中的平台身份、精度计划、conformance/result | 实验路径；不得成为首切片默认实现或被描述为等价 FP64 |
| 本地 MPS | `simulation/mps_local.py` 的无噪声指令循环；`mps_state.py`、`mps_factorization.py`、`static_mps.py`、`tebd.py`、`dense_island.py`、`mps_brickwork.py` | `simulation/mps_execution.py` 的兼容入口、adaptive 与 noisy trajectory；`mps.py` 门面 | 单设备无噪声数值路径已独立且不依赖 Runtime；轨迹持久化、rank 与 checkpoint 生命周期仍待迁移 |
| 分布式 MPS | `runtime/backends/mps/operations.py`、`factorization.py`、`site_kernels.py` 及 forward/reverse 的数值段 | state/distribution/communication/*transport/planning/training_engine/checkpointing/production/profiling | 算法与 rank ownership、collective、持久恢复、生产门禁混合 |
| 本地张量网络 | `simulation/tensor_state.py`、`tensor_contraction.py`、`tensor_stages.py`、`real_imag_kernels.py` | `tensor_execution.py`、`tensor_path_search.py`、`tensor.py` | 收缩/反向与路径、设备、环境拓扑判断混合；能力仍为 experimental |
| 分布式张量网络 | `runtime/backends/tensor_network/sharded_kernels.py`、`sliced_reverse.py`、`reverse_dag.py` 的数值段 | distributed_dag/execution/redistribution/multi_axis/partial_mesh/joint_planning/checkpoint/rematerialization/memory_evidence | sliced、sharded 与通信计划交织；生产 transport 未认证 |
| 密度矩阵 | `simulation/density_matrix.py` | `simulation/noise.py` 兼容门面、Runtime noise registry | 本地精确演化、Kraus 作用和测量已归 Simulation；噪声 lowering 与计划分派仍归 Runtime |
| 噪声模型与 lowering | Markovian Kraus 数值在 density kernels、`statevector/noisy.py`、`simulation/mps_state.py`/`mps_execution.py` | 语义由 `flagquantum/noise/`（Core 团队路径）拥有；lowering/选择由 `flagquantum/compilation/noise/` 拥有；轨迹公共设施在 `runtime/trajectories/` | Simulation 只应拥有 channel/trajectory 数值演化，不应复制 NoiseModel 或选择策略 |
| 轨迹 | statevector 分支在 `runtime/backends/statevector/noisy.py`；MPS 分支在 `simulation/mps_execution.py`/`mps_state.py` | `runtime/trajectories/` 拥有 seed、ownership、统计、checkpoint；执行文件还直接 all-reduce/保存 | 采样/归一化是数值算法；ID 分配、跨 rank 汇总、检查点和自适应停止生命周期属于 Runtime |
| 可微计算 | 本地状态向量依赖 PyTorch 图；MPS/TN 数值操作及 Triton autograd 在 `simulation/`；显式 sharded adjoint/reverse 在各 backend；JAX pullback/VJP 在 `runtime/backends/jax/` | gradient ownership/reduction、训练循环、优化器、检查点和 evidence 与其混合 | 必须保留参数梯度所有权、dtype、复数共轭约定和前向相同的分布语义 |

## 4. `simulation` 代码归属矩阵

| 路径 | 主分类 | Simulation 应拥有 | 应拆出的混合点 |
| --- | --- | --- | --- |
| `linalg.py` | 纯数值算法 | PyTorch 线性代数工具 | 无明显 Runtime 策略 |
| `small_statevector.py` | 纯数值算法 | 小规模精确演化与 Z 期望 | 常量缓存键含 device 合理；它不是通用执行契约 |
| `mps_state.py`、`mps_factorization.py`、`mps_low_rank.py` | 纯数值算法 | MPS 状态、门作用、分解、截断和误差 | 环境变量控制 kernel/分解策略应由请求/Runtime 决策后显式传入；dense correctness fallback 必须可见 |
| `static_mps.py`、`mps_brickwork.py`、`tebd.py`、`dense_island.py` | 数值算法 + Kernel 调用 | 固定形状图、TEBD、dense island、局部编译核 | 编译/缓存策略和能力选择需与 Runtime/Compiler 的决定区分 |
| `mps_local.py` | 纯数值执行 | 已初始化 MPS 上的 IR 门作用、融合与 bucket kernel 调用 | 无 Runtime 生命周期所有权 |
| `mps_execution.py` | 兼容入口与高级执行（混合） | Circuit/IR 到本地 MPS 初态的薄适配、单轨迹数值演化 | trajectory rank/world、checkpoint 文件、resume、adaptive stop 生命周期应迁至 Runtime |
| `mps_planning_mixin.py` | 资源/执行策略（混合） | 仅保留算法所需 shape/truncation 估计 | backend/kernel 环境开关与执行规划不应由状态对象决定 |
| `mps_models.py`、`mps.py` | 结果转换/兼容门面 | 算法内部诊断或短期门面 | 长期结果契约必须由 Core；门面退出条件是 Runtime 只经获批 Engine Contract 调用 |
| `tensor_state.py`、`tensor_contraction.py`、`tensor_stages.py` | 纯数值算法 | 网络表示、局部/分片收缩、显式反向、Kahan 等数值方法 | 执行计划和持久记录需由 Runtime/Core 契约提供 |
| `tensor_path_search.py` | 算法规划（混合） | contraction-order 搜索作为数值算法 | 设备/编译策略、全局资源预算决定属于 Runtime/Compiler 输入 |
| `tensor_execution.py` | 执行适配（混合） | IR 到 contraction 数值调用 | 读取 `WORLD_SIZE`/`LOCAL_WORLD_SIZE`、节点判断、执行模式选择属于 Runtime |
| `tensor_models.py`、`tensor.py` | 结果转换/兼容门面 | 算法内部结构 | 跨层结果与稳定类型应由 Core 提案定义 |
| `real_imag_kernels.py`、`triton_kernels/**` | Kernel 调用/纯数值算法 | eager/Triton 数值实现与 backward | 平台是否可用、是否允许 fallback 由 Platform 能力与 Runtime policy 决定 |
| `graph.py` | 非 Simulation：编译辅助 | 无长期归属 | 当前被 native compiler 使用，目标应归 Compiler；Simulation 不应成为编译图权威位置 |
| `noise.py` | 结果转换/兼容门面 | 无新语义 | NoiseModel 属 Core 路径、lowering 属 Compiler、执行入口属 Provider/Runtime；待调用归零后退出 |

## 5. `runtime/backends` 代码归属矩阵

| 路径 | 主分类 | Simulation 应拥有 | Runtime/Provider 应拥有 |
| --- | --- | --- | --- |
| `simulation/density_matrix.py` | 纯数值算法 | density 构造、算子展开、unitary/Kraus 演化、IR 数值循环和 density 测量 | 无；稳定结果投影仍由 Runtime/Core 负责 |
| `runtime/noise_registry.py` 的 density adapter | 执行适配 | 无数值实现 | plan 验证、选项过滤、lowering 调用和 executor 分派 |
| `statevector/triton.py`、`double_single_device_gates.py`、`split_real_imag*.py` | 数值算法 + Kernel 调用（混合） | 状态演化、精度扩展、expectation/VJP | 设备身份、provider evidence、精度/回退授权、conformance 汇总由 Platform/Runtime |
| `statevector/forward.py`、`reverse_adjoint.py` | 数值算法（高度混合） | shard-local 门、cross-shard 数学、adjoint/VJP | process group、collective 生命周期、rank/topology、环境开关、通信 evidence |
| `statevector/reverse.py`、`gradient_reduction.py` | Kernel/执行适配（混合） | autograd bridge 与局部梯度数学 | process group、bucket policy、all-reduce、ownership/evidence |
| `statevector/local_execution.py` | 执行适配（混合） | shard 数值 reference kernel | backend policy、模拟 rank 编排、真实 transport、结果报告 |
| `statevector/planning.py`、`models.py`、`environment.py`、`layout.py`、`kernel_dispatch.py` | 资源或通信编排 | 仅算法约束/代价模型输入 | Runtime plan/topology/policy/环境；Platform kernel capability；Core-owned records |
| `statevector/forward_executor.py`、`training.py`、`checkpointing.py` | 资源/生命周期编排 | 无训练生命周期所有权 | 执行循环、故障协调、优化器、检查点/恢复、进度与超时 |
| `statevector/noisy.py` | 数值算法 + Runtime 编排 | batched gate/Kraus 采样、归一化、观测量 | trajectory ownership、collective 汇总、检查点、失败处理、自适应停止 |
| `mps/operations.py`、`factorization.py`、`site_kernels.py` | 纯数值算法/Kernel 调用 | 门作用、环境 transfer、分解及 backward | kernel cache policy 可由 Platform/Runtime 注入 |
| `mps/forward.py`、`reverse.py`、`reverse_replay.py`、`reverse_*observables.py` | 数值算法（高度混合） | rank-local MPS math、tape/VJP、observable contraction | rank ownership、transport sequencing、collective 与执行生命周期 |
| `mps/state.py`、`records.py` | 结果/所有权模型（混合） | 算法内部张量状态可留 | topology ownership、跨层记录应由 Runtime/Core 契约 |
| `mps/communication.py`、`distribution.py`、`metadata_transport.py`、`reverse_transport.py` | 资源或通信编排 | 仅通信算子要求 | Runtime/Platform 实现 transport 和 process group |
| `mps/training*.py`、`checkpointing.py`、`production.py`、`profiling.py`、`device_resolution.py` | 资源/生命周期/结果 | 局部 loss/gradient kernel 可下沉 Engine | 设备选择、参数广播、优化器、持久化、生产门禁、观测 |
| `tensor_network/sharded_kernels.py`、`sliced_reverse.py`、`reverse_dag.py` | 数值算法（混合） | shard/slice contraction 与 reverse math | task ownership、checkpoint plan、跨 rank reduce/transport |
| `tensor_network/distributed_execution.py`、`distributed_sliced_reverse.py`、`redistribution.py`、`partial_mesh.py` | 通信/执行适配（混合） | 局部 contraction 调用 | process group、P2P/all-to-all、rank 生命周期与聚合 |
| `tensor_network/distributed_dag.py`、`sliced_tasks.py`、`multi_axis_sharding.py`、`joint_planning.py` | 资源/通信规划 | 算法可行性和 shape cost | Runtime ownership/topology/memory/communication plan；跨层类型归 Core |
| `tensor_network/dynamic_checkpoint.py`、`rematerialization.py`、`memory_evidence.py`、`distributed_optimizer.py` | 生命周期/资源/结果 | rematerialization 的数值代价模型、局部更新 math | durable checkpoint、预算/证据、optimizer ownership 与执行策略 |
| `jax/gate_primitives.py`、`kernel.py`、`mps_kernel.py`、`*_kernels.py`、`*_contraction.py`、`*_pullbacks.py` | Kernel 调用/数值算法 | JAX quantum kernel、VJP/pullback、contraction | backend/device 是否选择 JAX 由 Runtime/Platform |
| `jax/array_conversions.py` | 执行适配 | DLPack/array 数值边界的无拷贝语义 | 框架选择与 fallback policy 由 Runtime；外部对象不得越过边界 |
| `jax/*execution.py`、`backend_dispatch.py`、`statevector_training.py`、`mps_gradients.py`、`tensor_network_gradients.py` | 执行适配（混合） | 局部 kernel 调用 | profile/backend policy、device count、shard orchestration、训练生命周期 |
| `jax/*planning.py`、`planning_core.py`、`runtime_environment.py`、`transport.py` | 资源或通信编排 | 算法约束/代价输入 | Runtime/Platform topology、environment、transport 和 device lifecycle |
| `jax/*records.py`、`*result.py`、`evidence_collector.py`、`release_policy.py`、`compatibility_surface.py` | 结果转换/门面 | 算法内部 diagnostics | Core result/evidence、Runtime 汇总、release policy；兼容面应有退出条件 |

`__init__.py` 和各 facade 仅是导出/兼容层，不建立新的算法权威位置。

## 6. Simulation 不应拥有的逻辑

本次扫描未发现凭据、API token 或 secret 的实现进入两个目录；这是应保持的负面事实。
但以下现有逻辑不应成为未来 Simulation Engine 的组成部分：

1. **Runtime 策略与自动选择**：distributed profile/backend policy、mode 选择、world/local
   world 推断、kernel/fallback 授权、memory/precision policy、production/release claim gate。
2. **设备与平台选择**：`resolve_device`、CUDA/JAX device count、provider identity、厂商 route、
   platform capability 探测。Engine 只能消费已解析的 platform handle/capability。
3. **集群和通信生命周期**：process-group 建立/销毁、rank placement、P2P/collective 调度、
   timeout/watchdog、故障协调、跨节点拓扑和通信 evidence 汇总。Engine 可以定义某次数值步骤
   需要的通信操作，但不拥有集群策略或传输生命周期。
4. **持久化与训练生命周期**：checkpoint 路径/保留/恢复、 durable job、重试、自适应停止、
   optimizer step 调度、进度记录。数值 rematerialization/checkpoint placement 算法可返回建议，
   Runtime 决定何时及向何处保存。
5. **凭据与外部服务**：当前无实现；将来也只能位于 QPU/Remote Service Provider 或外部
   Compute Service，禁止进入 Simulation。
6. **跨层结果和公共类型**：Simulation 可产生内部数值 diagnostics，但稳定请求、结果、
   Evidence、Failure 和序列化 schema 均由 Core 唯一拥有。

优先迁移热点是 `simulation/tensor_execution.py` 的环境拓扑读取、
`simulation/mps_execution.py` 的 trajectory ownership/checkpoint，以及三个分布式 backend
中的 process-group/数值核交织。不能简单搬文件；必须先有 Core 契约和替换测试。

## 7. 第一个可替换切片

### 7.1 范围

候选名：**Local PyTorch Dense Statevector Engine**。

- 输入：已校验 `CircuitIR`、批量初态（缺省为 `|0...0>`）、已绑定参数、Runtime 已解析的
  dtype/device/platform 执行上下文；
- 输出：形状 `(batch, 2**n_wires)` 的完整复状态，以及仅描述实际数值路径的 diagnostics；
- 语义：`single_device_fast_path`，不初始化 distributed、不读取 rank/cluster 环境；
- 梯度：保留 PyTorch 一阶 autograd 图；参数梯度仍回到调用方拥有的 Tensor；
- 支持基线：现有本地状态向量 gate/custom matrix、batch、complex64/complex128 行为；
- 明确不含：规划、编译、测量结果包装、噪声、shots、分布式 sharding、checkpoint、设备选择、
  fallback、provider identity 和性能声明。

它比 density/MPS/TN 更适合作为首切片，因为能力矩阵已将本地状态向量训练标为
`production_supported`，数值基线成熟、输入输出最小、无需近似误差契约，并且可以直接用
现有 PyTorch 路径做真实现。执行循环和 kernel dispatch 已归入 Simulation；程序、初态与
生命周期缓存仍应逐步归入实现内部或由既有执行上下文承载，同时保持 Stable Core 行为完全不变。

### 7.2 消费者替换证明

`tests/team/simulation/test_statevector_engine_characterization.py` 固定：

- IR 重建与直接 `Circuit.state()` 的批量、wire order、dtype 和归一化一致；
- RY 期望梯度与解析值一致；
- 自定义可微矩阵不丢失 autograd 图。

`tests/team/simulation/test_statevector_engine_replacement_draft.py` 使用测试局部 fake，固定
`run_native(..., mode="statevector")` 消费者在调用形式不变时可以接受另一个 `.state()`
实现，并验证 fake 输出、计划 mode/world size 以及参数梯度所有权。测试没有向产品代码新增
Protocol、注册表或导出；因此它证明替换方向可行，但还没有证明最终 Core 契约完成。

## 8. Core 契约提案（未实施）

需要由集成/Core 团队先批准一个最小版本化 Simulation Contract。Simulation 团队建议复用
现有 `CircuitIR`、Core execution/accuracy/failure/evidence 词汇，不定义第二套字典。提案应
至少回答：

| 项目 | 建议约束 |
| --- | --- |
| `SimulationRequest` | 引用规范 executable/`CircuitIR`；包含已绑定参数引用、初态引用、目标 `full_state`、明确 precision/approximation/fallback 决定；首版只允许 world size 1 |
| `SimulationContext` | Runtime/Platform 已解析的 tensor/device/kernel 能力句柄；不得含凭据、cluster policy 或厂商 SDK 对象；是否允许 PyTorch Tensor 作为进程内非序列化字段需由 Core 明确 |
| `SimulationResult` | 状态 payload、实际 dtype/device、algorithm id/version、approximation/truncation/fallback facts；由 Runtime 投影到稳定 `ExecutionResult` |
| 失败 | unsupported instruction/dtype/device、invalid initial state、numerical failure 必须结构化且 fail closed，不得静默换 backend/CPU |
| Engine 行为 | `execute(request, context) -> result`；同一消费者可注入真实现或 fake；契约不得包含 Runtime planner、credential、checkpoint path、process group lifecycle |

由于 Core 必须基础设施中立，提案不能草率把 `torch.Tensor` 写入可序列化 Core schema。
建议区分稳定、可序列化的请求/结果信封与进程内 tensor payload handle，并由 Core/API 所有者
决定生命周期、设备驻留和 DLPack 表达。获批前不要在 `simulation` 下添加私有 Protocol 来
绕过跨团队顺序。

## 9. 数值一致性风险与后续门禁

| 风险 | 首切片验收要求 |
| --- | --- |
| wire/basis order 改变 | Bell、非对称输入、非相邻/反向 wires 与 IR round-trip 对比 |
| batch/broadcast 语义漂移 | scalar、batch 参数、自定义 batched matrix 与多初态覆盖 |
| complex dtype/设备转换 | complex64/128 分别设容差；禁止隐式 CPU 或精度降级 |
| autograd 断图或共轭错误 | 解析梯度、有限差分、gradcheck；custom matrix 和复值 loss 约定覆盖 |
| 融合/Triton 与 eager 不一致 | 前向、梯度分别与 eager reference 对比；记录实际 kernel/fallback |
| 缓存复用旧参数/图 | 同一 Circuit 多轮参数、refresh、训练 step 后结果与梯度覆盖 |
| 原位更新破坏叶张量 | 初态和参数无意 mutation 检查 |
| 自定义初态在 IR 边界丢失 | 当前 `CircuitIR` 不携带 `Circuit.inputs`；Core 请求必须显式承载初态 payload/reference，不能假设 `run_native(IR)` 与任意自定义初态 Circuit 等价 |
| Runtime 越权重新规划或换实现 | supplied plan identity、显式 mode、fallback evidence/fail-closed 测试 |
| 将 local fake 误作 scalability 证据 | 测试仅标 `integration` 与 `single_device_fast_path`；不产生 distributed claim |

下一阶段应让现有实现与 test fake 跑同一套 conformance，并继续把 Runtime 的 statevector
分支限制为请求组织和结果投影。执行循环已完成物理归位；后续只迁移有明确收益的 helper 和
生命周期状态。任何迁移都必须保持 `fq.Circuit`、`run`、`plan` 和 `ExecutionResult` 的
受保护签名、默认值、失败阶段与序列化语义不变。
