# Simulation Engine 现状盘点与首个替换切片

状态：Simulation 团队交付候选（事实盘点与测试草案，不是公共契约）

集成进展（2026-09-04）：本地状态向量、密度矩阵、无噪声 MPS 和张量网络计划构建已归入
各自的 Simulation 实现。本地 TN 的计划构建与状态入口位于
`flagquantum/simulation/tensor_local.py`，Pauli/Hamiltonian 观测量计划与 MPO 构造位于
`tensor_observables.py`，`tensor_execution.py` 仅保留兼容门面和振幅入口；
初态与生命周期缓存仍暂由 `Circuit` 持有，迁移不得改变 `fq.Circuit`、Runtime 或结果契约。

复核进展（2026-09-05）：Statevector 的门矩阵作用、对角门数值作用、基态块合并和
adjoint 局部数学已由 `simulation/statevector_ops.py` 与
`simulation/statevector_adjoint.py` 统一持有；本地分片调试路径仅处理索引、所有权和
结果组装。TN 的正反向局部收缩数学已归 `simulation/tensor_stages.py`；JAX 的 dtype、
门矩阵、状态作用、分片局部 observable/loss、MPS 批量更新和 pullback 已归
`simulation/jax_*.py`。剩余候选集中在仍与分布式 MPS/TN 记录和调度交织的路径，且真实
本地 Engine 与测试 fake 现已通过同一套首切片 conformance，但最终 Core Engine 契约尚未批准。
因此 `simulation_extraction` 必须保持 `in_progress`，不得以目录数量或单一 CPU 测试代替退出条件。

首次盘点日期：2026-09-03；最近复核日期：2026-09-05

共同基线：`d7c56603e363bba95d5e98b9a77a75adb3c52e0d`

原始团队分支：`codex/vnext-team-simulation`；当前集成分支：`codex/flagquantum-vnext-architecture`

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
| 本地状态向量 | `simulation/statevector.py` 的执行循环；`simulation/statevector_ops.py` 的布局、门作用、门矩阵组合、压缩基态索引展开和融合；`simulation/triton_kernels/statevector_gates.py` 的本地门及跨分片 CX control-one pack/unpack CUDA kernel | `runtime/execution.py` 的 statevector 分支、`Circuit.state()`/`Circuit.run()` | 生产支持；数值实现已归 Simulation，初态与生命周期缓存仍暂由 `Circuit` 持有；分布式执行器复用门矩阵组合与索引数值函数，仍负责通信策略与传输 |
| 小规模专用状态向量 | `simulation/small_statevector.py` | 特定模型/基准调用方 | 2--4 qubit 数据重上传专用核，不是通用 Engine |
| 分布式状态向量 | `simulation/statevector_ops.py`、`statevector_adjoint.py` 和 `simulation/triton_kernels/statevector_*` 的 rank-local 数值原语 | `runtime/backends/statevector/` 的 planning/models/forward/reverse/forward_executor/training/checkpointing/gradient_reduction | Runtime 保留 amplitude/qubit-address 所有权、通信、chunk 和生命周期；不再实现局部门矩阵数学 |
| Split real/imag 与 Double-Single | `simulation/split_real_imag_statevector.py` 的 P0/P1/P2 门矩阵、门作用、零态执行循环和 Pauli-term 数值归约；`simulation/double_single_host_gates.py` 与 `double_single_device_gates.py` 的隔离门矩阵生成；`simulation/double_single_statevector.py` 的 P3/P4 零态初始化、门执行、归一化和 Pauli-term 归约 | Runtime 文件中的参数绑定、平台身份、精度计划与授权、编码策略、observables、参数移位调度、P5 autograd/SGD 边界、conformance/result | P0--P4 的基础数值实现已归 Simulation；Runtime 只组合既有数值原语；P3/P4 适配器因主机摄取与路径证据不同而保持分离；P5 单行 SGD 更新尚不构成独立数值核，不为搬移而新增 helper；实验路径不得成为首切片默认实现或被描述为等价 FP64 |
| 本地 MPS | `simulation/mps_local.py` 的无噪声指令循环；`mps_noisy.py` 的已降低单轨迹数值循环；`mps_state.py`、`mps_factorization.py`、`static_mps.py`、`tebd.py`、`dense_island.py`、`mps_brickwork.py` | `mps_execution.py` 的兼容适配；`runtime/trajectories/mps.py` 的随机流、多轨迹调度、恢复与合并；`mps.py` 门面 | 单设备数值路径已独立；Simulation 数值函数只接收 lowered IR、初始化状态和显式 RNG |
| 分布式 MPS | `simulation/mps_rank_local.py`、`mps_site_kernels.py`、`mps_compiled_layers.py`、`mps_factorization.py`、`mps_canonicalization.py`、`mps_reverse.py`、`mps_observables.py` 及 reverse 尚未拆出的数值段 | `runtime/backends/mps/` 的 forward/reverse/state/distribution/communication/*transport/planning/training_engine/checkpointing/production/profiling | 前向与反向批量门收缩、反向 pair 分解与截断投影、canonical-site 分解与残差、基础门作用、site kernel、QR、local VJP 与 observable/MPO 局部扫描已归位；Runtime 保留所有权、通信、显存准入与微批决策、canonicalization sweep、重平衡、tape/checkpoint 生命周期、梯度 collective、跨 rank observable pipeline 和结果证据 |
| 本地张量网络 | `simulation/tensor_local.py` 的计划构建与数值状态入口；`tensor_observables.py` 的观测量计划与 MPO；`tensor_state.py`、`tensor_contraction.py`、`tensor_stages.py`、`real_imag_kernels.py` | `tensor_execution.py` 的兼容门面与振幅入口、`tensor_path_search.py`、`tensor.py` | 本地执行和观测量职责已独立；路径搜索仍待进一步收口，能力仍为 experimental |
| 分布式张量网络 | `simulation/tensor_stages.py` 的 pair contraction、pair pullback、高秩回退和补偿累加 | `runtime/backends/tensor_network/` 的 DAG/schedule、tape/checkpoint、task ownership、通信与结果证据 | 正反向局部收缩数学已归位；sliced/sharded reverse 的生命周期仍与 checkpoint 及通信计划交织；生产 transport 未认证 |
| 密度矩阵 | `simulation/density_matrix.py` | Runtime noise registry | 本地精确演化、Kraus 作用和测量已归 Simulation；噪声 lowering 与计划分派仍归 Runtime；旧 `simulation.noise` 门面已退出 |
| 噪声模型与 lowering | Markovian Kraus 数值在 density kernels、`simulation/noisy_statevector.py`、`simulation/mps_state.py`/`mps_execution.py` | 语义由 `flagquantum/noise/` 拥有；lowering 由 `flagquantum/compiler/noise.py` 拥有；选择由 `runtime/planner/noise_selection.py` 拥有；轨迹公共设施在 `runtime/trajectories/` | Simulation 只拥有 channel/trajectory 数值演化，不复制 NoiseModel、lowering 或选择策略 |
| 轨迹 | statevector 的已 lowering 单批次指令循环与数值核在 `simulation/noisy_statevector.py`，编排在 `runtime/backends/statevector/noisy.py`；MPS 分支在 `simulation/mps_execution.py`/`mps_state.py` | `runtime/trajectories/` 拥有 seed、ownership、统计、checkpoint；执行文件还直接 all-reduce/保存 | 采样/归一化是数值算法；ID 分配、随机流构造、读出误差、跨 rank 汇总、检查点和自适应停止生命周期属于 Runtime |
| 可微计算 | 本地状态向量依赖 PyTorch 图；MPS/TN 数值操作、Triton autograd 和已拆出的 JAX MPS pullback 在 `simulation/`；其余显式 sharded adjoint/reverse 仍在各 backend | gradient ownership/reduction、训练循环、优化器、检查点和 evidence 与其混合 | 必须保留参数梯度所有权、dtype、复数共轭约定和前向相同的分布语义 |

## 4. `simulation` 代码归属矩阵

| 路径 | 主分类 | Simulation 应拥有 | 应拆出的混合点 |
| --- | --- | --- | --- |
| `linalg.py` | 纯数值算法 | PyTorch 线性代数工具 | 无明显 Runtime 策略 |
| `small_statevector.py` | 纯数值算法 | 小规模精确演化与 Z 期望 | 常量缓存键含 device 合理；它不是通用执行契约 |
| `mps_state.py`、`mps_factorization.py`、`mps_low_rank.py` | 纯数值算法 | MPS 状态、门作用、分解、截断和误差 | 环境变量控制 kernel/分解策略应由请求/Runtime 决策后显式传入；dense correctness fallback 必须可见 |
| `static_mps.py`、`mps_brickwork.py`、`tebd.py`、`dense_island.py` | 数值算法 + Kernel 调用 | 固定形状图、TEBD、dense island、局部编译核 | 编译/缓存策略和能力选择需与 Runtime/Compiler 的决定区分 |
| `mps_local.py` | 纯数值执行 | 已初始化 MPS 上的 IR 门作用、融合与 bucket kernel 调用 | 无 Runtime 生命周期所有权 |
| `mps_noisy.py` | 纯数值执行 | 已降低 IR 上的 unitary/Kraus MPS 单轨迹演化 | 不导入 Compiler 或 Runtime，不派生 seed，不拥有 checkpoint |
| `mps_execution.py` | 兼容入口与适配 | Circuit/IR 到本地 MPS 初态、lowered IR 内部入口、adaptive bond rerun | 正式 `run_native` 路径由 Runtime 先调用 Compiler lowering；受保护的 legacy 直接入口仍保留同签名 lowering |
| `runtime/trajectories/mps.py` | Runtime 生命周期 | 多轨迹 ownership、随机流、统计收敛、失败重试、checkpoint/restart 和 rank 结果合并 | 通过调用方提供的单轨迹执行器调用 Simulation，不实现 MPS 门或 Kraus 数值算法 |
| `mps_planning_mixin.py` | 资源/执行策略（混合） | 仅保留算法所需 shape/truncation 估计 | backend/kernel 环境开关与执行规划不应由状态对象决定 |
| `mps_models.py`、`mps.py` | 结果转换/兼容门面 | 算法内部诊断或短期门面 | 长期结果契约必须由 Core；门面退出条件是 Runtime 只经获批 Engine Contract 调用 |
| `tensor_state.py`、`tensor_contraction.py`、`tensor_stages.py` | 纯数值算法 | 网络表示、局部/分片收缩、显式反向、Kahan 等数值方法 | 执行计划和持久记录需由 Runtime/Core 契约提供 |
| `tensor_path_search.py` | 算法规划（混合） | contraction-order 搜索作为数值算法 | 设备/编译策略、全局资源预算决定属于 Runtime/Compiler 输入 |
| `tensor_local.py` | 纯数值执行 | IR 到本地 contraction plan、状态入口和局部编译模板复用 | 无 Runtime/Provider 依赖；初态和程序缓存仍消费现有 Circuit 生命周期容器 |
| `tensor_observables.py` | 纯数值算法 | Pauli/Hamiltonian 计划、MPO 压缩和批量观测量 contraction | 压缩设备由调用方显式给定，不读取 rank 或集群环境 |
| `tensor_execution.py` | 兼容入口与振幅执行 | 公开入口薄适配、振幅投影与 contraction 调用 | 不拥有 Runtime/Provider 策略；振幅代码可在收益明确时再独立 |
| `tensor_models.py`、`tensor.py` | 结果转换/兼容门面 | 算法内部结构 | 跨层结果与稳定类型应由 Core 提案定义 |
| `real_imag_kernels.py`、`triton_kernels/**` | Kernel 调用/纯数值算法 | eager/Triton 数值实现与 backward | 平台是否可用、是否允许 fallback 由 Platform 能力与 Runtime policy 决定 |
| `graph.py` | 受保护兼容工具 | 当前仅由根 API 兼容导出，无 Compiler 调用方 | 不复制到 Compiler，不形成第二权威；只有出现具体 Compiler 消费者并批准公共 API 迁移后才归位 |

## 5. `runtime/backends` 代码归属矩阵

| 路径 | 主分类 | Simulation 应拥有 | Runtime/Provider 应拥有 |
| --- | --- | --- | --- |
| `simulation/density_matrix.py` | 纯数值算法 | density 构造、算子展开、unitary/Kraus 演化、IR 数值循环和 density 测量 | 无；稳定结果投影仍由 Runtime/Core 负责 |
| `runtime/noise_registry.py` 的 density adapter | 执行适配 | 无数值实现 | 计划执行仅验证并分派已 lowering IR；直接兼容入口按调用请求 Compiler lowering |
| `statevector/split_real_imag*.py` | 数值算法 + Kernel 调用（混合） | 状态演化、精度扩展、expectation/VJP；状态向量 adjoint/VJP Triton kernel 已归 `simulation/triton_kernels/statevector_adjoint.py` | 设备身份、provider evidence、精度/回退授权、conformance 汇总由 Platform/Runtime；Double-Single 门矩阵生成已移至 `simulation/double_single_*_gates.py` |
| `statevector/forward.py`、`reverse_adjoint.py` | 分布式执行适配 | 无 Runtime 类型依赖的 rank-local eager 普通门、对角门、rank-pair、gate-basis block 合并、旋转门导数与复内积已归 Simulation | process group、collective 生命周期、rank/topology、owner 与全局索引解析、chunk policy、Triton 路由、环境开关和通信 evidence；local expectation 分块遍历依赖这些 Runtime 语义，不强迁 |
| `statevector/reverse.py`、`gradient_reduction.py` | Kernel/执行适配（混合） | autograd bridge 与局部梯度数学 | process group、bucket policy、all-reduce、ownership/evidence |
| `statevector/local_execution.py` | 执行适配 | 门矩阵与对角门数值已委托 `simulation/statevector_ops.py` | backend policy、模拟 rank 编排、shard 索引/所有权、真实 transport 与结果报告；不得为搬文件而复制 plan 或 shard 类型 |
| `statevector/planning.py`、`models.py`、`environment.py`、`layout.py`、`kernel_dispatch.py` | 资源或通信编排 | 仅算法约束/代价模型输入 | Runtime plan/topology/policy/环境；Platform kernel capability；Core-owned records |
| `statevector/forward_executor.py`、`training.py`、`checkpointing.py` | 资源/生命周期编排 | 无训练生命周期所有权 | 执行循环、故障协调、优化器、检查点/恢复、进度与超时 |
| `statevector/noisy.py` | 数值算法 + Runtime 编排 | batched gate/Kraus 采样、归一化、观测量 | trajectory ownership、collective 汇总、检查点、失败处理、自适应停止 |
| `simulation/mps_rank_local.py`、`mps_site_kernels.py`、`mps_factorization.py` | 纯数值算法/Kernel 调用 | 门作用、环境 transfer、QR/SVD 分解 | Runtime 只保留显存预算、微批规划、workspace 池和 Runtime 错误翻译；原 Runtime 数值模块已删除 |
| `mps/forward.py`、`reverse.py`、`reverse_replay.py`、`reverse_*observables.py` | 数值算法（高度混合） | rank-local MPS math、tape/VJP、observable contraction | rank ownership、transport sequencing、collective 与执行生命周期 |
| `mps/state.py`、`records.py` | 结果/所有权模型（混合） | 算法内部张量状态可留 | topology ownership、跨层记录应由 Runtime/Core 契约 |
| `mps/communication.py`、`distribution.py`、`metadata_transport.py`、`reverse_transport.py` | 资源或通信编排 | 仅通信算子要求 | Runtime/Platform 实现 transport 和 process group |
| `mps/training*.py`、`checkpointing.py`、`production.py`、`profiling.py`、`device_resolution.py` | 资源/生命周期/结果 | 局部 loss/gradient kernel 可下沉 Engine | 设备选择、参数广播、优化器、持久化、生产门禁、观测 |
| `tensor_network/sharded_kernels.py`、`sliced_reverse.py`、`reverse_dag.py` | 数值执行适配（混合） | pair contraction、单项/批量 pair pullback、高秩 fallback 和 Kahan 累加已委托 Simulation | DAG/bucket schedule、tape/cotangent 生命周期、checkpoint plan、切片调度、跨 rank reduce/transport |
| `tensor_network/distributed_execution.py`、`distributed_sliced_reverse.py`、`redistribution.py`、`partial_mesh.py` | 通信/执行适配（混合） | 局部 contraction 调用 | process group、P2P/all-to-all、rank 生命周期与聚合 |
| `tensor_network/distributed_dag.py`、`sliced_tasks.py`、`multi_axis_sharding.py`、`joint_planning.py` | 资源/通信规划 | 算法可行性和 shape cost | Runtime ownership/topology/memory/communication plan；跨层类型归 Core |
| `tensor_network/dynamic_checkpoint.py`、`rematerialization.py`、`memory_evidence.py`、`distributed_optimizer.py` | 生命周期/资源/结果 | rematerialization 的数值代价模型、局部更新 math | durable checkpoint、预算/证据、optimizer ownership 与执行策略 |
| `simulation/jax_gate_primitives.py`、`simulation/jax_statevector.py`、`simulation/jax_tensor_network.py` | 纯数值算法 | JAX dtype、指令与 Pauli 矩阵、statevector 分片初态与本地执行、跨 rank 门数学、分片 observable/loss、张量网络 contraction 与输出 loss 计算 | 无 Runtime/Platform 依赖；Runtime 保留 shard 组织、通信置换、pmap/shard-map、collective 和执行证据 |
| `simulation/jax_mps.py`、`simulation/jax_mps_batched.py`、`simulation/jax_mps_pullbacks.py` | 纯数值算法 | JAX MPS 初态、单/双站点更新、批量 pair 分解、远程门 swap 路由、statevector 收缩、局部 observable、局部 VJP、边界及 QR/SVD pullback | 无 Runtime/Platform 依赖；Runtime 保留 circuit loop、shard 组织、参数所有权、通信、截断策略和执行证据 |
| `jax/kernel.py`、`mps_kernel.py`、`*_kernels.py`、`*_contraction.py`、`*_pullbacks.py` | Kernel 调用/数值算法 | 剩余 JAX quantum kernel、VJP/pullback、slice 与 contraction 编排 | backend/device 是否选择 JAX 由 Runtime/Platform；标签切片与基础 einsum 数学归 Simulation |
| `jax/array_conversions.py` | 执行适配 | DLPack/array 数值边界的无拷贝语义 | 框架选择与 fallback policy 由 Runtime；外部对象不得越过边界 |
| `jax/*execution.py`、`backend_dispatch.py`、`statevector_training.py`、`mps_gradients.py`、`tensor_network_gradients.py` | 执行适配（混合） | 局部 kernel 调用 | profile/backend policy、device count、shard orchestration、训练生命周期 |
| `jax/*planning.py`、`planning_core.py`、`runtime_environment.py`、`transport.py` | 资源或通信编排 | 算法约束/代价输入 | Runtime/Platform topology、environment、transport 和 device lifecycle |
| `jax/*records.py`、`*result.py`、`evidence_collector.py`、`release_policy.py`、`compatibility_surface.py` | 结果转换/门面 | 算法内部 diagnostics | Core result/evidence、Runtime 汇总、release policy；兼容面应有退出条件 |

`__init__.py` 和各 facade 仅是导出/兼容层，不建立新的算法权威位置。

### JAX Runtime 数值边界收口审计（2026-09-05）

剩余 JAX Runtime 数值调用按职责处理，不以“清零 `jnp` 调用”为目标：collective、
设备放置、结果整形和证据探针属于执行语义，继续留在 Runtime；参数化张量网络的节点
记录组装依赖 Runtime 记录，暂不为搬迁而引入 node factory。单行零值分配或矩阵组合仅在
形成重复算法权威时下沉，不拆成细碎公共函数。审计识别出的实质算法
`mps_gradient_ownership.py` 跨 rank 张量重建后的 MPS 环境传递与 Z 观测量计算已迁入
`simulation/jax_mps.py`，Runtime 仅保留 rank 张量记录到数值参数的适配。

Statevector 复核确认 `statevector_kernels.py` 只剩指令/计划适配、collective 置换与
`pmap`/`shard_map` 编排；初态、局部门、pair 合并、all-to-all delta 和 observable/loss
数学均已委托 `simulation/jax_statevector.py`。该路径已到停止点，不为移动文件而复制
Runtime shard/plan 类型。

参数化张量网络复核确认，通用节点构造、标签切片、einsum 收缩以及输出 observable/loss
数学已由 `simulation/jax_tensor_network.py` 统一负责；`tensor_network_gradients.py` 中
剩余节点组装直接消费 Runtime 的 `JAXTensorNetworkNode`、切片任务、后端和 collective
选择，并参与梯度生命周期与结果证据，因此继续属于 Runtime 适配。该路径已到停止点：
不得为消除 Runtime 中的 `jnp` 调用而复制节点记录或新增 node factory；只有不依赖
Runtime 计划、任务、记录、策略和 collective，且具有独立复用价值的数值操作才继续下沉。

JAX MPS 反向路径复核确认，参数局部 VJP、边界 RXX adjoint 以及 QR/SVD
canonicalization/truncation pullback 已由 `simulation/jax_mps_pullbacks.py` 统一负责。
`mps_backward.py`、`mps_pullbacks.py` 和 `mps_canonicalization.py` 剩余逻辑属于受限 rank
协议：设备放置、参数所有权、collective 交换、截断策略、优化器生命周期和证据记录。
其中少量解析解校验与张量 shape 用于验证协议证据，不是第二套通用 MPS 数值权威；在没有
第二条脱离 Runtime 策略和记录的生产路径复用前，不继续拆成细碎 helper。该路径已到停止点。

本轮同时删除 `mps_kernel.py` 中已无读取方的独立 JAX dtype `ContextVar`；JAX 数值精度
上下文继续以 `simulation/jax_gate_primitives.py` 中的实现为唯一权威。

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

`mps_execution.py` 的 Runtime 导入豁免暂时只服务受保护的 `run_noisy_mps*` 限定名称；应在公共 API 完成正式弃用迁移后删除，不得通过动态导入绕过。下一批迁移热点是三个分布式 backend
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

`tests/team/simulation/test_statevector_engine_replacement_draft.py` 让真实
`run_local_statevector()` 与测试局部 fake 通过同一个参数化 conformance，二者均经
`run_native(..., mode="statevector")` 的同一消费者接缝执行。共同检查 batch、dtype、计划
mode/world size 和参数梯度所有权，且替换实现不修改 Runtime 消费者。测试没有向产品代码新增
Protocol、注册表或导出；因此首切片的实现替换证据已经成立，但还没有证明最终 Core 契约完成。

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

对 `statevector/local_execution.py` 的边界复核确认：门矩阵和对角门数值作用已委托
`simulation/statevector_ops.py`；该文件剩余的 shard 初始化、索引分组、跨 shard 所有权、
reference rank 编排和结果组装均以 `DistributedStatevectorPlan`、`StatevectorShardState`
为直接输入或输出。继续整体下沉会让 Simulation 依赖 Runtime，或产生第二套
plan/shard 类型；两者都不可接受。因此该文件作为执行适配保留，真实 transport、dry-run、
backend policy 和结果报告继续归 Runtime。

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

## 10. MPS 边界收口审计（2026-09-04）

本轮已将编译层执行、反向分解、反向 bucket 数值核和 canonicalization 数值核归入
`simulation/`。Runtime 仅保留 rank/site 所有权、扫描与通信顺序、梯度 collective、
checkpoint 生命周期和证据汇总。规范化路径已用两进程真实通信测试证明迁移前后全局态一致。

Runtime 中剩余的 tensor 拼接、reshape 和 stack 主要用于通信打包、梯度 bucket 与分布式
结果组装，不因使用张量操作而自动属于数值算法；只有改变 MPS 数学语义的实现才应继续迁入
Simulation。已删除本轮确认无消费者的私有兼容别名，不因文件较大而机械拆分模块。

目前仅保留 MPS 受保护兼容面相关的反向依赖：`simulation/mps_execution.py` 调用轨迹
Runtime，`simulation/mps_models.py` 仅在类型检查时引用 Runtime trajectory result 类型。
前者支撑受保护的 `run_noisy_mps`/merge 入口，后者受 `MPSMonteCarloResult` 的既有合格名约束；
不得以 `Any`、镜像类型或第二套生命周期实现绕开。
`simulation/noise.py` 已在 manifest 和身份测试切换到规范路径后删除，对应架构白名单同步退出。
除此以外，本轮未发现新的 Simulation→Runtime 依赖。MPS 数值边界已达到可停止继续横向
抽象的条件；后续优先推进最小纵向链路和目录归位。

## 11. 退出条件复核（2026-09-05）

`simulation_extraction` 完成前必须同时满足：

1. 真实本地 Engine 与 contract fake 运行同一套 conformance，Runtime 消费者无需修改；
2. `runtime/backends/jax/` 的量子数值核和 pullback 移至 Simulation，Runtime 只保留 backend/device、shard 和训练编排；
3. 分布式 TN reverse 中不依赖 task ownership、checkpoint 或 process group 的数学移至 Simulation；
4. 已登记的 `simulation/mps_execution.py`→Runtime 兼容调用和 `simulation/mps_models.py` 类型引用须经公共 API 迁移退出；`simulation/noise.py` 反向依赖已经退出；
5. 完整 CPU 数值、替换、架构和公共 API 门禁通过。

当前第 1 项已对本地 Statevector 首切片建立测试证据，第 2、3 项已达到各自审计停止点，
第 5 项仍需随集成门禁持续复核；第 4 项涉及受保护公共 API 迁移，尚未获批。因此不修改
机器可读状态。

## 12. TN 编译前向边界复核（2026-09-05）

`execute_compiled_tn_forward_with_tape()` 与 Simulation 的本地
`execute_contraction_stages()` 都会按 shape-compatible bucket 调用相同的
`complex_einsum_pair()` 数值原语，但二者不构成重复执行权威：前者消费分布式 DAG 的稳定
value id，并为显式反向保留完整 tape；后者消费本地 contraction plan，并在中间值用尽后释放。

因此不通过对象转换复用整个本地 executor，也不增加仅转发 equation 和 tensor 的薄包装。
Runtime 保留 DAG、bucket 顺序和 tape 生命周期，Simulation 继续拥有实际 contraction 与
pullback 数值原语。以后只有两条执行路径出现可独立复用的第二项数值行为时，才提取新的
Simulation helper。

## 13. 分布式 TN 反向边界复核（2026-09-05）

`reverse_dag.py`、`sliced_reverse.py` 和 `distributed_sliced_reverse.py` 中的前向 pair
contraction、反向 pair pullback、高秩 fallback 与 Kahan 累加均已调用
`simulation/tensor_stages.py` 或 `simulation/real_imag_kernels.py` 的唯一数值实现。

剩余 tensor stack、切片合并、cotangent map 累加以及有限性统计均直接表达 Runtime 的
DAG/bucket schedule、slice/shard ownership、tape/checkpoint 生命周期、collective 与证据结果，
不构成可独立复用的数值算法。该路径已到停止点：不为减少 Runtime 中的 tensor 操作新增
批处理包装、镜像记录或通用 executor；只有完全不依赖 Runtime DAG、任务、checkpoint、
所有权、process group 和证据类型的第二个实际消费者出现时，才继续向 Simulation 下沉。

## 14. 过渡代码删除复核（2026-09-06）

本轮按“先找无调用项，再进入下一条迁移切片”的顺序复核了现存过渡目录。未发现可在不改变
受保护 API、序列化产物或执行语义的前提下直接删除的已跟踪实现：

- `runtime/backends` 中的私有定义均仍有代码或测试消费者；其余张量操作属于计划、所有权、
  通信、checkpoint 或证据组装，不能仅因位于 Runtime 就认定为死代码；
- `utils.qasm_exporter` 与 `utils.qcis_exporter` 仍由公开 API、Deployment 和测试使用。Compiler
  emitter 是受限静态目标的权威实现，但尚不能覆盖旧 exporter 的完整 gate 集和失败语义，
  因而本轮不以简单转发或删除制造兼容性回归；
- `compilation` 剩余模块承载受保护的执行计划、序列化和校准语义，须先经过公共契约迁移；
- `_gateways/mcp` 没有已跟踪的生产实现可删除。

结论是删除路径已到当前安全停止点。下一条代码切片应以真实调用链为单位迁移，而不是继续按
文件名清理：优先选择 `runtime/backends/statevector` 中一段不依赖计划、设备选择、通信、
checkpoint 或证据类型的独立数值行为，迁入 Simulation 并由原入口委托；若不存在这样的完整
行为，则保留边界，不新增包装层。

## 15. Double-Single 诊断转换收口（2026-09-06）

Statevector P2–P5 结果对象曾分别重写 high/low 到 CPU float64/complex128 的诊断转换。
`DoubleSingleTensor.to_float64()`、`DoubleSingleComplexTensor.to_complex128()` 和现有 `to()`
已经是该数值表示的权威实现，因此 Runtime 结果对象现只组合这些方法并在返回前切断梯度图。
本轮删除三组重复 helper 和一处内联重复实现，不改变公开结果类型、执行计划、设备选择或数值
Kernel；CPU conformance 覆盖状态、期望值、梯度和优化器诊断结果。后续不再为同类结果对象
增加 high/low 手工重建代码。

P1–P5 conformance 的 complex128 参考路径也不再自行调用门矩阵和 statevector 底层作用函数，
而是复用 `simulation.pauli.pauli_product_statevector_expectation()`。Runtime 仍负责构造测试线路、
参数移位和判定阈值，Simulation 继续唯一拥有 Pauli 乘积期望值的稠密数值实现。
