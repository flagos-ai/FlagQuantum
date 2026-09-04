# Platform Provider 能力盘点

状态日期：2026-09-03

负责团队：Platform Provider

分支：`codex/vnext-team-platform-providers`

目录进展（2026-09-04）：首条 CPU 纵向链路通过后，原
`flagquantum/runtime/platforms` 已整体迁入 `flagquantum/providers/platform`，所有仓内
调用方同步切换且旧包删除。本文其余能力结论不因物理迁移而升级。

## 结论

当前仓库已有一条小而明确的平台边界：
`flagquantum.providers.platform.PlatformRuntime` 负责设备发现、激活、同步、
内存快照、流/事件、RNG 和运行时身份；内建实现只有 PyTorch CPU、PyTorch CUDA
与懒加载的 Torch-FL FlagOS。扩展侧的唯一权威注册与生命周期机制仍是
`flagquantum.extensions.sdk.ExtensionRegistry`。本轮没有新增 Provider 注册体系，
也没有修改受保护契约。

必须区分接口、软件执行证据与真实硬件认证：

- CPU 是本地始终可用的真实路径，已有普通单元/集成测试与
  `local_statevector` 的 `production_supported` 能力记录；但 CPU 内存总量、拓扑和
  集合通信不是 `PlatformRuntime` 当前可发现字段。
- NVIDIA CUDA 是真实执行路径。仓库中有 A100/A800 单卡、多卡以及双节点 A800
  运行材料；这些材料能证明其各自记录的 CUDA/NCCL 工作负载，不能自动证明所有
  CUDA 设备、所有算子或所有精度组合。
- FlagOS 已有真实 Torch-FL 适配与 `flagos` 逻辑设备执行证据，但已签入材料使用的
  物理设备是 NVIDIA A800，Torch-FL 也以 CUDA 后端构建。因此这些是 FlagOS-on-CUDA
  适配/可移植性证据，不是国产卡证据。
- 仓库没有一份已签入且通过评审的国产真实卡结果。现有
  `tests/test_domestic_single_card_certification.py` 只有在外部提供
  `FLAGQUANTUM_DOMESTIC_ATTESTATION` 时才运行，并且即使候选执行通过，也明确保持
  `hardware_certification=false`、`production_claim_allowed=false` 与
  `scalability_claim_allowed=false`。
- `FLAGQUANTUM_ACCELERATOR` 环境提示、测试中的 Hygon/FlagOS 假对象，以及接口类型
  都只能证明发现/隔离逻辑；它们不会使设备变为 available，也不是硬件支持证据。

## 盘点口径

“真实能力”表示仓库内存在可执行实现，并有与声明范围匹配的测试或签入运行材料。
“适配能力”表示公共运行时边界可以发现、激活或调用提供方，但其物理设备或内部路由
仍可能未被认证。“接口/Mock”表示仅有协议、环境提示、猴子补丁或待外部主机运行的
门禁。硬件材料只按其中记录的设备、版本、节点数、dtype、工作负载和声明等级解释。

本次代码改动分类为 `single_device_fast_path` 的平台生命周期一致性验证；文档同时引用
既有 `sharded_across_ranks`、`rank_local_replicated_kernel` 与集合通信材料，但没有产生
新的硬件或可扩展性证据。

## Platform 能力矩阵

| 平台/证据层级 | 设备发现与生命周期 | 内存 | Kernel/执行 | 原生双精度 | Double-Single 软件精度 | P2P/集合通信 | 节点间通信/拓扑 | CPU 回退 | 证据结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch CPU（真实） | 始终 installed/activated/available；单一 `cpu` 设备；同步、空流上下文、主机事件、RNG 可执行 | `MemorySnapshot` 字段存在，但 CPU 实现全部为 unknown | PyTorch 本地 statevector/training 为真实路径；Platform contract 不逐项声明 kernel | PyTorch CPU `float64/complex128` 路径存在并被本地数值测试使用；不是逐算子认证 | P0-P5 实验路径在 CPU 有测试；P5 autograd 交付边界仍是 FP32，显式 Double-Single SGD 不是 `torch.optim` | Platform contract 无通信方法；Gloo/本地分布式属于 Runtime，不由 CPU provider 认证 | Platform contract 无 CPU 拓扑；CPU 分布式测试只证明语义 | 本身即 CPU；不存在“回退到 CPU”事件 | 真实本地能力；没有硬件容量/拓扑声明 |
| PyTorch CUDA（真实 NVIDIA） | `torch.cuda` 发现、设备名/显存、同步、stream/event、RNG 可执行 | allocated/reserved/free/total 可取；API 失败时 free/total 保持 unknown | PyTorch CUDA 与 Triton/本地、分片运行路径真实存在；具体 kernel 由 Runtime/Simulation 各自门禁 | A800 材料包含 `complex128` 执行；只能证明记录的算子与工作负载，不是全算子 FP64 认证 | A800 上 P0-P4 与显式 P5 SGD 有可移植性材料；仍为 experimental | NCCL、P2P、all-gather/all-reduce/broadcast 等在既有 A800 材料中被执行；应按每份 payload 的语义解释 | 有单节点 2/4/8/16 卡和双节点 A800 开发材料；部分双节点材料仍为 `development_smoke` 或 `requires_runtime_summary`，不能一概提升 | 没有平台级自动 CPU 回退；上层 fallback 必须显式记录 | 真实 NVIDIA CUDA 能力；支持范围由 capability matrix 与具体 JSON 限定 |
| Torch-FL FlagOS on CUDA（真实适配，非国产认证） | `torch_fl` 仅显式请求时导入；要求注册 `torch.flagos`；支持发现、同步、stream/event、RNG 与身份 | 可选读取 allocated/reserved/free/total；缺失 API 保持 unknown | 本地与分片 statevector、有限训练轨迹已在 A800 CUDA-backed Torch-FL 上执行 | 单卡和 2/4/8 卡材料含 `complex128`；证明 FlagOS-on-A800 路径，不证明国产设备原生 FP64 | P0-P4 与显式 P5 SGD 有 FlagOS-on-A800 可移植性材料；提供方内部路由未审计 | 单节点 2/4/8 卡执行 all-gather、all-reduce、broadcast、isend/irecv；complex `reduce_scatter_tensor` 在完整矩阵中不支持；FlagCX 内层路由与 host staging 未验证 | 只有单节点 FlagOS 材料；无 FlagOS 多节点证据，拓扑与内层链路未归因 | 缺失/不可用时 fail closed；错误文案说明 CPU/CUDA 仍可用，但不会静默替换请求 | `development_evidence`/experimental；不是国产硬件、生产或 FlagCX 认证 |
| 国产真实卡（未验证） | 只有 attestation 驱动的候选 harness 和测试 Mock | 未验证 | 未验证 | 未验证 | 设计为验证 P0-P5，但无签入真实卡通过材料 | FlagCX 未验证 | 单卡候选，无多卡/多节点 | 未验证；候选 payload 必须禁止静默回退 | 当前为证据缺口，不得写成支持 |
| 其他厂商/未知加速器（接口或 Mock） | 环境提示被标记 `available=false`；厂商名字不参与类型推断 | 无 | 无 | 无 | device-generic 算法接口不等于设备通过 | 无 | 无 | 不会被 auto 选中 | 只有接口/Mock，无硬件声明 |

## 分能力事实

### 设备发现与生命周期

`providers/platform/registry.py` 的固定内建表是当前平台发现入口。CPU/CUDA 不需要可选
依赖；FlagOS provider 在未激活时不会被全局发现流程导入。`resolve_platform_device()`
在分配工作负载前检查平台可用性和显式设备下标，失败时抛出
`PlatformUnavailableError`。扩展 SDK 另有 task-local、不可变的
`ExtensionRegistry` 与 start/invoke/close 错误隔离；两者职责不同，当前没有生产桥接
或重复注册表。

### 内存

CUDA provider 能读取 allocator allocated/reserved 和设备 free/total。FlagOS provider
只调用 Torch-FL 公开挂到 `torch.flagos` 的可选内存 API；缺失项不会被猜测。CPU provider
返回字段全为 unknown。现有 contract 没有峰值内存、workspace、NUMA/HBM 层级、统一
内存、OOM 分类或 per-rank 聚合字段；这些数据目前散落在具体 Runtime 和 benchmark
payload 中。

### Kernel

`PlatformRuntime` 没有 kernel 枚举、算子探针、编译能力或 fallback 字段。算子能力目前由
`runtime/operator_backends.py`、`runtime/operator_probes.py`、Simulation 内的 PyTorch/
Triton 实现和具体工作负载测试分别给出。因而“provider 已发现”不能推出 statevector、
MPS、TN、梯度或任意算子已支持。FlagOS 自动选择保持关闭，直到工作负载级算子与数值
证据通过。

### 精度

通用 backend registry 宣布 `complex64/complex128` 是 PyTorch backend 的候选 dtype，
这不是逐设备、逐 kernel 的保证。原生双精度必须以具体运行材料确认。Double-Single 是
四个 FP32 word 或 high/low 累积的软件实验路径，保留 FP32 指数范围，不等价于一般 FP64/
complex128；P3 仍在 CPU 生成 float64/complex128 gate 后搬运，P4 才在受限角度和门集合内
做设备侧生成，P5 的普通 autograd loss/grad 仍为 FP32 交付。

### 通信与拓扑

`PlatformRuntime` 没有 P2P、collective、process-group、rank placement 或 topology 方法。
CUDA/NCCL 与 FlagOS 分布式执行由 Runtime 拥有。FlagOS 身份记录刻意把
`outer_backend=flagos` 与 `inner_backend=flagcx` 分开；仅连接 Torch-FL process group
不能证明 FlagCX、无 host staging 或物理链路。现有 FlagOS transport observability 材料
虽然验证了逻辑 `flagos` 输入输出和若干 collective 正确性，但 CUPTI device activity
不完整，因此 `host_staging_observed` 仍为 unknown，communication claim 被关闭。

### 回退与证据

平台解析失败必须 fail closed。FlagOS 不可用时不会悄悄返回 CPU 或 CUDA；用户可另行选择
仍可用的平台。算法内部的 eager、dense、host reference 或 CPU correctness 路径必须由其
各自 result/evidence 显式记录，不能被平台发现状态掩盖。环境变量提示和设备名称只用于
诊断，不会提升成熟度。

## 真实证据来源

| 来源 | 能证明 | 不能证明 |
| --- | --- | --- |
| `tests/unit/test_platform_runtime.py` | CPU 生命周期；CUDA 发现逻辑；FlagOS 懒加载、公开内存/stream/event/身份适配；未知厂商不被自动分类 | 真实 CUDA/FlagOS/国产硬件执行 |
| `tests/team/platform/test_platform_consistency.py` | 三个内建 provider 都满足现有 `PlatformRuntime` 结构；CPU 生命周期自洽；平台可通过现有 SDK 的 device extension 生命周期而无需第二套 registry | 任何加速器、kernel、通信、性能或硬件支持 |
| `artifacts/flagos_cuda_reference_a800_20260824.json` | 单个 `flagos:0` 在 NVIDIA A800、CUDA-backed Torch-FL 上的本地参考执行 | 国产卡、生产性能、无 host fallback |
| `artifacts/flagos_workload_capability_f4_20260826.json` | 单节点 2/4/8 卡 `complex64/complex128` statevector forward/training 的开发证据；完整 collective 矩阵明确暴露 complex reduce-scatter 缺口 | FlagCX 路由、多节点、生产/发布认证 |
| `artifacts/flagos_statevector_capacity_f5_a800_20260827.json` | 同一 32-qubit complex128 workload 在单卡/复制路径 OOM，而 8-rank 分片路径完成；物理卡为 NVIDIA A800 | 国产硬件、反向/优化器容量、通用可扩展性、FlagCX 或无 host staging |
| `artifacts/flagos_transport_observability_f6_a800_20260827.json` | 单节点 2/4/8 rank 的 four-collective、两种 complex dtype 正确性与逻辑设备驻留 | 内层通信实现、FlagCX、host staging、性能、多节点 |
| `artifacts/split_real_imag_optimizer_a800_20260825.json` | native CUDA 与 FlagOS-on-CUDA 上显式 Double-Single SGD 轨迹的实验可移植性 | 一般 FP64 等价、`torch.optim`、收敛、国产卡或生产支持 |
| `tests/test_domestic_single_card_certification.py` | 国产单卡候选证据的 fail-closed 字段和 P0-P5 预期阶段 | 当前仓库已有国产真实卡通过结果 |
| `capability-maturity.toml` 与生成的 Known Limitations | 声明上限与证据引用 | 超出具体 capability 范围的推断 |

## 厂商泄漏与边界债务

没有发现 `torch_fl` 被 import 到用户 API、Simulation 或通用 Runtime；架构规则只允许
`providers/platform/flagos.py` import 它。也没有发现 Torch-FL Python 对象进入
`fq.Circuit`、FlagQuantum IR 或稳定结果对象。

仍有以下需要后续收敛的边界债务：

1. `runtime/distributed/flagos_runtime.py` 在平台边界外直接读取
   `torch.flagos.set_device/current_device`。这是 provider 挂载对象对通用 Runtime 的直接
   可见，应最终下沉为 `PlatformRuntime` 的设备选择/当前设备能力；本轮不改契约。
2. statevector、MPS、TN、training state 和 operator backend 多处直接调用
   `torch.cuda` 的 stream/event/memory/RNG/synchronize。`architecture.toml` 用逐文件
   ceiling 把它们登记为既有债务，但统一 PlatformRuntime 尚未替代这些调用。
3. Simulation 使用 `Tensor.is_cuda` 和 Triton CUDA 限制做 kernel 选择。这没有泄漏
   CUDA Python 对象，但把厂商设备分类写入了数值实现，阻止 FlagOS/其他
   PrivateUse1 平台复用同一优化 kernel；应由可验证的 kernel capability 决定。
4. `runtime/operator_backends.py` 读取并序列化 FlagGems `vendor_name`。值被降为字符串，
   没有把 vendor object 传入用户 API，但该探测尚未纳入 PlatformRuntime 证据模型。
5. `PlatformRuntime.stream()`/`event()` 返回 `Any`，`PlatformIdentity.metadata` 与
   `PlatformDevice.metadata` 也接受任意值。实现当前主要返回 PyTorch 对象或 JSON-like
   标量，但 contract 本身尚未禁止不可序列化 vendor object；这是最小 contract 评审项。

`flagquantum/api.py` 暴露的是通用 `AcceleratorInfo`，不包含 Torch-FL/Hygon/CUDA SDK
对象；不过 `AcceleratorInfo` 自身属于 Stable Core 相关表面，本轮未修改。

## Platform Provider Contract 最小字段提案

以下是提交给 Integration/Core 的评审输入，不是本分支上的契约修改。建议在现有
`PlatformRuntime` 演进，而不是建立新 Provider SPI；是否进入稳定层、字段命名与版本化
方式必须走 API/contract proposal。

| 组 | 最小字段/操作 | 失败与证据语义 |
| --- | --- | --- |
| 身份 | `schema_version`, `provider`, `provider_version`, `vendor`, `device_type`, `runtime_versions`, `physical_device_id/name` | 字符串/标量、JSON-safe；未知为 null；身份不提升能力 |
| 发现 | `installed`, `activated`, `available`, `device_count`, `devices`, `availability_blockers` | 环境 hint 与 Mock 必须标未验证，不得 available |
| 生命周期 | `activate`, `set_device`, `current_device`, `synchronize`, `stream`, `event`, `rng_state`, `restore_rng_state`, `close` | 可选能力必须显式 unsupported；vendor object 只能留在 adapter 内部，可向 Runtime 返回受控句柄 |
| 内存 | 每设备 `allocated`, `reserved`, `free`, `total`, `peak_allocated`, `workspace`, `memory_kind` | 未测为 null；必须标采样 API、时间点和 per-rank 归属 |
| Kernel | `operation`, `dtype`, `layout`, `gradient`, `compiled/eager`, `supported`, `fallback_policy`, `probe_id` | 接口存在不等于 supported；fallback 必须授权并进入 result/evidence |
| 精度 | `storage_dtypes`, `compute_dtypes`, `accumulation_dtypes`, `native_fp64`, `native_complex128`, `software_precision_modes`, `precision_blockers` | 原生与 Double-Single 分开；按 workload/kernel 给出证据 |
| 通信 | `process_group_backends`, `p2p`, `collectives`, `dtype_support`, `device_resident`, `host_staging_observed`, `inner_backend`, `inner_backend_verified` | logical backend 与物理 route 分开；未知不允许通信声明 |
| 拓扑 | `node_count`, `world_size`, `local_world_size`, `rank_placement`, `links`, `topology_source`, `topology_fingerprint` | 推断与实测分开；跨节点 route 未验证时 fail closed |
| 回退 | `requested_path`, `selected_path`, `fallback_allowed`, `fallback_used`, `fallback_reason`, `source_device`, `target_device` | CPU/backend substitute 不得静默发生 |
| 证据 | `evidence_level`, `artifact_uri`, `artifact_sha256`, `code_revision`, `environment`, `workload`, `timestamp`, `blockers`, `claim_allowed` | capability-specific；Mock/CPU semantics/skipped test 不得提升硬件成熟度 |

首个契约评审应优先处理三点：把 FlagOS `set/current_device` 下沉、定义 JSON-safe 的
身份/设备 metadata、为 kernel/precision/communication/topology 引入“未知/未测/不支持/
已验证”四态语义。扩展 SDK 的 `DeviceExtension` 可作为外部扩展生命周期入口，但其
`devices() -> Sequence[Mapping[str, Any]]` 也属于冻结协议，不能在团队分支直接收紧。

## 第一版一致性测试范围

新增测试只验证当前可由 CPU 环境真实执行的共同不变量：

1. `cpu`、`cuda`、`flagos` 三个内建对象结构上满足同一个 `PlatformRuntime`；
2. CPU 的 discover/devices、identity、memory、RNG、event、stream、synchronize 生命周期
   自洽且结果可序列化；
3. 用测试内 adapter 将 CPU PlatformRuntime 放入现有 `ExtensionRegistry` 的 `device`
   kind，验证协商、启动、调用、清理与 task-local 隔离；
4. 不修改 `fq` 根导出、不添加全局 registry、不激活可选 FlagOS 依赖。

它不运行或模拟 CUDA/FlagOS kernel，不把跳过当证据，也不形成任何硬件声明。后续真实
provider conformance 应由每个硬件 runner 生成带环境与 artifact digest 的 payload，
再由集成团队批准的 contract fake 与 conformance suite 做替换验证。

## 真实硬件验证缺口

- 至少一份经 provisioner attestation、可复现且签入的国产物理卡 P0-P5 单卡结果；
- 国产卡设备名、驱动、固件、Torch-FL/FlagOS 构建、kernel 来源和容器摘要；
- 原生 FP64/complex128 的逐 kernel 支持与误差证据，以及与 Double-Single 的清晰边界；
- 无 CPU/CUDA 隐式回退、无 host staging 的可观测证据；
- FlagCX 内层路由、complex dtype collective 矩阵，尤其 reduce-scatter；
- 国产多卡 P2P 带宽/正确性、集合通信、rank placement 与拓扑指纹；
- 国产多节点网络后端、链路、超时/故障恢复与 collective correctness；
- 分片 forward、backward、optimizer update 保持同一 distribution semantics 的训练证据；
- 峰值显存、workspace、通信字节、per-rank ownership 与单卡容量失败；
- 重复运行、确定性、长稳、性能、收敛和 release payload 审计。

在这些缺口关闭前，推荐表述是“FlagOS 适配已在 CUDA-backed NVIDIA A800 上获得开发/
可移植性证据；国产真实卡支持尚未验证”，不得简写为“已支持国产算力”。
