# TargetCapabilities 的 Platform 事实映射

状态日期：2026-09-03
负责团队：Platform Provider
基线：`vnext-phase1-contract-foundation`

## 结论

现有 `flagquantum.runtime.platforms.PlatformRuntime` 能真实观察的通用事实只有：provider
身份与版本、当前进程的 installed/activated/available、设备枚举与名称、部分设备总内存、
部分 allocator 内存，以及 stream/event/RNG/synchronize 句柄能否被调用。它没有 dtype、
原生双精度、Double-Single、kernel 驻留、拓扑、P2P、集合通信、节点间通信或 CPU fallback
事实字段。

因此 Platform 不能把 `_compiler.TargetCapabilities`、`BackendCapabilities.dtypes`、扩展
`CapabilityResponse`、方法存在、环境变量或厂商 metadata 直接提升为硬件支持。首版
Platform→Core 投影必须同时保留 `support_status`、`fact_exposure`、`evidence_level` 三条正交
轴；缺失字段输出 `unknown/not_exposed` 和 blocker，而不是补成 `false`、零、空拓扑或
“未发生回退”。本轮新增的是 Platform-owned CPU 窄适配
`cpu_platform_to_target_capability_snapshot`；它只接收注入式 probe、时间和 evidence，
不新增 Provider 注册体系、不改变现有 Platform API/default 行为，也不触碰受保护 Core/API。

代码路径分类为 `single_device_fast_path` 的 discovery 特征映射。引用的分片和通信材料仅说明
已有证据边界，不形成新的硬件或可扩展性证据。

## 三个不可合并的维度

| 维度 | 值域 | 回答的问题 | 本切片的失败关闭规则 |
| --- | --- | --- | --- |
| support status | `unknown` / `unmeasured` / `unsupported` / `verified` | 当前目标对具体能力是否支持 | 只有与目标、设备、dtype、kernel/workload 范围匹配的观测才能进入 `verified`；声明最多为 `unmeasured` |
| fact exposure | `observed` / `declared` / `not_exposed` / `unknown` / `not_applicable` | 值从何而来，或为何没有值 | SDK 未给字段为 `not_exposed`；来源不清为 `unknown`；`not_applicable` 必须提供适用性依据 |
| evidence level | `basic` / `observable` / `certification` | 证据强度上限 | 身份、版本、发现、接口调用和厂商声明默认只到 `basic`；workload-bound 路径观测才可能是 `observable`；批准矩阵和审计后才可能是 `certification` |

`support_status` 描述的是能力结论，不是“这次探测调用是否成功”。因此明确的
`available=false` 负向探测应为 `unsupported/observed/basic`；只有
`available=true` 才是 `verified/observed/basic`。这项生命周期结论仍不得传播为 dtype、
kernel、通信或无回退结论。

### Fact exposure 逐项对账

| exposure | 当前可接受来源/例子 | support status 可能值 | 约束 |
| --- | --- | --- | --- |
| `observed` | 生命周期探测、内存 API 数值、绑定 workload 的 operator/route probe | `verified` 或 `unsupported` | 正/负结论都必须绑定实际 scope；观测到失败不能写成 unknown |
| `declared` | provider JSON-safe metadata、backend dtype 候选、外部 attestation 声明 | 默认 `unmeasured` | 声明不能因 provider available 而晋级 verified |
| `not_exposed` | CPU 内存、当前 PlatformRuntime 的 dtype/topology/P2P/collective/kernel residency/fallback 字段 | `unknown` | Core nullable-fact 修复后，CPU adapter 生成 `value=null`、`unknown/not_exposed` 和非空 blocker；不得补 false、0 或空集合 |
| `unknown` | 来源不明、不可序列化 SDK 对象、冲突或无法归因的 route 值 | `unknown` | 丢弃 vendor object，只保留可移植 blocker |
| `not_applicable` | 例如 snapshot 明确限定 `world_size=1` 时该 scope 的 collective 事实 | `unknown` | 必须有 applicability reason；不能替代 unsupported 或 unmeasured |

另外必须分别表示：

- **API 模型支持**：类型、方法或请求词汇允许表达某能力；不说明设备可用。
- **当前环境可用**：当前进程的 provider/device 被探测到；不说明某 dtype/kernel/workload
  通过。
- **已验证证据**：绑定 provider、物理设备、版本、代码 revision、workload、时间与 digest
  的结果；只能支持材料覆盖的范围。

## 当前可观测字段表

| 能力域 | 当前通用来源 | 可输出的事实 | 默认三轴映射 | 不可推导 |
| --- | --- | --- | --- | --- |
| 身份 | `PlatformIdentity` | provider、device type、PyTorch/provider 版本、vendor 字符串 | 值存在时 `verified/observed/basic` 仅表示身份观测 | vendor 身份不是硬件能力或国产卡认证 |
| 生命周期 | `installed/activated/is_available` | 当前进程状态 | `true` 为 `verified/observed/basic`；明确的 `false` 为 `unsupported/observed/basic` | available 不等于 kernel、dtype、通信 verified |
| 设备发现 | `discover()/PlatformDevice` | logical device、index、name、available | 枚举结果为 `verified/observed/basic` | 名称不得用于厂商类别或能力推断 |
| 设备内存 | `PlatformDevice.memory_bytes`、`MemorySnapshot` | CUDA 通常可给 total/allocated/reserved/free；FlagOS 取决于公开 SDK；CPU 全缺失 | 数值存在为 `verified/observed/basic`；缺失为 `unknown/not_exposed/basic` | 缺失不是零；不代表容量、峰值、workspace 或 workload 可装入 |
| stream/event | `stream()` / `event()` | 受控调用可返回句柄 | 仅能说 API 调用已观测；语义、计时、并发和 kernel 归属仍需独立事实 | 方法存在不证明 timing、异步执行或设备驻留；句柄不可进入 snapshot |
| dtype | `BackendCapabilities.dtypes`、operator probe、artifact | 通用 backend 候选；probe/artifact 覆盖的逐算子事实 | backend 列表是 declared/unmeasured；通过的 runtime probe 可按其 operator/dtype 范围 verified | 不能从 `cuda`、`flagos` 或张量类型推断全设备 dtype 支持 |
| 原生双精度 | 无 Platform 字段；硬件/算子材料 | 特定 workload 的 float64/complex128 运行结果 | 无材料为 `unknown/not_exposed/basic`；材料必须记录原生路径后才可 observed | A800 上 complex128 结果不证明所有 CUDA，也不证明 FlagOS 国产卡原生 FP64 |
| 软件扩展精度 | Double-Single 合同、实现与 A800 artifacts | 受限 P0-P5 路径上的 high/low FP32 表示 | API/算法存在为 declared/unmeasured；具体 artifact 按覆盖范围升级 | 不等同通用 FP64/complex128；不证明普通 autograd/torch.optim 全链路 |
| kernel 驻留 | operator probes、workload evidence | 逐 operator/dtype 的 forward/backward 或逻辑设备驻留 | 必须绑定 probe/profile/workload；PlatformRuntime 单独为 `unknown/not_exposed` | provider 可用、stream 存在、输出 device label 都不足以证明无 host kernel/fallback |
| 拓扑/P2P | benchmark/runner 的 rank placement、topology fingerprint | 仅 artifact 记录的节点、rank、链接信息 | 无当前快照来源时 `unknown/not_exposed/basic` | 设备数不能推出 P2P；拓扑指纹不能自动推出带宽或 collective route |
| 节点内集合通信 | Runtime distributed conformance/artifact | 指定 backend、collective、dtype、world size 的结果 | 按矩阵单元映射；失败可为 `unsupported/observed`，未测为 `unmeasured` | `torch.distributed` 或 process group 初始化不证明所有集合通信 |
| 节点间通信 | 多节点 runner/artifact | 指定网络/backend/topology/workload 的结果 | 无匹配材料为 `unknown/not_exposed/basic` | 单节点 NCCL/FlagOS 材料不得晋级多节点 |
| CPU fallback | workload route audit/result evidence | requested/selected path、fallback allowed/used/reason | 真正观测后才可 `verified/observed/observable`；route 未暴露则 `unknown/not_exposed` | `not_exposed` 不能反证 `fallback_used=false` |

## 平台事实矩阵

| 平台 | API 模型 | 当前实现可观察 | 已验证范围 | 不能保证 |
| --- | --- | --- | --- | --- |
| CPU | 完整 Platform 生命周期；PyTorch backend 列出 complex64/complex128 | 单一 CPU、同步、host event、null stream、RNG；内存字段为空 | 本地 CPU 数值/训练和 operator probe 测试覆盖的具体路径 | Platform 不提供 CPU 内存、NUMA、拓扑、通信、逐 kernel 原生 FP64 或 fallback 事实 |
| NVIDIA CUDA | CUDA discovery、memory、stream/event、RNG；Runtime 有 NCCL 路径 | 当前 CUDA 进程的设备名/显存/allocator；可创建 CUDA handle | 已签入 A100/A800 artifact 覆盖的设备、版本、dtype、workload、world size | 所有 NVIDIA 型号、所有 kernel/dtype、event timing、P2P/多节点、无 CPU fallback 的普遍保证 |
| FlagOS | Torch-FL 懒激活和 `torch.flagos` 设备生命周期；Runtime 有 `backend=flagos` 路径 | SDK 实际暴露的设备、可选内存、stream/event/RNG、runtime identity | 已签入材料是 CUDA-backed Torch-FL 在 NVIDIA A800 上的开发证据 | 国产物理卡、FlagCX 内层 route、无 host staging、全 collective、FlagOS 多节点和生产支持 |
| 国产真实卡 | 有 attestation 驱动 P0-P5 候选 harness | 仓库当前无已签入真实卡 observation | 无经评审的真实国产卡结果 | 厂商、型号、驱动、原生 FP64、Double-Single、kernel 驻留、P2P、collective、多节点和无回退均未知 |
| 其他厂商/环境 hint | Extension/Platform 类型和 `FLAGQUANTUM_ACCELERATOR` 可表达名称 | hint 仅产生 `available=false` 的 unknown accelerator | Mock 只验证隔离和失败关闭 | 任何真实硬件支持 |

### 精度边界

`BackendCapabilities.dtypes=(complex64, complex128)` 是 PyTorch 执行模型的候选集合，不是
per-device hardware discovery。`CapabilityEvidence` 只有 `runtime_probe` 或 `hardware_ci`
通过时才把具体 operator/dtype 单元视为 verified，并且不能外推到未探测 kernel。

Double-Single 是软件扩展精度，必须独立于 `native_fp64/native_complex128`。现有 P3/P4/P5
分别覆盖 full-state high/low、受限设备侧门生成和显式 Double-Single SGD；其 FP32 指数范围、
受限门集合、普通 autograd FP32 边界及非 `torch.optim` 限制必须随事实保留。

### 通信与 fallback 边界

`PlatformRuntime` 没有 process group、P2P、collective、rank placement 或 topology 方法。
`build_distributed_identity()` 也明确区分 outer backend 与 inner route：FlagOS process group
初始化仍把 `inner_backend`、FlagCX 和 host staging 保留为未知，并关闭 communication claim。
CPU fallback 只可由执行路径观测或经审计的 provider attestation 给出；设备标签和成功结果
不能反推没有 host execution。

## 证据来源与解释上限

| 来源 | 可证明 | 不可证明 |
| --- | --- | --- |
| `tests/unit/test_platform_runtime.py` | CPU 生命周期、CUDA discovery fake、FlagOS lazy/public API 适配和厂商名不参与分类 | 真实 CUDA、FlagOS 或国产硬件执行 |
| `flagquantum/runtime/platforms/cpu_target_capabilities.py` | 注入式 CPU device/count/memory/precision 观察到 Core v1 snapshot；独立 CPU identity/scope；TTL/evidence 传递；缺失事实 blocker | 不发现 CUDA/FlagOS/QPU；不产生 requirements、fallback 或性能/硬件声明 |
| `tests/team/platform/test_cpu_target_capabilities.py` | CPU adapter 的 source/evidence round-trip、unavailable/missing/negative probe、TTL/scope 和默认行为不变 | 任何硬件能力；fake probe 不是真实硬件证据 |
| `tests/team/platform/test_target_capability_facts.py` | 候选投影的三轴分离；缺 SDK 字段保持 unknown；声明不晋级；对象不泄漏 | 任何硬件能力；test-only fixture 不是 Core 合同 |
| `runtime/operator_probes.py` + `CapabilityEvidence` | 特定 provider/device/profile/operator/dtype 的 forward/backward probe | 未探测算子、通信、拓扑、物理 route 或生产等级 |
| `artifacts/flagos_cuda_reference_a800_20260824.json` | NVIDIA A800 上单 `flagos:0` CUDA-backed Torch-FL 参考路径 | 国产卡、原生 FlagOS 硬件、无 host fallback |
| `artifacts/flagos_workload_capability_f4_20260826.json` | 单节点 2/4/8 卡 FlagOS-on-A800 指定 dtype/workload 开发矩阵及 reduce-scatter 缺口 | FlagCX route、host staging、多节点、认证/发布声明 |
| `artifacts/flagos_statevector_capacity_f5_a800_20260827.json` | 记录范围内 32-qubit complex128 单卡/复制 OOM 与 8-rank 分片完成 | 通用容量、反向/优化器容量、国产卡或生产扩展性 |
| `artifacts/flagos_transport_observability_f6_a800_20260827.json` | 单节点四类 collective、两种 complex dtype 的逻辑设备结果 | 内层通信实现、无 host staging、性能和多节点 |
| `artifacts/split_real_imag_*_a800_*.json` | 指定 A800 CUDA/FlagOS-on-CUDA P0-P5 Double-Single 开发路径 | 一般 FP64 等价、所有门/优化器、收敛、国产卡和生产能力 |
| `tools/validate_domestic_single_card.py` | 外部 attestation、P0-P5、no-fallback 候选证据的严格验收规则 | 当前已有国产卡证据；通过候选仍不等于 hardware certification |

`a800-node-0` 与 `a800-node-1` 是 NVIDIA A800 节点。会话中可用性或临时运行不能替代签入、
带 digest 和 revision 的审核 artifact；即使在这两节点成功，也只能形成 NVIDIA CUDA 范围的
证据，不能形成国产硬件或 FlagOS 证据。

## CPU Platform 窄适配实现

`flagquantum.runtime.platforms.cpu_target_capabilities` 是当前唯一实现切片。调用者注入
`CPUCapabilityProbe`，其 `observe()` 返回 `CPUCapabilityObservation`，同时提供独立的
`target_id`、`provider_version`、`target_revision`、`environment_id`、probe `source_ref` 和
静态 `target_class_source_ref`。适配器
固定 `target_class=local_runtime`、`provider=pytorch_cpu`，并用实际 `device_ids` 建立 scope；
同时生成 `target.class=local_runtime` 的权威静态声明 fact，使 Compiler 与 CPU candidate 可匹配。

- `available=true` 且设备数量/ID 经 probe 观察时，`device.kind`、`device.count` 为
  `verified/observed`；这些 observed facts 及内存、显式 precision probe 结果必须由至少
  `observable` 级 probe evidence 支撑。纯静态 `target.class` 可由 `basic` evidence 支撑。
- `available=false, device_count=0` 生成 `device.count=0` 的
  `unsupported/observed` fact 和 `cpu_unavailable` blocker；它不会被替换成 CPU 之外的目标。
- 内存或 precision 缺失时不填 0、空字符串或伪造 dtype。由于 Core v1 当前 typed fact value
  nullable，生成 `value=null`、`unknown/not_exposed` 与非空 blocker；Core matcher 会按
  fact status fail closed。
- probe 明确给出 `unsupported`/`unmeasured` 的 precision 时保留原状态和非空 blocker；
  adapter 不把 declared、Python dtype、`BackendCapabilities` 或 provider identity 升成 verified。
- 精度 `verified` 仅接受 `observed` exposure；`verified/declared` 会在 probe value object 层拒绝。
- `captured_at` 与正 TTL 由调用者注入，evidence refs 原样以 Core `EvidenceReference` 传入；
  adapter 不制造 evidence digest，不把 `stream/event` handle 放入 snapshot。

该适配器只生产 `TargetCapabilitySnapshot`，不生产 `CapabilityRequirement`，不选择目标，
不实现 Runtime fallback，不影响 `get_platform_runtime()`、平台 registry 或默认 backend。

## Platform→Core 最小投影提案

以下仅供 Integration/Core 评审。Core 应拥有最终 schema、枚举和序列化规则；Platform 只做
现有 `PlatformRuntime`/SDK 到该合同的 adapter，不建立 registry。

```text
PlatformCapabilitySnapshotCandidate
  schema_version
  snapshot_id / observed_at
  provider_identity
    provider, provider_version, device_type, vendor
    runtime_versions, physical_device_id/name
  environment
    installed, activated, available, availability_blockers
  device
    logical_device, index, memory facts
  facts[name]
    value
    support_status
    fact_exposure
    evidence_level
    source_kind / source_ref
    scope {device, dtype, kernel, workload, world_size, node_count}
    blockers
  evidence_refs[] {artifact_uri, sha256, code_revision, environment_id}
  snapshot_blockers[]
```

首批闭集 fact name 建议为：`storage_dtypes`、`compute_dtypes`、
`accumulation_dtypes`、`native_fp64`、`native_complex128`、
`software_precision_modes`、`device_memory_*`、`stream_semantics`、
`event_semantics`、`kernel_residency`、`topology`、`p2p`、
`intra_node_collectives`、`inter_node_communication`、`cpu_fallback_used`。

投影规则：

1. Platform identity 只作为 identity，不触发 capability promotion。
2. provider metadata 只能作为 JSON-safe `declared/basic` 输入；对象、callable、live handle 和
   未知自由字段不进入 Core snapshot。
3. SDK 字段缺失在候选层输出 `value=null`、`support_status=unknown`、
   `fact_exposure=not_exposed` 和具体 blocker；顶层 blocker 只保留整张 snapshot 不可用的
   全局问题。Core matcher 对 nullable non-verified fact 失败关闭。
4. `available=true` 只验证环境可用；不传播到 dtype、kernel、通信或 fallback。
5. `unsupported` 需要一次适用且权威的负向探测；未运行探测只能是 `unmeasured` 或 `unknown`。
6. `not_applicable` 必须携带 applicability reason；不能用它隐藏未实现或未测能力。
7. `observed` 值必须绑定 source、scope 和时间；snapshot 过期后不能继续作为当前环境事实。
8. claim 上限取所有所需 fact/evidence 的最低等级；任一 required fact 为 unknown、unmeasured、
   not_exposed 或 stale 时，preflight 失败关闭。
9. stream/event 等 vendor handle 留在 Platform adapter 内，以 Core-owned opaque handle id 或局部
   回调使用；不可序列化对象不进入 Runtime result、Simulation state 或用户 API。
10. CPU fallback 未观测且 route 不暴露时必须保留 unknown blocker，严禁填 `false`。
11. Core consumer 只有在所有 required fact 均为 `verified/observed`、未 stale，且证据等级达到
    claim 最低门槛时才可放行；`unsupported`、`unmeasured`、`unknown`、`not_exposed` 与
    无适用性理由的 `not_applicable` 一律失败关闭。

## 厂商泄漏复核

1. `runtime/distributed/flagos_runtime.py` 在 Platform 边界外直接读取
   `torch.flagos.set_device/current_device`。这是明确的边界债务；应在批准的合同中下沉为
   Platform 设备选择能力，本切片不改契约。
2. Runtime/Simulation 的 CUDA fast path 多处直接使用 `torch.cuda`；这些是已登记的既有
   architecture ceiling。它们会妨碍替换为其他平台，不能作为通用 capability snapshot 来源。
3. `runtime/operator_backends.py` 把 FlagGems `vendor_name` 降为字符串，未泄漏 vendor object，
   但其 operator catalog 是声明/探测输入，不是平台硬件认证。
4. `PlatformRuntime.stream/event -> Any` 与 identity/device `metadata: Any` 尚未在合同层强制
   JSON-safe。现实现可返回 PyTorch/provider handle；它们必须止于 adapter，投影只接受标量、
   闭集字符串列表与受控引用。
5. 未发现 `torch_fl` import 进入根用户 API 或 Simulation；根 API 暴露通用
   `AcceleratorInfo`，不会传递 Torch-FL/Hygon SDK 对象。

## 当前不可保证项

- 任一国产真实卡已经通过验证或获得硬件认证；
- 国产卡原生 FP64/complex128、Double-Single 全路径、设备 kernel 驻留或无 CPU/CUDA 回退；
- FlagCX 内层 route、无 host staging、完整 complex collective/P2P；
- FlagOS 多节点，或国产多卡/多节点 rank placement、拓扑、网络与容错；
- 仅由 CUDA/FlagOS available、设备名、环境 hint、Mock 或接口存在推出任何 workload 支持；
- 未签入远程节点临时结果可作为 release、production、certification 或国产算力证据。

在这些缺口关闭前，准确表述仍是：“CPU 是真实本地路径；NVIDIA CUDA 有范围受限的真实
硬件材料；FlagOS 有 CUDA-backed NVIDIA A800 的适配开发证据；国产真实卡能力未知且未验证。”
