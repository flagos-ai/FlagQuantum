# FlagQuantum Package Architecture

FlagQuantum keeps public compatibility at the package root while organizing
implementation code by product layer.

## Layers

- `core/`: circuit API and unified intermediate representation.
- `compilation/`: compiler passes, topology routing, scheduling, and planning.
- `runtime/`: stable runtime contracts, configuration, execution, training,
  backend boundaries, and distributed protocols.
- `simulation/`: statevector-adjacent kernels, MPS, noise, tensor contraction, and graph utilities.
- `algorithms/`: VQE, QAOA, Hamiltonians, ansatz builders, and quantum AI workflows.
- `deployment/`: trained-circuit packaging, quantum-cloud provider contracts, and inference results.
- `devices/`, `ops/`, `measurement/`, `encoding/`, `drawer/`, `utils/`: established FlagQuantum subsystems.

## Public API Policy

The top-level public style is:

```python
import flagquantum as fq
```

User-facing subsystems are also exposed as attributes, for example
`fq.compiler`, `fq.noise`, `fq.mps`, `fq.algorithms`, and `fq.deployment`.
Implementation files live in the organized layer directories rather than as
duplicate root modules.

## Train-To-Deploy Flow

FlagQuantum treats a trained circuit as a portable asset:

1. Build and train with `fq.Circuit`, `fq.algorithms`, and the native runtime.
2. Bind the optimized parameter tensor back into the parameterized quantum gates.
3. Compile with `fq.compiler` and optional backend topology.
4. Package with `fq.create_deployment_package`.
5. Submit through a `fq.QuantumProvider` implementation.
6. Fetch counts or expectation values for quantum-computer inference.

## Quantum Cloud Providers

FlagQuantum exposes provider adapters through one common deployment protocol:

- `fq.QuafuProvider`
- `fq.OriginQProvider`
- `fq.TencentQuantumProvider`
- `fq.TianyanProvider`
- `fq.GuodunProvider`
- `fq.FieldQuantumProvider`
- `fq.HttpQuantumProvider` for custom OpenQASM-style services

These adapters share `submit`, `query_status`, `fetch_result`, and
`discover_backends`. Production deployments configure each adapter with the
target platform endpoint, credentials, and endpoint mapping while the training
and packaging code remains unchanged.


┌─────────────────────────────────────────────────────────────────────────────┐
│                         用户输入 (量子AI任务 / 参数化电路)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ★ 规划与中间表示层 (IR & Planner)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     FlagQuantum IR (统一表示)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      统一规划器 (Unified Planner)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│          ┌───────────┬───────────┬───────────┬───────────┬───────────┐     │
│          ▼           ▼           ▼           ▼           ▼           ▼     │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ │
│  │ 执行计划   │ │ 内存规划   │ │ 通信规划   │ │ 可观测量   │ │ 梯度规划   │ │ 编译/部署  │ │
│  │Exec Plan   │ │Mem Plan   │ │Comm Plan  │ │Obs Plan   │ │Grad Plan  │ │Deploy Plan│ │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ★ 分布式运行时与执行引擎 (Runtime)                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   执行语义 (必须显式声明)                            │   │
│  │  ┌───────────────────────┐  ┌───────────────────────────────────┐  │   │
│  │  │ ⭐ sharded_across_ranks │  │  replicated_per_rank             │  │   │
│  │  │  (唯一可声明扩展性)     │  │  (仅限Smoke测试,不可声明扩展)     │  │   │
│  │  └───────────────────────┘  └───────────────────────────────────┘  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  data_parallel_replicated / observable_term_parallel / 其它   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              分片执行后端 (Sharded-First)                           │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐      │   │
│  │  │ 分布式态矢量│ │ 分布式MPS  │ │ 分布式TN   │ │ 分布式噪声  │      │   │
│  │  │(振幅/比特  │ │(格点/键    │ │(图分割/    │ │(密度矩阵/   │      │   │
│  │  │  分片)     │ │  分片)     │ │  切片)     │ │  轨迹分片)  │      │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ★ 关键约束: 若执行路径静默降级为复制 → 系统必须 Fail Closed               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  ★ 分布与通信策略 (Distribution & Comm)                     │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌────────────┐  │
│  │  分片策略      │  │  拓扑感知      │  │  通信原语      │  │ 计算/通信   │  │
│  │ (张量块/子图/  │  │ (节点内/节点间/ │  │ (P2P/AllReduce │  │  重叠       │  │
│  │  状态段)       │  │  存储/检查点)  │  │ /ReduceScatter)│  │ (掩盖延迟)  │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  └────────────┘  │
│                                                                             │
│  ★ 高频边界通信 → 保持在节点内 (NVLink/NVSwitch)                           │
│  ★ 跨节点通信 → 显式规划,低频化 (通过切片/批处理/检查点聚合)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                ★ 混合框架集成 (PyTorch + JAX Hybrid)                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   ┌─────────────┐    DLPack (零拷贝)    ┌─────────────┐           │   │
│  │   │  PyTorch    │ ◄──────────────────► │    JAX      │           │   │
│  │   │  (主训练接口)│                      │ (量子核加速) │           │   │
│  │   └─────────────┘                      └─────────────┘           │   │
│  │         │                                      │                  │   │
│  │         ▼                                      ▼                  │   │
│  │   ┌─────────────────────────────────────────────────────┐        │   │
│  │   │        torch.autograd.Function                      │        │   │
│  │   │        (跨框架梯度边界控制)                          │        │   │
│  │   └─────────────────────────────────────────────────────┘        │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────┐        │   │
│  │   │  分布式训练集成: torchrun / DDP / FSDP / DTensor   │        │   │
│  │   └─────────────────────────────────────────────────────┘        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ★ JAX 单卡加速不等于分布式容量扩展                                         │
│  ★ 只有量子态/收缩本身跨秩分片才算分布式扩展                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ★ 硬件与部署闭环 (Deployment Loop)                      │
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────┐   │
│  │     计算集群 (多节点)        │    │       量子硬件部署              │   │
│  │  ┌──────────┐ ┌──────────┐  │    │  ┌─────────────────────────┐  │   │
│  │  │ 节点 1   │ │ 节点 2   │  │    │  │  量子编译器              │  │   │
│  │  │ GPU 0..N │ │ GPU 0..N │  │    │  │  (布局/路由/校准感知)    │  │   │
│  │  └──────────┘ └──────────┘  │    │  └─────────────────────────┘  │   │
│  └─────────────────────────────┘    │              │                  │   │
│              │                       │              ▼                  │   │
│              │                       │  ┌─────────────────────────┐  │   │
│              └───────────────────────┼─►│  真实量子硬件 (QPU)     │  │   │
│                                      │  │  超导/离子阱/光量子等   │  │   │
│                                      │  └─────────────────────────┘  │   │
│                                      │              │                  │   │
│                                      │              ▼                  │   │
│                                      │  ┌─────────────────────────┐  │   │
│                                      │  │  测量结果/校准反馈      │  │   │
│                                      │  │  (闭环回到IR/Planner)   │  │   │
│                                      │  └─────────────────────────┘  │   │
│                                      └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ★ 质量门禁与基准测试 (Quality Gates)                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  预检诊断 (Preflight)                                               │   │
│  │  • NCCL/Gloo 可达性  • 带宽健康检查  • 超时行为  • 集合通信正确性  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  基准测试 JSON (必须包含)                                           │   │
│  │  {                                                                  │   │
│  │    "distribution_semantics": "sharded_across_ranks",               │   │
│  │    "scalability_claim_allowed": true,                              │   │
│  │    "local_memory_bytes_by_rank": [...],                            │   │
│  │    "communication_bytes": 0,                                       │   │
│  │    "single_gpu_expected_oom": true,                                │   │
│  │    "intra_node_communication_bytes": 0,                            │   │
│  │    "inter_node_communication_bytes": 0,                            │   │
│  │    "collective_counts": {...}                                      │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ★ 复制模式必须设置 scalability_claim_allowed: false                        │
└─────────────────────────────────────────────────────────────────────────────┘
