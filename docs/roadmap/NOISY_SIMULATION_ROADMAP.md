# FlagQuantum 含噪声模拟落地规划

## 1. 决策与定位

FlagQuantum 值得现在开始建设含噪声模拟能力。当前代码库已经具备
`KrausChannel`、`NoiseModel`、精确密度矩阵、MPS trajectory，以及
SV/MPS/TN 规划和多 GPU 执行基础，不需要从零搭建。

项目的首要目标不是简单增加若干噪声通道，而是构建：

> 一套可信、可扩展、可验证，并能在多 GPU 上以明确置信区间达到目标精度的
> quantum trajectory 系统。

建议以精确密度矩阵作为小规模正确性基准，以分布式 SV trajectory 作为通用
规模化路线，以 MPS trajectory 利用低纠缠结构。MPO 和 noisy TN 在前述能力
稳定后建设。

首个目标 API：

```python
result = fq.run(
    circuit,
    noise_model=noise,
    observables=hamiltonian,
    mode="auto",
    trajectories="auto",
    target_standard_error=1e-3,
    seed=42,
)
```

系统应完成：

1. 将噪声模型降低为统一 Channel IR；
2. 在 density matrix、SV trajectory、MPS trajectory 之间选择；
3. 根据目标统计误差决定轨迹数量；
4. 在多 GPU 上调度轨迹；
5. 返回期望值、方差、标准误、置信区间和截断误差；
6. 记录随机流、噪声模型身份和执行审计；
7. 使用小规模密度矩阵路径进行交叉验证。

## 2. 总体架构

```text
DeviceNoiseProfile / NoiseModel
                 │
                 ▼
          Unified Channel IR
                 │
                 ▼
        Noisy execution planner
          ├─ exact_density
          ├─ statevector_trajectory
          ├─ mps_trajectory
          └─ future
              ├─ mpo_density
              └─ tensor_network_trajectory
                 │
                 ▼
       statistics and error accounting
```

### 2.1 噪声描述层

建议逐步将当前噪声实现整理为：

```text
flagquantum/noise/
├── channels.py
├── model.py
├── device_profile.py
├── lowering.py
├── validation.py
└── serialization.py
```

噪声描述必须独立于执行后端。同一份 `NoiseModel` 应可交给密度矩阵、SV、
MPS，以及未来的 MPO/TN。

需要覆盖的噪声位置包括：

- 门前和门后噪声；
- 按持续时间计算的 idle noise；
- 初始化、reset 和 readout noise；
- coherent over-rotation；
- 后续的串扰、泄漏和时间相关噪声。

### 2.2 统一 Channel IR

通道指令除 Kraus 矩阵外，还应保存来源和时序：

```python
ChannelInstruction(
    channel_type="kraus",
    wires=(0,),
    parameters={...},
    placement="after_gate",
    source_gate_id="gate-17",
    duration_ns=35.0,
)
```

最小指令集合：

- `KrausChannel`
- `LindbladSegment`（后续）
- `ReadoutChannel`
- `ResetChannel`
- `ClassicalCondition`

### 2.3 执行计划

`NoisyExecutionPlan` 至少显式记录：

- trajectory 数量和 batch 大小；
- GPU/rank 分配；
- 状态与 workspace 显存预测；
- 目标标准误及最大轨迹数；
- MPS bond/cutoff 和截断预算；
- 随机流策略；
- checkpoint 策略；
- 候选后端拒绝原因。

## 3. 分阶段实施

## Phase 0：冻结语义与结果契约

预计 3–5 个工作日。

工作内容：

- 明确噪声通道与门的应用顺序；
- 定义 batch、wire、dtype 和 device 传播规则；
- 固定随机数与跨 world-size 可复现语义；
- 区分物理通道噪声和测量采样扰动；
- 定义版本化结果 schema 和模型 identity。

建议结果类型：

```python
@dataclass(frozen=True)
class NoisyExecutionResult:
    expectation: Tensor
    variance: Tensor
    standard_error: Tensor
    confidence_interval: Tensor
    trajectory_count: int
    effective_sample_size: float
    truncation_error_bound: float | None
    noise_model_identity: str
    execution_summary: dict
```

验收条件：

- 相同 seed 完全复现；
- 不同 seed 在统计容差内一致；
- NoiseModel 可稳定序列化并生成 identity；
- identity 或测量摘要被篡改时检查失败；
- 非 CPTP 或维度错误的通道 fail-closed。

## Phase 1：正确性黄金路径

预计 1 周。

第一批通道：

- bit flip；
- phase flip；
- depolarizing；
- amplitude damping；
- phase damping；
- thermal relaxation；
- reset error；
- readout confusion matrix；
- coherent over-rotation。

强化密度矩阵路径，使其支持任意单/双比特 Kraus、batch、任意 wires、
observable、sampling、complex64/complex128 和必要的 autograd。

需要验证：

```math
\rho=\rho^\dagger,\qquad
\operatorname{Tr}(\rho)=1,\qquad
\rho\succeq0.
```

测试覆盖 1–8 qubits、概率边界值、纠缠线路、多观测量，并同时对照解析解与
独立参考实现。

验收条件：

- trace、Hermitian 和 positivity 检查达到 dtype 对应容差；
- 所有内置通道的小规模解析结果匹配；
- trajectory 均值在合理统计置信区间内覆盖 density 参考值。

## Phase 2：生产级 MPS trajectory

预计 1–2 周。

在现有 MPS trajectory 上增加：

- 稳定的 global trajectory ID 和 seed 派生；
- Welford 在线均值/方差；
- adaptive stopping；
- trajectory checkpoint/resume；
- 截断误差累计与独立报告；
- 多观测量单次执行；
- 单轨迹异常隔离。

目标接口：

```python
result = fq.run_noisy_mps(
    circuit,
    noise,
    observables=observables,
    trajectories=1024,
    target_standard_error=1e-3,
    max_trajectories=8192,
    max_bond=256,
    cutoff=1e-8,
    seed=7,
)
```

每个 trajectory batch 后更新：

```math
\mathrm{SE}=\frac{s}{\sqrt{N}}.
```

当全部目标观测量满足绝对/相对误差阈值时停止，同时使用
`min_trajectories` 防止少量样本偶然产生零方差。

验收条件：

- 确定性通道与 density 路径一致；
- 随机通道置信区间覆盖 density 参考；
- adaptive stopping 不系统性低估误差；
- checkpoint 恢复与连续执行一致；
- statistical error 与 MPS truncation error 分开报告。

## Phase 3：Batched SV trajectory

预计 2 周。这是首版最关键的性能阶段。

当前进度（2026-08-06）：首个可验证里程碑已完成，包括真实 trajectory
batch 布局、任意 unitary、单/多比特 Kraus 的批量采样/归一化、Pauli 快路径、
branch-state-free 振幅阻尼路径、readout Z 后处理、global trajectory ID 随机流、
内存预算自动选择，以及 density matrix 交叉测试。CUDA 峰值、批量吞吐和多 GPU
性能证据仍属于本阶段未完成项，因此本阶段尚未整体验收。

状态布局：

```text
[trajectory_batch, circuit_batch, 2^n]
```

需要实现：

- batched single-qubit Kraus；
- batched two-qubit Kraus；
- probability、sampling、apply、normalize 融合；
- Pauli channel 快速路径；
- amplitude damping 快速路径；
- readout error 后处理路径。

Pauli noise 应采样后复用 X/Y/Z kernel，不走通用矩阵乘法。

显存规划：

```math
M \approx
B_{\mathrm{trajectory}}
\times B_{\mathrm{circuit}}
\times 2^n
\times \text{complex bytes}
+ M_{\mathrm{workspace}},
```

其中前四项实际为乘积。planner 根据可用显存确定 trajectory batch，超预算时
减小 batch 或 fail-closed。

验收条件：

- batched 与逐轨迹执行在相同 trajectory IDs 下匹配；
- 相比 Python 逐轨迹循环有明确吞吐提升；
- CUDA reserved peak 不超过预测预算；
- 1/2/4/8 GPU 均有时间、吞吐和显存实测证据。

## Phase 4：分布式 trajectory 调度

预计 1–2 周。

当前进度（2026-08-06）：statevector trajectory 已支持 rank-local round-robin
global ID ownership、world-size-independent seed，以及带完整性/重复 ID/noise model
identity 校验的结果合并；1/2/4 rank 单进程语义测试可与单 rank 保留态逐条匹配。
NCCL count/sum/sum² collective 已集成公共 `fq.run` 自动路径，并在 8×A800
完成固定 workload 的 1/2/4/8 卡强扩展实测，8 卡达到 7.29×、91.1% 并行效率。
分布式 adaptive stopping 已按 batch 执行全局标准误判断，并在 8×A800 完成
128/512 条统一早停实测；首版要求 trajectory 上限能整除 world size × batch。
statevector 已支持绑定 circuit/model/seed/schedule/rank ownership 的原子统计
checkpoint，并完成 8 卡 128/512 中断后重启恢复到 512/512 的实测。批次内失败
会降级为逐 trajectory 重试，成功样本继续统计，失败 ID 写入结果/checkpoint；
8 卡注入 global ID 3 故障后完成 127/128 且 collective 正常退出。Phase 4 的
核心语义已完成；更强的容错仍需覆盖进程/GPU/NCCL 级故障。

第一层使用 trajectory parallel。每个 rank 独立处理全局轨迹 ID 的一个子集，
最终只归约：

- `count`
- `sum(x)`
- `sum(x²)`
- failure count
- maximum truncation error

随机数使用 world-size-independent 的 counter-based schema：

```math
\operatorname{seed}_t =
H(\operatorname{global\ seed},\operatorname{trajectory\ id}).
```

第二层支持混合并行：

```text
trajectory groups × statevector shard ranks = world size
```

例如 8 GPU 可以选择 8×1、4×2、2×4 或 1×8。planner 根据单条状态大小、通信
成本和目标轨迹数搜索最合适的组合。

checkpoint 保存：

- 下一 global trajectory ID；
- count/sum/sum-square；
- RNG schema 和 seed；
- NoiseModel identity；
- circuit digest；
- MPS 截断统计；
- world-size-independent 调度状态。

验收条件：

- world size 1/2/4/8 统计结果一致；
- 改变 world size 不改变给定 trajectory ID；
- 中断恢复后结果与连续执行一致；
- 通信正确性、吞吐和尾延迟均有硬件证据。

## Phase 5：噪声感知自动选择器

预计 1 周。

当前进度（2026-08-06）：已实现 density matrix、batched statevector trajectory
和 MPS trajectory 的统一候选报告，包含 lowered noise event count、最大 Kraus
rank、内存估算、exact/sampling/truncation 误差语义、资格与拒绝原因。原有 auto
dispatch 已接入该选择器；所有候选违反预算时 fail-closed。A800 实测 workload
的版本化选择快照已保存。首个跨 8/10/12/16 qubits、depth 4/8、两种 noise
density 的 A800 校准集也已完成，确认 SV 时间与 lowered channel count 强相关，
且当前小规模 MPS trajectory 吞吐显著落后于 batched SV。版本化校准 artifact
现已可作为可选成本输入；只有 circuit digest、noise identity、通道数和 trajectory
batch 等严格匹配时才参与决策，否则保持解析策略回退。8-qubit A800 校准决策
快照已保存。目标统计误差现在也进入 time-to-solution 证据：没有 pilot variance
时显式使用 `Var(Pauli) <= 1` 上界估计轨迹数，并报告给定 trajectory ceiling 是否
足够；运行时仍以观测到的全局 standard error 决定停止。固定 seed/trajectory ID
的 pilot-variance 流程也已接入，保存逐 observable 的均值、方差和身份信息；单卡
A800 的 32-trajectory pilot 点估计对应 121 条；新增的 95% bounded-variance
上置信界经 8 observable Bonferroni 修正后仍为 2,500 条，并因此正确选择 3.40 s
的 exact density，而不是预计 31.41 s 的 SV trajectory。32/128/512/2048 的
单卡 A800 pilot-size 曲线也已完成：上界依次为 1.0/0.5745/0.2394/0.1104，
但 512 pilot 已耗时 6.51 s，超过 exact density 的 3.40 s，因此所有点都选择
density。该证据把 pilot 定位为 density 因规模或显存不可行后的策略。正式
512-trajectory scaling 数据现已转换为严格绑定 world size 的 1/2/4/8×A800
selector calibration，匹配耗时为 11.07/5.28/2.87/1.52 s，四份决策均记录
`exact_device_calibration`。审计确认 MPS 目前只有手工 rank-local 分片和事后
merge 原语，公共执行没有 rank 推断与 collective reduction；因此 world_size>1
时已显式拒绝 `distributed_collective_not_implemented`，未给显式 rank 的多卡
noisy_mps 请求也 fail-closed；手工 rank-local + merge 仍保留。下一步实现公共
分布式 MPS trajectory 统计归并后再采集成本证据。

候选成本模型：

- density matrix；
- SV trajectory；
- MPS trajectory；
- future MPO；
- future TN trajectory。

决策输入：

- qubits、batch 和 noise event count；
- 平均/最大 Kraus rank；
- 目标统计误差；
- MPS bond 估计；
- 输出数量和类型；
- world size 与显存限制；
- exact/approximate policy；
- 是否要求梯度。

初始策略：

```text
小规模且 exact_noise=True        → density_matrix
低纠缠/近邻且允许近似            → mps_trajectory
其他一般线路                     → statevector_trajectory
单卡显存不足                     → state sharding 或尝试 MPS
所有候选均违反预算               → fail-closed
```

选择结果必须包含拒绝原因、内存预测、轨迹数估计和误差策略。

## Phase 6：噪声训练与后向

前向稳定后启动，预计 2–3 周。

第一版范围：

- 电路参数梯度；
- 固定噪声参数；
- trajectory autograd；
- common random numbers；
- 多 GPU 梯度统计归约。

暂缓任意 Kraus 参数的无偏梯度和高阶梯度。

对于参数扰动或 paired execution，复用相同 trajectory IDs 和 Kraus 随机流。
结果报告 gradient mean、standard error、signal-to-noise ratio 和 trajectory
count。

验收条件：

- 小线路梯度与 density/autograd 参考匹配；
- common random numbers 明确降低配对梯度方差；
- 分布式梯度与单卡参考一致；
- 训练 checkpoint 包含 trajectory 状态。

## Phase 7：MPO 与 noisy TN

不作为首版阻塞项。

MPO 优先服务：

- 一维局域开放系统；
- 需要避免 Monte Carlo 方差；
- operator entanglement 可控的场景。

noisy TN 优先服务：

- 少量局部观测量；
- 可裁剪因果锥；
- 中低树宽；
- 低 Kraus rank；
- 稀疏噪声事件。

首个 noisy TN 版本应优先采用 trajectory TN，复用现有 TN 收缩、切片、校准
和多 GPU task parallel，避免直接构造可能使有效树宽急剧上升的完整双层密度
网络。

## 4. 测试与证据

### 4.1 正确性证据链

```text
解析解
  ↕
density matrix
  ↕
SV trajectory
  ↕
MPS trajectory
  ↕
future TN trajectory
```

trajectory 正确性采用置信区间覆盖和重复试验检验，不能仅用固定绝对误差。

### 4.2 性能指标

每个后端记录：

- trajectories/s；
- noisy gate-events/s；
- time-to-target-error；
- peak allocated/reserved memory；
- trajectory batch size；
- GPU utilization；
- communication time；
- final standard error。

首要指标是达到同一目标精度所需时间：

```math
\text{Time To Target Error},
```

而不是只比较单轨迹前向时间。

### 4.3 物理真实性证据

导入硬件校准后比较：

- ideal simulation；
- simplified noise；
- calibration-driven noise；
- real hardware result。

必须报告模型无法解释的残差、缺失校准参数和独立噪声假设，不能把简单
depolarizing 拟合描述为真实设备的完整复现。

## 5. Noisy Simulation v1 范围

首个里程碑命名为：

> **Noisy Simulation v1：可验证、可并行的量子轨迹**

包含：

- 9 类基础噪声；
- 统一 NoiseModel/Channel IR；
- density matrix 黄金路径；
- MPS trajectory；
- batched SV trajectory；
- 单节点 1/2/4/8 GPU trajectory parallel；
- adaptive stopping 和置信区间；
- checkpoint/resume；
- 自动后端选择；
- 前向 API。

不包含：

- 多节点生产声明；
- MPO；
- noisy TN 双层网络；
- leakage；
- 非马尔可夫噪声；
- 完整噪声参数梯度；
- 全套误差缓解。

## 6. 优先级与风险控制

实施优先级：

1. 正确性与结果 schema；
2. MPS trajectory 统计和恢复；
3. batched SV trajectory；
4. 多 GPU trajectory parallel；
5. adaptive error control；
6. 自动选择器；
7. 后向训练；
8. 硬件校准导入；
9. MPO/noisy TN。

主要风险：

| 风险 | 控制方法 |
|---|---|
| 轨迹统计误差被当作数值误差 | 强制分别报告方差、标准误和截断误差 |
| 改变 world size 后不可复现 | 使用 global trajectory ID 派生随机流 |
| trajectory batch 导致 OOM | reserved-memory 预检和自动降低 batch |
| MPS 截断偏差掩盖噪声结果 | 与 density 对照并独立累计 truncation error |
| 只增加通道但没有产品价值 | 用 time-to-target-error 和真实任务验收 |
| 过早投入 noisy TN/MPO | 放到 v1 前向与分布式 trajectory 稳定之后 |
| 简化模型被误认为真机复现 | identity、校准时间和建模假设进入审计结果 |

## 7. 完成定义

Noisy Simulation v1 完成需要同时满足：

1. 正确性：density、SV trajectory、MPS trajectory 的交叉验证通过；
2. 统计性：置信区间覆盖和 adaptive stopping 经过重复试验校准；
3. 可复现性：seed、trajectory ID、checkpoint 和 world-size 语义稳定；
4. 资源安全：所有 GPU 路径具备显存预检和 fail-closed；
5. 性能：1/2/4/8 GPU 有真实硬件 time-to-target-error 证据；
6. 可观察性：结果包含模型 identity、执行计划、误差分解和运行摘要；
7. Claim boundary：不把单节点证据扩展为多节点或真机真实性声明。
