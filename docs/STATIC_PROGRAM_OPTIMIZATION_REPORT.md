# FlagQuantum 静态程序与后端内核优化报告

更新日期：2026-07-15

## 1. 报告范围

本文记录 `Module` 参数化量子电路训练路径从“每步调用 builder 并重建
Circuit”演进到“静态程序、参数槽位、后端编译计划和融合 CUDA kernel”的过程。

本报告覆盖：

- Statevector（SV）、Matrix Product State（MPS）和 Tensor Network（TN）；
- Circuit、Instruction、CircuitIR 和参数绑定生命周期；
- 后端 execution plan、workspace、contraction stage 和 Triton kernel；
- CPU 回归测试和单卡 A100 工程 microbenchmark；
- 失败实验、正确性问题和当前限制。

本文中的计时是开发过程中的单机工程数据，用于比较实现方向，不是正式 release
benchmark 或多卡 scalability 证据。正式性能声明仍需使用仓库 benchmark contract、
固定环境清单和可审计结果文件。

## 2. 基准程序

可复现入口：

```bash
python examples/benchmark_static_program.py --mode sv
python examples/benchmark_static_program.py --mode mps
python examples/benchmark_static_program.py --mode tn
```

A100 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/benchmark_static_program.py \
  --mode tn \
  --device cuda \
  --qubits 8 \
  --depth 6 \
  --batch-size 16 \
  --steps 10 \
  --repeats 2
```

基准比较两条完整 `Module` 训练路径：

- `static`：builder 捕获一次，后续复用 Circuit、Instruction 和后端程序；
- `rebuild`：每步重新调用 builder，重新创建 Circuit 和 Instruction。

两条路径均包含 forward、backward 和相同 observable。warm-up 不计入稳态计时。

## 3. 初始问题

优化前，参数化量子训练每步包含：

```text
classical preprocessing
  -> Python circuit builder
  -> new Circuit
  -> new Instruction objects
  -> new CircuitIR
  -> backend planning
  -> backend execution
  -> observable
  -> backward
```

这种设计存在以下固定开销：

1. 电路拓扑不变，但 builder 和容器每步重建；
2. 动态 Tensor 被复制进新的 Instruction 参数字典；
3. 后端重复分析 wires、融合分组、labels 和 contraction path；
4. TN 为每个 pair equation 生成独立 specialization；
5. PyTorch Inductor 对 complex operator 支持有限，直接编译 complex einsum 收益低；
6. 小电路中 Python 和运行时管理开销容易掩盖结构缓存收益。

## 4. 阶段一：builder 捕获与静态参数槽位

`Module` 使用 `make_fx` 捕获 builder 中生成动态门参数的经典 Tensor 图。
Circuit 拓扑只捕获一次，后续步骤只执行经典前处理并生成参数槽位 Tensor tuple。

核心结构：

```text
Compiled builder graph
  -> dynamic tensor tuple
  -> task-local parameter bindings
  -> static Circuit
  -> static Instruction views
```

实现特性：

- builder 在输入 shape、dtype、device 不变时只执行一次；
- Tensor 相关控制流会被明确拒绝；
- shape、dtype 或 device 改变时重新捕获；
- 编译产物不进入 `state_dict`；
- `.to()` 后自动失效并重新编译；
- 执行结束清理动态绑定，避免保留上一轮 autograd 图。

运行时指标包括：

- `builder_compiled`；
- `builder_compile_count`；
- `builder_cache_hits`；
- `static_program_reused`。

## 5. 阶段二：后端直接读取参数槽位

静态 Instruction 增加按门 schema 对齐的 `parameter_slots`。三个后端在开始执行时
读取一次 binding tuple，热循环使用：

```python
parameter = bindings[slot_index]
```

Mapping 视图仅用于 IR 兼容和检查，不再参与逐门参数读取。MPS 的 RX、RY、RZ
快路径同样直接使用槽位索引。

这一阶段消除了逐门 `Mapping.__getitem__` 和逐门 ContextVar 查询，但 microbenchmark
显示端到端仍基本持平，说明主要瓶颈已转移到 IR、执行计划和数值 kernel。

## 6. 阶段三：静态 IR 与后端 compiled program

### 6.1 公共缓存

Circuit 缓存：

- `CircuitIR`；
- 后端程序字典；
- 无梯度初态 workspace。

调用 `gate()` 修改拓扑时，IR 和后端程序缓存一起失效。

### 6.2 Statevector

SV 缓存固定指令序列，以及每个 wires layout 的 permutation 和 inverse permutation。
训练步骤只生成动态门矩阵并执行预先确定的 layout。

### 6.3 MPS

`CompiledMPSProgram` 按以下 signature 专门化：

```text
n_wires
batch size
device
dtype
max_bond
cutoff
single-qubit fusion policy
```

操作预分类为：

- `one`；
- `fused_one`；
- `adjacent_two`；
- `remote_two`；
- `dense_fallback`。

### 6.4 TN

`CompiledTNProgram` 分离静态 node template 和动态 Tensor slots，并缓存：

- node labels；
- node metadata；
- output ordering；
- contraction path。

CPU 上创建和绑定大量小 node 的成本可能高于直接 eager 构建，因此 compiled node
binding 只在 CUDA 路径启用，CPU 保留 eager fast path。

## 7. 阶段四：real/imag 共享 contraction 基础设施

直接把 complex einsum 交给 Inductor 时，A100 验收出现：

```text
TorchInductor does not support code generation for complex operators.
Performance may be worse than eager.
```

因此新增共享 real/imag kernel 层，将复数乘法拆为：

```text
real = ar @ br - ai @ bi
imag = ar @ bi + ai @ br
```

该基础设施被 TN pair contraction 和 MPS 单站点 contraction 共用。CPU 使用原生
complex einsum；CUDA 使用 real/imag 路径。

工程过程中发现 `torch.conj()` 产生 unresolved conjugate view，不能直接传入
`view_as_real`。当前入口会先执行 `resolve_conj()` 和 `resolve_neg()`，并通过全部
TN bra-ket、gradient 和 contraction-strategy 回归。

## 8. 阶段五：TN staged/bucketed contraction

原 greedy contraction 是严格逐 pair Python 循环。优化后先把 contraction tree 按
依赖关系划分为 stage，再在每个 stage 内按以下信息分桶：

```text
equation
left shape
right shape
```

同一 bucket 中互不依赖的 pair 被堆叠为额外 batch 维度，一次执行 batched kernel。

策略：

- 首次执行只规划 path，不编译碎片 pair；
- 后续执行使用 staged program；
- bucket 输入在 kernel 前统一 `contiguous()`；
- 多元素 bucket 使用融合路径；
- 单元素 bucket 使用原生 complex einsum，避免无收益的 specialization。

这一步把大量小 kernel 收敛为较少的 stage bucket。

## 9. 阶段六：canonical GEMM lowering

可矩阵化的 pair equation 被统一降低为：

```text
left  -> permute -> contiguous -> [B, M, K]
right -> permute -> contiguous -> [B, K, N]
                           |
                           v
                    batched GEMM
                           |
                           v
                reshape -> output permute
```

支持：

- 无 batch 或多个 batch 维度；
- 多个 contracted 维度；
- 输出维度重新排序；
- staged bucket 增加的额外 batch 维度。

主要 specialization 从具体 equation/rank 收敛为 `(B, M, K, N, dtype, device)`。
在 8-qubit TN 工程基准中，主要 BMM shape 约收敛到 4 类。

## 10. 阶段七：自定义 fused complex Triton BMM

实现文件：

- `flagquantum/simulation/triton_kernels/complex_bmm.py`；
- `flagquantum/simulation/real_imag_kernels.py`。

旧的 `flagquantum/simulation/triton_complex_bmm.py` 保留为模块兼容别名。

forward kernel 在一次加载和一次输出中计算：

```text
real += ar * br - ai * bi
imag += ar * bi + ai * br
```

自定义 `torch.autograd.Function` backward 使用：

```text
dLeft  = grad * conj(right)^T
dRight = conj(left)^T * grad
```

两个梯度也调用 fused complex Triton BMM。适用条件是 CUDA + complex64；CPU、
complex128 和不支持的情况使用 PyTorch fallback。

首次 A100 forward 测试暴露了实虚 slice stride 为 2、kernel 却按连续矩阵寻址的
错误。入口现会分别把 `real` 和 `imag` slice 规范为 contiguous。修复后，三组不同
`(B,M,K,N)` 的 forward、left gradient 和 right gradient 均与 `torch.bmm` 对齐。

## 11. 性能记录

测试设备确认：8 × NVIDIA A100-SXM4-40GB。以下数据只使用 GPU 0。

工作负载：8 qubits、6 layers、batch 16。不同开发阶段的 `steps`、进程缓存和编译
状态可能不同，因此跨阶段数据只用于趋势判断，不应直接作为正式加速比声明。

| 阶段 | 后端 | 稳态时间 | 备注 |
|---|---|---:|---|
| 初始静态 Instruction | SV | 约 0.836 s | 20 steps × 3 repeats |
| 初始静态 Instruction | MPS | 约 1.091 s | 20 steps × 3 repeats |
| 初始静态 Instruction | TN | 约 1.526 s | 当时 static 慢于 rebuild |
| 后端 program cache | SV | 约 0.925 s | 约 1.047× vs rebuild |
| 后端 program cache | MPS | 约 1.206 s | 约 1.014× vs rebuild |
| 后端 program cache | TN | 约 1.487 s | 约 1.043× vs rebuild |
| A100 complex compile | TN | 约 1.091 s | complex Inductor 收益低 |
| A100 staged real/imag | TN | 约 0.537 s | 10 steps × 2 repeats |
| A100 canonical GEMM | TN | 约 0.547 s | 10 steps × 2 repeats |
| A100 fused Triton | TN | 约 0.497 s | 10 steps × 2 repeats |
| A100 fused Triton | MPS | 约 0.867 s | 10 steps × 2 repeats |

最后一组同配置 static/rebuild 数据：

```text
TN static  : 0.496710 s
TN rebuild : 0.500745 s

MPS static  : 0.866846 s
MPS rebuild : 0.871741 s
```

结构缓存本身在小电路中只贡献约 1% 左右；更显著的绝对时间改善来自 staged
contraction、canonical lowering 和 fused complex kernel。

## 12. 正确性与测试证据

本轮相关验证包括：

- builder 只捕获一次；
- Circuit 和 Instruction 对象身份跨步骤复用；
- SV、MPS、TN 动态参数和输入梯度；
- TN greedy、memory-greedy、beam、optimal 和 sliced contraction；
- unresolved conjugate bra 网络；
- canonical GEMM batch、multi-contract 和 output permutation；
- fused Triton forward、left backward 和 right backward；
- MPS specialization 随 max_bond/cutoff 改变；
- 编译产物不进入 checkpoint state_dict；
- Ruff 静态检查。

最近一次 CPU 相关集合：

```text
89 passed, 3 CUDA tests skipped
```

A100 fused Triton 专项：

```text
3 passed
```

## 13. 失败实验与经验

### 13.1 只缓存 Python 容器不足以加速

Circuit/Instruction 复用是必要基础，但在小电路中 Mapping、runtime summary、IR hash
和实际 Tensor 运算会掩盖收益。必须把缓存推进到后端 execution plan 和 kernel。

### 13.2 直接编译 complex einsum 不合适

Inductor 对 complex operator 的 codegen 有限制。real/imag 表示是当前 A100 路径的
正确基础。

### 13.3 每个 pair 一个 torch.compile specialization 会爆炸

不同 equation 和 rank 共用 Python code object 时曾在第 9 个 specialization 触发
Dynamo 默认 `recompile_limit=8`。staging、bucketing 和 canonical GEMM 从结构上解决
该问题，编译失败仍必须安全回退。

### 13.4 workspace 不能破坏 autograd

无梯度初态和静态索引可以跨步骤复用；SVD/QR 输出及参与 backward 的中间 Tensor
不能跨训练步原地复用，否则会破坏 autograd version counter 和计算图生命周期。

### 13.5 CPU 与 CUDA 应采用不同策略

CPU 原生 complex einsum 通常比四次实数 contraction 更快；CUDA 才启用 real/imag
和 Triton。compiled node binding 同样只在能产生收益的设备路径启用。

### 13.6 Naive observable batch 会造成 contraction peak 爆炸

第一版 `CompiledTNObservableProgram` 给所有 observable operator node 增加共享 batch
label，希望一次直接输出多个 Z expectation。CPU 小规模正确性和梯度通过，但在
8-qubit、batch 16 的 A100 验收中，greedy path 过早把 observable 维度带入 bra/ket
中间量，尝试申请 64 GiB，触发 OOM。

当前修复：

- 恢复 Module 在 12 qubits 以内的 dense-state observable fast path；
- direct observable batch 执行前计算 contraction profile；
- peak 超过 `2**24` elements 时回退逐 observable direct contraction；
- 保留 program 和正确性测试，作为后续 environment-prefix 实现的基础。

后续修复进一步把 observable program 的默认 contraction 从 balanced greedy 改为
memory-greedy。A100 上强制 `dense_observable_wires=0` 后，8 qubits、batch 16、8 个
Z observable 可在约 17.7 MiB peak allocated memory 下完成 forward/backward，且不
materialize full state。相同训练 microbenchmark 的 direct 路径约 0.585 s，仍慢于
小规模 dense-state 路径的约 0.497 s，因此最终采用自适应 crossover：12 qubits
以内优先 dense speed path，更大问题使用 memory-first direct path。

结论：observable batch 必须结合 memory-first path 和 peak guard。仅增加 batch
label 会 OOM；修正 path 后可安全执行，但小电路不应为了避免 full state 而牺牲速度。

### 13.7 串联通用 fused BMM 不等于双站点融合

曾尝试把 MPS `apply_two` 中的 site-site contraction 和 gate contraction 分别替换为
两个通用 fused complex BMM。A100 稳态由约 0.867 s 退化到约 1.255 s，原因是这些
矩阵较小，两次 layout 规范化、中间 `theta` 和两次 kernel launch 抵消了计算收益。
该修改已回退。真正的 MPS 双站点优化必须用一个三输入 kernel 直接完成
`left + gate + right -> split matrix`。

随后 fused BMM 改为直接读取 interleaved complex storage：kernel 使用
`2 * offset + {0,1}` 访问实部和虚部，并直接写 `[B,M,N,2]`，删除四份输入实虚
contiguous copy、两个输出分配和 `torch.complex` 组装。forward/backward A100 测试
继续通过；8-qubit MPS 工程数据约为 0.860 s。

后续实现了真正的三输入 forward kernel：

```text
left site + two-qubit gate + right site -> split matrix
```

固定门和 batched gate 的 forward、left/gate/right gradient 均通过 A100 对照。但
8-qubit 训练中自定义 backward 的三路 contraction 仍使总时间约 1.166 s，慢于
eager。当前按 contraction volume 自适应：需要梯度时阈值为 `2**18`，推理阈值为
`2**12`；小 bond 继续走 eager，大 bond 才启用三输入 kernel。该 kernel 是后续融合
backward 的正确性基础，不会强制影响 quick-start 小模型。

### 13.8 Observable program 必须缓存到 Circuit 生命周期

最初 `CompiledTNObservableProgram` 保存在 `TensorNetworkState` 内，但训练每一步都会
创建新的 state，因此只能复用同一步内的调用，不能消除下一步的 observable path
规划和 peak profile。现在 `TensorNetworkContractionPlan` 持有 Circuit 的
`_backend_programs` 缓存引用，并按以下静态签名缓存：

```text
(n_wires, batch size, observable wires, contraction strategy)
```

缓存值包含 observable program、规划得到的 peak size，以及 direct-batch/fallback
决策。后续训练步只把本步动态 Tensor 绑定到 program，不再重复 profile。运行摘要新增
`observable_program_cache_hit` 和 `observable_peak_size`，便于基准和诊断。CPU 回归覆盖
首次 miss、第二次 hit 和数值一致性；TN 测试当前为 24 passed。

## 14. 阶段八：stride-aware fused complex BMM backward

此前 fused complex BMM 的 backward 会执行三类显式 layout copy：

```text
gradient.contiguous()
conj(right.transpose()).contiguous()
conj(left.transpose()).contiguous()
```

现在 Triton kernel 直接接收左右矩阵的 batch/row/reduction/column stride，并通过
compile-time conjugate flag 在寄存器中改变虚部符号。backward 因而可以直接消费转置
view 和非连续上游梯度，无需生成三份 contiguous Tensor。forward 也支持任意合法的
rank-3 strided complex64 输入。

A100 正确性验收覆盖三组连续 shape，以及一组两侧输入和上游梯度均非连续的 shape；
forward、left gradient、right gradient 共 4 项全部通过。CPU 保持原生 `torch.bmm`
fallback，不改变用户 API。

随后 canonical einsum-to-BMM lowering 增加了独立的静态 layout cache。缓存键为
`(equation, left shape, right shape)`，缓存 permutation、矩阵维度、输出 shape 和输出
permutation；动态训练步不再重复解析 labels、构造维度字典和计算排列。不能 lower 的
layout 同样进行负缓存。

lowering 删除了 `permute(...).contiguous().reshape(...)` 中无条件的 contiguous，改为
`permute(...).reshape(...)`：布局可以折叠成 matrix view 时与原 Tensor 共享 storage，
只有确实无法表达为二维 stride 时才复制。TN contraction bucket 中 `torch.stack()` 后
的冗余 contiguous 也已删除，因为 stack 的输出本身已是新分配的连续 Tensor。

A100 集成验收新增两种 canonical equation，覆盖 view-friendly 和必须重排的 layout；
与原生 complex einsum 的 forward、左右 gradient 对照全部通过。相关 CUDA 单元测试
当前为 11 passed。

进一步审计 compiled TN stages 后发现，不能分桶的单 contraction 会传入
`compile_cuda=False`。为判断是否应让该路径无条件使用 fused BMM，新增
`benchmarks/tn_stage_contraction.py`，隔离测量单个 canonical stage 的 forward +
backward。A100-SXM4-40GB、complex64、batch 16 的结果如下（median steady step）：

| M=K=N | fused | native | native/fused | fused/native peak memory |
|---:|---:|---:|---:|---:|
| 8 | 0.444 ms | 0.340 ms | 0.766x | 8.59 / 17.10 MB |
| 32 | 0.443 ms | 0.337 ms | 0.760x | 9.57 / 18.09 MB |
| 128 | 0.370 ms | 0.292 ms | 0.788x | 25.30 / 33.82 MB |

测试配置分别为 10/100/5、10/100/5、5/50/5（warmup/iterations/repeats）。三点均
未出现训练延迟 crossover：fused 节省显存，但慢约 21%–24%。因此“canonical 始终
fused”的实验没有作为最终默认策略保留。

当前采用双目标自适应路由：

- CPU 始终使用原生 complex einsum；
- 小/中型 CUDA 训练和推理 contraction 均使用原生 cuBLAS 路径，优先 steady latency；
- 估算输入加输出 working set 达到 256 MiB 时使用 fused BMM，优先峰值显存并降低
  OOM 风险；
- `compile_cuda=False` 仍只禁止非 canonical 的 `torch.compile` specialization。

三份 JSON 保存在 `benchmarks/results/local/tn_stage_b16_*_a100.json`，属于 local
非发布证据。benchmark 随后扩展为 forward-only；batch 16 下，32³ 为 fused
0.0596 ms、native 0.0488 ms，128³ 为 fused 0.0605 ms、native 0.0484 ms，native
分别快约 18% 和 20%。重复运行中的亚毫秒 training 数字存在方向波动（32³ 曾测得
fused 快约 5%，128³ 基本持平），因此当前不把这些微基准用于宣称稳定 kernel speedup，
只用于拒绝“常规规模无条件 fused”策略。A100 最终路由、kernel、lowering 和 TN 联合
测试为 38 passed。

自适应路由随后进一步拆分为两个阶段：`_canonical_bmm_layout` 只读取或解析静态
layout metadata，`_canonical_bmm_inputs` 仅在最终选择 fused 路径后才执行
`permute().reshape()`。此前即使 working-set policy 最终选择 native einsum，也会先
materialize canonical matrices；对不能 view 化的 layout，这会产生一份真实复制并
立即丢弃。现在常规 native 训练和推理路径完全不构造 BMM 输入临时 Tensor。CUDA
测试通过 monkeypatch 将 materialization 函数改为必然抛错，确认 native 路由不会调用
它；显存压力 fused 路由仍通过 forward/backward 对照。A100 联合测试保持 38 passed。

## 15. 阶段九：编译期 contraction stage bucketization

此前 `_CONTRACTION_STAGE_CACHE` 只保存按依赖 level 划分的 operation tuple；每次训练
执行仍会创建 Python `dict`，重新按 `(equation, left shape, right shape)` 分桶，并再次
生成 batched equation。现在缓存值升级为固定的 `CompiledTNStagePlan`：

```text
CompiledTNStagePlan
  -> CompiledTNContractionStage[]
       -> CompiledTNContractionBucket[]
            equation
            batched_equation
            operations[]
```

分桶和 batched equation 生成只发生一次。steady step 直接遍历静态 Stage/Bucket plan，
只执行动态 Tensor lookup、必要的 stack 和 contraction，不再构造 bucket mapping。新增
单元测试验证同 level、同 shape 的两个 operation 在编译期合并为一个 bucket，最终输出
labels 也保存在 plan 中。CPU TN 专项为 25 passed；A100 kernel/lowering/TN 联合测试为
39 passed。一次将 TN 测试放在 CUDA kernel 测试之前的联合进程出现后续极小分配 OOM，
进程退出后设备占用恢复到 1–4 MiB，调整测试顺序后全部通过；因此不将该资源峰值记作
stage-plan correctness 失败，但后续完整 benchmark 仍应独立记录 peak memory。

## 16. 当前限制

- fused Triton BMM 当前只优化 complex64；
- layout-aware fused BMM 已完成 A100 编译、forward、左右 gradient 和 TN 集成正确性
  验收；尚未完成受控峰值显存和 steady latency 矩阵，因此暂不作为已验证加速结论；
- block size 使用简单规则，尚未建立受控 autotune policy；
- fused BMM 输出仍需分配新 Tensor；无法折叠为 strided matrix view 的高维 canonical
  layout 仍会由 `reshape` 生成输入副本；
- 小规模 TN observable 会按策略主动 materialize 完整 state；大规模路径使用 direct
  observable program 或逐 observable fallback；
- observable batch 只有在 peak-size guard 允许时启用，尚未实现真正共享的
  left/right environment prefix；
- 通用 MPS 双站点 gate contraction 尚未全部使用 fused kernel；
- MPS 的 SVD/QR 和动态 bond rank 仍是主要瓶颈；
- 当前报告没有多卡性能或 scalability 结论；
- 开发阶段 timing 尚未写成 benchmark JSON artifact。

## 17. 下一阶段计划

### P1：融合 layout 与 BMM

让 Triton kernel 接收高维静态 stride/permutation 描述，在 kernel 内完成 flatten
索引到原 Tensor offset 的映射，消除剩余无法 view 化的 reshape copy。

状态：已在阶段十和阶段十一完成 forward/backward 直接寻址，并通过 A100 编译与数值
验收。受控 local benchmark 见第 21 节。

### P1：受控 Triton autotune

只对高频和足够大的 `(B,M,K,N)` shape autotune；小矩阵使用固定配置，并缓存最佳
tile、warps 和 stages。

### P1：MPS 双站点 fused kernel

融合：

```text
left site + two-qubit gate + right site -> SVD/QR matrix
```

避免生成中间六维 Tensor。

### P2：MPS 分解与 workspace

- 无截断时使用 QR fast path；
- 需要截断时使用 batched/truncated SVD；
- 按 bond shape 分桶；
- 建立有界、autograd-safe workspace policy。

### P2：正式 A100 验收矩阵

至少覆盖：

| qubits | depth | batch |
|---:|---:|---:|
| 8 | 6 | 16 |
| 12 | 8 | 16 |
| 16 | 8 | 32 |
| 20 | 12 | 16 |

每组记录 cold compile、steady forward/backward、peak memory、kernel 数、Dynamo graph
数、是否 materialize full state，以及结果正确性。

## 18. 维护规则

后续每次优化应在本文追加：

1. 优化假设和目标瓶颈；
2. 修改的执行路径与失效条件；
3. correctness/gradient 测试；
4. CPU fallback 行为；
5. A100 cold 与 steady 数据；
6. 失败实验和回退原因；
7. 是否改变用户 API、checkpoint 或数值语义。

不得只记录最快数字。所有结果必须同时记录工作负载、设备、warm-up、steps、
repeats 和证据语义。

## 19. 阶段十：高维 stride/permutation 直接寻址

显存压力路由此前仍会执行：

```text
high-rank tensor -> permute -> reshape/copy -> rank-3 fused BMM
```

现在新增 layout-aware fused complex BMM。kernel 接收静态 batch/free/contracted shape、
原 Tensor stride 和 canonical permutation，在 kernel 内把扁平 `(B,M,K,N)` 索引反解为
原 Tensor storage offset。forward 因而直接读取高维输入，不再为无法 view 化的
permutation 创建 rank-3 输入副本。常规小/中 contraction 的 native einsum 路由不变，
CPU fallback 仍使用 PyTorch BMM。

自定义 autograd backward 继续复用已验证的 stride-aware rank-3 fused BMM；当前 backward
会按需构造 canonical matrix，下一步再把相同的高维寻址扩展到两路梯度 kernel。新增测试
明确将 `_canonical_bmm_inputs` 替换为抛错函数，验证显存压力 forward 路由不再依赖该
materialization helper，并对必须重排的 equation 检查 forward 和左右梯度。

初始 CPU-only 环境验证：

```text
real_imag + Triton unit: 5 passed, 10 CUDA skipped
Ruff: passed
```

随后在 NVIDIA A100-SXM4-40GB 上完成真实 Triton 编译和数值验收。首次编译发现当前
Triton JIT 不支持 constexpr tuple slicing、动态 tuple indexing 和 tuple iteration；
layout metadata 最终编码为静态 axis tuple，并将 flatten 寻址显式展开到每组最多 8 个
轴，超过限制时 fail-fast。修复后 fused BMM、canonical lowering 和 layout route CUDA
专项为 15 passed，TN 集成测试为 25 passed。峰值显存和 steady latency 仍需受控基准，
相关受控基准随后在第 21 节补齐。

## 20. 阶段十一：高维 stride-aware backward

layout-aware fused BMM 的 autograd backward 现在也直接读取原始高维 Tensor。两路梯度
通过同一个 layout kernel 计算：`dLeft` 将 right 的 free/contracted 维交换并在寄存器
中共轭，`dRight` 对 left 做对应交换和共轭。静态 permutation 只作为 kernel metadata，
不再执行：

```text
left.permute(...).reshape(B, M, K)
right.permute(...).reshape(B, K, N)
```

因此必须重排的 canonical contraction 在 forward 和 backward 均不再产生输入矩阵副本。
输出梯度仍以连续 canonical matrix 写出，再 reshape/permute 为输入布局；CPU 和非
complex64 路径继续使用原生 PyTorch BMM，用户 API 与数值语义不变。

CUDA 回归测试新增 fail-closed 路由断言：将旧的 rank-3 `_launch` 替换为必然抛错函数，
并对必须重排的 equation 执行 forward、左右 gradient 与 complex einsum 对照。A100
真实编译和数值验收已经通过；峰值显存和 steady latency 尚未形成受控 JSON artifact，
随后已在第 21 节形成受控 local JSON artifact。

## 21. 阶段十二：高维 layout 受控 A100 基准

新增 `benchmarks/tn_layout_contraction.py`，隔离比较两条相同 fused complex BMM 路径：

- `direct_layout`：forward/backward 均从原始高维 stride 直接寻址；
- `materialized_layout`：先 `permute().reshape()` 生成 canonical matrix，再调用 rank-3
  fused BMM。

设备为 NVIDIA A100-SXM4-40GB，complex64，batch 16；两侧 physical layout 均刻意排列为
无法直接 view 化的顺序。结果属于 local 非发布工程证据：

| B,M,K,N | warmup/iterations/repeats | direct | materialized | materialized/direct | direct/materialized peak |
|---:|---:|---:|---:|---:|---:|
| 16,32,32,32 | 10/50/5 | 0.466 ms | 0.398 ms | 0.853x | 9.83/10.09 MB |
| 16,128,128,128 | 5/20/5 | 0.483 ms | 0.512 ms | 1.061x | 29.49/33.69 MB |

最新复测中 128³ 点的 cold compile 为约 3.16 s，最大绝对误差为 `2.36e-05`。直接寻址在
128³ 点快约 6.1%，但在 32³ 点慢约 17.2%；32³ 早先一次运行曾显示 direct 快约 9%，
说明亚毫秒小点存在明显方向波动，不能作为稳定 latency speedup 证据。峰值临时分配增量
在两点均降低 40%：32³ 为 0.375/0.625 MiB，128³ 为 6/10 MiB。128³ 的总 peak
allocated 低约 12.5%。JSON 位于：

```text
benchmarks/results/local/tn_layout_b16_m32_k32_n32_a100.json
benchmarks/results/local/tn_layout_b16_m128_k128_n128_a100.json
```

这组证据验证“已选择 fused 路径后，直接寻址稳定减少物化显存，并在较大测试点改善
延迟”，不证明小矩阵 latency 收益，也不证明 fused kernel 在常规规模快于原生 cuBLAS。
因此不改变当前 256 MiB memory-first 路由阈值；layout-copy P1 子项可以关闭，受控
autotune 和更完整验收矩阵仍保持开放。

## 22. 阶段十三：Triton kernel 包边界

随着 complex BMM、layout-aware contraction 和 MPS 双站点 kernel 增加，底层 Triton
实现从 `simulation/` 平铺文件整理到私有包：

```text
flagquantum/simulation/triton_kernels/
  __init__.py
  complex_bmm.py
  mps_two_site.py
```

`simulation/real_imag_kernels.py`、TN/MPS backend 和 benchmark 负责路由、fallback 与执行
策略；`triton_kernels/` 只持有 CUDA kernel、autograd wrapper 和 launch 约束。旧模块
`triton_complex_bmm.py` 与 `mps_two_site_triton.py` 使用模块对象别名继续兼容，因而旧导入
以及针对私有 `_launch` 的 monkeypatch 语义均保持不变。用户 API、checkpoint 和数值语义
没有变化。

验证结果：CPU 专项 5 passed、12 CUDA skipped；A100 kernel 专项 17 passed；A100 MPS/TN
backend 集成 59 passed；Ruff 通过。

## 23. 失败实验：按 shape 的 Triton autotune

尝试只对 contraction volume 达到 `2**24` 的 complex BMM 启用受控 autotune，小矩阵
继续使用固定配置。候选集限制为四组 32/64 的 M/N tile、固定 K=32，并按 shape 缓存。

A100、complex64、`(B,M,K,N)=(16,128,128,128)` 的 layout forward+backward 结果：

```text
固定配置 cold    : 约 1.24 s
autotune cold    : 约 7.05 s
固定配置 steady  : 约 0.439 ms
autotune steady  : 约 0.512 ms
```

autotune 会分别覆盖 forward 和两路 backward 的相关 shape，候选编译与测量显著放大 cold
成本；亚毫秒 kernel 的选择也容易受测量噪声影响。本实验因此回退，默认路径继续使用固定
tile。原始 local 非发布结果保存在
`benchmarks/results/local/tn_layout_b16_m128_k128_n128_a100_autotuned.json`。

后续若重启 autotune，应离线生成 architecture/shape 配置表，而不是在用户首次训练时在线
搜索；验收还必须同时限制新增 specialization 数和总 cold compile budget。

回退后同时删除 layout kernel signature 中已被 packed axis metadata 取代的四个空 shape
constexpr 参数，避免无意义的 specialization metadata。A100 CUDA 专项 15 passed；固定
配置 128³ 复测 direct/materialized 为 0.483/0.512 ms，确认默认路径恢复且显存优势保持。
cold compile 在不同独立进程间约 1.24–3.16 s 波动，因此不把本次 signature 精简声明为
确定的 cold compile 加速。

## 24. 阶段十四：Triton inference crossover

为了寻找 Triton 相对原生 PyTorch/cuBLAS 的真实加速区间，layout benchmark 增加同一
高维 equation 的 native complex einsum，以及 direct/native 的 forward-only 和
forward+backward 对照。设备为 A100-SXM4-40GB、complex64、batch 16，physical layout
会迫使 canonical BMM 输入发生真实 reshape copy。

方阵 `(M,K,N)=128³` 到 `1024³` 的训练路径没有速度 crossover，native 仍明显更快；
Triton 临时显存增量持续低约 40%。非对称 TN contraction 出现了明确的 forward-only
crossover。固定 `M=N=64` 扫描 contracted dimension：

| B,M,K,N | Triton forward | native forward | native/Triton | Triton/native 临时显存增量 |
|---:|---:|---:|---:|---:|
| 16,64,128,64 | 0.078 ms | 0.083 ms | 1.06x | 0.5/2.5 MiB |
| 16,64,256,64 | 0.078 ms | 0.085 ms | 1.09x | 0.5/4.5 MiB |
| 16,64,512,64 | 0.118 ms | 0.154 ms | 1.30x | 0.5/8.5 MiB |
| 16,64,1024,64 | 0.228 ms | 0.295 ms | 1.29x | 0.5/16.5 MiB |
| 16,64,2048,64 | 0.318 ms | 0.460 ms | 1.45x | 0.5/32.5 MiB |
| 16,64,4096,64 | 0.679 ms | 0.904 ms | 1.33x | 0.5/64.5 MiB |
| 16,64,8192,64 | 1.497 ms | 1.802 ms | 1.20x | 0.5/128.5 MiB |

K=128/256 的收益接近微基准噪声；K=512 起连续多个点保持 1.20x–1.45x。因此新增保守
inference 路由，必须同时满足：

```text
no input requires grad
canonical reshape cannot be represented as a view
B * M * K * N >= 2**25
```

原有 256 MiB memory-first 路由继续适用于训练和推理；新规则只提前覆盖已经实测有速度
收益的 inference layout。A100 fused/lowering 专项为 17 passed。训练 backward 在这些
非对称 shape 上仍慢于 native，因此不扩大到 requires-grad 路径。

这些规模描述的是 TN contraction stage，而不是可直接换算的电路 qubit 数；实际 qubit
crossover 取决于 contraction path、bond/index dimension、batch 和 observable 网络。
JSON 保存在 `benchmarks/results/local/tn_layout_native_*_a100.json`，均属于 local 非发布
工程证据。

## 25. 阶段十五：persistent 单比特门循环

新增 `triton_kernels/single_qubit_loop.py`，验证电路中可融合 loop 的另一条 Triton 路径。
每层执行非对易的 RX+RZ，避免把同轴旋转预先求和形成不公平的算法简化；每个 Triton
program 将 amplitude pair 保留在寄存器中跨 depth 循环，整个 loop 只读写全局状态一次。

公平对照包括向量化 PyTorch eager、`torch.compile(fullgraph=True)` 和 JAX
`jit(lax.scan)`。A100、complex64、batch 16、65,536 amplitude pairs：

| depth | Triton | PyTorch eager | torch.compile | JAX scan | eager/Triton | JAX/Triton |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.058 ms | 0.915 ms | 1.339 ms | 0.397 ms | 15.9x | 6.9x |
| 8 | 0.106 ms | 1.909 ms | 2.612 ms | 0.599 ms | 17.9x | 5.6x |
| 16 | 0.204 ms | 3.701 ms | 4.915 ms | 1.021 ms | 18.1x | 5.0x |
| 32 | 0.400 ms | 7.646 ms | 9.714 ms | 1.557 ms | 19.1x | 3.9x |
| 64 | 0.789 ms | 14.972 ms | 19.686 ms | 3.038 ms | 19.0x | 3.9x |

depth 16 的 pair-count 扫描同样保持优势：4,096、16,384、65,536、262,144 pairs 下，
Triton 相对 eager 分别约为 87.5x、61.7x、18.1x、17.3x；相对 JAX scan 分别约为
9.4x、8.7x、5.0x、4.5x。小 pair 点的极高倍数主要来自 eager/JAX 固定调度成本，不能
外推为吞吐加速比。

depth 16、65,536 pairs 的最大幅度误差为 `1.77e-06`，相对范数误差为 `1.19e-07`；
depth 64 最大幅度误差为 `3.82e-06`。Inductor 明确报告 complex operator codegen 不受
支持，因此 `torch.compile` 在该对照中慢于 eager。

这证明 Triton 的主要速度优势区域不是通用 BMM，而是 IR 能识别并安全融合的局部门循环。
当前原型尚未接入 Circuit IR。任意跨 wire且需要全局同步的门序列不能放入同一个 kernel。
local JSON 位于
`benchmarks/results/local/single_qubit_loop_*_a100.json`。

随后实现自定义 fused backward。由于每层 RX/RZ 都是 unitary，backward 从最终输出开始
逆序应用 `RZ†` 和 `RX†`，同时传播 complex state gradient，并用解析导数计算每层 RX/RZ
参数梯度。该设计不保存 `depth` 份 forward activation，只保存最终输出和角度：

```text
saved memory: O(state + parameters)
reverse compute: O(depth * state)
parameter reduction: block reduction + atomic add per batch/layer
```

CPU fallback 的 state/RX/RZ 梯度对照已通过，Ruff 通过。CUDA backward correctness 和
训练 crossover 尚未完成：验证时外部 `pt_elastic/FlagScale` 作业占满 8 张 A100，每卡
约 40.3 GiB，连几十 KiB 的测试分配也 OOM。资源释放前不得把 forward speedup表述为训练
speedup；CUDA backward 仍需覆盖 depth 1/4/16/64、非整 block pair count、state gradient、
两类参数梯度、峰值显存和 steady forward+backward。

## 26. 阶段十六：persistent 单比特循环 backward 验收

GPU 资源释放后，在独立的 8 × NVIDIA A800-SXM4-80GB 节点上完成自定义 backward 的
真实 Triton 编译、正确性和训练 microbenchmark。新数据不能与第 25 节的 A100 forward
数据混算；本节只使用 GPU 0，证据语义为 `single_device_fast_path`、local 非发布工程数据。

CUDA correctness 覆盖 depth 1/4/16/64、513 个非整 block amplitude pairs，以及 state、
RX 参数、RZ 参数三类梯度，结果为 6 passed。受控训练 benchmark 配置为 complex64、batch
16、65,536 pairs、10 warmup、50 iterations、5 repeats，完整 step 包含 forward 和
backward：

| depth | Triton step | eager step | eager/Triton | Triton/eager peak increment |
|---:|---:|---:|---:|---:|
| 1 | 0.383 ms | 1.550 ms | 4.05x | 32/80 MiB |
| 4 | 0.313 ms | 5.413 ms | 17.27x | 32/176 MiB |
| 16 | 0.566 ms | 22.598 ms | 39.90x | 32/560 MiB |
| 64 | 1.758 ms | 63.432 ms | 36.09x | 32/2096 MiB |

最大 state gradient 绝对误差从 depth 1 的 `4.81e-07` 增至 depth 64 的 `4.34e-06`；
RX/RZ 参数梯度最大绝对误差均不超过 `6.11e-04`。参数梯度需要跨 65,536 pairs 做
float32 atomic reduction，因此绝对误差随 depth 和 reduction size 增长；专项测试仍在
`atol=rtol=2e-4` 的较小 513-pair correctness workload 上逐 depth 对照。

Triton cold step 在四个独立进程中约 0.48--0.50 s。若干点的第一个 steady repeat 仍有
明显额外延迟，尤其 depth 1/4/16；表格按预先约定报告 5 repeats median，并保留全部原始
samples，不能把最快 repeat 当成结果。JSON 位于：

```text
benchmarks/results/local/single_qubit_loop_backward_b16_p65536_d1_a800.json
benchmarks/results/local/single_qubit_loop_backward_b16_p65536_d4_a800.json
benchmarks/results/local/single_qubit_loop_backward_b16_p65536_d16_a800.json
benchmarks/results/local/single_qubit_loop_backward_b16_p65536_d64_a800.json
```

这组结果验证 persistent RX/RZ loop 在命中特定融合模式时具有训练 latency 和有界反向
activation memory 优势。它仍是隔离 kernel microbenchmark，尚未接入 Circuit IR，也不
代表任意单比特门序列、完整 Module 训练或多卡 scalability 加速。

## 27. 阶段十七：Circuit IR 融合与端到端 VQE

persistent RX/RZ loop 已接入本地 statevector compiled program，而不是由 benchmark 直接
调用 kernel。IR pass 仅识别由 RX/RZ 构成的连续区段；不同 wire 的操作可按量子门交换律
分组，但每条 wire 上必须至少包含两个严格 RX→RZ pair。自定义 matrix、其他 opcode、
顺序不匹配、CPU、complex128 或显式设置 `FQ_TRITON_SINGLE_QUBIT_LOOP=0` 时保持原 eager
gate 路径。runtime summary 报告开关状态、融合 region 数和融合 gate 数。

新增 `benchmarks/vqe_triton_runtime.py`，通过以下完整产品路径比较两种执行策略：

```text
fq.Module -> static builder -> FlagQuantum IR -> statevector
                 -> Hamiltonian expectation -> backward -> torch.optim.Adam
```

A800-SXM4-80GB GPU 0 的受控任务使用 16 qubits、16 层本地 RX/RZ、末端 CX chain、512 个
训练参数和 31 项 Hamiltonian `0.7 sum(ZZ) - 0.25 sum(X) + 0.05 Z0`。两条路径使用独立但
完全相同的初始参数；首轮实验曾因两个 Module 共享初始化 Tensor storage 而被判定为无效，
修复为显式 clone 并加入参数一致性断言后重新测量。

50 次 Adam iteration 的累计 wall-clock 为：

```text
FlagQuantum IR + Triton fusion : 6.607 s
FlagQuantum PyTorch eager      : 20.023 s
eager / Triton   : 3.031x
```

两条 energy trajectory 逐 iteration 对齐，最终能量绝对差 `8.58e-06`。runtime metadata
确认 Triton 路径命中 16 个 region、融合 512 个 gates，eager 路径命中数为零。计时包含
参数绑定、statevector、Hamiltonian、backward、optimizer step
和每步同步，不包含各自 warm-up。JSON 和累计运行时间 SVG 位于：

```text
benchmarks/results/local/vqe_triton_ir_16q_d16_i50_a800.json
benchmarks/results/local/vqe_triton_ir_16q_d16_i50_a800.svg
```

这是 FlagQuantum 真实 VQE product path 的 local single-GPU 工程证据，但 ansatz 特意包含
能被当前 pass 合法融合的深 RX/RZ commuting region。它证明该类 VQE 的端到端收益，不证明
任意 hardware-efficient ansatz、被 entangler 频繁打断的浅 rotation layer、其他 backend
或多卡 VQE 都有相同加速。

## 28. 阶段十八：20-qubit 500-step VQE 与容量扫描

端到端 VQE 主任务扩大到 20 qubits、32 层本地 RX/RZ、1,280 个参数和 39 项
Hamiltonian，并执行 500 次 Adam iteration。电路包含 20 个 H、1,280 个 RX/RZ 和 19 个
CX，共 1,319 gates；按可并行 H/RX/RZ 和顺序 CX chain 计算，逻辑深度约 84。

A800 的 warm steady 累计结果为：

```text
FlagQuantum IR + Triton fusion : 158.926 s (GPU 0)
FlagQuantum PyTorch eager      : 552.005 s (GPU 0)
FlagQuantum JAX jit            : 426.014 s (GPU 7)
eager / Triton                 : 3.473x
JAX / Triton                   : 2.681x
```

上述三条累计曲线均在各后端 warm-up 完成后开始计时，**不包含** Triton JIT、JAX/XLA
compile 或 eager 首次执行成本。本轮没有单独记录 cold-start 时间，不能事后从 steady-state
曲线反推；JSON 将该项明确标为未测量。benchmark 后续运行会另存一次首次
forward+backward 的 startup 时间，但是否命中持久编译缓存也必须随结果说明。

JAX 使用 `jax.jit(jax.value_and_grad(...))`、complex64 和 highest matmul precision；在独立
进程中固定到同型号 A800 GPU 7，并设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。三条路径使用
相同初始参数、Hamiltonian、Adam、warm-up、逐步同步和完整 product-path 计时边界。JAX 与
Triton 的最终 energy 绝对差为 `4.86e-05`。JAX 同进程 GPU 0 复测为 `421.173 s`，与独立
GPU 7 结果相差约 1.15%；最终图采用按要求独立测得的 GPU 7 trajectory。

这证明当前 IR-aware Triton fusion 对比当前通用 FlagQuantum JAX product path 仍有约
`2.68x` 端到端收益。但 JAX 路径尚未实现与 Triton 等价的 RX/RZ region lowering：它仍按
通用 statevector gate path 执行，而 Triton 将 1,280 个 RX/RZ 融合为 20 个 persistent
regions。因此该结果证明“当前专用融合有必要”，不证明 Triton 语言本身不可由具有同等 IR
融合的 JAX/XLA lowering 替代。局部 `JAX scan` forward microbenchmark 也不能代替这项
端到端训练测量。JAX 显存不能由 `torch.cuda.max_memory_allocated` 完整观测，所以 memory 图
不加入误导性的 JAX allocator 数值。

两条路径最终 energy 绝对差为 `1.53e-05`。runtime metadata 确认 Triton 路径命中 20 个
IR regions、融合 1,280 gates；eager 命中零个。逐 iteration 的 CUDA peak allocated
memory 保持平台，Triton 为 `0.539 GiB`，eager 为 `10.384 GiB`，没有随训练步数增长。
累计 speedup 随 iteration 收敛到稳态单步比值，而不是无界增长。

主任务 JSON 和三张横轴为 Number of Iterations 的图位于：

```text
benchmarks/results/local/vqe_triton_ir_20q_d32_i500_a800.json
benchmarks/results/local/vqe_triton_ir_20q_d32_i500_runtime_a800.svg
benchmarks/results/local/vqe_triton_ir_20q_d32_i500_speedup_a800.svg
benchmarks/results/local/vqe_triton_ir_20q_d32_i500_memory_a800.svg
benchmarks/results/local/vqe_triton_ir_20q_d32_i500_accuracy_a800.svg
```

精度图直接使用500步实际训练的两条 energy trajectory，逐步绘制
`abs(E_Triton - E_eager)`，不做平滑或抽样。图中标注当前数值策略为 complex64
statevector / float32 parameter reduction；final、mean、max deviation 分别为
`1.53e-05`、`2.14e-05`、`8.01e-05`，最大值出现在第119步。JAX/Triton trajectory 的
final、mean、max deviation 分别为 `4.86e-05`、`2.39e-05`、`9.92e-05`，最大值出现在
第197步；当前 accuracy SVG 仍专门展示 Triton/eager deviation。

另用独立进程执行 20/22/24/26-qubit、32 层、单个完整 VQE training step 容量 probe。
每个进程从空 allocator 开始；step time 包含首次 builder/kernel compile，只用于容量诊断，
不与主任务 steady timing 比较：

| qubits | Triton status / peak | eager status / peak |
|---:|---:|---:|
| 20 | success / 0.524 GiB | success / 10.369 GiB |
| 22 | success / 2.235 GiB | success / 45.549 GiB |
| 24 | success / 9.641 GiB | OOM / 78.634 GiB |
| 26 | success / 41.516 GiB | OOM / 78.508 GiB |

24q eager 在申请额外 128 MiB 时真实 OOM；26q eager 在申请额外 512 MiB 时真实 OOM。
两者均保留原始 `torch.cuda.OutOfMemoryError` 和峰值记录，没有生成伪造 runtime 或 speedup。
26q Triton 成功只证明单步可执行容量，不代表适合运行 500 steps。容量图横轴必须是 qubit
数而非 iteration，因为它表达的是问题规模扫描：

```text
benchmarks/results/local/vqe_capacity_20_22_24_26q_d32_a800.svg
benchmarks/results/local/vqe_capacity_*q_d32_*_a800.json
```

## 29. 阶段十九：batch=1 MPS VQE 与空间 bond bucket 负结果

MPS VQE 正式评测固定 circuit batch 为 1；扩大 batch 只代表并行运行多个独立 VQE，不能
作为单个大规模量子任务的加速证据。扫描轴改为 qubits、max bond、circuit depth 和
iterations，并记录实际最大 bond、累计 discarded weight、peak allocated memory 及逐步
energy deviation。

复数 truncated SVD 在重复奇异值处存在不唯一的奇异向量相位，原生 backward 会抛出
`svd_backward` 异常。训练截断路径现在保留 forward 最优 SVD 子空间，但对该子空间执行
stop-gradient，通过投影右因子回传稳定近似梯度。runtime 明确标记
`svd_gradient_method=projected_stop_subspace`；该方法不是精确 SVD 梯度，结果必须同时报告
截断误差和相对参考 trajectory 的偏差。16q、8 层、bond 64 的短测已连续完成三步，未再
触发异常，累计 discarded weight 约为 `1.9e-11` 至 `8.6e-11`。

通用 Circuit IR 原型进一步识别连续、空间不相交的相邻两比特门，按相同左右 bond shape
打包，将空间 bond 作为额外批维度，并直接把融合 contraction 输出交给 batched split。
64q、8 层、bond 64、batch 1 的 A800 结果为：

```text
spatial-bucket Triton steady : 5.28--5.51 s/step
spatial-bucket eager steady  : 5.28--5.54 s/step
original per-gate path       : about 1.94--1.99 s/step
```

空间 bucket 把 Triton 覆盖从每步 25 个区域提高到 78 个，但 `stack -> batched SVD ->
unstack` 成为更大的瓶颈，整体退化约 2.7 倍，显存也没有收益。因此该原型仅由
`FQ_MPS_SPATIAL_BUCKET=1` 显式开启，默认关闭；当前 crossover policy 对该 shape 的决策
是禁止 bucket。JSON 位于
`benchmarks/results/local/mps_vqe_64q_l8_bond64_spatial_bucket_a800.json`。

Z/最近邻 ZZ Hamiltonian 已通过 `MPSState.expectation_z_zz_chain` 在一次 environment sweep
中共同计算，不再逐 Pauli term 重建完整环境；仓库另有 padded real-channel scan 原型。
因此后续工作是测量并选择这两种 shared-environment 实现的 crossover，而不是宣称尚未
存在的 observable 优化。

截至本阶段，诚实结论是：MPS 提供远超 dense statevector 的 qubit 容量路线；当前 Triton
two-site contraction 和空间 bucket 均没有证明 batch=1 VQE 的端到端加速。

## 30. 阶段二十：fixed-rank range-QR 原型

为避免 `two-site contraction -> full matrix -> generic SVD`，实现了确定性 complex Gaussian
range projection 和 reduced QR reference，并增加 Triton 原型直接计算 `Y=MΩ`。kernel
forward 与显式 materialize `M` 的结果在 A800 上通过 complex64 对照，但第一版单 kernel
因重复遍历 right index，在 bond 16--32 上比 materialize+GEMM 慢 1.7--14.8 倍，因此没有
接入执行路径。

随后采用因式分解 contraction 顺序和窄 QR，完整 forward factorization microbenchmark
显示替换通用 SVD 的潜力：

| active dim | rank | range-QR | SVD | SVD / range-QR |
|---:|---:|---:|---:|---:|
| 16 | 16 | 0.965 ms | 1.003 ms | 1.04x |
| 32 | 16 | 0.903 ms | 3.396 ms | 3.76x |
| 32 | 32 | 0.993 ms | 2.858 ms | 2.88x |
| 64 | 32 | 1.010 ms | 6.969 ms | 6.90x |
| 64 | 64 | 1.194 ms | 6.977 ms | 5.84x |

该 microbenchmark 只测 forward factorization。`FQ_MPS_FIXED_RANK_QR=1` 实验路由接入
完整 batch=1 VQE 后，32q、12 层、bond 32 的稳态约 0.90 s/step，原 SVD 路径约
1.10 s/step；但初始 energy 从约 `-24.41` 偏到 `-10.63`，后续 trajectory 同样明显
失真。由于不 materialize `M`，快速路径也无法免费计算精确 discarded Frobenius weight，
runtime 将其标为未测量 `NaN`，而不是伪造为零。

结论：固定 rank QR 的计算方向有性能潜力，但单次随机 range finder 的精度不可接受；该
路由默认关闭，不构成 MPS 加速结论。下一步必须验证 power iteration、复用前一步 bond
subspace 或小型 core SVD 是否能在精度恢复后仍保留端到端收益。

## 31. 阶段二十一：Adaptive Dense-Island MPS 启动

后续 MPS/TN 算法创新转向 dense-island 混合表示：局部连续 qubit 使用已验证的 dense
Triton statevector region，island 边界使用受控 bond，并将压缩从每个两比特门后延迟到
跨边界 region 或 layer interval。完整语义、梯度合同、验收矩阵和失败退出条件见
`docs/ADAPTIVE_DENSE_ISLAND_MPS_PLAN.md`。该路径处于 Phase 0，不改变默认 backend，也
没有尚未测得的性能结论。

Phase 1 已实现相邻 island 的 immediate merge、跨边界 gate、按原边界 split 和真实
discarded-weight 记录。无截断时使用 identity-factor exact-autograd split，forward 和参数
gradient 均与完整 statevector 对齐；需要截断时保留 forward SVD 子空间，并使用明确标记
为 `retained_subspace_adjoint` 的稳定近似梯度。当前仍是每个跨边界 region 立即 split，
延迟多个 region/layer 的压缩和性能 benchmark 尚未完成。

新增原生 block-environment Z/最近邻 ZZ Hamiltonian。每个 dense island 内的全部 Z 与内部
ZZ 项合并为一个对角插入 contraction；island 边界 ZZ 共享预计算的 left/right
environment，全程不重建完整 statevector。6-qubit complex64 forward 与参数 gradient 均
通过完整 statevector 对照；16-qubit 诊断中，与同一 Dense-Island 状态重建后的能量差为
`5.72e-06`。

A800 GPU 0、batch 1、16q、8层、island width 4 的首轮完整
forward + shared-environment energy + backward + Adam 结果：

| max bond | Dense-Island steady | gatewise MPS steady | MPS / island | peak GiB | final energy delta |
|---:|---:|---:|---:|---:|---:|
| 16 | 约 0.23--0.27 s | 约 0.29--0.31 s | 1.225x | 0.025 / 0.017 | 0.2013 |
| 32 | 约 0.22--0.31 s | 约 0.31--0.33 s | 1.348x | 0.028 / 0.018 | 0.0150 |

Dense-Island 每步只在12个跨 island gate处 merge/split，逐门MPS执行60个two-site更新，
因此出现真实稳态性能信号；但两组均未达到`1e-4` energy误差门槛，不能作为成功加速
结论。block Hamiltonian 已排除为主要误差来源，当前阻塞是截断后的
`retained_subspace_adjoint` 训练轨迹。JSON位于
`benchmarks/results/local/dense_island_vqe_16q_l8_w4_b{16,32}_i10_a800.json`。

## 32. 阶段二十二：TensorCircuit-NG 风格静态 MPS 原型

新增 `flagquantum/simulation/static_mps.py`，将 MPS 改写为固定 bond-capacity profile、
函数式 tensor tuple 和纯函数 brickwork 程序。每个 site 的物理 shape 在 program compile
时确定，gate loop 可被 Python 展开为固定计算图；Z/最近邻 ZZ 能量继续使用共享
left/right environment。该设计借鉴 TensorCircuit-NG 的静态函数式程序思想，不声称复刻
其 JAX/XLA 实现，也不把 `triton.jit` 等同于 `jax.jit`。

6q、2 层、complex64、未截断路径已经与完整 statevector 的 state、energy 和全部参数
gradient 对齐。未截断 split 使用 identity-factor 精确分解，避免低纠缠状态的重复零奇异
值触发未定义 SVD backward。发生截断时，forward 仍取 top-chi SVD 子空间，backward 冻结
本步保留子空间并对投影求导；该路径稳定但属于 approximate truncated-MPS gradient。

整段 loss 提供 `torch.compile(dynamic=False, fullgraph=True)` 缓存入口。A800 上的首次
8q、2 层、bond 8 forward+backward 探针未能生成 Inductor kernel：Dynamo 已捕获静态图，
但 AOTAutograd 在 complex einsum backward 分解中因非连续 stride 执行
`view ComplexFloat as Float` 失败。当前不能报告静态 Inductor/Triton 加速数字。下一步需要
把捕获区域改写为 real/imag 双实数通道，或为 complex contraction/environment 提供自定义
backward；静态 eager 实现保留为 correctness 与后续 benchmark 基线。

real/imag 双实数静态路径随后完成：状态原生保存为 `[..., 2]` float32，RY、RZ、CX、
two-site contraction、精确 split 和共享 Z/ZZ environment 均不在捕获图内创建 complex
tensor。6q、2 层下，双实数与 complex static 的 energy 和完整参数 gradient 通过对照。

A800、8q、2 层、batch 1、bond profile 16（shape 上无截断）的整图
forward+backward 已由 Inductor 成功编译，原 complex stride 错误消失：

| 指标 | 数值 |
|---|---:|
| 首次 compile + step | 85.23 s |
| static real/imag eager steady mean | 157.91 ms/step |
| compiled steady mean | 19.43 ms/step |
| steady speedup | 8.13x |
| compiled/eager gradient max abs delta | 2.09e-4 |

这是一个真实但边界明确的静态图加速信号：当前 compiled real/imag 路径只接受不需要
top-chi 截断的 bond profile；通用截断 MPS 仍走 complex eager 的近似 retained-subspace
gradient。按本次冷编译成本和每步节省约 138 ms 粗略估计，约 616 步后才摊平编译成本；
若把冷编译计入 500-step 总 runtime，不能宣称 8.13x 端到端加速。

## 33. 阶段二十三：Circuit IR 推导 exact bond profile

`StaticMPSProgram.from_ir` 现在逐条读取通用 Circuit IR，对每条 cut 统计跨越它的两比特门，
以门的 operator-Schmidt rank 累乘 bond 上界，并同时限制在左右 Hilbert 空间维度
`2**min(cut, n-cut)` 内。CX/CY/CZ 使用 rank 2，SWAP 使用 rank 4；自定义门可通过
`instruction.metadata["operator_schmidt_rank"]` 声明，否则按通用两比特门 rank 4 保守处理。
如果推导出的精确上界超过用户 `max_bond`，编译阶段直接报错，不静默截断。

CX 执行也改为 rank-2 MPO 更新：未饱和 cut 直接把有效 bond 扩为两倍，不 materialize
two-site matrix、不调用 SVD；达到左右 Hilbert 上界的 cut 使用无 SVD identity factorization
精确消除冗余表示。128q、8 层交错最近邻 CX 的 IR 推导结果为最大跨-cut次数4、最大
exact bond 16。

A800 上 `128q x 8 layers x batch 1 x chi 16` 的 real/imag static eager 完整
forward + shared Z/ZZ environment + backward smoke 成功：单步6.446秒，peak allocated
0.0342 GiB，loss `-110.4896`，全部参数梯度有限。whole-program `torch.compile` 探针超过
4分钟仍停留在 AOT/Inductor 编译阶段，已人工终止，因此没有伪造 compiled runtime 或
speedup。结论是 exact profile 和128q执行已完成，但最终 compiled runner 必须按重复
layer/bond-shape bucket 缓存，而不能把128q全部操作展开成一个巨型FX图。
