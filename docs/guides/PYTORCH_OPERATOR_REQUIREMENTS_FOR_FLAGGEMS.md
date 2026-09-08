# FlagQuantum PyTorch 后端算子与精度需求清单

生成日期：2026-07-09

本文面向算子/编译器团队，用于梳理 FlagQuantum native PyTorch 后端在 statevector、MPS、tensor network、density matrix/noise 与训练梯度中实际涉及的 PyTorch 算子、精度要求和 FlagGems 替换优先级。

这份清单只描述 PyTorch native operator acceleration 需求，不把多卡复制运行描述为分布式扩展性。真正分布式能力仍以一个逻辑量子任务跨 rank sharding 为准。

## 1. 总体结论

FlagQuantum 当前 PyTorch 后端对外承诺的量子核心精度是：

| 类别 | 当前使用精度 | 说明 |
| --- | --- | --- |
| 量子态、门矩阵、MPS tensor、TN node、密度矩阵 | `torch.complex64`, `torch.complex128` | 默认 `complex64`，高精度路径 `complex128` |
| 参数、概率、期望值、loss、实数 Hamiltonian 系数 | `torch.float32`, `torch.float64` | 与 complex dtype 成对出现：`complex64 -> float32`，`complex128 -> float64` |
| 采样、basis index、rank transport metadata | `torch.int64`, `torch.long`, `torch.int` | 不要求 autograd |
| `float16/bfloat16` | 非量子核心默认精度 | 可作为经典侧/未来混合精度优化，不应作为当前量子核心正确性主线 |

给算子部门的最重要需求是：**complex64 + autograd + 非连续张量布局**。只支持 float32/bfloat16/fp16 不足以加速 FlagQuantum 的量子核心。

## 2. 代码路径与热点算子

### 2.1 dense statevector / `fq.Circuit`

主要文件：

- `flagquantum/circuit.py`
- `flagquantum/runtime/execution.py`
- `flagquantum/runtime/executors/statevector/`

核心形状：

- statevector: `[batch, 2**n_wires]`
- k-qubit gate matrix: `[2**k, 2**k]` or `[batch, 2**k, 2**k]`
- apply-gate working tensor: reshape/permute 后变为 `[batch, 2**k, rest]`

热点算子：

| 算子 | 用途 | dtype | autograd | 优先级 |
| --- | --- | --- | --- | --- |
| `torch.bmm` | dense gate application | `complex64/complex128` | 必须 | P0 |
| `matmul` / `@` | sharded local gate group update、密度矩阵辅助 | `complex64/complex128` | 必须 | P0 |
| `reshape`, `permute`, `transpose`, `expand`, `clone` | gate application layout 变换 | complex/int | layout 必须保持 PyTorch 语义 | P0 |
| `torch.zeros`, `torch.empty`, `torch.as_tensor`, `torch.stack` | 初始化、参数矩阵组装 | complex/float/int64 | 部分需要 | P0 |
| `torch.abs`, `torch.conj`, `torch.real`, `sum`, elementwise `mul/add/sub/div` | 概率、期望值、loss | complex -> float | 必须 | P0 |
| `torch.diag`, `torch.diagonal`, `torch.count_nonzero` | 对角门 fast path | complex/bool | 建议 | P1 |
| `torch.arange`, `torch.nonzero`, bitwise index ops | shard basis index | int64/long | 不需要 | P1 |
| `torch.multinomial`, `torch.unique` | sampling/counts | float32/int64 | 不需要 | P2 |

### 2.2 MPS

主要文件：

- `flagquantum/simulation/mps/`
- `flagquantum/simulation/mps/state.py`
- `flagquantum/simulation/mps/factorization.py`

核心形状：

- MPS site tensor: `[batch, left_bond, 2, right_bond]`
- two-site tensor: `[batch, left_bond, 2, 2, right_bond]`
- split matrix: `[batch, left_bond * 2, 2 * right_bond]`

热点算子：

| 算子 | 用途 | dtype | autograd | 优先级 |
| --- | --- | --- | --- | --- |
| `torch.einsum` | one-site/two-site update、环境收缩、期望值 | `complex64/complex128` | 必须 | P0 |
| `torch.linalg.qr` | exact/no-truncation MPS split | `complex64/complex128` | 必须 | P0 |
| `torch.linalg.svd` | truncation、`from_statevector`、近似 MPS | `complex64/complex128` | 必须，但需要稳定 backward | P0 |
| `torch.stack`, `reshape`, `transpose`, `expand`, `clone` | MPS tensor layout | complex | 必须 | P0 |
| `torch.exp`, `torch.cos`, `torch.sin` | RX/RY/RZ fast path | real -> complex | 必须 | P0 |
| `torch.conj`, `torch.real`, `torch.abs`, `sum` | norm、observable、loss | complex -> real | 必须 | P0 |
| `torch.eye`, `torch.zeros`, `torch.ones` | exact split、environment init | complex | 建议 | P1 |
| `torch.clamp`, `torch.sqrt`, `scatter_`, `multinomial`, `unique` | sampling、trajectory、normalization | float/complex/int64 | 部分需要 | P2 |
| `torch.linalg.eigh` | noisy trajectory 多 qubit channel fallback | complex | 建议 | P2 |

MPS 需要特别强调：

- `torch.linalg.svd` 对 complex backward 有相位/gauge 不唯一问题，算子部门如果做 SVD 加速，必须给出稳定的 gradient 语义或可控的 custom backward。
- 当前训练主路径会尽量走 exact/autograd-friendly split，但通用 MPS、truncation、capacity 模式仍需要稳定 complex SVD/QR。
- 输入经常是非连续张量，算子必须兼容 stride/layout，不能假设 contiguous。

### 2.3 Tensor Network

主要文件：

- `flagquantum/simulation/tensor.py`

核心形状：

- TN node tensor: 任意 rank，标签由 integer labels 管理
- pair contraction: `left_tensor`, `right_tensor` -> intermediate tensor
- expectation contraction: bra/operator/ket 网络，输出通常为 `[batch]`

热点算子：

| 算子 | 用途 | dtype | autograd | 优先级 |
| --- | --- | --- | --- | --- |
| `torch.einsum` | TN contraction 核心 | `complex64/complex128` | 必须 | P0 |
| `torch.empty`, `torch.zeros`, `torch.eye` | dry run/profile、identity node、initial node | complex | 建议 | P1 |
| `torch.conj`, `torch.real`, `sum`, elementwise add/mul | expectation、partial reduction | complex -> real | 必须 | P0 |
| `reshape`, `stack` | node construction/output reshape | complex | 必须 | P0 |

TN 需求重点：

- `einsum` 必须支持 complex64/complex128、高 rank、动态 contraction equation、非连续 tensor。
- backward 必须与 PyTorch autograd 对齐；否则无法支撑量子机器学习训练。
- 对 slicing/reduction 路径，未来会需要 complex partial-sum reduction 和 deterministic reduction。

### 2.4 Density Matrix / Noise

主要文件：

- `flagquantum/noise/`
- `flagquantum/simulation/density_matrix.py`

热点算子：

| 算子 | 用途 | dtype | autograd | 优先级 |
| --- | --- | --- | --- | --- |
| `torch.bmm` | `U rho U^dagger`、Kraus channel | `complex64/complex128` | 建议 | P1 |
| `torch.conj`, `transpose`, `zeros_like`, elementwise add/mul | channel accumulation | complex | 建议 | P1 |
| `torch.diagonal`, `torch.real`, `sum`, `stack` | density expectation | complex -> real | 建议 | P1 |
| `torch.sqrt`, `torch.zeros`, `torch.ones` | channel matrix construction | float/complex | 建议 | P2 |

### 2.5 Gate Matrix Construction

主要文件：

- `flagquantum/ops/matrices.py`

热点算子：

| 算子 | 用途 | dtype | autograd | 优先级 |
| --- | --- | --- | --- | --- |
| `torch.exp`, `torch.cos`, `torch.sin` | parameterized gates | real/complex | 必须 | P0 |
| `torch.stack`, `torch.cat`, `torch.zeros`, `torch.eye`, `torch.ones_like`, `torch.zeros_like` | gate assembly | complex | 必须 | P0 |
| `torch.conj` | RZ/RZZ 等相位共轭 | complex | 必须 | P0 |
| `torch.outer`, `torch.arange`, `torch.sqrt` | QFT matrix | float/complex | 通常不关键 | P2 |

### 2.6 Hamiltonian / Algorithms / Training

主要文件：

- `flagquantum/algorithms/core.py`
- `flagquantum/runtime/training.py`
- `flagquantum/runtime/hybrid.py` 中的 PyTorch interface 包装

热点算子：

| 算子 | 用途 | dtype | autograd | 优先级 |
| --- | --- | --- | --- | --- |
| `torch.kron` | dense Pauli operator construction | `complex64/complex128` | 建议 | P1 |
| `torch.matmul`, `torch.diagonal`, `sum`, `real` | density/Hamiltonian expectation | complex -> real | 必须 | P1 |
| `torch.optim.*`, `.backward()` | 训练 loop | float parameters | PyTorch 原生 | 不替换 |
| `torch.as_tensor`, `torch.linspace`, `torch.randn` | parameter init / benchmarks | float32/float64 | 部分需要 | P2 |

## 3. P0/P1/P2 需求汇总

### P0：量子训练核心，必须优先支持

这些算子直接决定 statevector/MPS/TN 是否能在 PyTorch native 路径里训练：

| 算子/算子族 | 必须支持 dtype | 关键要求 |
| --- | --- | --- |
| `bmm`, `matmul` / `@`, `mm` | `complex64`, `complex128` | batched complex GEMM，支持 autograd 和非连续输入 |
| `einsum` | `complex64`, `complex128` | 高 rank contraction，动态 equation，backward 对齐 PyTorch |
| elementwise `add/sub/mul/div` | `complex64`, `complex128`, `float32`, `float64` | broadcasting、autograd、不可静默降精度 |
| `abs`, `conj`, `real` | complex -> real/complex | 期望值/loss 必需，gradient 必须正确 |
| `sum` / reduction | complex and real | 支持 dim/keepdim、determinism、backward |
| `exp`, `cos`, `sin`, `sqrt` | real/complex | 参数门矩阵构造，autograd 必须正确 |
| `stack`, `cat`, `zeros`, `ones`, `eye`, `empty`, `as_tensor` | complex/float/int64 | 创建与组装必须保 dtype/device |
| `torch.linalg.qr` | `complex64`, `complex128` | MPS exact split，autograd 必须稳定 |
| `torch.linalg.svd` | `complex64`, `complex128` | MPS truncation，需解决 complex gauge/phase backward 稳定性 |

### P1：重要性能路径，建议第二批支持

| 算子/算子族 | dtype | 用途 |
| --- | --- | --- |
| `kron`, `outer`, `dot`, `vdot`, `addmm` | complex/float | Hamiltonian、QFT、dense observable |
| `diagonal`, `diag`, `count_nonzero` | complex/bool | diagonal gate fast path、density expectation |
| `gather`, `index_select`, `nonzero`, `arange` | int64/complex | sharded basis/index path |
| `where`, `clamp` | float/complex/bool | probabilities、JAX parity/debug、sampling |
| `scatter_`, `scatter_add_`, `scatter_reduce` | complex/float/int64 | sampling、future sharded transport/reduction |

### P2：辅助/工程路径

| 算子/算子族 | dtype | 用途 |
| --- | --- | --- |
| `multinomial`, `unique`, `argmax` | float/int64 | sampling/counts/noise trajectory |
| `rand`, `randn`, `randint`, `manual_seed` | float/int64 | 初始化、测试、benchmark |
| `log2`, `ceil`, `min`, `max`, `prod`, `var` | float/int | profiling、shape、统计 |
| `torch.linalg.eigh` | complex | noisy trajectory fallback |

## 4. 当前 FlagGems 接入现状

当前 FlagQuantum 已经有 operator backend 入口：

- `flagquantum/compute/flaggems.py`
- `benchmarks/operator_backend_compare.py`

当前安全候选算子：

```text
abs, add, addmm, bmm, dot, gather, index_select, kron, mm, mul,
outer, sum, vdot, where_self, zeros, zeros_like
```

当前实验候选算子：

```text
einsum, scatter, scatter_, scatter_add_, scatter_reduce,
scatter_reduce_, svd
```

注意：

- “算子注册成功”不等于“支持 complex64”。
- 集群上出现过 `KeyError: 'complex64'`，说明至少部分 FlagGems 注册算子尚未覆盖 complex64。
- 对 FlagQuantum 而言，`complex64` 支持优先级高于 `float16/bfloat16`。
- 如果 FlagGems 只支持 float32/bfloat16/fp16，对经典 AI 部分有价值，但不能证明量子模拟核心被加速。

## 5. 推荐给算子部门的需求表达

### 5.1 必须满足的正确性要求

1. `complex64` 是第一目标精度；`complex128` 是高精度验证目标。
2. 所有 P0 complex 算子必须支持 PyTorch autograd backward。
3. 不允许静默把 `complex64` 降成 `float32` 或把 `complex128` 降成 `complex64`。
4. 输出 dtype 必须与 PyTorch 对齐：
   - `abs(complex64) -> float32`
   - `real(complex64) -> float32`
   - complex reduction 默认仍保持 complex，除非 PyTorch 语义返回 real。
5. 必须支持非连续输入，因为量子 gate application 经常来自 `reshape/permute/transpose`。
6. reduction 类算子需要给出确定性或可解释的数值误差边界，便于量子梯度对齐。

### 5.2 必须覆盖的最小端到端场景

| 场景 | 关键算子 | dtype | 验收目标 |
| --- | --- | --- | --- |
| 8-20q statevector VQE/QML | `bmm`, `abs`, `sum`, `conj`, `real`, `exp/cos/sin` | complex64 | loss/grad 与 PyTorch 对齐 |
| 60-1000q low-bond MPS training | `einsum`, `qr/svd`, `sum`, `conj`, `real`, `exp/cos/sin` | complex64 | loss/grad 与 PyTorch/JAX reference 对齐 |
| general TN contraction | `einsum`, reduction, `conj`, `real` | complex64 | forward/backward 与 PyTorch 对齐 |
| density/noise simulation | `bmm`, `conj`, `transpose`, `diagonal`, `zeros_like` | complex64 | expectation 与 PyTorch 对齐 |

## 6. 本地验证命令

查看 FlagQuantum PyTorch 后端静态算子：

```bash
rg -o --no-filename "torch\.[A-Za-z_][A-Za-z0-9_]*" flagquantum -g "*.py" \
  | sort | uniq -c | sort -nr
```

验证 FlagGems 对某个 dtype 的实际支持：

```python
from flagquantum.compute.flaggems import validate_flaggems_ops

ops = [
    "mm", "bmm", "sum", "mul", "abs", "add", "addmm",
    "gather", "index_select", "kron", "vdot", "dot",
    "where_self", "zeros", "zeros_like",
]

for dtype in ["float32", "float16", "bfloat16", "complex64", "complex128"]:
    result = validate_flaggems_ops(ops, device="cuda", dtype=dtype)
    print("\n==", dtype, "==")
    print("passed:", result.passed_ops)
    print("failed:", result.failed_ops)
```

运行 FlagGems 替换 benchmark：

```bash
python benchmarks/operator_backend_compare.py \
  --device cuda \
  --mode mps \
  --max-bond 32 \
  --n-wires 12 \
  --layers 3 \
  --batch-size 8 \
  --observable ising \
  --iters 50 \
  --warmup 10 \
  --flaggems-strict \
  --flaggems-validation-dtype complex64 \
  --json-output operator_backend_mps_cuda_flaggems_complex64.json
```

## 7. 不属于 FlagGems 本地算子替换的需求

下面这些是 FlagQuantum 分布式/混合后端能力，不应要求 FlagGems 通过本地 ATen 替换直接解决：

| 能力 | 归属 |
| --- | --- |
| `torch.distributed.all_to_all`, `batch_isend_irecv`, `all_gather` | FlagQuantum distributed runtime + NCCL/Gloo |
| JAX `pmap`, `shard_map`, DLPack bridge | FlagQuantum JAX quantum kernel runtime |
| MPS site shard、TN slice/reduction、statevector amplitude shard | FlagQuantum distributed planner/executor |
| 多节点 rank placement、inter-node communication planning | FlagQuantum distributed planner |

FlagGems 的最佳切入点是：**把单机/单卡 PyTorch native 量子核心算子做快、做准、支持 complex autograd**。这不会影响 FlagQuantum 的分布式设计，反而能让单卡/CPU 用户体验和生产级局部 kernel 同时变强。
