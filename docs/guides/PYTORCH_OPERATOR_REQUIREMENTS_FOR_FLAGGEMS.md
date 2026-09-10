# FlagQuantum PyTorch Operator and Precision Requirements

Generated: 2026-07-09

This inventory is for operator and compiler teams. It covers the PyTorch operators,
precision requirements, and FlagGems replacement priorities used by FlagQuantum's
native PyTorch statevector, MPS, tensor-network, density-matrix/noise, and training
paths.

It describes native PyTorch operator acceleration only. Replicated multi-GPU
execution is not distributed scalability; true distributed capability requires
one logical quantum workload to be sharded across ranks.

## 1. Summary

The current quantum-core precision commitments of the PyTorch backend are:

| Category | Current precision | Notes |
| --- | --- | --- |
| Quantum states, gate matrices, MPS tensors, TN nodes, density matrices | `torch.complex64`, `torch.complex128` | Default: `complex64`; high-precision path: `complex128` |
| Parameters, probabilities, expectations, loss, real Hamiltonian coefficients | `torch.float32`, `torch.float64` | Paired with complex dtype: `complex64 -> float32`, `complex128 -> float64` |
| Samples, basis indices, rank transport metadata | `torch.int64`, `torch.long`, `torch.int` | No autograd requirement |
| `float16/bfloat16` | Not the default quantum-core precision | Potential classical-side or future mixed-precision optimization; not the current quantum-core correctness baseline |

The primary request to operator teams is **complex64, autograd, and noncontiguous
tensor layouts**. Supporting only float32/bfloat16/fp16 cannot accelerate the
FlagQuantum quantum core.

## 2. Code Paths and Hot Operators

### 2.1 Dense Statevector / `fq.Circuit`

Primary files:

- `flagquantum/circuit.py`
- `flagquantum/runtime/execution.py`
- `flagquantum/runtime/executors/statevector/`

Core shapes:

- Statevector: `[batch, 2**n_wires]`.
- k-qubit gate matrix: `[2**k, 2**k]` or `[batch, 2**k, 2**k]`.
- Gate-application working tensor after reshape/permute: `[batch, 2**k, rest]`.

| Operator | Purpose | dtype | Autograd | Priority |
| --- | --- | --- | --- | --- |
| `torch.bmm` | Dense gate application | `complex64/complex128` | Required | P0 |
| `matmul` / `@` | Sharded local gate-group updates; density-matrix support | `complex64/complex128` | Required | P0 |
| `reshape`, `permute`, `transpose`, `expand`, `clone` | Gate-application layout transforms | complex/int | Preserve PyTorch layout semantics | P0 |
| `torch.zeros`, `torch.empty`, `torch.as_tensor`, `torch.stack` | Initialization and parameterized matrix assembly | complex/float/int64 | Required for some paths | P0 |
| `torch.abs`, `torch.conj`, `torch.real`, `sum`, elementwise `mul/add/sub/div` | Probabilities, expectations, loss | complex -> float | Required | P0 |
| `torch.diag`, `torch.diagonal`, `torch.count_nonzero` | Diagonal-gate fast path | complex/bool | Recommended | P1 |
| `torch.arange`, `torch.nonzero`, bitwise index ops | Shard basis indices | int64/long | Not required | P1 |
| `torch.multinomial`, `torch.unique` | Sampling/counts | float32/int64 | Not required | P2 |

### 2.2 MPS

Primary files:

- `flagquantum/simulation/mps/`
- `flagquantum/simulation/mps/state.py`
- `flagquantum/simulation/mps/factorization.py`

Core shapes:

- MPS site tensor: `[batch, left_bond, 2, right_bond]`.
- Two-site tensor: `[batch, left_bond, 2, 2, right_bond]`.
- Split matrix: `[batch, left_bond * 2, 2 * right_bond]`.

| Operator | Purpose | dtype | Autograd | Priority |
| --- | --- | --- | --- | --- |
| `torch.einsum` | One-/two-site updates, environment contraction, expectations | `complex64/complex128` | Required | P0 |
| `torch.linalg.qr` | Exact MPS splitting without truncation | `complex64/complex128` | Required | P0 |
| `torch.linalg.svd` | Truncation, `from_statevector`, approximate MPS | `complex64/complex128` | Required, with stable backward | P0 |
| `torch.stack`, `reshape`, `transpose`, `expand`, `clone` | MPS tensor layout | complex | Required | P0 |
| `torch.exp`, `torch.cos`, `torch.sin` | RX/RY/RZ fast paths | real -> complex | Required | P0 |
| `torch.conj`, `torch.real`, `torch.abs`, `sum` | Norms, observables, loss | complex -> real | Required | P0 |
| `torch.eye`, `torch.zeros`, `torch.ones` | Exact splitting and environment initialization | complex | Recommended | P1 |
| `torch.clamp`, `torch.sqrt`, `scatter_`, `multinomial`, `unique` | Sampling, trajectories, normalization | float/complex/int64 | Required for some paths | P2 |
| `torch.linalg.eigh` | Noisy trajectory fallback for multiqubit channels | complex | Recommended | P2 |

Specific MPS requirements:

- Complex `torch.linalg.svd` backward has phase/gauge nonuniqueness. SVD
  acceleration must provide stable gradient semantics or a controlled custom backward.
- The main training path favors exact splits compatible with autograd, but
  general MPS, truncation, and capacity modes still require stable complex SVD/QR.
- Inputs are often noncontiguous. Operators must respect strides/layouts and
  cannot assume contiguous storage.

### 2.3 Tensor Network

Primary file:

- `flagquantum/simulation/tensor.py`

Core shapes:

- TN node tensors have arbitrary rank and integer labels.
- Pair contraction: `left_tensor`, `right_tensor` -> intermediate tensor.
- Expectation contraction uses a bra/operator/ket network, usually returning `[batch]`.

| Operator | Purpose | dtype | Autograd | Priority |
| --- | --- | --- | --- | --- |
| `torch.einsum` | Core TN contraction | `complex64/complex128` | Required | P0 |
| `torch.empty`, `torch.zeros`, `torch.eye` | Dry runs/profiling, identity nodes, initial nodes | complex | Recommended | P1 |
| `torch.conj`, `torch.real`, `sum`, elementwise add/mul | Expectations and partial reductions | complex -> real | Required | P0 |
| `reshape`, `stack` | Node construction and output reshaping | complex | Required | P0 |

Specific TN requirements:

- `einsum` must support complex64/complex128, high tensor rank, dynamic
  contraction equations, and noncontiguous tensors.
- Backward must agree with PyTorch autograd to support quantum machine learning.
- Slicing/reduction paths will require complex partial-sum and deterministic reductions.

### 2.4 Density Matrix / Noise

Primary files:

- `flagquantum/noise/`
- `flagquantum/simulation/density_matrix.py`

| Operator | Purpose | dtype | Autograd | Priority |
| --- | --- | --- | --- | --- |
| `torch.bmm` | `U rho U^dagger`, Kraus channels | `complex64/complex128` | Recommended | P1 |
| `torch.conj`, `transpose`, `zeros_like`, elementwise add/mul | Channel accumulation | complex | Recommended | P1 |
| `torch.diagonal`, `torch.real`, `sum`, `stack` | Density expectations | complex -> real | Recommended | P1 |
| `torch.sqrt`, `torch.zeros`, `torch.ones` | Channel matrix construction | float/complex | Recommended | P2 |

### 2.5 Gate Matrix Construction

Primary file:

- `flagquantum/ops/matrices.py`

| Operator | Purpose | dtype | Autograd | Priority |
| --- | --- | --- | --- | --- |
| `torch.exp`, `torch.cos`, `torch.sin` | Parameterized gates | real/complex | Required | P0 |
| `torch.stack`, `torch.cat`, `torch.zeros`, `torch.eye`, `torch.ones_like`, `torch.zeros_like` | Gate assembly | complex | Required | P0 |
| `torch.conj` | Phase conjugation for RZ/RZZ and related gates | complex | Required | P0 |
| `torch.outer`, `torch.arange`, `torch.sqrt` | QFT matrices | float/complex | Usually not critical | P2 |

### 2.6 Hamiltonians / Algorithms / Training

Primary files:

- `flagquantum/algorithms/core.py`
- `flagquantum/runtime/training.py`
- PyTorch interface wrappers in `flagquantum/runtime/hybrid.py`.

| Operator | Purpose | dtype | Autograd | Priority |
| --- | --- | --- | --- | --- |
| `torch.kron` | Dense Pauli operator construction | `complex64/complex128` | Recommended | P1 |
| `torch.matmul`, `torch.diagonal`, `sum`, `real` | Density/Hamiltonian expectations | complex -> real | Required | P1 |
| `torch.optim.*`, `.backward()` | Training loops | float parameters | Native PyTorch | Do not replace |
| `torch.as_tensor`, `torch.linspace`, `torch.randn` | Parameter initialization / benchmarks | float32/float64 | Required for some paths | P2 |

## 3. P0/P1/P2 Requirements

### P0: Quantum Training Core — First Priority

These operators determine whether statevector/MPS/TN training works through the
native PyTorch path:

| Operator/family | Required dtype | Key requirements |
| --- | --- | --- |
| `bmm`, `matmul` / `@`, `mm` | `complex64`, `complex128` | Batched complex GEMM, autograd, noncontiguous inputs |
| `einsum` | `complex64`, `complex128` | High-rank contraction, dynamic equations, backward agreement with PyTorch |
| elementwise `add/sub/mul/div` | `complex64`, `complex128`, `float32`, `float64` | Broadcasting, autograd, no silent precision downgrade |
| `abs`, `conj`, `real` | complex -> real/complex | Correct gradients for expectations/loss |
| `sum` / reduction | complex and real | dim/keepdim, determinism, backward |
| `exp`, `cos`, `sin`, `sqrt` | real/complex | Correct autograd for parameterized gate matrices |
| `stack`, `cat`, `zeros`, `ones`, `eye`, `empty`, `as_tensor` | complex/float/int64 | Preserve dtype/device during creation and assembly |
| `torch.linalg.qr` | `complex64`, `complex128` | Stable autograd for exact MPS splitting |
| `torch.linalg.svd` | `complex64`, `complex128` | Complex gauge/phase backward stability for MPS truncation |

### P1: Important Performance Paths — Second Priority

| Operator/family | dtype | Purpose |
| --- | --- | --- |
| `kron`, `outer`, `dot`, `vdot`, `addmm` | complex/float | Hamiltonians, QFT, dense observables |
| `diagonal`, `diag`, `count_nonzero` | complex/bool | Diagonal-gate fast paths, density expectations |
| `gather`, `index_select`, `nonzero`, `arange` | int64/complex | Sharded basis/index paths |
| `where`, `clamp` | float/complex/bool | Probabilities, JAX parity/debugging, sampling |
| `scatter_`, `scatter_add_`, `scatter_reduce` | complex/float/int64 | Sampling, future sharded transport/reduction |

### P2: Supporting and Engineering Paths

| Operator/family | dtype | Purpose |
| --- | --- | --- |
| `multinomial`, `unique`, `argmax` | float/int64 | Sampling/counts/noise trajectories |
| `rand`, `randn`, `randint`, `manual_seed` | float/int64 | Initialization, tests, benchmarks |
| `log2`, `ceil`, `min`, `max`, `prod`, `var` | float/int | Profiling, shapes, statistics |
| `torch.linalg.eigh` | complex | Noisy trajectory fallback |

## 4. Current FlagGems Integration

FlagQuantum already has operator-backend entry points:

- `flagquantum/compute/flaggems.py`
- `benchmarks/operator_backend_compare.py`

Current safe candidate operators:

```text
abs, add, addmm, bmm, dot, gather, index_select, kron, mm, mul,
outer, sum, vdot, where_self, zeros, zeros_like
```

Current experimental candidates:

```text
einsum, scatter, scatter_, scatter_add_, scatter_reduce,
scatter_reduce_, svd
```

Caveats:

- Successful operator registration does not establish complex64 support.
- A cluster run produced `KeyError: 'complex64'`, showing that at least some
  registered FlagGems operators did not cover complex64.
- For FlagQuantum, `complex64` support takes priority over `float16/bfloat16`.
- Support limited to float32/bfloat16/fp16 benefits classical AI but does not
  establish acceleration of the quantum simulation core.

## 5. Recommended Requirements for Operator Teams

### 5.1 Required Correctness Properties

1. `complex64` is the primary precision target; `complex128` is the high-precision
   validation target.
2. All P0 complex operators must support PyTorch autograd backward.
3. Never silently downgrade `complex64` to `float32` or `complex128` to `complex64`.
4. Output dtypes must agree with PyTorch:
   - `abs(complex64) -> float32`
   - `real(complex64) -> float32`
   - Complex reductions remain complex by default unless PyTorch semantics return real.
5. Support noncontiguous inputs: quantum gate application often uses
   `reshape/permute/transpose`.
6. Reductions must provide determinism or explainable numerical error bounds for
   quantum gradient comparisons.

### 5.2 Minimum End-to-End Scenarios

| Scenario | Key operators | dtype | Acceptance target |
| --- | --- | --- | --- |
| 8-20q statevector VQE/QML | `bmm`, `abs`, `sum`, `conj`, `real`, `exp/cos/sin` | complex64 | Loss/grad agreement with PyTorch |
| 60-1000q low-bond MPS training | `einsum`, `qr/svd`, `sum`, `conj`, `real`, `exp/cos/sin` | complex64 | Loss/grad agreement with PyTorch/JAX references |
| General TN contraction | `einsum`, reduction, `conj`, `real` | complex64 | Forward/backward agreement with PyTorch |
| Density/noise simulation | `bmm`, `conj`, `transpose`, `diagonal`, `zeros_like` | complex64 | Expectation agreement with PyTorch |

## 6. Local Verification Commands

Inspect static PyTorch operator references in FlagQuantum:

```bash
rg -o --no-filename "torch\.[A-Za-z_][A-Za-z0-9_]*" flagquantum -g "*.py" \
  | sort | uniq -c | sort -nr
```

Verify actual FlagGems support for a dtype:

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

Run the FlagGems replacement benchmark:

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

## 7. Outside Local FlagGems Operator Replacement

The following are FlagQuantum distributed/hybrid capabilities. Local ATen
replacement by FlagGems is not responsible for providing them:

| Capability | Owner |
| --- | --- |
| `torch.distributed.all_to_all`, `batch_isend_irecv`, `all_gather` | FlagQuantum distributed runtime + NCCL/Gloo |
| JAX `pmap`, `shard_map`, DLPack bridge | FlagQuantum JAX quantum kernel runtime |
| MPS site sharding, TN slicing/reduction, statevector amplitude sharding | FlagQuantum distributed planner/executor |
| Multinode rank placement, internode communication planning | FlagQuantum distributed planner |

The best FlagGems integration point is **fast, accurate native PyTorch quantum
operators on a single machine/device, with complex autograd support**. This
strengthens local CPU/single-GPU usability and production local kernels while
preserving FlagQuantum's distributed architecture.
