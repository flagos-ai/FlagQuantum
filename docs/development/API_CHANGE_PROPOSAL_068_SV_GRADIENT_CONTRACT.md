# API Change Proposal 068: The statevector adjoint gradient contract

## Status

**Proposed. No code is written by this proposal and no behaviour changes until
the repository owner approves it and the acceptance items below are measured.**

This proposal fixes the contract for the statevector adjoint gradient. Plan row
`W2-03` ("the statevector backward pass and its autograd contract") names this
document as its deliverable, and
plan row `W4-06` explicitly moved adjoint gradients out of its own scope and
handed them here, so two rows refer to this file and leaving it unwritten keeps
both of them unresolved rather than one. Every premise below was measured on
this checkout before the proposal was written, because engineering decision
principle 6 requires inspecting what already exists and principle 10 forbids
claiming a gap that has not been observed. Each measured premise carries the
literal command and the literal output that produced it. Everything in
"Required evidence" is a requirement on the implementation, not a result: no
line of production code has been written for this proposal.

## Problem

### 1. The plan row names an artifact that does not exist, and two differently named adjoint modules do

The row's definition of done names `statevector_adjoint.py` as the module to
productionize. No such module exists under `flagquantum/simulation/statevector/`.
Measured:

```
$ find flagquantum -name 'statevector_adjoint.py'
flagquantum/kernels/triton/statevector_adjoint.py
```

The only match is a Triton kernel module of 406 lines, which is not a
simulation-layer numerical module and is not the file the row's phrase
"`statevector_adjoint.py` reaching production" can mean. Measured:

```
$ ls flagquantum/simulation/statevector/
README.md
__init__.py
__pycache__
adjoint.py
clifford_matching.py
controlled_phase.py
cx_sequence_dispatch.py
cz_graph.py
diagonal_cpu.py
double_single.py
double_single_device_gates.py
double_single_host_gates.py
dynamic_noise.py
local.py
noisy.py
operations.py
product_state.py
program.py
rx_rz_dispatch.py
ry_rz_dispatch.py
single_qubit_matrix_dispatch.py
small.py
split_real_imag.py
two_qubit_cpu.py
wire_permutation.py
```

The numerically-owned adjoint module is `flagquantum/simulation/statevector/adjoint.py`,
127 lines, and it is named in its domain README (`flagquantum/simulation/statevector/README.md:14`,
"Start in `adjoint.py` for local adjoint and gradient primitives."). A proposal
that created a second file spelled `statevector_adjoint.py` would put two
sources of truth for one arithmetic in one directory, so this document refuses
that spelling and states the contract over the existing module.

### 2. The adjoint arithmetic exists in Simulation and is reachable only through private imports

`adjoint.py` declares six names in its `__all__` tuple:
`analytic_rotation_derivative` (`flagquantum/simulation/statevector/adjoint.py:12`),
`real_conjugate_inner_sum` (`:36`), `z_expectation_adjoint_chunk` (`:56`),
`z_expectation_chunk` (`:42`), `z_hamiltonian_chunk` (`:70`), and
`z_hamiltonian_weights` (`:90`). The line numbers are not in `__all__` order
because the tuple at the bottom of the file lists the two expectation helpers
in the reverse of their definition order; the quoted output below is the
authority. Nothing in the package `__init__` re-exports any of them. Measured:

```
$ grep -n '' flagquantum/simulation/statevector/adjoint.py | sed -n '120,127p'
120:__all__ = (
121:    "analytic_rotation_derivative",
122:    "real_conjugate_inner_sum",
123:    "z_expectation_adjoint_chunk",
124:    "z_expectation_chunk",
125:    "z_hamiltonian_chunk",
126:    "z_hamiltonian_weights",
127:)
$ wc -c flagquantum/simulation/statevector/__init__.py
      51 flagquantum/simulation/statevector/__init__.py
$ cat flagquantum/simulation/statevector/__init__.py
"""Dense statevector numerical implementations."""
---END---
```

One docstring line, 51 bytes, no import and no `__all__`. The only importers of
the module are Runtime-private. Measured:

```
$ grep -rn 'statevector\.adjoint\|statevector import adjoint\|from \.adjoint\|from \.\.adjoint' flagquantum tests tools benchmarks --include='*.py'
flagquantum/runtime/executors/statevector/reverse_adjoint.py:14:from ....simulation.statevector.adjoint import (
flagquantum/runtime/executors/statevector/reverse_adjoint.py:17:from ....simulation.statevector.adjoint import (
flagquantum/runtime/executors/statevector/reverse_adjoint.py:20:from ....simulation.statevector.adjoint import (
flagquantum/runtime/executors/statevector/reverse_adjoint_sweep.py:22:from ....simulation.statevector.adjoint import (
flagquantum/runtime/executors/statevector/reverse_adjoint_sweep.py:25:from ....simulation.statevector.adjoint import (
flagquantum/runtime/executors/statevector/reverse_adjoint_sweep.py:28:from ....simulation.statevector.adjoint import z_hamiltonian_chunk
flagquantum/simulation/native_cpu/__init__.py:3:from .adjoint import (
flagquantum/simulation/native_cpu/rzz.py:10:from .adjoint import _load_extension
flagquantum/simulation/native_cpu/permutation.py:11:from .adjoint import _load_extension, native_cpu_compact_cx_index_available
flagquantum/simulation/native_cpu/rotation.py:10:from .adjoint import _load_extension
tests/team/simulation/test_statevector_adjoint_numerics.py:9:from flagquantum.simulation.statevector.adjoint import (
```

Four of those eleven hits are a same-named but different module
(`flagquantum/simulation/native_cpu/adjoint.py`); the remaining seven are a
Runtime executor reaching into Simulation, or a test. The module is therefore
implemented, tested
(`tests/team/simulation/test_statevector_adjoint_numerics.py`), and
simultaneously unreachable through any documented path. That is the actual
condition of "productionize", and it is not a missing-arithmetic problem.

### 3. The `torch.autograd.Function` wrapper the row asks for already exists, and the public gradient route is a different arithmetic

The row's second definition-of-done item is a `torch.autograd.Function` wrapper
around the adjoint. That wrapper exists. Measured:

```
$ grep -rn '_ShardedStatevectorExpectation' flagquantum --include='*.py'
flagquantum/runtime/executors/statevector/reverse.py:339:class _ShardedStatevectorExpectation(torch.autograd.Function):
flagquantum/runtime/executors/statevector/reverse.py:628:    apply: Callable[..., object] = _ShardedStatevectorExpectation.apply
```

Its `forward` runs the sharded forward executor and evaluates the weighted Z/ZZ
expectation, its `backward` runs `_explicit_sharded_adjoint`, and it is applied
from `execute_torch_distributed_statevector_reverse`
(`flagquantum/runtime/executors/statevector/reverse.py:511`), which is exported
from `flagquantum/runtime/executors/statevector/__init__.py:12`. Runtime is not
Stable Core, so this boundary is a real wrapper that no user is told about.

Meanwhile the documented user route does not use it. Measured, with the reverse
executor itself instrumented by monkeypatch:

```
reverse executor calls from fq.run(..., require_gradients=True) = 0
t.grad = -0.3986093279844229
value = 0.917120822816605
summary keys mentioning grad/adjoint/method: []
out.runtime = {
  "mode": "statevector",
  "execution_path": "local_statevector",
  "simulation_engine": "pytorch_statevector",
  "platform_provider": "pytorch_cpu",
  "device": "cpu",
  "distribution_semantics": "single_device_fast_path"
}
```

`fq.run(circuit, options=fq.ExecutionOptions(require_gradients=True), outputs=fq.expectation(fq.Z(0)))`
returns a tensor whose `grad_fn` is `MulBackward0`, whose gradient is correct to
`-0.3986093279844229` against the closed form `-sin(0.41) = -0.3986093279844229`,
and which reaches the reverse executor zero times. The gradient the user gets
today is the dense local statevector autograd of `flagquantum/simulation/statevector/local.py`
(`_expectation_z` at line 1121, `state` at line 873), not the adjoint. Two
gradient arithmetics for one observable exist in the tree, the public one is
selected by no stated rule, and neither the result record nor the plan names
which one ran: `out.runtime` has no gradient field at all, and
`out.summary()` has no key containing `grad` or `adjoint`. This is the gap the
row's third item — "gradient must agree with parameter shift within a tolerance"
— is really about, because agreement cannot be asserted for a route the caller
cannot identify.

### 4. The adjoint path is one forward and zero additional forwards, and it is not the full-state replay `W4-06` delivered

`W4-06` recorded the boundary honestly in the plan. At
`cudaq-parity-pr-plan.md:944` its out-of-scope list states, in the plan's own
words, that adjoint gradients were not implemented, that the half-sentence
about adding them was moved out of that row and left to `W2-03` in
`API_CHANGE_PROPOSAL_068`, and that what the row delivered is the live autograd
graph of the existing integrators, whose reverse cost grows as a constant
multiple of the forward count rather than being adjoint's one forward and one
reverse. The statevector adjoint is a different object and this proposal must
not inherit that sentence. Measured on the three-wire circuit
`fq.Circuit(3, dtype=torch.complex128).rx(0, t0).ry(1, t1).rz(2, t2).cx(0, 1).cx(1, 2)`
with `(t0, t1, t2) = (0.41, -0.37, 0.83)` and
`observable_terms=[(1.0, (0,)), (1.0, (2,))]`, with
`execute_torch_distributed_statevector` instrumented by monkeypatch:

```
  forward calls during forward(): 1
  forward calls during backward(): 0  total: 1
  saved_forward_state_reused: True
  backward_uses_full_state_replay: False
  gradient_method: 'statevector_adjoint'
  backward_status: 'completed'
  analytic_rotation_derivative_count: 3
  value: tensor(1.7722, dtype=torch.float64, grad_fn=<_ShardedStatevectorExpectationBackward>)
  gradient: [-0.770243704677945, 0.33164504250688803, 0.0]
```

One forward, zero extra forwards during backward, no full-state replay, on a
circuit with two entangling gates rather than one, so the counts are not an
artifact of a single-`CX` fixture. The same run reports
`gradient_method: 'statevector_adjoint'` and
`executor: 'pytorch_native_sharded_statevector_reverse_v1'`, and its checkpoint
record selects strategy `reversible_adjoint` with
`selection_reason: forward_state_reuse_fits_budget` when no policy is passed and
`selection_reason: explicit_strategy` when one is, which matters because the
reason field is what distinguishes a chosen strategy from an inherited default.

`peak_backward_scratch_bytes` is the one field in that record that cannot be
quoted without its circuit, and the reason is stronger than "it scales with the
circuit": it is a measured allocator high-water mark that is not monotone in wire
count or gate count. Measured with the parameter values above and each circuit's
observables listed with it:

```
  circuit                                    peak_backward_scratch_bytes
  2w ry(0) ry(1) cx(0,1)          Z0,Z1                         64
  2w ry(0) ry(1)                  Z0,Z1                          0
  3w ry(0) ry(1) ry(2) cx(0,1)    Z0,Z1,Z2                     128
  3w ry(0) ry(1) ry(2) cx(0,1) cx(1,2)   Z0,Z2                   0
  3w rx(0) ry(1) rz(2) cx(0,1) cx(1,2)   Z0,Z2                   0
  3w h(0) ry(2) rxx(0,2) rz(1)    Z2                           64
  4w ry(0..3)                     Z0..Z3                         0
  6w ry(0..5)                     Z0..Z5                         0
```

The `3w ry cx cx` row and the `3w ry cx` row are the decisive pair: adding a
second entangling gate takes the measured high-water mark from `128` to `0`,
which no monotone reading of "scratch" predicts. Both runs produce byte-identical
checkpoint records —
`strategy='reversible_adjoint'`, `selection_reason='forward_state_reuse_fits_budget'`,
`estimated_local_state_bytes=128`, `estimated_required_bytes=640`,
`estimated_compact_reversible_bytes=0`, `low_memory_cpu_cx=False` — so the plan
side cannot be used to derive the measured number, and the number must always be
quoted with the exact gate list it came from. The existing assertion at
`tests/unit/test_statevector_reverse.py:640`
(`assert completed["peak_backward_scratch_bytes"] > 0`) is true for its own
circuit, which measured `64`, and is not a universal property of the adjoint
route: a `complex128` three-wire run measured `0`.

The same field also reads `0` before `backward()` is called, together with
`parameter_gradient_ready=False` and `backward_status='pending'`, so the
pre-backward summary is a distinguishable pending state rather than a second
reading of the result; every scratch figure above was read after `backward()`.
The kernel dispatch record from the same three-wire run shows the backward ran on
eager PyTorch with no Triton execution and no silent substitution:

```
{
  "decisions": [
    {
      "feature": "local_reversible_vjp",
      "selected": "pytorch",
      "reason": "disabled_by_policy",
      "device_runtime": {"provider": "pytorch", "device_type": "cpu"},
      "kernel_compiler": null,
      "kernel_route": {
        "semantic_id": "gradient.vjp.reversible_1q.local",
        "implementation": "pytorch_eager",
        "integration_path": "pytorch",
        "fallback": false,
        "implementation_id": null,
        "catalog_mismatches": ["device", "dtype"]
      },
      "count": 3
    }
  ],
  "triton_execution_count": 0,
  "pytorch_fallback_count": 3
}
```

One field in that record is a trap for a contract. `kernel_route.fallback` is
`false` while `pytorch_fallback_count` is `3`. The counter is summed over
selected routes whose chosen implementation is the eager PyTorch one; it is not
a count of substitutions. A caller reading `pytorch_fallback_count = 3` will
conclude that three planned fast paths were replaced, when the record's own
`fallback` field says none were. This proposal names the reading that is true
and leaves the counter alone, because renaming a serialized evidence field is a
separate change with its own compatibility analysis.

### 5. The observable domain is Z and ZZ, and every other shape is refused by name

`_normalize_z_observable_terms` (`flagquantum/runtime/executors/statevector/reverse.py:115`)
is the gate. Measured, on a two-wire circuit with a genuinely trainable angle
present so that the parameter-layout refusal cannot mask the observable refusal:

```
  three-wire term                -> ValueError: statevector adjoint supports only Z and ZZ terms
  repeated wire                  -> ValueError: an observable term cannot repeat a wire
  wire outside circuit           -> ValueError: observable term wire is outside the circuit
  empty terms                    -> ValueError: observable_terms must contain at least one Z or ZZ term
  NaN coefficient                -> ValueError: observable coefficients must be finite
  both wire spellings            -> ValueError: provide observable_wires or observable_terms, not both
  inf coefficient                -> ValueError: observable coefficients must be finite
  observable_wires out of range  -> ValueError: observable term wire is outside the circuit
```

The trainable-tensor gate is separate and also named
(`flagquantum/runtime/executors/statevector/reverse.py:111`,
`"sharded reverse mode requires at least one trainable tensor"`). Measured on
the same circuit with a detached bound tensor:

```
=== C. bound tensor detached (no requires_grad) ===
  ValueError sharded reverse mode requires at least one trainable tensor
```

Both spellings of a trainable angle are accepted, which the contract must
record because only one of them is used in the existing tests. Measured:

```
=== A. trainable tensor placed directly as the gate angle ===
   OK  value= 0.917120822816605  grad= -0.3986093279844229
=== B. the same angle routed through fq.Parameter + bind_parameters with a requires_grad tensor ===
   OK  value= 0.917120822816605  grad= -0.3986093279844229
```

### 6. Three independent routes agree today, and no document states the tolerance or the worst measured margin

Measured on `fq.Circuit(3, dtype=...)` built as
`.rx(0, t0).ry(1, t1).rz(2, t2).cx(0, 1).cx(1, 2)` with
`(t0, t1, t2) = (0.41, -0.37, 0.83)` and `observable_terms=[(1.0, (0,)), (1.0, (2,))]`,
comparing the public `fq.run` route, the adjoint reverse executor route, and
`flagquantum.gradients.parameter_shift_gradient`. The loss is the arithmetic sum
of the two requested expectations, formed identically for all three routes:

```
=== complex dtype torch.complex128 / param dtype torch.float64 ===
  value: public 1.7721776451532325 adjoint 1.7721776451532327 diff 2.220446049250313e-16
  public   gradient: [-0.770243704677945, 0.33164504250688803, 0.0]
  adjoint  gradient: [-0.770243704677945, 0.33164504250688803, 0.0]
  pshift   gradient: [-0.7702437046779449, 0.3316450425068881, 0.0]
  worst |public-adjoint|: 0.0
  worst |adjoint-pshift|: 1.1102230246251565e-16
  worst |public-pshift|: 1.1102230246251565e-16
=== complex dtype torch.complex64 / param dtype torch.float32 ===
  value: public 1.7721774578094482 adjoint 1.7721776962280273 diff 2.384185791015625e-07
  public   gradient: [-0.7702436447143555, 0.33164504170417786, 5.960464477539063e-08]
  adjoint  gradient: [-0.7702436447143555, 0.33164504170417786, 0.0]
  pshift   gradient: [-0.7702437043190002, 0.33164501190185547, 5.960464477539063e-08]
  worst |public-adjoint|: 5.960464477539063e-08
  worst |adjoint-pshift|: 5.960464477539063e-08
  worst |public-pshift|: 5.960464477539063e-08
```

In `complex128` the public and adjoint gradients are bit-identical on this
fixture and the parameter-shift route differs by one unit in the last place; the
`complex64` routes differ at the `5.960464477539063e-08` level, which is the
resolvable step of that dtype near `0.33` and not a disagreement about
arithmetic. The third component is exactly `0.0` for the adjoint route in both
dtypes because an `rz` rotation commutes with a Z observable, which decision 6
addresses separately rather than treating as a broken gradient.

The three routes were invoked as follows, and the parameter-shift spelling is
recorded because it is not a root export and its signature is three positional
arguments:

```python
# public route
result = fq.run(circuit, options=fq.ExecutionOptions(require_gradients=True),
                outputs=[fq.expectation(fq.Z(0), name="z0"),
                         fq.expectation(fq.Z(2), name="z2")])
(result.expectation("z0") + result.expectation("z2")).backward()

# adjoint route
gradient_result = execute_torch_distributed_statevector_reverse(
    circuit, observable_terms=[(1.0, (0,)), (1.0, (2,))],
    checkpoint_policy=StatevectorCheckpointPolicy(strategy="reversible_adjoint"))
gradient_result.backward()

# parameter-shift route, from flagquantum.gradients
parameter_shift_gradient(builder, torch.tensor([0.41, -0.37, 0.83], dtype=pd), loss)
```

with `builder(params)` rebuilding the same circuit from a parameter tensor and
`loss(circuit)` returning the same sum of the two expectations. Its presence in
`flagquantum.gradients.__all__`
(`['batched_parameter_shift_gradient', 'parameter_shift_gradient']`) is a
namespace export and not a Stable Core export: `hasattr(fq,
'parameter_shift_gradient')` measured `False`, so the reference route this
contract compares against is itself only reachable through a non-root import.

The small fixtures above are not the worst case, because a single-qubit
observation can be satisfied by a shortcut. The same comparison on an
eight-wire, six-parameter circuit measured:

```
=== dtype=torch.complex128 ===
  adjoint gradient: [-0.21557943223654327, 0.2651193380136144, 1.3358520797068208e-17, 0.9848185375519395, -0.16389537131333418, 0.0]
  pshift  gradient: [-0.21557943223654297, 0.265119338013615, 0.0, 0.9848185375519393, -0.16389537131333398, 0.0]
  component diffs: [3.0531133177191805e-16, 5.551115123125783e-16, 1.3358520797068208e-17, 2.220446049250313e-16, 1.942890293094024e-16, 0.0]
  max |adjoint - pshift|: 5.551115123125783e-16
  adjoint value: 3.6719233468698627
=== dtype=torch.complex64 ===
  adjoint gradient: [-0.21557952463626862, 0.2651194930076599, -2.302127057873804e-07, 0.9848188161849976, -0.1638953983783722, 0.0]
  pshift  gradient: [-0.21557950973510742, 0.26511967182159424, 3.5762786865234375e-07, 0.9848186373710632, -0.16389548778533936, -3.5762786865234375e-07]
  component diffs: [1.4901161193847656e-08, 1.7881393432617188e-07, 5.878405744397242e-07, 1.7881393432617188e-07, 8.940696716308594e-08, 3.5762786865234375e-07]
  max |adjoint - pshift|: 5.878405744397242e-07
  adjoint value: 3.671924114227295
```

built as `fq.Circuit(8, dtype=...)` with `.rx(0, t0).ry(1, t1).rz(2, t2).rx(4, t3).ry(6, t4).rz(7, t5)`
and `.cx(0, 1).cx(2, 3).cx(4, 5).cx(6, 7).cx(1, 2).cx(5, 6)`, parameters
`(0.11, -0.27, 0.83, -0.52, 0.19, -0.66)`, and observable terms
`[(1.0, (0,)), (1.0, (2,)), (1.0, (4,)), (1.0, (6,))]`, with the loss formed as
the explicit left-to-right sum of the four expectations on both routes. The
worst component difference over all six parameters is therefore
`5.551115123125783e-16` for `complex128` and `5.878405744397242e-07` for
`complex64`.

Two features of that block matter for the tolerance. The two wires the
observable does not touch (parameters 3 and 5) carry a difference of `0.0` in
`complex128`, so the comparison is not uniformly noisy. The dominant
`complex64` difference is on parameter 2, whose true value is exactly `0.0`:
the adjoint route reports `-2.302127057873804e-07` and the parameter-shift route
reports `+3.5762786865234375e-07`, and the `5.878405744397242e-07` gap between
them is almost entirely the disagreement of two float32 evaluations of a zero.
That is the honest shape of the `complex64` number: the tolerance is spent on a
mathematically zero component, not on a component that either route gets wrong
by a visible amount. It also means the `complex64` worst case moves with the loss
reduction order, because a different summation order shifts both routes' rounding
of a zero; an independent run of the same block using `sum(expectations)` in
place of the explicit chain measured `5.878405744397242e-07` for the same
component and `1.7881393432617188e-07` for the same maximum, so the figure is
stable at the order of the tolerance and not at the last bit.

Against the same measured gradient vector
`[(-0.770243704677945+0j), (0.33164504250688803+0j), 0j]`, each plausible wrong
rule is an exact complex multiplier on the true derivative, so its predicted
gradient is `ratio * measured`. Measured:

```
measured adjoint gradient: [(-0.770243704677945+0j), (0.33164504250688803+0j), 0j]

  generator factor +0.5j instead of -0.5j (sign flipped)   ratio=(-1+1.2246467991473532e-16j) predicted=['0.770244-0.000000j', '-0.331645+0.000000j', '-0.000000+0.000000j']
                                                           max margin = 1.54048740935589
  generator factor -1.0j instead of -0.5j (doubled)        ratio=(2+0j)                     predicted=['-1.540487+0.000000j', '0.663290+0.000000j', '0.000000+0.000000j']
                                                           max margin = 0.770243704677945
  generator factor dropped, i.e. +1.0                      ratio=(-0+2j)                    predicted=['0.000000-1.540487j', '-0.000000+0.663290j', '-0.000000+0.000000j']
                                                           max margin = 1.7223172829011577
  adjoint seed ket*w instead of 2*ket*w (halved)           ratio=(0.5+0j)                   predicted=['-0.385122+0.000000j', '0.165823+0.000000j', '0.000000+0.000000j']
                                                           max margin = 0.3851218523389725
  adjoint seed conjugated                                  ratio=None                       predicted=['-0.770244-0.000000j', '0.331645-0.000000j', '0.000000-0.000000j']
                                                           max margin = 0.0
```

The smallest of those margins is `3.851218523389725e-01`, eight and a half
orders of magnitude above the `1e-9` tolerance this proposal sets for
`complex128`.

The last row is a finding about the evidence rather than about the
implementation. Conjugating the adjoint seed is not detectable here: within the
Z/ZZ observable domain the expectation of a real-angle circuit is real, so the
entire backward chain is real and conjugation is a symmetry of the result. A
tolerance computed only over Z/ZZ observables and real angles therefore cannot
catch a conjugated seed, and this proposal does not claim that it does. Any
evidence item that purports to close that hole must say which fixture makes the
intermediate complex; the Z/ZZ, real-angle fixture used here does not.

The value each wrong-rule margin is measured against is the seed the
implementation actually uses, which is literal in Simulation. Measured:

```
$ grep -n '' flagquantum/simulation/statevector/adjoint.py | sed -n '56,67p;85,87p'
56:def z_expectation_adjoint_chunk(
57:    amplitudes: torch.Tensor,
58:    global_indices: torch.Tensor,
59:    *,
60:    n_wires: int,
61:    wire: int,
62:) -> torch.Tensor:
63:    """Return one amplitude chunk's derivative of a Z expectation."""
64:
65:    bit = (global_indices >> (n_wires - wire - 1)) & 1
66:    signs = (1 - 2 * bit).to(dtype=amplitudes.real.dtype)
67:    return 2 * amplitudes * signs.reshape(1, -1)
85:    shaped = weights.reshape(1, -1)
86:    value = (amplitudes.abs().square() * shaped).sum()
87:    return value, 2 * amplitudes * shaped
```

The `2` in `2 * amplitudes * signs` (`adjoint.py:67`) and `2 * amplitudes * shaped`
(`adjoint.py:87`) is the factor the fourth wrong rule removes, so that rule is a
statement about a specific literal in this file rather than about a hypothetical.

The observable-isolation control that keeps the Z/ZZ shortcut honest was also
measured, on `fq.Circuit(2, dtype=...).rx(0, t0).ry(1, t1).cx(0, 1)` with
`(t0, t1) = (0.41, -0.37)`:

```
=== observable isolation: only Z0 is observed ===
  dtype=torch.complex128  gradient=[-0.3986093279844229, 0.0]  value=0.9171208228166051
  dtype=torch.complex64  gradient=[-0.39860936999320984, 0.0]  value=0.9171208143234253
=== the same circuit with both Z0 and Z1 observed ===
  dtype=torch.complex128  gradient=[-0.770243704677945, 0.33164504250688803]  value=1.7721776451532327
  dtype=torch.complex64  gradient=[-0.770243763923645, 0.33164504170417786]  value=1.7721775770187378
=== weighted ZZ term, coefficient 0.5 ===
  dtype=torch.complex128  gradient=[0.0, 0.180807715982481]  value=0.46616367280301724
  dtype=torch.complex64  gradient=[0.0, 0.18080772459506989]  value=0.466163694858551
```

Observing only `Z0` makes the second component exactly `0.0` in both dtypes,
and observing both makes it non-zero; the weighted `ZZ` term moves the gradient
of the entangling pair whose single-wire `Z0` contribution is exactly `0.0`.
A run that reports the two-observable gradient while only `Z0` was requested
would therefore be detectable, which is the property the checklist item on
observable isolation asks for.

Every number in this section was produced in an eager, non-traced,
non-scripted run. The literal control line for the interpreter that produced
them is:

```
CONTROLS torch.jit.is_scripting()=False torch.jit.is_tracing()=False torch.get_num_threads()=4
torch.__version__ = 2.14.0
```

with one warmup iteration discarded before each timed fixture, and the
`UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to
unexpected behavior` that `float(tensor)` emits on this PyTorch version filtered
out of the quoted output rather than suppressed in the run.

The measurements also confirm the forward route is not itself the source of the
agreement, against an independent dense matrix-vector reference built from
Kronecker products and `torch.vdot` that calls neither the circuit state loop
nor the adjoint. Measured for `fq.Circuit(2, dtype=...)` as
`.ry(0, theta).ry(1, phi).cx(0, 1)`, `theta = 0.41`, `phi = 0.23`:

```
=== circuit dtype = torch.complex128 ===
  state dtype                        = torch.complex128
  max |state - dense_matvec|         = 0.000000e+00
  Z0     flagquantum=0.9171208228166051         dense=0.9171208228166051         abs_diff=0.000000e+00
  Z0Z1   flagquantum=0.9736663950053749         dense=0.9736663950053749         abs_diff=0.000000e+00
  X0     flagquantum=0.09087396745191441        dense=0.09087396745191441        abs_diff=0.000000e+00
  Y0     flagquantum=0.0                        dense=0.0                        abs_diff=0.000000e+00
  wrong rule 'cx dropped' on X0      = 0.39860932798442283  margin(vs X0)=3.077354e-01
  wrong rule 'angle*2' on X0         = 0.16668481560421827  margin=7.581085e-02
=== circuit dtype = torch.complex64 ===
  state dtype                        = torch.complex64
  max |state - dense_matvec|         = 6.808282e-09
  Z0     flagquantum=0.9171208143234253         dense=0.9171208228166051         abs_diff=8.493180e-09
  Z0Z1   flagquantum=0.973666407279508          dense=0.9736663950053749         abs_diff=1.227413e-08
  X0     flagquantum=0.09087397135200614        dense=0.09087396745191441        abs_diff=3.900092e-09
  Y0     flagquantum=0.0                        dense=0.0                        abs_diff=0.000000e+00
  wrong rule 'cx dropped' on X0      = 0.39860932798442283  margin(vs X0)=3.077354e-01
  wrong rule 'angle*2' on X0         = 0.16668481560421827  margin=7.581085e-02
```

`Z0` is read from `Circuit.expectation_z(0)`; `Z0Z1`, `X0`, and `Y0` are read
from the circuit's own state vector through the same `torch.vdot` reduction as
the reference, because the local API's `expectation_z` is Z-only and returns one
value per wire rather than a joint expectation. The reference is assembled with
`torch.kron(A, B)` placing `A` on wire `0` and `B` on wire `1`; that ordering is
not assumed but measured, by checking that
`<kron(Z, I)>` on a `.ry(0, t)` circuit equals `Circuit.expectation_z(0)`, which
holds for `t = 0.41` and is what makes the comparison a comparison rather than
two conventions agreeing with themselves. The two wrong rules are built from the
same reference: `cx dropped` deletes the `CX` and `angle*2` doubles `theta`
before the `CX`. For `complex128` every compared value is bit-identical to the
dense reference, including the state vector itself at
`max |state - dense_matvec| = 0.000000e+00`. For `complex64` the forward route
itself is only accurate to about `6.808282e-09` on the state and `8e-09` on the
observables, which is the precision floor of that dtype and is one of the
reasons the `complex64` agreement tolerance cannot be tighter than it is. The
two wrong-rule margins are identical in both dtypes because they are properties
of the reference construction, not of the circuit's dtype.

Note `fq.Circuit(2)` defaults to `dtype=torch.complex64` (the circuit dtype
default is read from the runtime config at `flagquantum/circuit.py:207`), so a
caller who does not pass `dtype` gets the `complex64` accuracy class whether or
not they asked for it.

### 7. The tolerances that exist are local, ad hoc, and mutually inconsistent

No document states the tolerance for adjoint-versus-parameter-shift agreement.
What exists is a scatter of per-test and per-tool constants. Narrowing the
search to the files that own the adjoint and the local gradient contract gives a
literal, complete list:

```
$ grep -rnE '(atol|rtol|abs|tol|tolerance)[^a-zA-Z]*=[^a-zA-Z]*[0-9]' tests/unit/test_statevector_reverse.py tests/unit/test_native_cpu_adjoint.py tests/unit/test_correctness_properties.py --include='*.py' | grep -i 'grad'
tests/unit/test_statevector_reverse.py:625:    assert float(theta.grad) == pytest.approx(float(dense_grad[0]), abs=2e-5)
tests/unit/test_statevector_reverse.py:626:    assert float(phi.grad) == pytest.approx(float(dense_grad[1]), abs=2e-5)
tests/unit/test_statevector_reverse.py:663:    assert float(theta.grad) == pytest.approx(parameter_shift, abs=1e-5)
tests/unit/test_statevector_reverse.py:664:    assert float(theta.grad) == pytest.approx(finite_difference, abs=2e-4)
tests/unit/test_statevector_reverse.py:697:    torch.testing.assert_close(theta.grad, expected[0], atol=1e-10, rtol=1e-10)
tests/unit/test_statevector_reverse.py:698:    torch.testing.assert_close(phi.grad, expected[1], atol=1e-10, rtol=1e-10)
tests/unit/test_statevector_reverse.py:1147:        torch.testing.assert_close(parameter.grad, reference, atol=3e-5, rtol=3e-5)
tests/unit/test_native_cpu_adjoint.py:1557:        actual_gradients, torch.stack(expected_gradients), atol=1e-11, rtol=1e-11
tests/unit/test_native_cpu_adjoint.py:1604:    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)
tests/unit/test_native_cpu_adjoint.py:1645:    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)
tests/unit/test_native_cpu_adjoint.py:1684:    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)
tests/unit/test_native_cpu_adjoint.py:1723:    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)
tests/unit/test_native_cpu_adjoint.py:1789:        parameters.grad, expected_gradient, atol=2e-12, rtol=2e-12
tests/unit/test_native_cpu_adjoint.py:1831:    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)
tests/unit/test_native_cpu_adjoint.py:1995:        gradients[1:], torch.zeros_like(gradients[1:]), atol=0, rtol=0
tests/unit/test_correctness_properties.py:87:    assert float(theta.grad) == pytest.approx(expected, abs=1e-5)
```

The same pattern is not confined to those three files: the unrestricted command

```
$ grep -rnE '(atol|rtol|abs|tol|tolerance)[^a-zA-Z]*=[^a-zA-Z]*[0-9]' tests --include='*.py' | grep -ic 'grad'
139
```

returns 139 gradient tolerance assertions across the test tree, with no shared
constant and no file that defines one.

Beyond the tests, the tool-level constants are:

```
flagquantum/benchmarking/differentiable_simulator_corpus.py:51:_ABSOLUTE_TOLERANCE = 1e-9
tools/validate_flagos_statevector_training.py:216:    tolerance = 6e-5 if dtype_name == "complex64" else 3e-10
tools/diagnose_flagos_statevector_reverse.py:108:    tolerance = 3e-5 if dtype_name == "complex64" else 3e-12
tools/diagnose_flagos_statevector_reverse.py:207:    tolerance = 6e-5 if dtype_name == "complex64" else 3e-10
```

Twelve distinct numbers bound an adjoint or gradient difference across those
assertions and tools — `1e-5`, `2e-5`, `2e-4`, `1e-10`, `3e-5`, `1e-11`, `2e-12`,
and `0` in the tests, and `1e-9`, `3e-10`, `3e-12`, `6e-5` in the tools — and
none of them is derived from a measured margin. The span is seven orders of
magnitude, from `6e-5` down to `3e-12`, and it is a span of numbers rather than a
span of requirements: nothing in the tree records which circuit, dtype, or
reference each number was calibrated against, so the difference between a
`complex64` and a `complex128` figure cannot be recovered from the constants
alone. In particular the two `complex64` figures disagree with each other — the
tests assert `abs=1e-5` at
`tests/unit/test_statevector_reverse.py:663` while
`tools/validate_flagos_statevector_training.py:216` and
`tools/diagnose_flagos_statevector_reverse.py:207` allow `6e-5`, a factor of six
— and the closest pair to a stated contract,
`tests/unit/test_statevector_reverse.py:663`'s `1e-5` against
`flagquantum/benchmarking/differentiable_simulator_corpus.py:51`'s `1e-9`, is not
one either: the first is a `complex64` assertion on a one-qubit circuit and the
second is a value-and-gradient band used to compare engines, so reading either as
the adjoint's tolerance is an inference no file states.
`tests/unit/test_statevector_reverse.py:653` compares the adjoint
against a hand-written closed form
`parameter_shift = (math.cos(0.41 + math.pi / 2) - math.cos(0.41 - math.pi / 2)) / 2`,
which is a correct reference for one gate on one wire and does not exercise an
entangling circuit, a second parameter, or the `complex64` class at all.

### 8. `analytic_rotation_derivative` degrades to `None` by name, but a matrix whose shape contradicts the opcode reaches `torch.matmul`

The primitive does validate something. Its return type is
`torch.Tensor | None` and it refuses an unsupported opcode or parameter tuple by
returning `None` (`flagquantum/simulation/statevector/adjoint.py:29-31`), which
its three call sites do handle
(`flagquantum/runtime/executors/statevector/reverse_adjoint_sweep.py:565`,
`:590`, `:992`). Measured by calling it directly and reporting the raw return
value, with no attribute access on the result:

```
  rx    wires=1 params={'theta': 0.1}                     returned Tensor shape=(2, 2) head=-0j
  ry    wires=1 params={'theta': 0.1}                     returned Tensor shape=(2, 2) head=-0j
  rz    wires=1 params={'theta': 0.1}                     returned Tensor shape=(2, 2) head=-0.5j
  rzz   wires=2 params={'theta': 0.1}                     RAISED RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x4 and 2x2)
  h     wires=1 params={}                                 returned None
  rx    wires=1 params={'theta': 0.1, 'phi': 0.2}         returned None
  u3    wires=1 params={'theta': 0.1}                     INSTRUCTION REFUSED IRValidationError: opcode 'u3' is missing parameter(s): phi, lbd
```

Three facts follow, and the third is the defect. First, `h` and a rotation whose
parameter tuple is not exactly `("theta",)` return `None` rather than raising;
that is a documented return value, not an accident. Second, `u3` never reaches
the function at all — Core's IR refuses to construct the `Instruction`, so that
refusal belongs to `flagquantum/core/ir/__init__.py` and not here. Third, `rzz` is
accepted by the opcode-and-params guard while its generator is `4x4`, so when the
matrix passed in is `2x2` the multiplication fails inside `torch.matmul` with
`RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x4 and 2x2)`. The same
call with a `4x4` matrix is correct:

```
  rzz wires=2 matrix 4x4 -> shape (4, 4) head -0.5j
```

The guard therefore tests the opcode name and the parameter tuple but not that
the matrix shape agrees with the opcode's wire count, and the failure surfaces as
a linear-algebra error from inside a library call rather than as a refusal that
names the contradiction. The `None` path is also silent in the evidence: a
caller reading `analytic_rotation_derivative_count` sees a smaller number and
nothing else, so a rotation that quietly stopped taking the analytic path and
started taking the JVP fallback at
`flagquantum/runtime/executors/statevector/reverse_adjoint_sweep.py:997` is
visible only as a counter that did not increment.

## Decision

### 1. The contract is stated over the existing modules; no second adjoint module is created

`flagquantum/simulation/statevector/adjoint.py` is the one authoritative
implementation of the local adjoint primitives, and
`flagquantum/runtime/executors/statevector/reverse.py` is the one authoritative
adjoint gradient boundary. A new file named
`flagquantum/simulation/statevector/statevector_adjoint.py` is refused: the row's
phrase names an artifact that does not exist, and creating it would put two
sources of truth for one arithmetic in one directory. The row's intent —
productionize the adjoint and wrap it — is satisfied by (a) giving
`adjoint.py`'s public names a documented reachability path, and (b) documenting
the wrapper that already exists, rather than by adding a file whose only
distinguishing feature is its name. Recorded here so the plan row can be read
against the tree.

### 2. One named tolerance per dtype, derived from the measured margin

The adjoint gradient must agree with the parameter-shift gradient within
`1e-9` absolute for `complex128` parameters and `1e-5` absolute for `complex64`
parameters, compared componentwise on the same circuit, the same observable
terms, and the same parameter values. Each number is chosen from the measured
margin, not from taste:

- `complex128`: the worst measured component difference across the fixtures in
  premise 6 is `5.551115123125783e-16`, which leaves more than six orders of
  magnitude of headroom under `1e-9`, while the nearest wrong rule misses by
  `3.851218523389725e-01` — eight and a half orders of magnitude above the
  tolerance.
- `complex64`: the worst measured component difference is
  `5.878405744397242e-07`, which leaves a factor of about `17` under `1e-5` —
  roughly one order of magnitude, not the comfortable two that a dtype-only
  argument would assume — while the nearest wrong rule misses by
  `3.851218523389725e-01`, four and a half orders of magnitude above it. The
  headroom is thin enough that the tolerance must not be tightened to `1e-6`
  without re-measuring, and it is stated with the circuit and the loss order it
  was measured on, because premise 6 shows the `complex64` figure is set by the
  rounding of a mathematically zero component and therefore moves with the loss
  reduction order.

`1e-5` for `complex64` is the value already in force at
`tests/unit/test_statevector_reverse.py:663`, where the adjoint is compared
against a closed-form parameter shift on `fq.Circuit(1).ry(0, theta)` — a
default-`complex64` circuit with a `float32` parameter, which is why the citation
is a `complex64` citation. `1e-9` for `complex128` is the value already in force
at `flagquantum/benchmarking/differentiable_simulator_corpus.py:51`, where the
same constant is applied to `value_error` and `gradient_error` together
(`:662-663`) and serialized as `"absolute_tolerance"` (`:750`); it is a
corpus-level engine-comparison band rather than an adjoint-versus-parameter-shift
assertion, and the contract adopts its value rather than its scope. The contract
therefore adopts two values that are already relied on rather than inventing a
third and fourth, and it deliberately does **not** adopt the stricter `1e-10`
at `tests/unit/test_statevector_reverse.py:697` as the contract for a route the
caller cannot yet identify, or the `2e-12` at
`tests/unit/test_native_cpu_adjoint.py:1604`, which is a statement about a
different, extension-backed boundary. Loosening any route that already asserts
a stricter value is out of scope: this contract sets a floor for the adjoint
comparison, and it does not authorise relaxing an existing assertion.
`complex128` is the accuracy class the contract is stated for; a `complex64`
agreement is an accuracy statement about that dtype and not evidence for the
`complex128` one, and the reverse is equally true.

### 3. The gradient route must be identifiable from the result, and no route may be substituted silently

Today the public `fq.run(...)` route and the adjoint route compute the same
observable by different arithmetic and neither the result nor the plan says
which ran (premise 3). This is refused as a contract: a gradient assertion is
meaningless against a route the caller cannot name. The contract therefore
requires that a differentiable execution record its gradient method in the
execution evidence, using the vocabulary the reverse executor already uses —
`TorchDistributedStatevectorGradientResult.summary()` already returns
`"gradient_method": "statevector_adjoint"`
(`flagquantum/runtime/executors/statevector/reverse.py:197`) — and that the
name it reports is the arithmetic that actually ran. A record that reports
`statevector_adjoint` while the dense local loop produced the gradient, or the
reverse, is a defect of the same class as an unlabelled fallback, and
engineering decision principle 9 forbids it.

This decision deliberately does not require the public route to *use* the
adjoint. Choosing the adjoint for a local circuit would be a performance
change with no correctness content, and the local dense route is a first-class
fast path under non-negotiable rule 2. What is required is that whichever route
runs is named.

### 4. The observable domain stays Z and ZZ, and widens only by measurement

Weighted single-wire Z and two-wire ZZ terms are the contract. Everything else
stays refused by the named errors measured in premise 5, and the refusals move
to the earliest knowable stage rather than at `_normalize_z_observable_terms`
after planning has begun. Widening to X, Y, or arbitrary Pauli strings is
refused here: an X or Y term cannot be expressed by the Z/ZZ weight vector the
adjoint seeds with, and a term the implementer cannot express must be refused
by name rather than folded into the Z/ZZ structure and silently dropped. A
future widening needs its own measured evidence for the new terms, because the
seeding identity `2 * amplitudes * weights`
(`flagquantum/simulation/statevector/adjoint.py:67` and `:87`) is a statement
about the diagonal observable and not about a general one. A Y or X term has no
such weight vector, so the identity that carries the Z/ZZ gradient does not
transfer, and saying that it does would be a claim this proposal cannot measure.

### 5. `analytic_rotation_derivative` keeps its `None` return and refuses a shape contradiction by name

The `None` return is the primitive's existing contract, its three call sites
already branch on it, and this proposal keeps it. What changes is the case in
premise 8 where the opcode and the matrix disagree: a matrix whose shape does not
match the opcode's wire count must be refused with a message that names the
instruction, its wires, and the shape it received, instead of failing inside
`torch.matmul` with `mat1 and mat2 shapes cannot be multiplied (4x4 and 2x2)`.
A caller cannot distinguish that `RuntimeError` from a defect in the primitive,
and it names neither the instruction nor the field at fault, so it is refused as
contract behaviour. This is the only behaviour change this proposal asks for in
the primitive: the accepted set is unchanged, the `None` cases keep returning
`None`, and no gate is added to the rotation set.

The silent `None` is a second, smaller requirement. The primitive itself need
not report anything, but the caller that decides between the analytic path and
the JVP fallback must record which one it took, because
`analytic_rotation_derivative_count` is a count and not a verdict: a rotation
that stopped being analytic is visible today only as a counter that did not
increment. Engineering decision principle 9 permits an approximation or a
substitution only when it is recorded, and a fallback from the analytic
derivative to a JVP is exactly such a substitution. The tolerance contract in
decision 2 is not extended to a gate the primitive does not implement; the JVP
fallback and the analytic path answer different questions and a single number
must not be read as covering both.

### 6. `rz` stays in the domain even though it can be exactly zero, and no result may be inferred from a zero

`rz` rotations commute with the Z/ZZ observable, so an `rz` angle that acts
before only Z-basis measurements legitimately has a zero gradient — measured as
the third component of `[-0.770243704677945, 0.33164504250688803, 0.0]` in
premise 6. The contract states that a zero component is a value and not an
absence, and that no test may treat a zero as evidence that the adjoint ran.
`tests/unit/test_native_cpu_adjoint.py:1995` asserts
`torch.testing.assert_close(gradients[1:], torch.zeros_like(gradients[1:]), atol=0, rtol=0)`,
which is a correct zero assertion for a real zero; the contract requires the
accompanying positive assertion on a component that is known non-zero, so that a
uniformly-zero gradient cannot pass.

### 7. Adjoint gradients stay out of the Stable Core and add no public name

No root export is added, no `docs/public_api_v1.json` entry is added, and no
signature, default, or serialized field in the Stable Core changes. `adjoint.py`
gains a documented reachability path inside the Simulation namespace it already
lives in; the zero new public name count is the same posture
`API_CHANGE_PROPOSAL_065` took for a different row. A user-facing
`fq.adjoint_gradient`-shaped entry point is refused here because the arithmetic
is selected by execution policy rather than called directly, and exposing it as
a callable would create a second way to obtain one gradient with no rule
choosing between them.

### 8. Higher-order autograd and `torch.func` transforms are refused

The wrapper's `backward` is a hand-written adjoint that recomputes from saved
state and does not build a differentiable graph over its own inputs. It is not
marked, so the current failure is PyTorch's rather than FlagQuantum's. Measured
on the same one-qubit adjoint fixture, asking for a graph on the first
derivative:

```
first derivative: tensor(-0.3986, dtype=torch.float64) requires_grad: False grad_fn: None
second derivative REFUSED: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The first derivative is detached, so the second attempt fails with a message
that names no FlagQuantum concept, no capability, and no remedy; a caller cannot
tell from it whether higher-order gradients are unsupported by design or broken
by accident. The contract refuses a second-order gradient claim:
`grad(grad(...))`, `torch.autograd.functional.jacobian` over the adjoint
boundary, and `torch.func` transforms over it. The wrapper must be marked
double-backward-incapable with the honest marker (`once_differentiable`-style
refusal) so that the refusal is a statement rather than an accident, rather than
being left to produce a silent wrong second derivative. Every existing
higher-order path in the tree that works —
`flagquantum/kernels/triton/single_qubit_loop.py:38`,
`flagquantum/algorithms/optimization.py:41` — works through
`torch.autograd.functional.jacobian` over ordinary differentiable tensor
operations, which is a different mechanism and is not claimed here.

### 9. The capability matrix is not raised by this proposal, and this proposal states no maturity claim

`capability-maturity.toml` records `gradient_support = "exact"` for
`local_statevector` (line 53) and for `sharded_statevector_training` (line 73),
both at level `production_supported`. This proposal does not change either
entry, does not add a `gradient_support` value, and does not assert a maturity
level. The vocabulary is already wide and inconsistent, and adding a
`gradient_support` value for the adjoint would be a vocabulary change requiring
its own approval rather than a consequence of this contract. Measured over the
51 capabilities:

```
  1  adapter_defined
  1  bound_parameters_only
  1  development_evidence
  1  development_evidence_exact_autograd
  3  exact
  1  exact_autograd_through_the_integrators
  1  experimental
  1  experimental_composed_primitives
  1  explicit_parameter_shift_double_single_optimizer_experimental
  1  extension_defined
  1  first_order_parameter_shift_internal_double_single_float32_delivery_experimental
  1  forward_only_development_evidence
 17  not_applicable
  1  parameter_shift_device_double_single_experimental
  1  parameter_shift_experimental
  1  parameter_shift_full_double_single_experimental
  1  parameter_shift_selective_double_single_experimental
  1  symbolic_parameters_only
 15  unsupported
distinct: 19
```

Only three capabilities carry the bare `exact` value, and `local_statevector`
and `sharded_statevector_training` are two of them; the rest of the vocabulary
mixes support levels with implementation identifiers, including
`first_order_parameter_shift_internal_double_single_float32_delivery_experimental`,
which names a dtype, a delivery path, and a maturity level in one string. A
proposal cannot raise a capability maturity level, and it should not deepen that
vocabulary as a side effect; the acceptance checklist below measures, and the
measurement is reported to the owner.

## Public API

No public name is added, removed, renamed, reordered, or retyped. Measured:

```
policy_version: 1.0-alpha package: flagquantum
stable_exports type: list len: 34
  stable_exports contains 'adjoint': False
  stable_exports contains 'gradient': False
  stable_exports contains 'reverse': False
  stable_exports contains 'Reverse': False
  stable_exports contains 'expectation': True
  stable_exports contains 'training': False
  exports: ['Circuit', 'CircuitIR', 'ExecutionOptions', 'ExecutionPlan', 'ExecutionResult', 'I', 'IRSerializationError', 'IRValidationError', 'IR_VERSION', 'Instruction', 'MeasurementResult', 'Module', 'Observable', 'OutputRequest', 'Parameter', 'ParameterExpression', 'RuntimePolicy', 'TrainingResult', 'X', 'Y', 'Z', '__version__', 'compile', 'counts', 'expectation', 'experimental', 'plan', 'probabilities', 'restore_job', 'run', 'samples', 'submit', 'train', 'twin']
candidate keys: ['schema_version', 'status', 'target', 'source_manifest', 'approval', 'stable_core', 'stable_extensions', 'approved_namespace_additions', 'experimental', 'remove_before_public', 'pre_public_renames', 'rules']
  candidate contains 'adjoint': False   raw-lines: []
  candidate contains 'gradient': False   raw-lines: []
  candidate contains 'reverse': False   raw-lines: []
  candidate contains 'Reverse': True   raw-lines: [195]
  candidate contains 'expectation': True   raw-lines: [37, 73, 83, 169, 181, 182]
  candidate contains 'training': True   raw-lines: [140]
  candidate contains 'parameter_shift': False   raw-lines: []
$ sed -n '140p;195p' contracts/public-api-v1-candidate.json
      "namespace": "flagquantum.training",
        "MPSReverseCheckpointPolicy",
```

`docs/public_api_v1.json` holds 34 stable exports, and none of its 34 members
matches `adjoint`, `gradient`, `reverse`, or `Reverse`: all four checks are
`False`. The only `gradient` and `adjoint` occurrences in
`contracts/public-api-v1-candidate.json` are zero lines, checked
case-insensitively; the single `Reverse` hit is `MPSReverseCheckpointPolicy` at
line 195, which is an MPS checkpoint field and not a statevector adjoint
surface, and `flagquantum.training` at line 140 is a namespace entry, not a
gradient surface. Both contract files are therefore unchanged by this proposal,
which is the same finding `API_CHANGE_PROPOSAL_065` reported for its own entry
points.

The names this proposal touches are the six in
`flagquantum/simulation/statevector/adjoint.py`'s `__all__`
(`analytic_rotation_derivative`, `real_conjugate_inner_sum`,
`z_expectation_adjoint_chunk`, `z_expectation_chunk`, `z_hamiltonian_chunk`,
`z_hamiltonian_weights`) and the six in
`flagquantum/runtime/executors/statevector/reverse.py`'s `__all__`
(`ParameterGradientOwnership`, `BackwardExecutionEvidence`,
`StatevectorCheckpointPolicy`, `TorchDistributedStatevectorGradientResult`,
`execute_torch_distributed_statevector_reverse`, `resolve_checkpoint_policy`).
All are unchanged in name, signature, and default. The reachability path the
contract adds is a documented import inside the Simulation namespace, not a root
export. The new public name count is zero.

The contract's user-visible surface is the tolerance and the recorded gradient
method, and neither is a new callable. The user code below is unchanged from
today; what changes is the evidence beside it, which is what makes the tolerance
a statement about a known route rather than about whichever arithmetic the
selection rule happened to pick. This block documents the surface the acceptance
items must produce, not the behaviour of the current tree:

```python
import torch
import flagquantum as fq

theta = torch.tensor(0.41, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta)

result = fq.run(
    circuit,
    options=fq.ExecutionOptions(require_gradients=True),
    outputs=fq.expectation(fq.Z(0), name="z0"),
)
value = result.expectation("z0")
value.backward()
# The execution evidence names the arithmetic that produced this gradient, so
# the agreement tolerance is a statement about a known route. Measured today on
# this fixture: value 0.917120822816605, gradient -0.3986093279844229, against
# the closed form -sin(0.41) = -0.3986093279844229. The adjoint route was
# measured on the same fixture with the same pair of numbers, so the assertion
# below holds before and after the route is named; what the contract adds is
# that the name is present, not that the number changes.
assert abs(float(theta.grad) - (-0.3986093279844229)) <= 1e-9
```

## Compatibility

- The public gradient route is not re-routed. `fq.run(circuit, outputs=fq.expectation(...))`
  keeps the arithmetic it has today; this proposal adds a record of which
  arithmetic that is. No existing result, plan, or serialized payload changes
  meaning.
- The adjoint path's accepted domain is unchanged: the same Z/ZZ terms, the same
  errors with the same messages. Refusals that premise 5 measured move to an
  earlier stage but keep their text, so a caller matching on the message is not
  broken.
- Accepting a `Parameter`-bound trainable tensor as well as a directly placed
  one is not a new behaviour; it is already true (premise 5, case B) and this
  proposal records it rather than changing it.
- The kernel dispatch record keeps its field names. `pytorch_fallback_count`
  is documented with the reading that is true for it — a count of selected
  eager-PyTorch routes, not a count of substitutions — and is not renamed. A
  rename is a serialized-evidence change and needs its own proposal.
- `docs/public_api_v1.json`, `contracts/public-api-v1-candidate.json`,
  `capability-maturity.toml`, and every serialized plan and result schema are
  unchanged. The new public name count is zero, so no deprecation window is
  owed.
- A caller who relied on `analytic_rotation_derivative` raising
  `RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x4 and 2x2)` when
  the matrix shape contradicts the opcode is broken deliberately, because that
  exception is not a contract, it does not name the instruction, and the
  replacement does. This is recorded as a behaviour change rather than presented
  as a pure addition. The `None` return for an unsupported opcode or parameter
  tuple is not part of that change: it is the existing contract and it stays.

## Required evidence before this proposal can be accepted

Each item names the measurement, not the intention. A green suite does not show
that a guard is exercised, so each refusal is additionally proven by deleting it
and observing the test fail.

- [ ] The three-route fixture in premise 6 is reproduced with the executed
      gradient method read back from the result evidence, and the adjoint
      route's worst component difference from the parameter-shift route is
      reported as a number for `complex128` and for `complex64`. The observed
      values must be at or below `5.551115123125783e-16` and
      `5.878405744397242e-07` respectively, each quoted with the circuit, the
      parameter values, and the loss reduction order it was measured on, or the
      increase must be explained before the tolerance is accepted. Because the
      `complex64` margin under `1e-5` is a factor of about `17` rather than two
      orders, an increase in that number is a tolerance failure and not noise.
- [ ] The same fixture under each detectable wrong rule in premise 6 is shown to
      fail the `1e-9` / `1e-5` tolerance, so the tolerance is evidence rather
      than decoration. The smallest margin over all detectable wrong rules must
      be reported, and it must exceed the tolerance by at least three orders of
      magnitude. The conjugated-seed rule is excluded from this item unless the
      run names a fixture in which the backward intermediate has a non-zero
      imaginary part; the Z/ZZ, real-angle fixture measured in premise 6 does
      not, and claiming that rule as caught without such a fixture would be a
      false statement about the evidence.
- [ ] The gradient method reported in the execution evidence is the arithmetic
      that actually ran, proven by a run in which the reverse executor is
      instrumented and its call count is compared against the reported method.
      A run that reports `statevector_adjoint` while the dense local loop
      produced the gradient must fail this item.
- [ ] The adjoint route's forward and backward execution counts are re-measured
      on a circuit of at least three wires with at least two entangling gates,
      and the recorded numbers are `1` forward and `0` additional forwards
      during backward, with `backward_uses_full_state_replay` false. Every
      `peak_backward_scratch_bytes` figure is quoted after `backward()` and
      together with the exact gate list and observable terms it was measured on,
      because the field is not monotone in wire count or gate count: the pair
      `3w ry(0) ry(1) ry(2) cx(0,1)` at `128` and
      `3w ry(0) ry(1) ry(2) cx(0,1) cx(1,2)` at `0` shows a second entangling
      gate lowering the measured peak while the checkpoint record is unchanged.
      A nonzero scratch figure must not be used as a proxy for "the adjoint
      ran", and neither must a zero one as evidence that it did not. A count
      that differs must be reported as a change in the contract, not absorbed.
- [ ] The adjoint route is compared against a **third** independent reference
      that calls neither the integrators under test nor the adjoint: a direct
      Kronecker-product matrix-vector construction with `torch.vdot`, at a
      tolerance derived from the measured difference between the two and not
      chosen in advance. The premise 6 forward check
      (`max |state - dense_matvec| = 0.000000e+00` for `complex128` and
      `6.808282e-09` for `complex64`) is the pattern; the gradient equivalent
      must be measured.
- [ ] Both spellings of a trainable angle — a `requires_grad` tensor placed
      directly as the gate angle, and the same tensor routed through
      `fq.Parameter` plus `bind_parameters` — are measured to produce the same
      gradient value bit for bit, and a detached bound tensor is measured to be
      refused with the text
      `sharded reverse mode requires at least one trainable tensor`.
- [ ] Every refusal in premise 5 is raised before any state is allocated, names
      the field it blames, and is proven by a mutation that deletes it and fails
      a test. The mutation count and the killed count are recorded.
- [ ] `analytic_rotation_derivative` is re-run on the seven measured inputs of
      premise 8 and the return value or exception type and text of each is
      recorded: `rx`/`ry`/`rz` unchanged, `rzz` with a `2x2` matrix refused by a
      message naming the instruction, its wires, and the received shape, `h` and
      the two-parameter `rx` still returning `None`, `rzz` with a `4x4` matrix
      still returning shape `(4, 4)`, and `u3` still refused by Core before the
      call. A mutation that removes the new shape check must fail a test.
- [ ] A test asserts a non-zero gradient component alongside the zero component
      measured for `rz`, and is proven by mutating the implementation to return
      a uniformly zero gradient and observing the test fail.
- [ ] A second-order gradient attempt over the adjoint boundary is measured to
      raise a refusal that names FlagQuantum's own reason, rather than the
      current `RuntimeError: element 0 of tensors does not require grad and does
      not have a grad_fn`: `torch.autograd.grad(g, t)` where `g` is the first
      derivative, `torch.autograd.functional.jacobian` over the boundary, and one
      `torch.func` transform over it. For each, the exception type and text are
      recorded, and a mutation that removes the `once_differentiable` marker
      fails a test.
- [ ] The wide-circuit and observable-isolation controls are all present in the
      recorded run that produced the tolerance: the forward-only value reported
      separately from the adjoint value, an eight-wire circuit that a
      single-wire shortcut cannot satisfy, and one fixture in which a requested
      observable makes a parameter's gradient exactly `0.0`. The premise 6 run
      recorded `gradient=[-0.3986093279844229, 0.0]` for the `Z0`-only
      observable against `gradient=[-0.770243704677945, 0.33164504250688803]`
      when `Z0` and `Z1` were both requested, and recorded
      `torch.jit.is_scripting()=False torch.jit.is_tracing()=False
      torch.get_num_threads()=4` on `torch 2.14.0` with one warmup iteration
      discarded; the accepted run must record the same controls.
- [ ] `docs/public_api_v1.json` still holds 34 stable exports, contains no
      `adjoint`, `gradient`, or `reverse` entry, and
      `contracts/public-api-v1-candidate.json` is unchanged; the comparison is
      recorded as a diff of the two files before and after.
- [ ] `capability-maturity.toml` is unchanged, its `gradient_support` vocabulary
      still has the 19 distinct values it has today, and `local_statevector`
      remains at line 53 with `gradient_support = "exact"` so no level moved.
- [ ] `python tools/ci_tier.py pr-runtime` and `python tools/ci_tier.py
      pr-default` pass, with the observed passed/failed/skipped counts recorded
      here. `python tools/ci_tier.py gpu-scheduled` is recorded as the tier the
      plan row declares, and if it cannot run in this environment the reason is
      stated rather than the row being reported as satisfied.
- [ ] `python -m mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum`
      reports exactly the two known pre-existing errors and no new one:
      `flagquantum/ecosystem/qiskit/conversion.py:624` and
      `flagquantum/runtime/executors/statevector/training.py:250`. The measured
      baseline on this checkout before any change is:

      ```
      flagquantum/ecosystem/qiskit/conversion.py:624: error: Incompatible types in assignment (expression has type "ndarray[tuple[Any, ...], dtype[Any]]", variable has type "Tensor | None")  [assignment]
      flagquantum/runtime/executors/statevector/training.py:250: error: Argument "group" to "monitored_barrier" has incompatible type "ProcessGroup | Literal[-100]"; expected "ProcessGroup | None"  [arg-type]
      Found 2 errors in 2 files (checked 598 source files)
      ```

- [ ] `flagquantum/simulation/statevector/adjoint.py`'s six public names have a
      documented reachability path inside the Simulation namespace, and a test
      imports them through that path rather than through
      `flagquantum.runtime.executors.statevector.*`; the import path is recorded.
- [ ] `python tools/check_repository_language.py` reports zero files containing
      Han-script text, and the count is recorded.
- [ ] Repository owner approves. Not yet requested; see `## Status`.

## Non-goals

- A file named `flagquantum/simulation/statevector/statevector_adjoint.py`. The
  plan row's phrase names an artifact that does not exist; the contract is
  stated over the module that does, and creating a second name for one
  arithmetic is refused.
- X, Y, or arbitrary Pauli-string observables through the adjoint, and
  multi-wire terms longer than two wires. The seeding identity is a statement
  about a diagonal observable, and each widening needs its own measured
  evidence.
- Selecting the adjoint automatically for a local circuit, or removing the
  dense local autograd fast path. Route selection with a performance motive is a
  separate change; this proposal only requires that the route that ran be named.
- Higher-order derivatives, `torch.func` transforms, and `jacobian` over the
  adjoint boundary.
- The accelerator and multi-rank adjoint paths. `W2-03` is declared
  `pr-runtime + gpu-scheduled`, and the Triton module
  `flagquantum/kernels/triton/statevector_adjoint.py` and the distributed
  sharded paths (`_explicit_sharded_adjoint`,
  `flagquantum/runtime/executors/statevector/reverse_adjoint.py:238`) are
  measured here only as reachable code; no distributed or accelerator
  correctness, performance, or capacity claim is made, and
  `scalability_claim_allowed` stays false.
- Renaming or removing `pytorch_fallback_count`, or changing any serialized
  evidence, plan, or result field. The misleading reading is documented; the
  rename is a separate compatibility change.
- A new `gradient_support` value, a capability maturity change, or any
  `docs/reference/KNOWN_LIMITATIONS.md` maturity wording. A proposal cannot
  raise a maturity level, and the checklist measures rather than promotes.
- Raising `flagquantum/benchmarking/differentiable_simulator_corpus.py:51`'s
  `_ABSOLUTE_TOLERANCE = 1e-9`, or any of the tool-level tolerances measured in
  premise 7. That constant is applied to value and gradient together at `:662-663`
  and serialized into the corpus payload at `:750`, so changing it changes a
  recorded payload field and not only a test band. The `1e-9` that appears in
  `flagquantum/benchmarking/differentiable_adjoint_report.py:96` is prose inside a
  formatted report string and is likewise not retyped here. Unifying any of these
  onto the contract's two values is a follow-up that must not silently narrow an
  existing tool's pass band.
- `W4-05`'s time-dependent generators and `W4-04`'s GPU and distributed
  batching. Both are named as excluded by `W4-06` and neither is in scope here.
