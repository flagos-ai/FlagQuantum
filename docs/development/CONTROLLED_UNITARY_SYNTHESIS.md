# Phase-exact recursive control synthesis

The native owner is `compiler/optimization/controlled_unitary_synthesis.py`.
It transforms one supplied concrete 2×2 unitary into native local matrix,
phase, CX and CCX instructions. It neither executes a state nor constructs a
whole-block matrix. Integer controls are ordered, positive and most-significant
before the target; zero activation wrapping remains a caller responsibility.
The input matrix and its absolute scalar phase are retained. No SDK object,
measurement slot or clean allocation is owned by this transform.

`borrowed_controlled_x_ir(controls, target, borrowed)` implements the companion
multi-X recursion. Borrowed wires can have arbitrary values and entanglement;
they are restored rather than assumed to start zero. All operands must be
distinct nonnegative native wire indices. `controlled_unitary_ir` rejects
nonfinite, nonunitary or non-2×2 inputs before emitting a recipe.

## Complete dependency mapping

This is the complete recursive-control dependency invoked by the locked
Qualtran v0.7.0 controlled-H path, not a replacement for that whole front end.
The immutable reference used for comparison is Cirq 1.7.0's
`transformers/analytical_decompositions/controlled_gate_decomposition.py`,
SHA256 `4c5e4d7b0faa0b28f55a65ea786cb869f5684a9d7a1a9698d56bc0a6ccbf0ce8`.
The original is Copyright 2020 The Cirq Developers, Apache-2.0.

| Source function | Native responsibility |
| --- | --- |
| `_unitary_power` | `_power`: principal eigenvalue powers and spectral reconstruction of the local unitary |
| `_is_identity` | Source allclose-to-identity removal at single-control leaves |
| `_flatten` | Ordered flattening of congruent-CCNOT blocks |
| `_decompose_abc` | `_abc`: determinant phase and three special-unitary factors; the phase is not discarded |
| `_decompose_single_ctrl` | `_single`: control phase, C, CX, B, CX, A, removing only source identity leaves |
| `_ccnot_congruent` | `_congruent`: original seven-instruction Ry/CX recipe including its relative sign |
| `decompose_multi_controlled_x` | `_multi_x` and public borrowed-wire entry: zero/one/two-control cases, both borrowed recursive constructions and the no-free-wire fallback |
| `_decompose_su` | Special-unitary branch of `controlled_unitary_ir`, retaining source factor and multi-X order |
| `_decompose_recursive` | `_recursive`: positive/negative half powers, intermediate borrowed-controlled X, recursive smaller control word |
| `decompose_multi_controlled_rotation` | `controlled_unitary_ir`: no/one/many controls and special-unitary dispatch |

The native 2×2 spectral helper uses NumPy eig/QR rather than introducing the
reference's SciPy Schur dependency. Repeated scalar matrices are handled directly.
This is a numerical backend adaptation, not a change to reference code; parity
claims below are for the tested emitted streams at explicit tolerances. It is
not a bitwise floating-point equivalence claim for every possible unitary.
Existing one/two-qubit optimization synthesis drops global phase and is therefore
unsuitable for this coherent-control requirement; it remains unchanged.

## Append workflow

The existing register emitter binds these owned instructions without adding
another public operation type. For this self-adjoint example the same exact
snapshot supplies both directions:

```python
import flagquantum as fq
from flagquantum.core.ir import CircuitIR
from flagquantum.compiler.optimization.controlled_unitary_synthesis import controlled_unitary_ir
from flagquantum.algorithms.primitives.circuit_register_emitter import CircuitRegisterEmitter
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare

ir = CircuitIR(3, controlled_unitary_ir([[0, 1], [1, 0]], (0, 1), 2))
emitter = CircuitRegisterEmitter((("data", 3),), (ir, ir))
op = BlackBoxPrepare((("data", 3),), (), emitter).make_on(
    selection=(0, 1, 2), junk=()
)
circuit = fq.Circuit(3).x(0).x(1).append(op)
assert circuit.state().abs().argmax().item() == 7
```

## Evidence and remaining consumer gap

The reference file is hash-checked and Qualtran 0.7.0, Cirq 1.7.0 and native
Kickmix 0.1 Simulator are required rather than skipped. For 0–5 controls,
H, half-Y, generic special-unitary, scalar-phase and general complex unitary
inputs match source instruction count, opcode, operand order, phase parameters
and local matrices. Complete operators, absolute complex phase and native IR JSON
round trips also match. All borrowed-X source branches are checked over every
basis column, including arbitrary borrowed bits, for 0–5 controls and 0/1/3
borrowed wires. Four invalid-matrix admissions fail as specified.

The local matrices remain local native IR atoms, exactly at the compared source
leaf level. Their counts are not synthesized Clifford+T costs. This module
supplies no invented independent resource model; consumers must retain their
separate declared estimates. No source whole-block tensor is used to emit gates.

The uniform-preparation consumer now connects through the shared
[controlled-H front end](CONTROLLED_HADAMARD.md): source HPow, controlled Y/X,
extracted control phases and optimized dispatch are preserved. The source Bloq
adjoint's Hadamard normalization is distinct from raw Cirq HPow(-1). Q-0695
remains open only for its recorded symbolic source model failures.

Acceptance: all 52 source/ownership/operator cases pass (3.29 s). The
concentrated synthesis, uniform and register-import suite passes 488 cases
(9.11 s), without skips. Serial coverage reports all 80 statements and 22
branches covered (52 cases, 8.23 s). The production module passes strict typing;
Black/Ruff, Compiler scope, architecture, multilevel IR and protected API checks
pass. The append example above is executed directly and returns basis state 111.
