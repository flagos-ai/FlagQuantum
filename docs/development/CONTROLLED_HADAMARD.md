# Source controlled Hadamard and uniform preparation

`compiler/registers/primitives/controlled_hadamard.py` owns the concrete
controlled-H front end. It reuses the phase-exact recursive unitary transform
rather than replacing a controlled H with a measured conjunction. Controls are
ordered native wires with explicit binary activation values; the target and
controls must be distinct nonnegative indices. The primitive owns no quantum
workspace or classical slots. Numerical state execution stays in Simulation.

The source is the Cirq 1.7.0 dependency invoked by locked Qualtran v0.7.0
`PrepareUniformSuperposition`, Copyright The Cirq Developers, Apache-2.0.
Tests freeze the hashes of `common_gates.py`, `eigen_gate.py`,
`controlled_gate.py`, `global_phase_op.py`, `diagonal_gate.py` and
`three_qubit_gates.py`, in addition to the recursive-control dependency.
No reference file is modified.

| Actual source dispatch | Native implementation |
| --- | --- |
| Bare H / positive single-control H | Existing H / CH primitive, without manufacturing a decomposition |
| HPow exponent +1 | Controlled half-Y, shifted X, then the extracted control phase |
| Raw Cirq HPow exponent -1 | Controlled quarter-Y, inverse X and controlled negative quarter-Y; its emitted stream differs from exponent +1 |
| Canonical all-positive two-control Y power | CCY → S-dagger, H, CCZ-power, H, S |
| Canonical controlled X | Native CX/CCX where source admits them; otherwise the shared recursive local-unitary synthesis |
| Controlled extracted phase with last activation 1 | Z-power control dispatch, including canonical CZ/CCZ powers and the generic local diagonal matrix fallback |
| Extracted phase with last activation 0 | Original Walsh/Gray diagonal traversal with its scalar phase retained |
| CCZ-power | Original adjacency-sensitive CX sweep and seven phase powers, including noncontiguous LineQubit layouts |

Local matrix payloads are native JSON-serializable values, not SDK objects or
whole-block matrix synthesis. The Walsh/Gray branch uses a length-2**controls
coefficient vector, matching the source's exponential instruction construction;
this is not a scalability improvement or a dense quantum state simulation.
Determinant phases near fixed Clifford/T exponents use the same 1e-12 leaf
classification tolerance as the source importer, while retaining general phase
parameters otherwise.

## Qualtran adjoint and uniform integration

Qualtran's Hadamard Bloq is self-adjoint. Its composite adjoint canonicalizes
that Bloq before exporting Cirq, so its H child uses exponent +1 even in the
inverse uniform circuit. Native uniform preparation follows that entry point:
reverse the target order and preserve each canonical H child. Directly requesting
the primitive's `adjoint=True` instead follows raw Cirq HPow(-1), a separately
verified path. These operators coincide, but their gate streams must not be
silently interchanged when comparing source execution.

`uniform_preparation_ir` now uses this shared primitive for every initial H.
It keeps the original comparator, threshold and zero-predicate phases and their
measured erasure. A power-of-two count needs no clean work or classical slots.
The independent source resource model still includes its declared measured
control peaks; it is not recalculated from the new literal work requirement.

```python
from collections import Counter
import flagquantum as fq
from examples.algorithms.uniform_preparation_catalog import UniformPreparationExample

op = UniformPreparationExample(5, controls=(0, 1))
circuit = fq.Circuit(op.qubit_count).append(op)
ir = circuit.to_ir()
counts = Counter(g.name for g in ir.instructions)
assert (ir.n_wires, len(ir.instructions), counts["measure"]) == (10, 375, 10)
assert counts["t"] + counts["tdg"] == 49
```

The same append interface and true inverse remain available. Run
`python -m examples.algorithms.register_uniform_preparation` for the six-state
example with coherent inactive controls, clean work and inverse validation.
It reports the independent model, explicit T/T-dagger count, remaining Rz and
local matrix leaves separately.

## Acceptance and limits

The primitive is checked against actual source H exports for every polarity of
0–4 controls in both raw Cirq directions. Exact stream checks include opcodes,
operands, phase parameters and local matrices; full complex operators and IR
JSON round trips retain absolute phase. Noncontiguous CCZ adjacency cases and
invalid polarity/direction admissions are included. No measurements are added
by the primitive.

Uniform tests no longer substitute measured conjunctions in their reference
helper. They read the actual source controlled-H lowering, then compare the
complete uniform stream, including feedback, scalar/rotation parameters and
local matrices. Resolved source/native consumers compare full logical operators
and all erasure branches, with every clean work wire restored. Actual source
uniform forward and Bloq-adjoint streams both have 375 instructions for n=5,
controls(0,1); raw Cirq HPow(-1) is deliberately not substituted there.

Source QEC cost declarations and actual matrix-leaf counts remain distinct.
Q-0695 still retains the original symbolic call-graph omission of controls and
its documented singleton/symbolic resource-query failures. This change does
not repair those reference defects or price local matrices as synthesized T gates.

Batch acceptance: 649 concentrated cases pass (23.12 s), including the twelve
resolved source/native complete-operator consumers and exact source-root opcode
counts. The source-front-end and existing uniform focused suite passes 174 cases
before the additional resolved fixtures; the final concentrated suite includes
them all. Strict typing passes all three production files; Black/Ruff, both
applicable scopes, architecture, multilevel IR and protected API checks pass.
Serial coverage passes 299 cases (26.57 s): both recursive synthesis and the H
front end have 100% statement/branch coverage, and the independent uniform owner
has 98%; the uncovered guard is invalid cost-profile admission, not an execution
path. The local CPU/representation suite passes 60 cases (40.57 s), the SDK-free
uniform notebook passes in its independent kernel (2.42 s), and both append
examples execute successfully.
