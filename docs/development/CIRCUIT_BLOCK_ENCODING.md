# Native block snapshots at the LCU boundary

`CircuitBlockEncoding` owns explicit forward and adjoint `CircuitIR` snapshots,
with an optional positive-controlled directional pair. It implements the existing
block metadata and positive-controlled child factory; LCU itself is unchanged.

Logical wire order is optional control, system, ancilla, resource. Private clean
work follows those ports. Both ordinary directions share their maximum work and
slot requirement; the controlled pair has its own requirement. Logical ancilla
and resource are retained through outputs, including nonzero coherent inputs.
Only private work is required to return to zero. Classical slots are explicitly
relocated through the existing primitive binder. A measured stream is never
reversed to manufacture an adjoint.

The Algorithms owner imports no external framework types. `_circuit_snapshot`
contains the same validation, binding and model composition used by sparse oracle
providers; the prior duplicate implementation was moved, not retained alongside
another binder. Returned-wire swaps are charged as explicit binding overhead on
top of the caller's independent directional model, not hidden in source costs.

The shortest ordinary append flow is:

```python
from math import pi
from flagquantum import Circuit
from flagquantum.algorithms.primitives.circuit_block_encoding import CircuitBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import CircuitIR, Instruction

snapshots = (
    CircuitIR(1, (Instruction("t", (0,)),)),
    CircuitIR(1, (Instruction("tdg", (0,)),)),
    CircuitIR(2, (Instruction("cphase", (0, 1), {"theta": pi / 4}),)),
    CircuitIR(2, (Instruction("cphase", (0, 1), {"theta": -pi / 4}),)),
)
t_model = ComponentCost("T or T adjoint", 1, GateCost(t=1))
controlled_model = ComponentCost("Controlled phase", 2, GateCost(rotation=1))
models = (t_model, t_model, controlled_model, controlled_model)
block = CircuitBlockEncoding(1, 0, 0, 1, 0, snapshots, models)
circuit = Circuit(1).append(block.make_on(system=(0,)))
```

Models above are illustrative supplied declarations, not a gate-count query over
the snapshots. In LCU, pass `block.bind_child` as the existing child factory and
`block.requirements(controlled=True)` as its clean-work/slot requirement. Call
`resource_model(controlled=True, adjoint=...)` for the corresponding independently
declared cost. There is no new registry, compiler pass or numerical backend.

## Optional external conversion

`flagquantum.ecosystem.cirq._block_encoding.import_block_encoding` captures the
source's four actual operations immediately, using shared `_component_snapshot`.
The optional source dependency stays at Ecosystem. The caller may supply four
independent models. If no model is available, emission remains usable while
`resource_model` reports that the independent cost is unavailable. Only concrete
scalar packed ports and identity signal states
are admitted, consistent with the existing LCU zero-signal requirement. Unknown
primitive decompositions, dirty borrowing, unsupported feedback and changed port
shapes fail explicitly. Explicit dense matrix leaves retain a copied, finite,
validated unitary tensor,
including controlled matrices and absolute scalar phase. They remain single
matrix instructions; no lower-level synthesis or T cost is inferred. Generic
unitary contraction of arbitrary components and symbolic physical execution
remain outside this conversion.

Fractional Z/CZ powers retain native phase/cphase semantics, including their
absolute scalar phase. Positive-controlled H and canonical rotations remain
native controlled primitives. The static Cirq API retains its previous contract.

## Evidence

`tests/unit/test_circuit_block_encoding.py` checks four measured interfaces on a
coherent input spanning all logical auxiliary values and an external spectator.
Every small measurement branch has the expected absolute complex amplitude and
zero work; nonconsecutive quantum and classical bindings are included.

`tests/unit/test_foreign_block_encoding.py` converts actual H/T source blocks,
with and without logical auxiliary ports, then compares all four complete complex
matrices and independent source costs. Converted children replace the existing
LCU factories without changing LCU consumers; both directions and outer-control
modes compare all small branches, including complete logical outputs and work.
Symbolic width and nonidentity signal objects are rejected before capture.

`test_fractional_phase_powers_and_controlled_h_preserve_absolute_phase` compares
against Cirq's independent gate matrices and feedback resolution on every branch.
Existing sparse oracle and LCU suites cover the shared extraction's consumers.


## Explicit matrix leaves

The feedback importer accepts Cirq MatrixGate and external gates that explicitly
provide a matrix and unitary protocol, including their quantum control wrappers.
It reads the actual controlled unitary rather than reconstructing controls from
assumed conventions. The tensor is copied, checked for finite values, checked
against operand dimension and checked for unitarity at absolute tolerance 1e-10.
Classical feedback retains the matrix operand and each gate's native conditions.
This also fixes matrix preservation when imported effects are wrapped with their
feedback metadata. Source data mutation cannot alter the owned native tensor.

The locked MatrixGate provides `QubitCount` but raises `DecomposeNotImplementedError`
for `QECGatesCost`, for both ordinary and controlled blocks. Those source outcomes
are recorded separately: physical snapshot width is known, while primitive
Clifford/T synthesis cost is unavailable. A missing independent native model is
not represented by an all-zero ComponentCost.


The original ten seeded dense coefficient families are converted in both
directions and compared instruction by instruction with the previously accepted
explicit-matrix LCU recipe. Gate names, wire operands, parameters, feedback and
every matrix element are compared. Each source matrix leaf's four interfaces
also match the full locked complex operator. The complete stream replacement
reuses the existing catalogue's projection/complex-output evidence; one family
in both directions additionally checks the occupied complex output support at
two numerical measurement histories and zero private work.

An initial run redundantly interpreted both complete streams numerically for
every family and stopped at its 150-second bound before completing all 20 cases.
It is not passing evidence. The final bounded run retains all 20 stream checks
and concentrates repeated numerical execution in the representative family.
Numerical samples are not called exhaustive histories; whole-catalogue channel
certification remains a separate open Q-0286 obligation.


## Explicitly requested directional availability

The optional block importer defaults to all four original interfaces. Set
`include_controlled=False` when requesting only ordinary forward/adjoint capture.
The source control is then never requested; this is an explicit capture request,
not an error-catching fallback. A missing controlled pair is represented by None,
and controlled binding/requirements refuse it. Partial controlled pairs and
binding-swap claims for uncaptured directions are invalid.

Supplied models are also directional: each slot may be None independently.
Known ordinary costs remain queryable when another direction has no model;
missing costs never become zero gates. The original independent declarations and
physical snapshot/binding inventories keep their existing separate accounting.
