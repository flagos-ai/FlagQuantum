# Native block snapshots at the LCU boundary

`CircuitBlockEncoding` owns explicit forward, adjoint, positive-controlled forward
and positive-controlled adjoint `CircuitIR` snapshots. It implements the existing
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
The optional source dependency stays at Ecosystem. The caller supplies four
independent models. Only concrete scalar packed ports and identity signal states
are admitted, consistent with the existing LCU zero-signal requirement. Unknown
primitive decompositions, dirty borrowing, unsupported feedback and changed port
shapes fail explicitly. This path does not claim generic synthesis of dense
matrix leaves or symbolic physical execution.

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
