# Live register graph scalar execution

## Production gap and complete owner changes

The unchanged Qualtran v0.7.0 SU2 source graphs execute, but the native live
register graph importer previously refused them because a zero-port GlobalPhase
has no connection edge. Resource counts retain its definition, but cannot
recover its occurrence identity, multiplicity or execution position.

The complete existing live graph adapter now supplies actual occurrences from
`CompositeBloq.iter_bloqsoqs()`. Compiler's existing library resolver has a
private ordered input for that live producer. It resolves all explicit children,
validates occurrence uniqueness and connected-node completeness, and uses the
existing typed builder to replay their actual order. Live leaf discovery and
physical capture retain the original objects at the SDK boundary. The adapter terminates all
foreign objects before returning an owned Program.

The ordinary stored-library entry retains its signature and refusal when the
wire omits execution information. Counts never become an execution schedule.
Graphs without zero-port children retain their existing scheduling path.
Allocate/Free, borrowed storage, word views, child closures and parameter
conversion remain with their existing owners. No Core operation, public export,
serialized field or numerical executor is added. Proposal 119's different
registerless scalar-effect opcode remains pending.

## Signature, execution and ownership

Root ports retain the original names, MSB bit order, array shapes and sides.
An isolated scalar call has no logical operands or results. Its accepted physical
leaf recipe allocates a clean private carrier, applies the existing X/phase
sequence, restores zero and frees the carrier. Repeated source occurrences
remain repeated calls even when they share one definition. A purely scalar root
has an empty logical signature and needs an explicitly supplied clean physical
carrier pool. An entangled caller spectator remains untouched.

The SU2 source live traversal visits the three connected rotations and then its
isolated scalar, despite that scalar having source instance ID zero. This order
is copied from the actual traversal, rather than sorted by ID or inferred from
constructor position. Scalar calls can also follow measured arithmetic without
changing branch phase, feedback or private-work cleanup.

```python
import torch
from qualtran.bloqs.basic_gates.su2_rotation import _hadamard
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.ecosystem.cirq._register_graph import import_register_graph

program = import_register_graph(_hadamard.make().decompose_bloq())
bound = bind_register_program(program, ((0,),), clean=(1,))
circuit = Circuit(2, dtype=torch.complex128)
circuit.append(FieldOperation(2, bound.circuit.instructions))
state = circuit.state().flatten()
torch.testing.assert_close(
    state[[0, 2]], torch.ones(2, dtype=state.dtype) / 2**0.5,
    atol=2e-13, rtol=0,
)
assert torch.count_nonzero(state[1::2]) == 0
```

The SDK is needed for capture. Binding and ordinary append execute the owned
Program without source imports; the tests block those imports explicitly.

## Actual primitives and independent source models

| Source live graph | Actual native primitives | Independent original QEC model |
|---|---|---|
| Hadamard factory, either direction | Rz 2, Rx 1, X 2, phase 2 | Clifford 3 |
| T factory, forward | Rz 2, Rx 1, X 2, phase 2 | T 1, Clifford 2 |
| Three equal scalar occurrences | X 6, phase 6 | zero cost, zero logical qubits |
| Two-bit Add followed by a scalar | CX 9, H 3, T 2, TDG 2, S 1, measure 1, CZ 1, reset 1, X 2, phase 2 | And 1, Clifford 4, measurement 1 |

The scalar is atomic in the source and has zero logical cost. Its carrier recipe
is not described as an upstream primitive decomposition. The source T-adjoint
model reports one rotation rather than the forward T category; that original
classification is retained separately. The arbitrary SU2 factory's two
source directions also have different literal scalar-carrier inventories.
Eight resource rows retain their full measured histograms and original model
categories in the batch receipt. Reserved pool capacity and idle spectators
are not reported as live private work.

## Direct live graphs without constructor serialization

The live adapter also constructs owned connections and occurrences directly
from the original graph. It no longer serializes leaf constructor arguments or
matches physical leaves by lossy serialized declarations. Nominal register
signatures reuse the original register codec; live Montgomery moduli omitted by
that codec are restored into the existing native register types. This preserves
distinct typed Split/Join views and Allocate/Free lifetimes without changing the
stored wire format or mutating source objects.

This closes the actual LP preparation graph import failure for widths 1, 3 and
7: the unchanged source tensor works while its constructor serializer rejects
a NumPy uint8 control. Every live call and operand now enters the native Program.
Full complex prepared states include a coherent spectator; symbolic measured-And
identities certify every raw history, and sparse numerical execution verifies
output amplitudes and clean private work. The certificate applies to the
prepared-input subspace, not an arbitrary-input unitary. Original LP inverse
phase counterexamples remain independently recorded.

The LP measured stream contains 8, 16 and 32 T/TDG primitives respectively.
Its independent source model has 2, 4 and 8 And nodes, plus 12, 24 and 48
rotations; the model's default rotation-synthesis T estimate is not a measured
primitive count. Three-control Majorana graphs and opaque unitary leaves also
retain their actual physical callbacks without requiring constructor JSON.
Unknown leaves lacking a supported physical protocol still fail conversion.

Run `tests/unit/test_lp_direct_live_graph.py` for the complete live LP path and
`tests/unit/test_luis_perina_state.py` for the existing native module, direct
source dependencies and retained inverse-phase counterexample. LP/LPRS stored
constructor bindings are now implemented; see [the LP audit](LUIS_PERINA_STATE_FILE_AUDIT.md).
The native root retains the original fifteen calls without redundant output
views, and the catalogue uses its complete native decomposition.

## Verification and remaining boundaries

The original three SU2 factories and their adjoints retain every root call,
operand connection and returned alias. Full complex states, an entangled
spectator, both floating complex precisions, ordinary append, JSON and work-zero
checks pass. Repeated scalar occurrences and an entirely registerless root are
executed. The measured Add graph checks every raw measurement history and actual
native trajectories. Stored-wire information-loss rejection, missing connected
occurrences and repeated instance diagnostics remain tested.

Primary related-module regression passes 102 cases and local integration passes
145. The Cirq 1.6.1 profile passes all 101 other cases, including every changed
path; its unrelated whole-library source graph snapshot has 3692 nodes against
a hard-coded 3693. Loading both production modules from the pre-change published
commit reproduces that same failure. Neither that test nor its expected size is
changed. This does not certify the complete secondary SDK profile or G2.

Nine direct source modules match the frozen blobs. Source protobuf boolean
argument deprecation warnings remain visible. Compiler and Ecosystem scopes,
strict typing, format, architecture, internal IR and public API checks are
required before publication. Original serialization omissions, unsupported
leaves and unresolved symbolic physical inputs retain their existing errors.
