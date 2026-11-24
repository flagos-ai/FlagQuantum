# Named gate conversion and direct dependency audit

Reference: unmodified Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/cirq_interop/_cirq_to_bloq.py::cirq_gate_to_bloq` and its register
import, wrapper, control and inverse dependencies. The frozen-source fixture
verifies four interop blobs, Qualtran 0.7.0 and actual Kickmix 0.1 Simulator.

## Ownership and interface

SDK objects stop at Ecosystem. A per-import Mapping selects existing native
FunctionRecord roots and their reachable Program helpers. Compiler fixed,
exchange, rotation, scalar, predicate and exact-adjoint owners remain
authoritative. Production adaptation does not import source constructors.
Unknown selections continue through the existing SDK decomposition or explicit
local tensor path. That continuation does not establish a named source graph.

Native signatures retain source names and order, including separate ctrl/target
ports and shaped q arrays. Existing optree placement bridges the raw wrapper's
operand grouping. Operand tuples are MSB-first. Existing component owners retain
logical outputs, private allocation, measurement feedback, exact erasure and
release. A scalar phase uses its existing clean carrier and restores it.

The raw CirqGateAsBloq wrapper first executes its SDK one-step decomposition.
Named selection applies to the exposed children; the wrapper layer is retained.
The private register gate importer forwards its existing definition map to
optree and preserves the legacy default. Explicit attached native definitions
still precede SDK semantics and normalization.

## Complete selection and dependency disposition

The sixteen original constant instances precede general control and power
selection. Three radian types, four zero-shift power types and scalar phases
reuse their existing native factories. Positive ZPow, Rz and scalar controls
select CZPow, CRz and ZPow. Other supported polarities and conjunctions reuse
ControlledViaAnd with explicit positive-component providers and the shared
predicate/erasure owner. Source control shape, bit order, new-control-first
nesting and single-AND-clause refusal are retained. The positive port adapter
only regroups existing one-bit bindings; it does not synthesize another gate.

Inverse composite wrappers import the original decomposition and reverse typed
calls using explicitly matched forward/inverse component pairs. A measured
CZPow child selects the negative-exponent compute/phase/exact-erasure component;
its measurements are never reversed. Unknown exact pairs refuse. Even an
unchecked inverse wrapper around a measurement is rejected. Native adapter
pass-through reuses the existing attached-definition path.

Symbolic scalar coefficients reuse the existing proven real-angle codec and
owned expression grammar. The source records arg(coefficient)/pi and declares
GlobalPhase atomic. Native execution preserves the coefficient for exp(i*f),
including values beyond the principal argument interval; this is a native scalar
carrier, not a claimed source primitive decomposition. Tests retain the complex
phase for nonlinear f and its controlled version. Closed SymPy constants are
converted to finite real scalars at selection; unresolved expressions remain
owned symbols. The nested arbitrary-control regression exposed this distinction.

## Gate, amplitude and ownership evidence

The reversed-operand CZPow child matches both direct native binding and the
unmodified source graph's per-wire primitive stream, parameters and feedback.
Its two raw measurement histories equal the exact complex input transformation
scaled by 1/sqrt(2), and the work wire is zero. Negative and multi-control cases
check every small measurement history against the original tensor, including
complex inputs, output work and independent source measurement counts. Fixed H/Y
controls additionally compare source primitive streams and operands. An inverse
containing H, CZPow and Rz verifies both exact-erasure histories. Existing source
replacement, stored-array, source-import-blocked binding and control tests are
reused through the now-automatic payload path.

The selected CZPow emits 3 H, 2 T, 2 Tdag, 6 CX, 1 S, 1 phase, 1 measurement,
1 conditional CZ and 1 reset on 3 used wires, with one work allocation/release.
Its separate source model is 1 And, 1 Clifford, 1 rotation, 1 measurement and
3 logical wires. A serialized CXPow(-0.375) emits 7 H, 2 Sdag, 3 phase, 3 S,
2 T, 6 CX, 2 Tdag and one each measurement/conditional CZ/reset on 3 used
wires; its source model is 1 And, 3 Clifford, 1 rotation and 1 measurement.
A two-target explicit matrix now emits 3 exact And erasures rather than three
atomic SDK cphase gates. Complete measured inventories and independent source
models are retained in g2-named-gate-definitions-resources-20261124.json.
Atomic rotation owners and scalar carriers have native primitive realizations;
logical source model nodes must not be reported as those primitive counts.

## Explicit boundaries

- A generic positive control of atomic XPow has no original gate body:
  the unmodified source raises DecomposeTypeError("atomic"). SDK execution of
  that case remains separate from a nonexistent source primitive stream.
- Arbitrary unknown controls or inverse leaves need an owned decomposition or
  explicit exact pair. Local matrix fallback is confined to external atoms;
  no algorithm root is contracted to manufacture source call-chain fidelity.
- Shifted powers retain the existing external decomposition path. Original
  nonbinary ports, custom JSON resolvers and source-only object metadata remain
  outside the concrete binary serialized payload admission.
- Original JSON serialization still rejects sin, and the arbitrary two-target
  source tensor still fails on non-through C[And]. Native direct symbolic
  selection and the original available Cirq emitter are distinct evidence.
- Original constructor metadata loss, QROM admission differences, pending
  Proposals 116/119 and repository/GPU/performance scope remain open in the ledger.

This batch does not add a complete generic constructor identifier or close its
parent item. Whole-source selection and dependency disposition do not imply that
missing original bodies or the remaining G2 acceptance scope have disappeared.

## Acceptance receipt

The final concentrated regression passes 188 cases without skips; related local
integration passes 145 cases. All 11 unchanged original interop and decomposition
protocol tests pass in the existing Cirq 1.6 compatibility environment. Strict
mypy checks six Python files. Black, Ruff, Ecosystem scope, language,
architecture, multi-level IR and public API checks pass. Dependency analysis
exactly matches the 720 existing findings. No protected API or serialized
vocabulary changed. This receipt retains the source-owned bodyless and admission
boundaries above; the generic constructor coverage remains 112/303.
