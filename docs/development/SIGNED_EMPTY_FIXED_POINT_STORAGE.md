# Approved signed and empty fixed-point storage

Proposal 116 was explicitly approved on 2026-11-25. Core's existing
`executable_register_width` now admits canonical positive-width signed quantum
fixed-point words and zero-width unsigned words. Logical descriptor validation
still checks parameter order, boolean sign, fractional constraints and nominal
identity. Physical storage still requires concrete integer widths/fractions and
zero classical width. Signed zero-width remains outside Proposal 116.

No arithmetic, value codec, gate, public entry or serialized vocabulary changes.
Compiler continues owning typed calls, Split/Join, casts and linear allocation.
Runtime and Simulation keep their execution and numerical responsibilities.
The user continues to append existing native operations.

## Complete dependency evidence

Signed OnEach at negative and positive fractional widths retains Split,
source-ordered child calls and Join. Actual original constructor libraries at
depths zero/one/two preserve metadata and resolve through native child factories.
X, H, S, S-adjoint, T and T-adjoint run on wires `(3, 0, 2)` with an independent
spectator. Complete complex operator columns, source SDK gates, actual adjoint
restoration, caller ownership and zero private work/feedback are compared.

Empty unsigned words keep an empty linear handle and zero child calls. Their
source forward and generic adjoint run in a real one-wire H/T spectator context;
the native circuit matches its full complex operator and preserves the empty
output. This does not assert that the standalone zero-wire source unitary
protocol works: it fails in the generic empty tensor-network contraction. That
failure is retained separately, without patching source or inventing a matrix.

Shaped cast/identity witnesses cover scalar, array, multidimensional and empty
array storage. They preserve raw MSB-first words and exact dtype metadata.
Creation, Split/Join and consumption leave the caller spectator untouched.
Symbolic fractional widths still pass logical verification and fail physical
lowering. The IR gate now names both an approved concrete-storage witness and
the independent symbolic-refusal witness. Signed/multi-bit fixed-point control
and classical/malformed descriptors keep their prior rejection.

| Operation scope | Actual primitives and work | Original model scope |
| --- | --- | --- |
| Three-bit X/H/S family OnEach | Three corresponding primitives, zero private work/feedback | Three Cliffords, peak three |
| Three-bit T family OnEach | Three T or T-adjoint primitives, zero private work/feedback | Three T, peak three |
| Empty unsigned OnEach | Zero instructions/work/feedback, empty logical output; one caller spectator needed for the physical plan | Zero root gate cost; spectator context counted separately |
| Cast/identity, creation/consumption | Zero gates, existing ownership events where needed | Cast source model zero gates |

## Acceptance and remaining whole-file gap

The affected concentrated suite passes 1,078 cases. The Cirq 1.6.1 follow-up
passes 185 cases. All four unchanged source OnEach tests and 145 local
integration cases pass. Four owner/focused Python files pass strict typing;
the additional existing logical/IR witness modules receive regression, formatting
and scope checks without claiming their entire legacy modules are strict-clean.
Formatting, approved integration scope, language, architecture, IR/public API
and existing SDK contracts pass. Locked Kickmix 0.1 with Simulator is present;
these specific gates use Qualtran source, not a substituted Kickmix reference.

Receipts under the reference environment:
`g2-signed-empty-storage-regression-20261125.log`,
`g2-signed-empty-storage-cirq16-20261125.log`,
`g2-signed-empty-storage-original-20261125.log` and
`g2-signed-empty-storage-quality-20261125.json`.

The complete OnEach audit additionally found that the unchanged source admits
signed zero-width QFxp at negative fractional widths. It decomposes into
Split/Join and its real spectator context executes. Proposal 116 excludes that
profile, so native rejection is retained. Proposal 124 records the exact new
approval requirement and receipt; Q-0246 stays open for it. Existing invalid
child, symbolic physical shape and source serialization boundaries remain in
`ON_EACH_FILE_AUDIT.md`. No broader acceptance is inferred from this storage
dependency delivery.
