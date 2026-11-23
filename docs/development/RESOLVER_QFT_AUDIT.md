# Textbook Fourier constructor disposition

## Complete source and ownership

The complete qft/qft_text_book.py and its original tests are frozen at Qualtran
v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. The constructor, both original
examples and direct Hadamard, PhaseGradientUnitary and TwoBitSwap dependencies
are mapped below. Six frozen source/test/dependency/resolver Git blobs and the
actual locked Kickmix 0.1 Simulator are verified in
`g2-textbook-constructor-intake-20261123.json`.

| Source definition | Native owner |
|---|---|
| QFTTextBook bitsize and with_reverse | Compiler formats/legacy_qft_factories.py resolves the existing integer metadata and serialized Boolean fields, with reversal defaulting to true. |
| signature | The existing unsigned register type retains the MSB-first q through word and original width. |
| decompose_from_registers | Existing phase/textbook_qft_component.py retains initial H, controlled phase-gradient calls on reversed prefixes, each later H and optional final reversal. |
| Direct gradient, Hadamard and swap closure | Existing phase_gradient_component.py, controlled_rotation_component.py, bit gate and word-view owners retain the accepted graph and measured And cleanup. |
| build_call_graph | Existing Algorithms qft_resource_model.py retains concrete and symbolic calls and independent QEC cost classification. |
| _qft_text_book, _symbolic_qft and DocSpec | Both unchanged examples resolve through the original serialized wire, preserving their concrete or bodyless symbolic declaration. |

The format adapter has no numerical implementation or independent resource
model. Caller q remains live. Shared controlled phases allocate and erase one
private reusable qubit when width exceeds one. Existing resource and native
execution receipts remain authoritative. Both reversal settings and stored
library depths zero, one and two resolve the same complete graph through the
existing binding and ordinary circuit.append(op) workflow.

## Exact execution and separate accounting

The constructor tests compare the bound graph, allocation events and output
ownership to the accepted component. They compare every original top-level
call dependency and operand, and reuse the accepted phase-child primitive and
feedback checks. Complete complex source operators on widths one through three
include a noncontiguous reversed binding, an entangled spectator, every raw
measurement history and zero private work. Existing Fourier and phase-window
suites retain exact adjoints, source child graphs and larger schedule checks.

For r=n(n-1)/2, the stored literal graph emits n+3r H, 6r CX, 2r T, 2r T-dagger,
r S, r phase gates, r measurements, r conditional CZ and r resets, plus floor(n/2)
SWAP when reversal is enabled. Peak physical use is n plus one reusable qubit
for n greater than one. Actual measurements are counted independently from
phase-synthesis estimates. The source analytical model reports r And and r
measurements, with special-angle T/Clifford and generic rotation categories.
The existing flat algorithm retains an independently accepted controlled-phase
primitive vocabulary. Its inventory remains separately documented in
TEXTBOOK_QFT_FILE_AUDIT.md.

## Retained source and constructor boundaries

The original zero-width decomposition raises a missing-q TypeError. Symbolic
widths, including an explicitly supplied Sympy Integer, raise DecomposeTypeError.
Direct symbolic constructors remain bodyless, and physical binding refuses.
The original wire serializes Sympy Integer(3) as Python int(3), so reconstructing
that wire admits the concrete three-bit graph. This source codec normalization
is recorded separately from direct symbolic construction.

Serialized Boolean reversal uses integer zero or one. Other values refuse at
the typed native constructor boundary. Original width-123 small-angle cost
assertion remains an expected source failure. Original automatic count checking
also remains disabled. Fresh receipt g2-textbook-constructor-original-20261123.log records 22
unchanged original passes, three original skips and two xfails. Full
symbolic execution, unapproved signed storage and generic control/adjoint
constructor work remain outside this module's acceptance.

## Acceptance receipts

The complete constructor and affected Fourier regression passes 347 cases.
The local execution integration scope passes 145 cases. Strict checking covers
both changed Python files. Black, Ruff, Compiler scope, repository language,
architecture, multilevel IR and public API checks pass. Dependency checking
retains exactly the existing 720 findings. Receipts are retained under the
reference environment with prefix g2-textbook-constructor and date 20261123.
