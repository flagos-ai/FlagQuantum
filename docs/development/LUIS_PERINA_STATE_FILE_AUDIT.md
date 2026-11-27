# Complete Luis–Perina resource-state source

## Frozen source and ownership

Q-0624 maps the complete Qualtran v0.7.0 file
`phase_estimation/lp_resource_state.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`55466291aba7d239ad4992b3aa3e09b44e33afe2`. Q-0625 maps its complete tests,
blob `41b84e5cb8df63838ddbd843d2e4202bd1ad6c18`.

Algorithms owns `luis_perina_state.py`. LPRSInterimPrep maps to
LuisPerinaInterim and LPResourceState maps to LuisPerinaState. The shared window
base owns RIGHT-only fractional output descriptors. Interim `m` is big-endian
THRU data and `anc` is a THRU logical qubit. Preparation accepts clean outputs
through the existing named PREPARE binder. The caller uses
`circuit.append(state.bind().make_on(selection=phase, junk=(), workspace=work,
classical_bits=slots))`.

The caller owns n+4 clean work wires and n+1 reusable result slots. The first
work wire is the source ancilla, the second is the source flag, and the shared
reflection owns the remainder. All work returns to zero in every branch.
Interim uses no scratch, measurement or result slot. Core owns exact unitary
inversion. Compiler owns OnEach, CRz, predicate measurement and reflection.
No production code imports Qualtran or Cirq.

## Complete stored constructors

`luis_perina_state_program` composes the complete source forward graph through
the existing interim, exact interim adjoint and zero-state reflection Programs.
Allocation, source call order and private frees remain explicit. Existing nominal
compatibility accepts the source QAny child return under its declared fractional
output descriptor. The root retains exactly the original fifteen calls and
their return mapping without additional Split/Join views. This does not change the
source's defective generic full-window adjoint or identify it with the exact
mathematical inverse.

Ecosystem's `_phase_window_library.phase_window_factories` adds the exact
LPRSInterimPrep and LPResourceState identifiers to the existing rectangular and
Kaiser providers. Compiler does not import Algorithms. The interim eps field
is validated metadata and does not alter source rotations. Symbolic declarations
retain the existing wire's symbols without inventing an executable body; source
symbol serialization omits assumptions. Shallow and one-level source libraries
resolve to the same complete native definition.

Both stored constructors preserve coherent inputs or prepared outputs at widths
1, 2, 3, 5 and 7 with ordinary `circuit.append(op)`, JSON and SDK-free resolution
and execution. Every original root call and operand matches construction order.
All measurement histories share the certified complex channel, and numerical
prepared outputs verify private zero work with a coherent spectator. Existing
full source/module evidence, including n=13 and the inverse counterexample,
remains applicable to unchanged dependencies.

The catalogue now uses this actual concrete LP Program instead of a port-only
stub. Allocation/free glyphs, interim and adjoint labels, Ry, CZ, X and reflection
symbols retain the original root geometry. The dedicated presentation owner is
`examples/algorithms/luis_perina_decomposition.py`; the shared window dispatcher
delegates to it. Fresh source scores are compared at widths 1, 3, 5 and 7.
Symbolic LP retains its unresolved body boundary. Unqualified native private
helpers are not misrepresented as exact source decompositions.

The shared window dependency dispatcher also recognizes the actual LP context,
its forward and exact-adjoint interim definitions, and each angle-specific CRz
in both directions. Recognition compares complete definitions, not port names
alone. The original generic adjoints retain their dagger-decorated word and
control glyphs; they are not relabeled as a forward circuit at a negated angle.
At widths 1, 3, 5 and 7, every interim and controlled-rotation child score matches
the unchanged source geometry and symbols. The actual catalogue callback
qualifies both interim directions. Reflection descendants and other private
helpers remain explicitly native until their independent source qualification.
These are presentation changes only: the Algorithms, Compiler, Core and
Simulation trees, executable Programs, work ownership, primitive inventories
and independent resource estimates are unchanged from `cab32760d`.

Measured inventories and source estimates remain separate. Stored LP emits
8/16/32 T or TDG primitives and 2/4/8 measurements at widths 1/3/7. Its independent
source model reports 2/4/8 And calls and 12/24/48 rotations. All non-identity
primitive inventories match live original capture. Atomic source Identity(n)
has no primitive recipe: native word lowering emits n one-bit I instructions,
whereas Cirq removes multi-bit Identity and retains one-bit I. Consequently
stored LP has 6/10/18 I instructions versus live capture's 6/4/4. This boundary
is explicit; total primitive counts across these two lowerings are not claimed
equal, and their identical complex actions do not waive the count difference.

## Complete responsibility mapping

- Interim H on the word, H on anc, reversed word traversal with exactly
  `-2*pi*2**i/(2**n+1)` controlled Rz, bare anc Rz and final H are preserved.
- Default eps remains 1e-11 metadata. The source emitter does not use it to change
  these rotations. Generic adjoint uses the shared opcode inversion rules.
- The window's standard-deviation factory preserves ceil(log2(pi/eps)). Its
  signature reuses the complete shared fractional window descriptor.
- The full amplitude-amplification order is Ry, interim, CZ, interim inverse,
  inverse Ry, zero reflection with coefficient i, interim, Ry, X flag, X anc.
- The seven original direct call keys and weights map to the native source
  graph. Interim retains each angle-specific CRz call, including the original
  symbolic aggregation. All three original examples map to the same constructors
  and shared signature/graph drawing. Notebook responsibility is Q-0623.

## Execution and resource evidence

The complete original suite, including every slow case to n=13, reports
38 passed, 1 original strict-type skip, 2 original count xfails and 2 original
unregistered slow-mark warnings. The exact log is `g2-lp-state-source.log`.

Interim source capture matches every primitive, operand, angle and direction at
n=1,2,3,5,13. Full operators through n=5, prepared tensors through n=13 and
complex coherent forward/inverse restoration are checked. The n=13 test avoids
forming the dense operator. Replacing the interim emitter with the captured
source stream preserves the complete measured preparation consumer.

At n=1,2,3, every forward measurement history and every independent joint
forward/inverse history is enumerated. Unnormalized complete complex amplitudes
match the original resource-state tensor tensored with a complex spectator and
clean work. Exact inverse restores that input, including its global phase.
Native execution is bounded to these widths for exhaustive dynamic histories.
The complete source tensor slow suite covers through n=13.

The actual positive-width interim inventory is n+2 H, 2n phase, 2n CX and one Rz.
The actual full preparation inventory is three Ry, 6n+9 H, 6n+1 phase, 12n+6 CX,
three Rz, n+3 CZ, 2n+4 identity, 4n+12 X, 2n+2 T, 2n+2 T-dagger, n+1 S,
n+1 measurement and n+1 reset. These identities are execution evidence and are
asserted at all exhaustive widths.

Independently, the declared QEC interim model is 3n+2 Clifford and 2n+1 rotation.
The concrete full model is n+1 And, 14n+21 Clifford, 6n+6 rotation and n+1
measurement. The original symbolic model is instead n+1 And, 10n+13 Clifford,
6n+6 rotation and n+1 measurement. The shared symbolic predicate model retains
that distinction. Every category is compared to the untouched source for all
original resource widths 1,3,5,7,9,11,13 and symbolic n. Original legacy T
formula tests also pass in the complete source suite. Physical inventories are
not presented as declared model counts. Inventory and extension preflight are
in `g2-lp-state-inventory.log`.

## Original defects and uncovered boundaries

The original strict check skips the same QFxp/QAny boundary between allocation
and interim preparation. Its original two count autotests remain xfailed.
The original symbolic QubitCount query raises for both classes. Native code
exposes the symbolic gate graph and concrete physical storage separately.

The source's nonreal zero-reflection adjoint returns the forward reflection.
Consequently the complete original LP adjoint tensor is the negative of the
mathematical bra at n=1 and n=2. The native explicit reverse emitter uses the
shared mathematical inverse and is tested separately. This source defect
remains an open parity boundary.

The generic SDK snapshot of the full LP source chooses a different MultiControlZ
synthesis with opaque matrix leaves and fewer allocated wires. It differs from
the source-specific measured predicate call chain used by the existing native
reflection. Interim primitives and the source call/model boundaries are exact.
Full SDK-default primitive inventory equality is not certified. Prior reflection
all-branch and source-declared-graph evidence is reused, not relabeled as a new
SDK-default synthesis comparison. Q-0625 remains open for these specific source
and execution-path boundaries. Zero and unresolved widths retain declarations
but have no native physical admission in this batch.

## Batch checks

The concentrated window, PREPARE, reflection, controlled-rotation, LP and GQSP
regression reports 310 passed. Black, Ruff, strict mypy for both new Python
files, Algorithms scope, architecture, multi-level IR and protected API checks
pass. Logs use `g2-lp-state-` in the external reference workspace. Kickmix 0.1
and its native Simulator were verified before execution. Q-0624 closes the
complete source responsibility mapping. Q-0625 retains the boundaries above.

## All original numerical widths and historical resource mode, November 25

The complete original test file has nine test functions and two independent
closed-form state helpers. The current batch fills the remaining positive-width
numerical and historical-accounting obligations while retaining the open stored
root and SDK-default obligations. Q-0625 remains unchecked.

| Complete original assertions | Current native evidence |
|---|---|
| Interim and full LP autotesters | Each of the two actual examples is observed independently in all five raw protocol modes. Both make/decompose/serialize pass upstream; both default count checks FAIL and become original xfails. Interim strict typing passes; full LP strict typing is UNVERIFIED. Original statuses are preserved. Full stored LP execution still needs the approved scalar-effect contract under pending Proposal 119. |
| Symbolic LP T complexity | Native source_t_complexity reuses the existing typed predicate and historical budget query. Full symbolic T, Clifford and rotation fields match unchanged upstream, including 4n+4 T. |
| Quick interim/full state at n=3 | Existing full complex and exhaustive raw-branch certificates remain applicable. |
| Slow interim tensors at n=1,3,5,7,9,11,13 | Native source-ordered interim stream, full prepared/coherent complex state and mathematical inverse now cover every original width, with no new source path or matrix fallback. |
| Slow full LP tensors at all seven widths | Actual native append IR is evaluated by the existing test-only sparse interpreter, including a coherent spectator. Every output amplitude matches the original get_resource_state helper. Final work residual is independently bounded. |
| Full LP historical resource equations at all seven widths | Native historical T/Clifford/rotation fields match the unchanged original and its exact 4n+4 / 26n+33 / 6n+6 formulas. |
| Interim historical resource equations at all seven widths | Existing modern interim categories and the shared historical budget give 0 T, 3n+2 Clifford and 2n+1 rotations, exactly matching upstream. |

### Why historical counting needs the typed call chain

Folding an already aggregated modern LP QEC cost loses the different historical
Clifford charge of the measured And inverse. It underestimates the original
historical Clifford field by 2(n+1). The new source_t_complexity query retains
the all-zero predicate's existing compute/erase resource calls until the shared
historical query prices them. It reuses the interim owner and accounts for the
two reflection carrier X plus final CZ/X/X and three Ry. No independent
conjunction formula, primitive synthesis or SDK dependency is introduced.

For resolved positive widths the historical fields are 4n+4 T, 26n+33 Clifford
and 6n+6 rotations. For symbolic width, the original unspecified control values
instead give 22n+25 Clifford; that source symbolic/concrete distinction is
preserved. The historical query supports positive integer widths and symbolic
declarations. Unsupported resolved widths fail explicitly. The existing modern
QEC query and its input contract remain unchanged.

At n=13 the emitted inventory has 56 T-family, 162 CX, 14 measurements and
17 clean work wires. The independent historical model is 56 T, 371 Clifford
and 84 rotations. The modern QEC model is 14 And, 203 Clifford, 84 rotations
and 14 measurements. These are separate accounts, with no claim that modeled
Cliffords equal the scan of the emitted stream.

### Branch and cleanup proof at the original large widths

The new state comparison retains all complex amplitudes, including an entangled
spectator, over native output widths up to 13 and physical widths up to 31.
It uses sparse primitive evaluation as a test diagnostic, not a production
backend, scalability benchmark or dense-to-sparse runtime fallback. The helper
adds literal Ry and identity handling; other supported gate semantics are
unchanged. Dense interim execution remains independently checked.

The existing erasure certificate proves every measured predicate inverse has
the same operator with positive scalar 1/sqrt(2) in both outcomes. It proves
branch identity, with final interference cleanup checked separately on the full
prepared input state. Consequently the independently evaluated branch and its
workspace residual apply to every raw history. Existing small-width tests still
enumerate the forward and forward/reverse Cartesian histories explicitly.

Across the original seven widths the maximum complete complex error is below
2.1e-16 and work leakage probability below 2.4e-31. The acceptance bounds are
2e-12 per amplitude and 1e-28 total work probability. Literal bitwise-zero
assertions would reject floating-point Ry/H cancellation at about 2e-16; the
residual is measured and recorded, not dropped from the cleanup assertion.
The full state norm stays within 2e-15 of one.

The complete unchanged original suite still reports 38 passes, one source
strict-type skip and two source count xfails. No original failure is patched.
The generic LP adjoint's minus sign and SDK-default opaque leaf synthesis remain
separate open boundaries. The new numerical and resource proof does not close
the full stored scalar-effect graph or its constructor. Receipts use
g2-lp-full-widths-*20261125 and g2-lp-original-protocol-matrix-20261125.json.

Current batch acceptance passes 387 concentrated cases, 42 LP cases on Cirq 1.6
and 145 local integration cases. Strict typing covers all three changed Python
files. Black, Ruff, Integration and Algorithms scope, links, language,
architecture, IR and protected API checks pass. The existing 720 dependency
findings are unchanged; the global dependency check is not green. The original
41-case suite includes every slow parameter and has no deselected test. Full
LP stored-root/scalar-effect and SDK-default boundaries still prevent Q-0625
closure. No parent or constructor count is increased by this batch.

## Complete original test-file closure review

Q-0625 is eligible for acceptance under the approved
[faithful-migration rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md) after
publication and final local checks. The earlier stored-root condition was
resolved by the complete forward LP/LPRS constructor delivery recorded above.
Proposal 119 is not required for that existing scalar-carrier path. This review
does not silently introduce a general scalar-effect contract.

Every responsibility of the complete original test file now has evidence:

| Original definitions | Native verification |
| --- | --- |
| Both interim/full autotesters | Actual native stored constructors at shallow and expanded source-library depths, real SSA calls and output ownership, serialized native execution, and original protocol outcomes kept separately |
| Symbolic LP complexity | Complete symbolic historical T/Clifford/rotation fields, including the original symbolic versus concrete Clifford distinction |
| Both closed-form state helpers and both quick-state tests | Interim cos/imaginary-sin and final real-sin vectors compared against actual prepared states, retaining absolute phase |
| Both slow parameterized state tests | Every original width 1/3/5/7/9/11/13, complete complex state, coherent spectator and separately measured cleanup residual |
| Full LP historical complexity assertions | All seven original widths, exact 4n+4 T, 26n+33 Clifford and 6n+6 rotations, with the declared source call counts retained |
| Interim historical complexity assertions | All seven widths, exact 0 T, 3n+2 Clifford and 2n+1 rotations |

This table covers all nine original test functions and both state helper
definitions. The fresh complete original suite has 38 passes, one original
QDType skip and two original count xfails, including every slow parameter.
The skip identifies QFxp(5,5) to QAny(5); the native source-compatible opaque
word binding follows its already-approved contract without patching that source
validator. Original protocol failures remain visible, not converted to passes.

### SDK-default execution is now verified

The existing Ecosystem directional emitter can already import the default SDK
synthesis, including its explicit small matrix atoms. The missing condition was
an execution certificate, not an unavailable native conversion implementation.
The new whole-width scenario captures both source directions, serializes both
owned CircuitIR snapshots, then blocks Qualtran, Cirq and Kickmix imports before
binding reversed/offset physical ports and using ordinary circuit.append(op).
No external source object remains in the emitter or executes during that append.

All seven original widths now pass that separate SDK-default path. Forward
states match the unchanged source tensor with a coherent spectator, and private
work leakage is below 1.6e-29 in the recorded run. The accepted probability bound
is 1e-24; the complete amplitude bound is 2e-12. The captured generic adjoint
returns minus the original input on the prepared image, preserving the source's
known extra minus sign. Its maximum complete error is below 2.9e-14 against that
defective result. The independent native mathematical inverse remains correct
and separately labeled; it is not substituted into the literal SDK snapshot.

The captured default synthesis uses three private work wires and no measurement
slots at every checked width. Every explicit matrix atom acts on one qubit;
the importer does not contract the entire LP algorithm into a dense matrix.
This certificate uses the existing native execution path and makes no backend
substitution, acceleration or scalability claim. The canonical measured source
graph retains its separate all-history certificates and source lifetimes.

### Actual inventories versus source models

| Output width | Canonical native instructions | SDK-default instructions | Canonical private work / measurements | SDK private work / measurements | Original historical T / Clifford / rotations |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 86 | 82 | 5 / 2 | 3 / 0 | 8 / 59 / 12 |
| 3 | 162 | 238 | 7 / 4 | 3 / 0 | 16 / 111 / 24 |
| 5 | 238 | 736 | 9 / 6 | 3 / 0 | 24 / 163 / 36 |
| 7 | 314 | 1626 | 11 / 8 | 3 / 0 | 32 / 215 / 48 |
| 9 | 390 | 2908 | 13 / 10 | 3 / 0 | 40 / 267 / 60 |
| 11 | 466 | 4582 | 15 / 12 | 3 / 0 | 48 / 319 / 72 |
| 13 | 542 | 6648 | 17 / 14 | 3 / 0 | 56 / 371 / 84 |

These are three distinct accounts. SDK matrix atoms and rotations are not
silently priced as synthesized T gates. Canonical one-bit I instructions and
the source's atomic/wide Identity lowering difference remain explicit above;
cross-policy total primitive equality is not claimed. Every comparable root
call, interim operation, source-controlled rotation and non-identity canonical
inventory retains its existing source comparison. Default SDK snapshots retain
their own actual source operations, output aliases and global phases.

The modern seven-category model is independently checked at all seven widths.
At width 13 it remains And 14, Clifford 203, rotations 84 and measurements 14,
with source peak 30. Its peak excludes the external coherent spectator. It is
not rewritten to match the three-work-wire SDK synthesis or the scan of emitted
canonical primitives. Receipts use `g2-lp-test-closure` and
`g2-lp-sdk-resources-20261127.json` in the reference workspace.

The generic nonreal-reflection adjoint defect, original count/type failures,
atomic Identity granularity, unresolved physical sizes and unqualified private
presentation helpers remain separate limitations. They do not leave an original
assertion or source-defined execution path in this test file unimplemented.
