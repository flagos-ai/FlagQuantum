# Cost-function phasing source-file audit

## Current complete-file disposition (2026-11-28)

This review applies the approved
[faithful-migration closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md).
The definition map below covers all six inventory identifiers, both oracle
fields, the nested `_extract_soqs`, both original example factories and their
documentation specification. Compiler owns validation, named typed composition,
source-order calls and lifetimes. The existing evaluator, phase and exact
erasure owners supply their definitions; numerical execution remains outside
Compiler. Original notebooks and test assets retain their published evidence.

A real remaining supported path was reproduced and implemented. The source
permits `phase_oracle.cost_registers == ()`; its validation loop is empty,
but evaluation, phase and evaluation adjoint still execute. Native composition
previously rejected that empty set. It now accepts it while retaining duplicate
name rejection and every nonempty cost width, shape and RIGHT/THRU check. This
uses the existing Program builder and constructor callbacks, with no new Core
contract, opcode, public entry or automatic inverse inference.

New cases use the unchanged original `PhasingViaCostFunction` with explicit
test oracles implementing its original `QvrInterface`. They compare the actual
three-call root and producer/operand/result edges through both the direct
composer and the existing constructor. The phase extras cover scalar and
two-element arrays in all three directions:

| Extra direction | Native and source ownership |
| --- | --- |
| THRU | Caller retains the input/output word, including arbitrary coherence with input and spectator. |
| RIGHT | Phase produces an owned logical output; allocation ownership transfers to the caller. The prepared one state is not described as zero private work. |
| LEFT | Caller transfers a valid prepared input image for consumption. Physical output is zero before release. The source effect equals the native clean-output matrix block; arbitrary invalid inputs are not claimed to undergo a runtime projection. |

Full complex matrices of the source unitary/isometry/effect and every valid
coherent execution agree. Source SDK gates, operands, angles and each wire's
ordering match. Direct and constructor binding have identical complete IR,
outputs and allocation/release events; ordinary append and JSON remain valid.
The SDK reference uses its real qubit manager to allocate transferred LEFT
inputs, rather than asking it to free unrelated LineQubits. It does not patch
source code or weaken ownership checks. Independent gates may commute under
the already-accepted scheduler; the three stored stage calls remain ordered.

A separate nontrivial evaluator computes a private RIGHT copy which the phase
never reads. It is still evaluated and exactly erased. Full logical operators,
spectator-entangled complex amplitudes, private-bit zero, precise allocation
and release, and the original analytical model all match. Native constructor
and append also run with source SDK imports blocked. Unknown evaluator
semantics still require an explicit exact-adjoint owner; a supplied inverse is
not certified merely because its type fits.

Q-0674 is eligible for acceptance after local checks and publication. Original
Square remains atomic and has no source gate decomposition. Its existing
native measured arithmetic is an independently specified realization of the
source value/isometry contract, not a copied source primitive stream. Mixed
dtype, empty-junk, recursive width and generic transport/SDK failures keep
their original witnesses and follow-up owners. Unresolved numerical leaves
remain declarations/refusals. The approved rule changes their disposition,
not their observed results or the reference code. Generic oracle/adjoint,
transport, scalar and verification-workflow parents are not closed here.

Inventories and models stay separate. The new empty-cost identity cases emit
one phase per extra bit, plus one X per bit for RIGHT/LEFT, with zero
measurements. The private-copy case emits two CX and one phase, allocates and
frees one clean bit, and has no feedback. Its original model reports two
Cliffords, one rotation and peak three logical qubits; the numerical test has
four physical wires because it adds a spectator. Original five-bit tutorial
tables below retain their separate Toffoli, phase, measurement and width units.

Three additional symbolic-width Square roots retain exact original signatures,
ordered edges and all three calls: direct QVR with a numerical or symbolic
coefficient, and gradient QVR with a symbolic coefficient. They remain logical
Programs with unresolved atomic/physical leaves. The corresponding numerical
gradient coefficient fails in the unchanged source while constructing an Fxp
template whose nominally constant widths are still SymPy expressions. This is
retained as a source failure, not normalized to a different Python-integer path.

Local batch validation passes 170 concentrated cost-phase/QVR/Square cases and
32 integration cases. Independent Cirq 1.6.1 verification passes 31 constructor,
port and recursive-decomposition cases. Both changed Python files pass strict
mypy, Black and Ruff; Compiler scope, architecture, IR, required-check mapping,
English, Markdown links, public API baseline and whitespace pass. Sixteen
complete source/asset/dependency blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator
are verified. Receipts use `g2-cost-phase-full-closure-*20261128` and the
explicit symbolic intake in the reference workspace. Existing original grids,
notebook runs, source refusals, every applicable small measured branch and
large bounded coherent/resource certificates are reused within their stated
scopes. No exponential large-statevector or failed original path is claimed.

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `phasing_via_cost_function.ipynb` | `b1c45ff7e36c3d82b43166e98c39d5ba89c68762` | Q-0673 |
| `phasing_via_cost_function.py` | `7e4204a0dcc72fa5785b2e58f4c209d0f083dd76` | Q-0674 |
| `phasing_via_cost_function_test.py` | `959d1c2e1b94b83c505aa8d27747002c61540406` | Q-0675 |

## Tutorial and port bindings

Run `jupyter notebook examples/notebooks/phasing_via_cost_function.ipynb`.
The native notebook preserves the twelve original cells and adds a body
compilation example using `circuit.append(op)`. The two original factories
use five input bits, ten fractional cost bits, coefficient 0.1234, and
tolerance 1e-8. Inputs and the caller-owned 32-bit gradient are through
registers. The ten-bit internal result and scratch wires are initialized to
zero and restored after the phase body. The source order is
`Square → QvrZPow / QvrPhaseGradient → Square†`.

The native notebook rendered three SVGs in an independent kernel with
Qualtran, Cirq, and quimb imports forbidden. The original notebook also ran
in an independent kernel.

## Source grids and measurement evidence

The original suite returned 10 passes, two skips, and 29 expected failures.
The Hamming grid explicitly marks 24 Cirq simulation cases as expected
failures. Five Square cases hit the source flattening boundary. The source
autotester skipped the inferred call graph and the `QUInt(10) → QFxp(10,10)`
type check.

The native grid checks both register encodings for each of the 24 original
Hamming and 12 original Square parameter combinations. The original
cost and rotation graphs determine the phase and gradient translation.
For comparable leaves the tests compare gate names, operands, angles,
measurement slots, and feedback. The source Square leaf has no physical
decomposition. The comparison binds its published classical permutation
to the independently verified column compressor. The gradient walker
follows the original scaled-add Cirq children to their bound Add leaves.

The measurement tests project every small raw history. For larger circuits,
the balance and clean-work certificate checks every erasure invariant over
all input values with either all small gradient words or representative
large words including both ends of the word range. Two sparse trajectories
compare the full supplied complex states. The 32-bit tutorial body is
verified over all five-bit inputs and six gradient basis words, including
wrap boundaries. Full Fourier-state kickback execution is checked on the
smaller fixtures.

| Five-bit body | Literal T/T† | Literal CX | Literal phase | Literal measurements | Source QEC categories | Reserved / source wires |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Square with Z powers | 160 | 394 | 10 | 40 | 40 Toffoli, 10 rotations | 35 / 15 |
| Square with gradient | 1636 | 4062 | 0 | 409 | 520 Toffoli | 98 / 47 |

Measurements and width describe the compiled body. The caller owns gradient
preparation and optional unpreparation. The direct phase gates are counted
before synthesis.

## Remaining source boundaries

Q-0673 and Q-0675 are covered. Q-0666's export inventory is accepted.
Q-0674 remains open for the unchanged source Square decomposition and mixed
fixed-point type boundary. Concrete binding of symbolic fields and external
RIGHT cost oracles is now exercised; unresolved symbolic values are rejected
before physical execution. This does not promise execution for every external
oracle or every fixed-point format.

## Complete source binding and ownership

The whole source file maps as follows:

| Source definition | Owned implementation and evidence |
| --- | --- |
| `cost_eval_oracle`, `phase_oracle` and post-init validation | Three bound stages in `compiler/registers/phase/cost_function_phasing.py`; source-side RIGHT cost and THRU phase admission is retained. |
| Cached `signature` | Source left ports and phase extras bind to named native registers. Cost output is private; the gradient remains caller-owned. |
| `build_composite_bloq`, including `_extract_soqs` | Compute, phase, and true oracle adjoint at the same `BlackBoxPrepare` append consumer, with named output and scratch ownership. |
| Both Square factories | Actual original objects are imported in both directions; smaller resolved instances exercise complete logical operators and every erasure history. |
| Documentation specification | Both original examples remain in the native notebook and catalogue; earlier independent kernel evidence is retained. |

`test_cost_phasing_source_binding.py` supplies a real optional-source oracle
with two copy CNOTs into a newly allocated `QFxp(2,2)` cost register. Symbolic
fields are explicitly resolved, then both signs of the coefficient and both
QVR strategies are captured without changing the reference. Native callbacks
replace these three stages at the same append consumer. Every logical column,
including every small gradient basis word, agrees in full complex amplitude;
all private cost and carry wires return to zero. The tests check owned JSON
round trips, true separately captured adjoints, all-history channel identities,
and exact opcode histograms. Finite-precision Fourier kickback is tested against
its approximation tolerance, separately from exact basis-word translation.

The optional Cirq boundary now recognizes the exact source `Square` leaf and
delegates to the existing native integer-square column compressor. This is a
realization of the source's value/isometry contract using the paper construction
(arXiv:2105.12767, page 76), **not** a newly discovered source gate recipe.
The frozen arithmetic source blob is
`efed2d6652c7dcc943c82e0c5d35ae5b3b627c09`. Its decomposition still raises
`DecomposeNotImplementedError`; no reference is patched. Widths one through
three compare every valid input against the original numerical tensor and
restore input/result/work with the true source adjoint. RIGHT result starts
at zero; LEFT consumes the matching square. Arbitrary invalid LEFT inputs
are outside that source contract. Reusable measurement storage and carry
allocation remain owned by the existing exporter and qubit manager.

The original five-bit direct factory is also executed on a coherent state
containing all 32 input values with two sparse trajectories. Its every-history
erasure certificate is checked separately. The original 32-bit-gradient factory
is captured in both directions and its primitive counts are checked; this is
not a claim of exhaustive execution over all 37 logical input bits. Earlier
large-word arithmetic evidence and the bounded full operators remain separate.
The table above describes the native tutorial's reserved layout; imported
source snapshots can reuse lifetimes and reserve fewer physical wires.

## Validation of the binding extension

The two new test files pass all 12 cases (8.46 seconds); 668 concentrated
regression cases pass (83.70 seconds), followed by the final focused run after
adding the original gradient-factory capture. CPU/representation integration
passes 60 cases (41.29 seconds), and the native cost-function notebook runs
in an independent kernel (2.49 seconds). Strict typing of all four changed
Python files, Black, Ruff, Ecosystem scope, architecture, multilevel IR, and
protected API checks pass. Serial branch coverage of the two adapter files
is 82% combined; the new square adapter is 88%. Defensive admission guards
remain uncovered, so this is not a complete branch-coverage claim.

## Stored composition interface

The next catalog closure composes separately supplied evaluation, phase and
exact erasure Programs. Cost names identify RIGHT evaluation outputs and THRU
phase inputs of matching width/shape. The outer signature contains evaluation
inputs and phase extras. The evaluator and its exact inverse own private results
and work, while the outer consumer retains input words and the through gradient.
The Square realization remains explicit native arithmetic below the unchanged
source decomposition refusal. User execution remains `circuit.append(op)`.

## Complete stored dependency graphs — 2026-11-22

Both original cost-phasing factories now compose evaluation, QVR and exact
evaluation erasure as three stored typed calls. Their combined recursive
source comparison accepts five exact graphs and retains 42 atomic contexts,
including both Square directions. Eight original source graphs and all four
factory execution paths are covered by the shared
[QVR batch evidence](QVR_FILE_AUDIT.md#complete-stored-dependency-graphs--2026-11-22).

The original root filename collision retains its existing first-file behavior.
Contextual descendant owners include the actual phase strategy so both complete
bodies remain navigable. A regression checks each node's outgoing edges and
Program ownership. Small complete source tensors, every forward and inverse
raw history, spectator amplitudes, private work zero, append, copy and JSON
round trips pass. Original Square remains atomic. Its execution delegates to
the previously accepted native arithmetic and carries no source gate-recipe
claim.

The original five-bit Z-power instance emits 160 T/Tdag, 394 CX, ten phases
and 40 measurements on 35 physical wires. Its independent source model is
40 Toffoli plus ten rotations on 15 wires. The gradient instance emits
1,636 T/Tdag, 4,062 CX and 409 measurements on 78 used wires. The previous
reserved layout used 98 wires. The original model remains 520 Toffoli on
47 wires. Minimal contiguous clean-pool rebinding retains all instructions
and allocation events. Pending source and contract boundaries remain open.

Batch acceptance passes 851 concentrated catalog checks with three existing
deselections and 14 existing notebook warnings, 465 execution/SDK checks with
one original deprecated-cost warning, and 29 relevant local integration checks.
Strict typing of eight Python files, Black, Ruff, Compiler and integration
scope, architecture, multilevel IR, protected API and whitespace checks pass.
All thirteen frozen source/test/notebook/dependency blobs remain unchanged.
Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified. No source parent is
closed by catalog-only evidence. The recorded source defects remain open.
