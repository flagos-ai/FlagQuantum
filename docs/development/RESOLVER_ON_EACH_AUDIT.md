# OnEach constructor dependency closure

## Source and ownership

The complete basic_gates/on_each.py and its original tests remain locked at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Existing whole-file
metadata, presentation, classical evaluation, SDK and repeated-body acceptance
in ON_EACH_FILE_AUDIT.md is reused. This batch closes the executable scalar QBit
child-helper dependency gap in formats/legacy_word_factories.py.

The resolver already retains each referenced child's real function and complete
helper closure. OnEach now builds its repeated calls against a declaration view
of that child. The existing component_graph helper isolates the wrapper's local
Split/Join definitions while retaining the external child identity and omitting
its declaration from the returned local module. The final resolver module owns
the real child once. Standalone fragments still fail verification when the
external child or one of its helpers is missing. No referenced-argument contract,
public entry, wire schema or storage admission changes.

The root takes and returns the same typed q word in MSB order. Repetition zero
retains its explicit empty word and does not execute the child. Symbolic width
retains a bodyless declaration and the resolved child closure. Finite binding
requires concrete dimensions. The normal user entry remains circuit.append(op).

## Execution and resource evidence

An actual serialized OnEach containing the accepted SU2RotationGate constructor
resolves at depths zero and one with the complete GlobalPhase/Rx/Rz closure.
Both directions and widths zero, one and three compare the original typed root
traversal and the exact accepted child primitive stream. Noncontiguous reversed
bindings, coherent spectators and full complex64/128 amplitudes are checked.
Every scalar carrier is restored to zero and freed between child invocations.
There are no measurements in this child, so the applicable branch set has one
history. Previously accepted measured-child primitive evidence is reused only
for its own components and is not claimed as a new OnEach measured execution.

| Account | Observation for width n and nonzero SU2 scalar |
|---|---|
| Actual primitives | 2n Rz, n Rx, 2n X and 2n phase |
| Actual private work | n carrier allocations and frees, each carrier restored to zero |
| Original direct model | n child calls |
| Original QEC model | n times the child's independent synthesis estimate |
| Original logical width | n data qubits, separate from the actual private carrier |

A replacement child built from a nested X function retains the same independently
owned leaf identity. It emits one X per bit and fails verification if the real
child closure is omitted. The earlier self-contained X/H/S/T, complete bitwise,
arithmetic and constructor consumers remain in the concentrated regression.

## Remaining boundaries

OnEach remains a partial catalogue mapping. Scalar QBit THRU execution is closed
for independently resolved children and arbitrary internal helper depth. Source
shape/direction and permissive non-QBit child metadata retain their accepted
logical owner and original failed-decomposition witnesses. Generic constructor
admission for those child signatures remains open. Source-valid signed and
zero-width fixed-point storage remains behind pending Proposal 116. No complete
identifier is added, leaving 66/303 complete and 237 remaining. Q-0246 and Q-0876
remain open. Generic Adjoint/Controlled/Power, arbitrary-target Pauli, pending
116-119 and whole-repository/GPU/performance acceptance remain separate gaps.

Batch receipts use ftqc-reference-env/g2-on-each-constructor-*.

## Batch acceptance

The concentrated dependency regression passes 3270 cases and local integration
passes 145. Focused constructor checks pass 89 cases, and all four unchanged
original OnEach tests pass. Four files pass strict typing, Black and Ruff.
Compiler scope, repository language, architecture, IR and public API checks pass.
Dependency policy retains the identical 720-line baseline. Five frozen source
blobs and the real Kickmix 0.1 Simulator are verified.

## Publication verification

The accepted module is published in the seven-commit range ending at
05031474d26059a3c531269738aa7b16fa07a923 on PR #1084. The GitHub Git
database API preserved every local blob, tree and commit SHA and updated the
integration branch with force disabled. The verified publication supersedes
the historical local-only transport status above. The combined published
catalogue covers 85/303 constructor identifiers, with 218 remaining.
The per-module execution, symbolic, codec and model boundaries remain recorded
at their original scopes. Exact publication evidence is retained in
ftqc-reference-env/g2-git-object-publication-20261123.json.
