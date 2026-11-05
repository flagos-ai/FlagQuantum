# UP-06 checked amplitude-amplification iterations

## Source-driven scope and reuse

The remaining corpus contains seven amplitude_amplification calls. The three
nearest original paths are quantum_counting.qmod, quantum_counting_iqae.qmod and
qmci_pi_estimation.qmod. Their shared blockers are a diagonal Z oracle with a
length-relative array index, a static iteration combinator, record interpretation
of an untyped qubit-array slice and phase effects inside binding-only apply.
The IQAE variant additionally declares one local flat record. Other callers still
have independent numeric, callback or arithmetic gaps and gain no coverage credit.

Reuse existing flat record fields and offsets, predicate XOR-to-H/phase/H,
classical integer binding, finite repeat expansion, borrowed phase validation,
phase kernels, binding scopes, physical scheduling and resource budgets. Add
private static source composition and pure-effect inversion, not an IR opcode,
numerical kernel, external SDK or stable export.

## Semantic contract

Admit amplitude_amplification(reps, oracle, space_transform, target) with named
single-array callbacks, a borrowed Const-array sign oracle, an inplace array
preparation and one whole initialized writable array or bound view. Static reps
remain in 0..256 and all effect/helper/scope/physical caps remain active. Prepare
A once and apply Q^reps, where Q=-A*S0*A^-1*S_chi and S0=I-2|0><0|. Preserve the
absolute minus sign on every round, including all-marked or empty-marked cases.
No amplitude parameter, normalization repair or cleanup assumption is introduced.

Expand through existing statement records and repeat syntax. Validate callbacks
even for zero repetitions. The sign-oracle profile permits indexed Z, global
phase(pi), indexed-bit phase(bit,pi), checked Boolean-controlled phase(pi) and
closed forwarding among such borrowed helpers. Fractional phases, non-diagonal
writes, allocation, captures, wrong roles and Output escape refuse. Existing
borrowed-phase source validation still checks unused definitions and indices.
Length-relative indices use bounded closed integer expressions; unknown quantum
values cannot become static indices. Actual bounds are checked when width binds.

Preserve existing inverses and add checked inverse composition of pure indexed
phase and pair-gate helper effects. Reverse order and invert every rotation/global
phase. Indexed controlled X, CX/CZ/SWAP remain self-adjoint. Ownership changes or
unsupported effects refuse before inverse emission. Binding-only apply may also
contain these validated phase effects. X apply_to_all and a phase flip on a
bound array view use existing indexed effects; whole-register routes remain
unchanged so prior resources do not drift.

A declared local flat record owns retained semantic storage just like an existing
local array. An untyped unsigned array or static slice can be interpreted as a
record helper argument only at the exact declared width and field order. Two
named records still require identical nominal types. No signed/fractional cast,
truncation, overlapping writable arguments or implicit fresh output is admitted.
Input/Const/drop/free ownership contracts remain unchanged. Dropped problem data
is retained and need not be zero; compiler scratch is separately verified zero.

## Acceptance

Keep all original bytes and license hashes. Run the three original paths for
reps=0,1,2, compare complete complex128 vectors to an independently constructed
public Circuit and analytic marked/unmarked amplitudes, without phase alignment.
Counting uses a+b<=2 on two two-bit fields. QMCI uses x*x+y*y<4 on one-bit fields,
so every state is marked and the per-round global sign must be observed. Validate
nonzero marker XOR, inverse coherence, field/array order, nominal type and width,
initialization, aliases, callback escape/capture, fractional oracle refusal,
negative/over-limit repetitions, zero-round validation, actual scratch and all
resource budgets. Provide runnable original Qmod and a valid native scalar
qfunc/Circuit comparison using current public APIs, without claiming a new native
amplification or array export. Recheck all 216 files and retain all previous 136
successful resource rows, original source blobs, AO/PF baselines and both known
source-semantic refusals. Update the ledger and proceed automatically; full CI
and a single integration PR remain behind the complete corpus acceptance gate.

Official semantics:
https://docs.classiq.io/sdk-reference/qmod/functions/open_library/amplitude_amplification
https://docs.classiq.io/qmod-reference/library-reference/open-library-functions/grover_operator/grover_operator
