# Private predicate phase construction and public-entry proposal

Historical #974 delivery record. The subsequent
[public phase delivery](SYNTH_PUBLIC_PHASE_DELIVERY.md) supersedes the public
keyword status and contributor command below. This record retains the original
private evidence and target limitations; its proposal/approval statements
refer to that earlier stage.


Date: 2026-10-31. Development evidence, unreleased. #972 merged as
d6df7bca0d5f68b760518af023f00f0669b66145 after 57 successful checks and three
configured skips. The user's merge-and-next instruction authorizes this private
Compiler slice. The public output_mode keyword is proposed and not implemented;
API-owner approval is null, independent user trials remain 0/2, and release
permission remains false. See the [effect/proposal](../api-changes/FQ-SYNTH-PREDICATE-PHASE-20261031.md)
and [pending contract](../../contracts/synthesis-phase-admission-proposal.json).

## Effect and shared implementation

For verified one-bit f, the circuit implements
O_f |x,0_work> = (-1)**f(x) |x,0_work>. Inputs retain their encoded values and
intentionally acquire relative phase. No result flag is exported. The existing
Compiler closed profile, source capture, bindings, lifetime analysis and both
schedules are reused. The selected forward instruction prefix U is captured
once at the point the predicate is available; Z marks its one-bit value and the
literal reverse of that same X/CX/CCX prefix implements U-adjoint. No second
selection or resynthesis constructs the inverse. Non-admitted gates, parameters,
custom matrices and metadata are refused by this closed-prefix proof.

The allocation trace mirrors compute/cleanup events in reverse, including the
same physical bindings and scratch, and counts inverse rematerializations.
Both instructions and trace events remain bounded. Phase binding has no output
ports, requires a one-bit return and declared zero workspace; it does not prove
an external caller's asserted-zero precondition. Arbitrary integer truthiness
and dirty-workspace recovery are outside this contract.

Preparation remains outside U and touches inputs only. Shared target selection,
constraints and actual resource checks remain authoritative. The same private
Services complete factory creates the canonical Program, and the existing
fq.run/program.probabilities execute/read its bound plan. No alternate IR,
executor, wrapper, registry, root export or public serialization was added.
The private arithmetic diagnostic is profile-specific; only common complete
source/actual resource fields are used as phase costs. The future public proposal
omits arithmetic_resources for phase while retaining existing value diagnostics.

## Actual contributor journey

Run python -m examples.synthesis_phase_oracle. This is explicitly an expert
private bridge, not a new maintained user entry. It captures x*x+y<5, selects a
phase schedule, prepares a uniform input outside the oracle, creates the same
checked Program and executes it. A full complex-state reference verifies every
marked sign and clean workspace at tolerance 1e-12; it does not infer success
from a uniform probability table. The measured residual was 3.33e-16.

Raw lowering without preparation/target optimization:

| Predicate | Schedule | Total qubits | Workspace | Operations | Depth |
| --- | --- | ---: | ---: | ---: | ---: |
| x*x+y<5, two-bit inputs | retain | 21 | 17 | 207 | 140 |
| x*x+y<5, two-bit inputs | recompute | 18 | 14 | 295 | 188 |
| x==y, two-bit inputs | either | 9 | 5 | 63 | 56 |

The uniform contributor workflow actually executes 18 qubits with 14 workspace
qubits, 273 operations and depth 178. Its planned complex128 state buffer is
4 MiB; this is not process peak memory. Source costs and actual transformed
costs remain separate. A value-mode flag for the same polynomial uses one more
visible qubit; no performance/optimality/hardware advantage is claimed.

## Target and controlled-conjugation boundary

Native logical CPU targets pass complete phase references. The private
PHASE_BASIS line profile preserves phase/rz/u3 plus z/cz as native, disables
target optimization and admits plain native preparation only. This prevents
projective single-gate rewrites at this boundary. Arbitrary requested bases,
control metadata in preparation, and unverified profiles fail explicitly.

Controlled conjugation changes only the marker to CZ: U-adjoint C(Z) U. The
same compute and inverse remain uncontrolled, so the control-zero branch is
exact identity. This is verified on entangled data/control amplitudes under
logical and conservative line compilation, including constant true/false
predicates where a lost global phase becomes observable. This does not admit
general controlled source syntax or generic Circuit.control legalization.

Line evidence is separate: the polynomial phase workflow passes; two-bit
x==y phase is explicitly refused by the existing target.layout SWAP-witness
guard after normal Runtime optimization. That refusal is retained and tested,
not counted as a successful oracle. Existing projective cphase conversion is
also unsuitable for arbitrary later controls. Global-phase IR is an existing
[approved but unimplemented proposal](../api-changes/FQ-IR-GLOBAL-PHASE-20261006.md).
No protected IR field, evaluator threshold, optimizer contract or routing guard
was changed to make this slice appear to support more targets.

Therefore the public output_mode proposal initially permits Python one-bit
predicates on logical CPU targets only, with basis=None and topology=None.
General line phase admission requires routing-witness work and more phase
preservation evidence. Native Qmod phase, arbitrary angles and within-apply DSL
are separate future work. Existing public value/comparison/Qmod behavior stays
unchanged and public phase keyword calls still raise TypeError.

## Acceptance

The suite executes all 16 two-bit inputs for polynomial less-than and equality
under both schedules (64 basis executions), compares complex phase-bearing
states and clean workspace, and checks selected physical layouts. Controlled
marker tests cover arbitrary entangled input/control coefficients, both schedules,
logical/line targets and exact sign at tolerance 1e-12 without global-phase
quotients. Interference changes the observed one-bit outcome, and applying the
oracle twice restores the prepared state. Binding, predicate width, prefix
adjoint, operation/search/memory caps, target profiles, source acquisition refusal
and the canonical Program owner are covered. The actual private example runs in
a subprocess; the documented future public code is clearly marked as proposed.

## Local verification record

- New Compiler phase, real CPU/CLI and public-proposal gates: 53 passed.
- Existing arithmetic, comparison, source, public API, Algorithms and compiler
  comparison regressions: 1361 passed, 21 optional-dependency skips. The three
  unchanged Qmod originals pass all six public file/text full-size runs.
- Focused instrumented phase tests: 43 passed, ten already-verified larger cases
  excluded from that coverage subset. The three instrumented shared modules
  have aggregate 85% statement/branch coverage in this subset; this is not a
  repository-wide coverage claim.
- Strict mypy: all 876 package sources plus the private example passed. Current
  and minimum Black 24.1.0, Ruff, API protection, architecture/dependencies,
  vocabulary, language, maturity, generated docs, links and scope checks passed.
- Default smoke/unit: 4562 passed, 121 skipped, stopped at the same three
  previously observed macOS native-QFT operation-count failures (44 versus 28)
  in tests/team/simulation/test_controlled_phase_graph.py. Those implementations
  and expectations are unchanged; local default is not claimed green.

Existing public signatures, exports, historical admission contracts and API
snapshots are unchanged. Remote CI must establish its own exact-head result.
