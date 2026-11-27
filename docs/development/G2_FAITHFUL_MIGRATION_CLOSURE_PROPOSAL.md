# Proposal: complete G2 migration with explicit source and platform limits

Status: explicitly approved by the user on 2026-11-27. Apply the rule below
individually; approval alone closes no parent. The user separately approved
Proposals 118, 124, 126 and 127 in the same decision.

## Current conflict

The ledger records 1113 accepted parents and 167 open parents. Recent SDK,
typing and reference-check repairs improved verification but closed no new
parent. They must not be presented as new algorithm delivery.

The original plan permits a source component that provides only formulas to
retain those formulas and their applicability. Its current section 99 also
requires original acceptance conditions for source defects, with only three
explicit exceptions: Q-0649, Q-0766 and Q-0924. The approved, limited decision is
recorded in [the source-defect disposition](G2_UPSTREAM_DEFECT_DISPOSITION.md).
That historical decision was limited to those three parents. The user approved
the extension below on 2026-11-27; its evidence requirements remain binding.

## Proposed G2 closure rule

Accept a complete source file only after every definition, direct dependency,
original asset and source-defined supported execution path has an authoritative
FlagQuantum owner and adequate evidence. Preserve the ordinary
`circuit.append(op)` entry. For comparable decompositions, retain calls,
operands, order, measurement feedback, full complex amplitudes, ownership and
workspace cleanup. Keep physical primitive inventories separate from modeled
resource estimates. Local mandatory checks and the published branch must pass.

Apply these dispositions individually, after inspecting each complete audit:

| Remaining condition | Proposed disposition |
|---|---|
| A supported, defined source path succeeds but native implementation, invocation or execution is missing | Keep the parent open; implement the missing path. |
| A source method provides only a formula, signature, abstract protocol or explicit decomposition refusal | Port that complete responsibility and retain its refusal. Additional physical algorithms need an independent specification and evidence; they are not invented as a prerequisite for copying a formula-only source. |
| A reproduced upstream defect is the only remaining condition after complete faithful migration | Permit G2 migration acceptance while retaining the failing input, source behavior, native impact and owning follow-up. Never call the defective result mathematically correct. |
| A corrected native path differs from the frozen source | Label and validate it separately. It cannot stand in for literal source parity or an unapproved behavior change. |
| A benchmark workload is completely ported and its state/output is verified, but its original speed target is unmet | Permit functional migration acceptance only with an explicit unmet-performance entry. Keep original timing targets, measurements and follow-up; claim no performance equivalence. |
| A platform-specific runtime or ISA is unavailable on the current host | Require complete source responsibility and available defined-domain evidence. Track the missing platform certificate independently; claim no execution, acceleration or cross-platform equivalence without it. Missing architecture functionality remains open. |
| Native checks fail, evidence or original assets are missing, or a protected contract awaits approval | Keep the parent open. This proposal does not waive these gaps. |

This is not approval to mark all 167 remaining parents complete. Each parent
still needs a complete, reviewed evidence bundle. Unsupported native inputs,
missing constructors for valid source transports, skipped required tests and
unverified native-only behavior cannot be converted into source defects.
Defined source behavior must not be relabeled as undefined behavior merely to
avoid implementing it.

## Concrete review queue

* Q-0477, clean QROAM: [complete file audit](QROAM_CLEAN_FILE_AUDIT.md),
  [constructor closure](RESOLVER_QROAM_CLEAN_AUDIT.md) and
  [forward/erase evidence](REGISTER_SOURCE_QROAM_CLEAN.md) already record the
  normal forward stream, signed banks, measurement branches, complex amplitudes
  and private zero-work behavior. The source adjoint leaf has no gate body;
  original omitted-selector failures and wrapper/leaf cost differences remain.
  Review these as source limits rather than asking for a nonexistent literal
  erasure stream. Preserve the independently specified native erasure.
* Finite-field square, multiplication and inverse: retain the source's
  ignored-adjoint and measurement-adjoint counterexamples. Inspect the full
  audits before accepting anything; do not equate an exact native inverse
  with the defective source decomposition. See
  [the inverse audit](FIELD_INVERSE_FILE_AUDIT.md).
* Hamiltonian and GQSP cleanup/precision: retain every measured residual and
  ownership limit. A source defect does not justify advertising nonzero logical
  junk as clean private work. A different native result signature or missing
  supported execution path still needs its own resolved contract and evidence.
* Performance and ISA parents: reuse verified original workload/state evidence,
  but retain unmet timing targets and unexecuted AVX/AVX512 certificates as
  explicit follow-ups. Missing implementations cannot close under this rule.
* The global reference dependency findings remain native verification gaps.
  They are not upstream defects. Q-0009/Q-0011 remain open until their required
  workflow evidence is complete.

## Separate protected interface decisions

The user explicitly approved all four prepared proposals on 2026-11-27,
separately from the closure rule. Implement Proposal 118 first:

| Proposal | Concrete scope |
|---|---|
| [118](API_CHANGE_PROPOSAL_118_CLASSICAL_REGISTER_CASTS.md) | Reinterpret an existing classical word with an equal-width nominal descriptor while retaining the same physical slots and immutable old snapshots. |
| [124](API_CHANGE_PROPOSAL_124_SIGNED_EMPTY_FIXED_POINT_STORAGE.md) | Admit the source-valid signed zero-width fixed-point storage profiles with negative fractional width. |
| [126](API_CHANGE_PROPOSAL_126_SYMBOLIC_QUANTUM_FIELD_DECLARATIONS.md) | Retain symbolic quantum finite-field and polynomial logical declarations; physical execution still requires concrete supported dimensions. |
| [127](API_CHANGE_PROPOSAL_127_STATIC_CONTROLLED_Y.md) | Support the existing positive-control CY instruction in static Cirq conversion without adding a Core instruction. |

Root AGENTS rule 8 requires explicit authorization and an approved proposal for
protected API behavior or vocabulary changes. Existing read/write approvals do
not authorize Proposal 118's nominal storage reinterpretation.

## Execution after a decision

Approval is recorded here and in the plan before changing acceptance. Review complete core owners first, reuse their published
evidence and commit implementation gaps independently. Move only the approved
source/performance/platform limits to explicit follow-up entries; keep their
evidence and original conditions. Update parent checkboxes, stage overview and
top status together through the existing progress script, then verify the PR
head. Report complete modules closed and exact remaining gaps. Continue G2
without merging the PR or entering G3 implementation.
