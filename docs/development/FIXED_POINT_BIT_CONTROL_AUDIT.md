# Single-bit fixed-point execution dependency

Approved Proposal 113 closes the concrete unsigned QFxp(1,0/1) storage dependency
of the complete controlled-source review. The frozen source remains Qualtran
v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`, copyright Google LLC,
Apache-2.0. Core owns canonical physical admission. Compiler reuses the existing
unsigned predicate, typed Split/Join, component calls and exact adjoint owners.
Simulation retains numerical conversion. Users continue through
`circuit.append(op)`.

## Compared execution

Both fractional widths, both polarities and scalar/singleton-array controls
match the original decomposition calls, operands and feedback. The explicit
zero-gate native Split/Join views preserve descriptor identity and output aliases.
Every complex operator column and an entangled spectator state match the
unchanged source tensor. Applying the controlled X twice restores the input.
The existing measured-child composition matches both unnormalized measurement
branches, cleans its reused work bit and matches all four forward/inverse history
pairs. Stored append and specialized-child replacement reuse the established
component witnesses, including expanded and stored calls.

Source QFxp `from_bits([1])` returns raw integer 1 for both fractional widths.
At fractional width 1, `to_fixed_width_int(0.5, require_exact=True)` returns 1.
The native real-value codec agrees. The control activation is raw integer 1. Earlier intake wording about numerical values must be read with this
explicit distinction.

## Resources

| Operation | Observed primitive stream | Original analytical model |
| --- | --- | --- |
| Positive controlled X | 1 CX | 1 Clifford |
| Negative controlled X | 2 X + 1 CX | 3 Clifford |
| Positive measured child | 1 CH, 3 H, 2 T, 2 Tdg, 6 CX, 1 S, 1 phase, 1 measurement, 1 CZ, 1 reset | 1 T, 1 AND, 5 Clifford, 2 rotations, 1 measurement |
| Negative measured child | Positive stream plus 2 X | 1 T, 1 AND, 7 Clifford, 2 rotations, 1 measurement |

The direct X needs zero clean work and zero classical slots. The measured child
uses one clean work bit and one measurement slot. Its source model estimates 27
T-equivalents. Actual primitive T/Tdg count is four and is reported independently.
The full counters, models and source blob checks are in
`ftqc-reference-env/g2-fixed-point-control-evidence.json`.

## Boundaries and existing-test correction

The source matrix-singleton decomposition, tensor-contract and SDK-unitary
assertion failures remain explicit, with its raw tensor inspected separately.
Signed one-bit construction and wider QFxp CtrlSpecAnd are original refusals.
Canonical wider, symbolic and malformed native storage remain refused. The
symbolic fractional-width source control also refuses single-bit extraction.

Negative integer fractional widths are a genuine remaining source-supported
physical gap. Six unchanged original cases at -1, -2 and -7 successfully emit
CX or X/CX/X, while approved Proposal 113 deliberately permits only 0 and 1.
Proposal 114 records this separate extension and awaits explicit approval.
Q-0158 and Q-0159 remain open. No new parent or capability acceptance is counted.

The old logical-register test expected bounded unsigned words to be refused,
although their earlier approved physical implementation already accepts them.
All three failures reproduce using the pre-batch verifier from `3d85c62d8`.
The corrected assertion checks that admitted bounded storage returns width four,
while both wider fixed-point records retain their physical refusal.

## Validation evidence

The focused run passes 117 checks, including the extra symbolic-source witness.
Strict typing passes all six changed Python files with the frozen-source type
path. Black, Ruff, Integration and Compiler scope, architecture, IR and public
API checks pass. The unchanged original controlled test file passes all 42
cases, including its real notebook, with three source collection warnings.
Source kernel paths are explicitly inherited by its child process.

The default smoke/unit command passes 434 checks and skips 51 before the existing
public-docstring coverage failure. Its nine named modules and test file are
byte-identical to pre-batch `3d85c62d8`. Receipts preserve the failure rather than
reporting a clean default tier. The initial concentrated run passed 10,675 checks
and encountered 13 Tk setup errors. Its headless Agg rerun passes all
10,688 checks with 32 warnings, including the 30 existing paired overflow
warnings. Local integration passes 17 checks. The extra symbolic-source witness
was added after broad-run collection and is included in the final 117 focused
checks. All three source blobs and Qualtran 0.7.0/Kickmix 0.1 Simulator presence
are verified. Dependency policy reports 687 existing whole-repository findings
and zero batch-path findings.
