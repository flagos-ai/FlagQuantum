# Complete QFT original test acceptance

This batch covers the four complete frozen Qualtran v0.7.0 QFT test files,
323 lines in total, and their rotation/gradient/arithmetic/swap dependencies.
Reference commit: `bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC,
Apache-2.0. All eight implementation/test Git blobs are verified by
`test_qft_source_acceptance.py`. No reference source or test is modified.

## Interfaces and ownership

Data words are caller-owned THRU unsigned registers, in MSB order. Gradient
words are caller-owned fractional THRU registers. Shared arithmetic owns clean
carry/product workspace and measured feedback slots. Algorithm roots retain
source call order, actual operand slices and optional final data reversal.
Compiler/algorithm owners emit IR; Simulation/Runtime execute it. Normal use
remains `circuit.append(op)`.

The original AddIntoPhaseGrad and PlusEqualProduct children have no lower
primitive decomposition. Their native arithmetic implementations are verified
against every raw child input and original classical callback. This is a
realization of an atomic source contract, not a claimed upstream primitive body.
The unchanged source root's full boxed call/operand sequence is compared exactly
with the emitted native stream.

## Complete original assertion map

| Original file and every test | Owned evidence |
| --- | --- |
| Q-0644 `qft_text_book_test.py`: `test_qft_text_book_quick`, `test_qft_text_book` | Existing `test_textbook_qft_file.py` executes every full complex operator at widths 2..7, both reversals and directions, through ordinary Circuit append, against unchanged source and Cirq matrices. |
| Same file: `test_qft_text_book_t_complexity`, `test_qft_text_book_t_complexity_symbolic` | Existing source/native complete category formulas and generalization assertions include widths 10/123 and symbolic n. The original 123-bit small-angle xfail remains an upstream model limit. No 123-bit statevector is claimed. |
| Same file: `test_qft_text_book_auto`, `test_symbolic_qft_auto` | Both actual factories, all raw source protocols, native constructor/signature/root graph/byte-roundtrip and existing concrete binding evidence. |
| Q-0641 `qft_phase_gradient_test.py`: `test_qft_with_phase_gradient` | New full complex operators for all original widths 2..5, both reversals and directions, including exact inverse restoration, gradient restoration and zero arithmetic work. Existing small full-joint operators and physical runtime evidence are reused. |
| Same file: `test_qft_phase_gradient_t_complexity`, `test_qft_phase_gradient_small_auto` | Existing symbolic/concrete model owner covers original widths 10/123, original bound and zero rotations. The actual factory's raw count failure remains a failure. The original duplicate even-width child annotation remains distinct from literal execution. |
| Q-0638 `approximate_qft_test.py`: `test_approximate_qft_small_auto`, `test_approximate_qft_from_epsilon_auto` | Both actual source factories, all ten raw modes and existing concrete constructor/body/physical replacements. |
| Same file: `test_approximate_qft_exact` | New complete operators cover every original width 2..5, both reversals/directions; gradients and work are restored. |
| Same file: `test_approximate_qft` | New seven-data/six-gradient full 128-column complex operator proof, original 1e-2 relative/absolute tolerance, all measured histories and independent 24-wire native Runtime append with coherent spectator and gradient/work zero. |
| Same file: `test_approximate_qft_with_eps`, `test_approximate_qft_t_complexity` | Existing precision owner covers every original n=1000/2000/10000 and precision=4/7/10 combination, default bounds and symbolic expression. Existing resource owner covers n=10/123 and both reversal settings. Source model mismatches/negative small-width estimates are retained separately. |
| Q-0647 `two_bit_ffft_test.py`: `test_t_gate`, `test_call_graph`, `test_tensors` | Actual factory raw protocols; existing normalized T/T-dagger graph and all twelve k=0..3,n=1..3 matrices. Exact source scalar exp(-i*pi*k/n), both analytic tensors and native physical phases are retained. Raw count failure is not replaced by the generalized count pass. |

The source helper classes TestQFTWithPhaseGradient and TestApproximateQFT own
only their declared width/reversal fields, signature and prepare/root/unprepare
test envelope. Native physical phase-gradient preparation/inverse and the same
boxed source envelopes provide their corresponding evidence.

## Full operator and measurement proof

The test reads the original root decomposition with its atomic arithmetic
children retained. Each actual native child stream must match that root call's
literal operands and direction. The existing certified_and_channel_map checks
every raw child basis input, every compute/erase predicate and final zero work.
Its independently tested local primitive lemma gives the same positive
2^(-events/2) scalar for every measurement history. Thus arbitrary complex
superpositions and entangled external spectators retain their relative phases.

Only after proving those physical blocks are their measured work wires removed
from the numerical column composition. No ideal adder replaces an unchecked
block. All data input columns retain the full joint gradient amplitudes; gradient
factorization is checked numerically. Actual native gradient preparation and
inverse are independently executed and return to zero. The complete root and
envelope are also constructed through DynamicCircuit.append.

Seven-bit approximate QFT has maximum entrywise error about 0.0043383173 and
gradient factorization residual below 2e-17. Both reversal settings and exact
inverse composition pass. Each direction has 40 erasures, so the channel proof
covers all 1,099,511,627,776 raw histories. The independent native Runtime check
uses one concrete trajectory; it does not stand in for that exhaustive algebraic
proof. Its full 24-wire complex state includes an entangled spectator and zeros
on all gradient/work wires after the envelope.

## Separate resource accounts

For the seven-bit/six-gradient root, actual emission is H127, CX336, T80,
T-dagger80, S40, measurement40, feedback CZ40 and reset40. Reversal adds three
SWAPs. Gradient prepare/unprepare adds twelve H and twelve phase instructions.
Root physical width is 23, including ten clean arithmetic work wires; the
independent Runtime witness uses one additional spectator.

Original source QEC declarations and logical-width queries are recorded
independently in the JUnit properties. Abstract Toffoli estimates are not these
measured primitive inventories. Existing textbook, recursive gradient and FFFT
resource owners retain their own explicit model/physical distinctions.
The original seven-bit declaration is 28 Toffoli, zero measurements and
seven Clifford without reversal or ten with reversal, with logical peak 13.
It is not substituted for the 23-wire physical root inventory above.

## Batch verification

The concentrated batch passes 210 cases, including the full 24-wire native
Runtime witness and all sixteen exact gradient/approximate operator-grid cases.
The latter have maximum operator error below 2.7e-16 and gradient residual
below 6.6e-17. The independent Cirq 1.6 selection passes 53 cases, with only the
already-passed large native Runtime witness explicitly deselected. The unchanged
complete original four files, including their slow selections, retain their
earlier 83 passes, three skips and six xfails. Eight source/test Git blobs and
actual native Kickmix 0.1 Simulator availability are verified.

Strict typing of the complete new test file, Black, Ruff, Integration scope,
repository language, links, architecture, IR and protected API checks pass.
Dependency policy retains its existing whole-repository findings, with no new
file diagnostics after guarding the optional Cirq imports. Logs and JUnit
per-direction primitive/model/numerical properties are retained under
`g2-qft-source-closure-*20261125`. Production algorithms and source files are
unchanged; the batch closes verification gaps in their existing execution paths.

## Retained source boundaries

Six actual factories have 30 raw modes: 22 PASS, five FAIL, two NA and one
UNVERIFIED. All five original count failures remain visible, before source skip
policy. The symbolic textbook decomposition/type modes are NA and counts are
unverified. All six serializers pass.

The original FFFT is_adjoint decomposition still emits the forward body; its
analytic tensor and native mathematical inverse are separate. Original QFT
bookkeeping/count differences, duplicate gradient-child annotation, small-angle
classification and negative precision-limited model estimates remain recorded.
Implementation parents Q-0637/Q-0640/Q-0646 stay open. No unapproved scalar-phase
contract, symbolic physical domain, G3 work or PR merge is introduced.
