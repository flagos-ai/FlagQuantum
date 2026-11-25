# Original QPE walk-channel acceptance

This batch addresses the complete frozen `text_book_qpe_test.py` and
`qubitization_qpe_test.py` uniform-Pauli numerical scenarios and their direct
window, walk, reflection, controlled-walk and Fourier dependencies. It does
**not** close either whole-file parent Q-0634 or Q-0631. The remaining GQSP and
large-factory obligations below remain explicit. Original Qualtran v0.7.0 source
at `bcb17bb25360ed68ce3bc084645db3e929799389` is unchanged.

## Interface and ownership

The actual original helper supplies Hamiltonians with two, three or four Pauli
terms and two, three or four target bits, respectively. The selection register
is a bounded unsigned MSB-first word of width one or two, with its original
iteration length. Target bits retain the opaque native word descriptor. The
five phase bits are MSB-first. No source register is silently dropped.

Algorithms owns QPE composition. Ecosystem captures the actual original child
streams into native register emitters; numerical execution remains in Runtime
and Simulation. Compiler owns no execution. This is a `single_device_fast_path`
verification change and introduces no production algorithm, public entry or
protected schema. Users still append the existing bound operation with
`circuit.append(op)`.

Each captured child declares both work and classical-slot requirements. Callbacks
receive the shared maximum pool and slice it to each child's own requirements.
Private work is borrowed clean and returned clean before sequential reuse.
Measured slots are reused only by the existing child protocols. The former
one-selection-bit, one-target-bit test fixture cannot express these original
three- and four-bit cases and is not reused as their wiring oracle.

## Complex channels and original source schedule

`tests/unit/test_qpe_original_walk_channels.py` checks all six original
width/window scenarios for each QPE family: rectangular and Luis-Perina five-bit
windows, with target widths two, three and four. There are 56 original target
eigenvectors per family, 112 total. All full complex output amplitudes are
compared without fitting a global phase. Maximum observed error is below
`3.5e-14`. All phase distributions are normalized and their exact modes satisfy
the original 0.05 energy predicate; this avoids the original stochastic
1,000-shot mode estimator. Qubitization retains the original unresolved
sine-or-cosine predicate and paper Figure 2 discrepancy.

For each of the nine walk/controlled-walk/zero-controlled-reflection entries,
the actual native emitted forward and independently emitted reverse streams
are evaluated on every logical basis column. Original forward tensors and
conjugate transposes provide independent references. Every private work
amplitude is checked, rather than merely observing that allocation leases end.
The existing erasure certificate proves every raw child measurement history
has the same positive scalar channel. The window is evaluated from its clean
input, compared with the original five-bit state, and explicitly unprepared
with zero work. The shared inverse QFT's complete matrix matches the original.

The unmodified root's boxed operation schedule, powers and operands are checked
before literal concatenation of actual native child instructions is compared
with `DynamicCircuit.append(op)`. The inverse instruction stream is checked
against reversed independently emitted children and window unpreparation.
Composing these certified channels checks all eigenvectors and all histories
without enumerating exponentially many full-state executions. The largest
forward case has 428 raw measurements: the proof covers all `2^428` histories;
it is not a claim that they were individually simulated. Reverse channels and
prepared-image restoration are checked independently, including every work bit.

An additional complete 20-wire Runtime execution covers the three-term,
Luis-Perina, qubitization case with a complex coherent spectator entangled with
all eight target eigenvectors. The entire final state, including all nine clean
work bits, is compared with the source channel. This one trajectory is an
execution cross-check; it does not replace the all-history proof.

## Actual instructions versus source models

The SDK capture includes native `any` matrix instructions. They are counted
explicitly as opaque leaves. An independent observer identified five distinct
one-bit diagonal `cirq.MatrixGate` values across the actual forward/reverse
child captures; none is an entire walk contracted into a matrix. They are **not** relabeled as elementary primitives or
assigned synthesized T costs. The following inventory is the actual forward
emitted instruction count, including those leaves, measurement and reset.
The JUnit receipt also records every instruction category and opaque leaf width.

| Target width | QPE | Window | Instructions | Opaque leaves | Measurements | Physical wires |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2 | Textbook | Rectangular | 1107 | 93 | 31 | 9 |
| 2 | Textbook | Luis-Perina | 1340 | 93 | 37 | 17 |
| 2 | Qubitization | Rectangular | 821 | 117 | 1 | 9 |
| 2 | Qubitization | Luis-Perina | 1054 | 117 | 7 | 17 |
| 3 | Textbook | Rectangular | 9973 | 279 | 372 | 15 |
| 3 | Textbook | Luis-Perina | 10206 | 279 | 378 | 19 |
| 3 | Qubitization | Rectangular | 11799 | 351 | 422 | 15 |
| 3 | Qubitization | Luis-Perina | 12032 | 351 | 428 | 19 |
| 4 | Textbook | Rectangular | 3029 | 279 | 93 | 13 |
| 4 | Textbook | Luis-Perina | 3262 | 279 | 99 | 20 |
| 4 | Qubitization | Rectangular | 2887 | 351 | 63 | 13 |
| 4 | Qubitization | Luis-Perina | 3120 | 351 | 69 | 20 |

Separately, the source root model charges one window, 31 controlled walks and
one inverse QFT for textbook QPE. Qubitization charges one window, one controlled
walk, eight zero-controlled reflections, 30 uncontrolled walks and one inverse
QFT. The actual source model dictionary is recorded independently in each
receipt. It is not derived from the emitted instruction inventory. Existing
native composed model/category/peak tests are reused in the concentrated
regression; this batch does not invent elementary costs for opaque captures.

## Remaining whole-file obligations

- Q-0634: the actual original Hubbard GQSP root's native constructor/decomposition
  path at time 5 and precision `1e-7` still needs complete acceptance; its existing
  original-source pass is retained separately.
- Q-0631: five actual large QubitizationQPE factory physical paths remain distinct
  from their already-accepted declarations and models. Source Ising object-array
  serialization failure remains an original failure.
- Source-default SDK opaque leaves are now visible in this numerical path;
  replacement with source-comparable elementary streams is still required where
  an upstream elementary decomposition exists. Matrix action does not establish
  primitive gate equality.
- The ordinary Luis-Perina emitter works here. Complete stored LP/scalar roots
  still depend on pending Proposal 119. Other pending proposals are unchanged.
- Original strict-dtype/count/serialization skips and expected failures, both
  separately tracked notebooks, and the qubitization paper discrepancy remain
  in their existing audits. ReadTheDocs remains deferred.

The complete file parents are deliberately left open. Evidence names in the
reference workspace start with `g2-qpe-original-walk-`; initial child captures
and operators use `g2-qpe-original-walk-*-intake-20261125`. The original source
and test Git blobs, Qualtran 0.7.0, and native Kickmix 0.1 Simulator are verified.

## Batch verification

All twelve original width/window/family configurations pass, covering 112
Hamiltonian eigenstate columns with maximum complex discrepancy below
`3.5e-14`. The complete 20-wire Runtime witness also passes. Concentrated native
QPE/window/Hubbard-GQSP regression: 53 passed. Independent Cirq 1.6 QPE
regression: 34 passed, including the Runtime witness. Original complete-source
10 textbook and 19 qubitization passes, original skips/expected failures and
notebook evidence are reused from the existing audits; no source test is edited.

Strict typing, Black, Ruff, Integration scope, repository language, documentation
links, architecture, multi-level IR and protected API checks pass. The dependency
policy retains its existing 720 diagnostics with none in the new file; this is
not a whole-repository dependency-policy pass. No production owner changed, so
prior 145-case local integration evidence remains applicable. A first concentrated
invocation omitted the existing OpenFermion reference path; the diagnosed missing
module was corrected using that installed path before the passing 53-case run.
No dependency was installed or reference code modified.
