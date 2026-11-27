# Native CX pulse synthesis

Numeric two-qubit matrices targeting CX, SX and RZ can use fewer local pulses
without loading Qiskit. The existing native synthesis boundary, source
annotations, global-phase quotient and preferred CX direction are preserved.
This is a `single_device_fast_path` compiler change, not a distributed feature.

## Upstream implementation and native modifications

The implementation adapts Qiskit 2.6.2 commit
`0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4`, specifically
[`get_sx_vz_2cx_efficient_euler` and `get_sx_vz_3cx_efficient_euler`](https://github.com/Qiskit/qiskit/blob/0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4/crates/synthesis/src/two_qubit_decompose/basis_decomposer.rs).
These are implementation adaptations, beyond citing KAK or the upstream pass.
The inspected Rust source SHA-256 is
`a74fbc72072ea6faf03043b15458123a4171880ca73f7ab60ec34c684473aa6c`.
The IBM 2026 source notice is retained alongside the earlier IBM notices and
Apache-2.0 notice in the modified module;
the repository LICENSE supplies the license text. Compiler owns maintenance.
No production dependency, upstream circuit object or additional IR is introduced.

The algorithm expresses control factors in ZXZ and target factors in XZX,
commutes exterior rotations across CX, and combines boundary matrices before
native one-qubit synthesis. The three-CX template uses Hadamard conjugation
internally. FlagQuantum exchanges both pairs of Weyl local factors and their
output labels before using this template. The canonical interaction commutes
with exchanging the two qubits, so this preserves the source operator and makes
the emitted CXs retain the caller's direction. No second KAK decomposition,
SWAP, final-layout change or direction-repair pass is required for this step.

The port reuses the existing ZYZ extraction, matrix multiplication, matrix
readers and single-qubit synthesis. It does not reproduce upstream global-phase
bookkeeping because the existing boundary explicitly identifies operators up to
one scalar phase. This path must not be used to drop a relative phase in a
controlled construction. It retains the upstream missing-RZ correction by
emitting the control RZ in the three-CX interior; phase verification uses a
largest-magnitude entry rather than assuming entry (0,0) is nonzero.

## Admission and resource bounds

The general factor sequence remains the baseline. Its cost is read from the
same numeric one-qubit leaf recipes used by emission, so an accepted proposal
does not require constructing both complete IR sequences. A proposed replacement is
admitted only when it has no more SX pulses, fewer total instructions, and a
complete 4x4 operator residual at most `1e-12` after removing one unit-modulus
phase. A rejected proposal retains that baseline. This is an internal candidate
selection, not execution fallback or acceptance of an approximate result.

The proposal handles exactly two or three CXs and fixed 2x2/4x4 algebra. Its work
and storage are bounded independently of circuit width. It does not change
compilation resource limits. Weyl classification tolerances are unchanged;
near-multiple branching is checked against the complete operator, not admitted
by a rounded fidelity. Other entanglers, pulse bases and symbolic workflows keep
their established paths. There is no new API, configuration option or schema.

## Development observations and limits

A fixed local diagnostic uses twelve QR-generated complex matrices with NumPy
seeds 27000 through 27011. Both participants use the exact CX/SX/RZ vocabulary;
Qiskit uses `pulse_optimize=True` and `approximate=False`.

| Implementation | CX total | SX total | RZ total |
| --- | ---: | ---: | ---: |
| FlagQuantum before this port | 36 | 180 | 234 |
| FlagQuantum with this port | 36 | 120 | 180 |
| Qiskit 2.6.2 with pulse optimization | 36 | 120 | 180 |

The modified native path's largest phase-aligned entry error was below `1e-14`;
the optimized reference was below `3e-14`. The common acceptance threshold is
`1e-12`. These are development-selected numeric matrix cases, not held-out
applications, qualified SDKBench evidence, timing or memory measurements, a
package release, or a global SDK ranking. Matching these counts does not prove
complete Qiskit synthesis parity. Extra proposal construction and verification
add compiler work; no compilation-speed improvement is claimed. The CZ block
path and previously reported cross-SDK studies are separate scopes.

Reproduce the current diagnostic with explicitly installed Qiskit and NumPy:

```bash
python -m benchmarks.compiler_cx_pulse_synthesis
```

The command prints actual package versions, aggregate counts and both complete
operator errors, and fails if either participant exceeds `1e-12`. It does not
assert that every upstream version must produce identical gate counts. The
historical before row is the pre-change source control, not produced by disabling
Qiskit's optimizer. The repository's existing formula-only anchor is distinct
from this optimized comparator.

## Retained compilation-time tradeoff

A serial local macOS CPU source control compared the same twelve matrices in
one process, with three warmup blocks per implementation and eleven measured
blocks in alternating order. Both numerical synthesis modules for the before
control were loaded from pre-change commit
`c9b6d5ab29e807ee5c4f582c39ca60ff82a51cd4`; Core and the
installed environment were shared. The median time per twelve-matrix block was
`0.0041105419950326905` seconds before and `0.006969959009438753` seconds after,
a ratio of approximately `1.70`. The reduced pulses therefore retain a measured
compilation-time cost on these development inputs. This is not a speed win.

The complete paired block times are retained below, in observation order, in
seconds. This is a small local diagnostic without Linux host qualification,
CPU pinning, exclusivity attestation or process-memory acceptance. It is not
SDKBench evidence, a hardware/runtime comparison or a statistically established
general regression magnitude. The initial prototype's separate local control
was about `1.91`; do not divide metrics across those runs or treat them as a
controlled measurement of the refinement alone. An earlier concurrent-test
probe is excluded from these serial observations.

```text
before_seconds,candidate_seconds
0.0041564170096535236,0.0069934170023771
0.00408475000585895,0.007034665992250666
0.00412066699936986,0.006968874993617646
0.004168042010860518,0.00705266599834431
0.004192082997178659,0.0069924999988870695
0.004029792005894706,0.006967875000555068
0.004091750000952743,0.007010291999904439
0.0042197089933324605,0.006904291993123479
0.0041105419950326905,0.006969959009438753
0.004090082991751842,0.00694145800662227
0.004052167001646012,0.006928959002834745
```

That source control checks both instruction economics and full numerical
correctness. Extra proposal algebra and complete-operator acceptance remain
explicit compiler costs; qualified whole-workflow time and memory evaluation
is still required before a broader performance conclusion.

### Row-pair verification refinement

Profiling identified dense multiplication of lifted one-qubit matrices as
avoidable verifier work. A local gate now combines the affected row pairs of the
complete operator directly, reusing the authoritative one-qubit matrix reader.
This computes all sixteen entries without multiplying structural zeros; it does
not inspect fewer columns or relax the phase or error thresholds. Two-qubit
instructions, reversed operands and unreadable inputs keep their matrix-reader
and rejection paths.

A subsequent source control used the same pre-change revision, twelve matrices,
three warmup blocks and eleven alternating measured blocks. The before median
was `0.004150291002588347` seconds and the revised candidate median was
`0.005828666995512322` seconds, a ratio of approximately `1.40`. Compilation was
still slower than the pre-port path. The earlier `1.70` observations remain
above; measurements from different runs must not be divided to assert a speedup.
These observations retain the same local-development limitations and do not
qualify as SDKBench results. The complete paired observations were:

```text
before_seconds,candidate_seconds
0.004243915987899527,0.005896208007470705
0.004249791993061081,0.005812125004013069
0.004121457997825928,0.005825792002724484
0.004137042007641867,0.005791624993435107
0.004111375004868023,0.005875417002243921
0.004144457998336293,0.005795999997644685
0.004116250012884848,0.005831542002852075
0.0042472500063013285,0.005751000004238449
0.004164957994362339,0.005980750007438473
0.004150374996243045,0.005844832994625904
0.004150291002588347,0.005828666995512322
```

### Fixed entangler preparation reuse

The target entangler's Weyl factors and closed-form constants depend only on its
published opcode. Preparation is now retained in a process-local LRU cache
bounded to the six supported entanglers. Invalid names raise before insertion.
The private objects are read without mutation, including their matrix factors;
no source unitary, circuit, output instruction, layout or caller metadata enters
the cache. The standard-library cache protects its index across threads; two
simultaneous cold calls may prepare the same deterministic constants twice.
Concurrent synthesis tests cover all six bases and distinct caller metadata.
This reuses preparation, rather than memoizing benchmark input matrices.

A new control retained the same twelve matrices, three warmup blocks and eleven
alternating measured blocks. Its before median was `0.004095957992831245`
seconds and its candidate median was `0.004311166994739324` seconds, a ratio of
approximately `1.05`. The candidate still did not beat the pre-port path. This
warm comparison excludes first-use basis preparation: a cold call still pays
that preparation, and each worker process has its own bounded cache. The earlier
`1.70` and `1.40` observations remain above; no ratio across runs is asserted.
The raw pairs below retain the same local-development limitations and provide
neither qualified SDKBench performance nor process-memory evidence.

```text
before_seconds,candidate_seconds
0.004233500003465451,0.0043257499928586185
0.004095957992831245,0.004323707995354198
0.004099584009964019,0.004412333000800572
0.0040964170038932934,0.004232332998071797
0.004038833008962683,0.0043196250044275075
0.004160750002483837,0.004274042003089562
0.0041185419977409765,0.0041918329952750355
0.003990875004092231,0.004273416998330504
0.0040893749974202365,0.004384166997624561
0.004067458008648828,0.004223791998811066
0.004071000003023073,0.004311166994739324
```

### HHL workflow resource regression

The first full CI attempt exposed a fixed CX-count expectation in the portfolio
HHL workflow: the new path emitted `(498, 498, 672)` instead of the earlier
`(612, 612, 824)`. This was investigated rather than retried as infrastructure.
A same-source local control rejects only the optional pulse proposal and
recovers the general factor sequence and all three earlier counts. Its U3 counts
are `(1887, 1887, 2549)`; admitted pulse proposals produce `(1653, 1653, 2245)`.
The CX reduction occurs in the complete synthesis/basis-conversion workflow;
it is not the three-CX random-matrix diagnostic above.

Both paths preserve the exact finite outcomes `(1154, 799, 91, 4)` for each
reduced profile and `(1132, 797, 108, 11)` for the basic profile, the same overlap
estimates and widths `(9, 9, 10)`. Complete converted statevectors are checked
against their native statevectors at `1e-12` up to the existing global phase;
observed maximum residuals were below `5e-15` on both paths. Separate existing
tests compare each native profile's absolute state with an independently built
HHL circuit. These are full state checks for the declared input, not complete
operators or held-out SDKBench evidence. The regression now checks both exact
resource tuples and both paths' state acceptance; no threshold was relaxed and
historical notebook evidence was not rewritten.

## Acceptance and supported workflow

The focused tests cover fixed random SU(4) matrices, framed two-CX interactions,
near-zero and near-Weyl-boundary inputs, both operand orders, nonconsecutive qubit
labels, annotations, zero matrix entries, deterministic output, corrupted
proposals and baseline retention. Existing conformance tests retain their
resource refusal just below the actual instruction count and success at it.
An end-to-end target test declares only CX/SX/RZ on a single directed edge and
sets the direction-repair budget to zero. It checks the complete operator and
proves that this optimization does not add an H requirement or reversed CXs.

```bash
python -m pytest tests/unit/test_compilation_cx_pulse_synthesis.py \
  tests/unit/test_compilation_two_qubit_synthesis.py \
  tests/team/compiler/test_two_qubit_decomposition_conformance.py \
  tests/team/compiler/test_direction_operand_semantics.py \
  tests/hybrid_compiler/test_target_legalization.py \
  tests/hybrid_compiler/test_native_gate_legalization.py \
  tests/hybrid_compiler/test_directional_topology.py -q
```

The existing user-facing target compilation and execution workflow remains:

```bash
python -m examples.circuits.compiler.target_aware_compilation
```

Expected: topology edges are valid, the mapping is restored, and the compiled
complex128 state agrees with the original circuit to `1e-12`. This existing
example demonstrates the enclosing workflow; the matrix-specific acceptance
and diagnostic above exercise the new numerical path directly.
