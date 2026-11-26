# Complete Hubbard PREPARE constructor and dependency closure

Reference: unchanged Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`, Apache-2.0, Google LLC.
Implementation/test/notebook blobs are
`cfdbf20b0baf5ef1f5fa5a7e21826d7f51f5aec5`,
`ef956c7d0b68de1a5805d76f2c6dd4144adc9485` and
`11862cbbb6894da65d4636f06b9e8c654a1d02cc`.

## Ownership and interface

The existing Ecosystem packed-oracle provider already supported the ordinary
PREPARE constructor. This delivery replaces its direct Algorithms construction
with an authoritative Compiler factory; it does not claim that the prior
constructor was absent. Coefficient conversion uses the shared Simulation numerical
formula and hands two owned angles to Compiler. The native numerical operation
retains its existing finite nonnegative-weight validator. The literal source
constructor can also execute the original negative-coefficient examples whose
angles are real, without changing that validator.

`layout/hubbard_signatures.py` retains PREPARE's concrete integer bit_length
sizing, independently of SELECT's ceil/log2 sizing. All words are MSB first.
Selection ports are U, V, p_x, p_y, alpha, q_x, q_y, beta; the final temp word
has two caller-owned THRU bits. Those logical direction bits are not private
clean work and are not erased after forward preparation.

`preparation/hubbard_prepare_component.py` owns the complete typed call chain:
Ry, controlled Ry, two uniform preparations, direction Hadamards, source
controlled-H and copying, controlled word exchanges, atomic ModAddK, and
MultiAnd compute/exchange/erase. The nested copy-CNOT moment barrier stays
explicit. Existing uniform, increment, conjunction and swap owners retain
their own clean work, measurement feedback and exact inverses. Compiler
performs no coefficient evaluation or numerical state execution.

`formats/legacy_hubbard_prepare_factories.py` owns constructor validation and
scoped inverse recipes, with an injected angle converter and optional pair
observer. Unknown adjoints delegate to the existing packed owner. Algorithms'
old stored-builder entry delegates to the same typed implementation. Exact
source adjoints reverse calls with explicit child inverses, not reversed
measurement instructions.

## Source fidelity and active path replacement

The normalization remains `2*N*t + (N*u)//2`, including floor before angle
evaluation. The numerator is evaluated as `t*N`; NumPy arccos and the original
square-root conventions are preserved. Native integer coefficients reach that
conversion before any float rounding. Both the existing numerical operation
and source constructor use this single formula. Thirty-two complete forward
and inverse primitive streams match published `a597ccee0` exactly, including
angles, operands, feedback, output bindings and lifetimes. No angle drift was
observed in this baseline; the reference was not changed.

The first Ry model now uses the existing source-compatible rotation classifier.
At t=0 or u=0, it is a Clifford instead of an unconditionally charged general
rotation. The controlled-Ry historical source shim is unchanged. Tests compare
all QEC categories and peak width at generic, fractional and endpoint weights.
Actual named primitives remain independent of these model categories.

The same Ecosystem owner now materializes the previously bodyless uncontrolled
SELECT using its already accepted private scalar carrier. Both original packed
SELECT directions, all four original SELECT/LCU forward examples and both
Hubbard specialized-control LCU paths bind complete native primitive bodies.
The packed SELECT stream equals the direct shared SELECT stream exactly;
physical lease offsets and ownership match, while extra logical partition
operations naturally shift logical operation positions. Original controlled
packed SELECT missing-control partitions remain an explicit failure.

## Verification

The new 83-case constructor scope passes in both Cirq 1.7.0 and 1.6.1 and
covers the original factory and DocSpec,
serialization depths zero through two, both directions, default/fractional/
endpoint/negative source weights and exact source Ry parameter values. All
recursive source calls, named operands, output aliases and typed signatures
are compared. Frozen assets, Qualtran 0.7.0 and a real Kickmix 0.1 Simulator
are required; native source checks do not silently skip.

For fractional, endpoint and negative source weights, every one of the eight
small raw measurement histories is checked in both directions with arbitrary
complex logical input, including dirty logical temp. The reference executes
the unchanged original generator and uses the original atomic ModAddK classical
permutation where its primitive decomposition is absent. Full absolute complex
amplitudes and zero private work are checked, without quotienting global phase.
Normal append also executes both local strategies with source SDK imports
blocked: eight shots execute 24 measurements and preserve the same channel.

Affected primary consumers pass 113 cases, including packed and block
constructors, preparation, notebook models and package/GQSP model consumers.
Existing larger PREPARE and numerical notebook evidence is reused. Baseline
receipts compare named primitives, operands, feedback, private lifetimes and
output bindings against published `a597ccee0`. The complete original test file
passes ten cases with two source-owned skips: decomposition-derived counts
and strict bounded-to-opaque dtype checking. Those skips are not native passes.

Eight new resource rows separate literal execution from the unchanged original
model boundary. Local integration passes 145 cases. Strict typing covers ten
Python files, with format, lint, all three owner scopes, language/links,
architecture, IR and protected API checks.

| Preparation | Actual named primitive inventory | Independent original model |
| --- | --- | --- |
| Two-by-two, generic weights | 26 T + 21 T-dagger, 66 CX, 23 H, 19 X, 3 S, 3 CZ, 1 Ry, 1 controlled Ry, 1 CH, 3 measurements/resets; live peak 12 | 5 CSwap, 2 And, 11 Clifford, 4 rotations, 2 measurements; peak 12 |
| Two-by-two, t=0 or u=0 | Same literal gate names and counts, with the source endpoint angle | 5 CSwap, 2 And, 12 Clifford, 3 rotations, 2 measurements; peak 12 |
| Three-by-three, generic weights | 82 T + 73 T-dagger, 262 CX, 101 H, 79 X, 23 S, 23 CZ, 4 Rz, 1 Ry, 1 controlled Ry, 1 CH, 23 measurements/resets; live peak 20 | 9 CSwap, 17 And, 116 Clifford, 8 rotations, 17 measurements; peak 18 |

The source ModAddK leaf has no primitive decomposition. Native increment
work and gates are counted in execution, not assigned retroactively to that
abstract source model. CH and continuous rotations remain named primitives;
these rows do not claim a fully synthesized Clifford+T inventory. Pool capacity,
private allocation events and simultaneous live peak remain distinct.

## Remaining boundaries

- Singleton source decomposition and resource failures remain explicit. The
  independent existing native singleton operation stays available; the literal
  stored source constructor does not acquire an identity body.
- Symbolic weights retain their constructor metadata and logical declaration.
  Physical acos/floor expressions are outside Proposal 107's approved
  sin/cos/exp/log/pow/div vocabulary and remain refused. No new Core opcode or
  parameter operator is introduced.
- Zero normalization reproduces the original division failure. Non-square
  dimensions and malformed metadata retain their admission diagnostics.
- Deeper packed AutoPartition tuple serialization and original LCU specialized
  inverse enum serialization retain their original failures. The source is
  unchanged. Source-connected graph versus numerical primitive boundaries are
  not inferred from resource counts.
- The prior Cirq 1.6 versus 1.7 CY port representation boundary belongs to
  SELECT descendants and remains separately recorded. No SDK-dependent
  representation selection is added to Compiler.

The ordinary user workflow is unchanged:

```bash
python -m examples.algorithms.hubbard_lcu_catalog
```

It appends the packed two-by-two preparation and reports 47 literal T-family
gates, three measurements and twelve declared wires. The model and supplied
work pool are not substituted for those executed primitive counts.

## Shared coefficient numerical owner

Normalization and angle evaluation now belong to Simulation
numerics/hubbard_coefficients.py. Existing Algorithms helper signatures
delegate unchanged; Ecosystem coefficient conversion imports the numerical
owner directly. See [exact arithmetic and replacement evidence](HUBBARD_COEFFICIENT_OWNER.md).
The 70 numerical/warning/refusal/stream/model and prior 22 constructor records
remain identical. No source formula, validator or atomic decomposition changes.
