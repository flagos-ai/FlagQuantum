# Population count through shared register bit operations

This batch follows unmodified Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`arithmetic/bitwise.py` and `arithmetic/hamming_weight.py`.
Google copyright and Apache-2.0 notices are retained. Compiler modules emit
FlagQuantum IR with explicit wire ownership; they do not allocate runtime
resources or expose reference-package types. No source algorithm is patched.

## Ownership and contracts

`compiler/registers/bitwise.py` owns `xor_ir(source,target)`,
`xor_constant_ir(target,value)` and `bitwise_not_ir(target)`. Registers are
nonempty big-endian tuples, and all quantum wires are distinct/nonnegative.
XOR inputs have equal width and the destination may contain arbitrary coherent
data. The constant emitter accepts an unsigned bit pattern fitting the register;
dtype conversion and signed encoding remain outside it. All three permutations
are self-inverse and use no quantum/classical scratch, T gates or measurements.
Equality's Xor and negation's BitwiseNot now use these functions without changing
their original source streams. No new public root API or compatibility facade
is introduced.

`compiler/registers/hamming_weight.py` owns the source three-to-two compression
tree and its measured adjoint. `hamming_weight_requirements(n)` returns
`(n - n.bit_count(), n.bit_length())`: logical junk and count-output widths.
`hamming_weight_ir(source,junk,output)` requires zero junk and output. Each tree
level retains the source little-endian traversal, reversed-junk pop order and
original And decomposition. The single-bit count copies use shared `xor_ir`.

Forward execution overwrites some source bits and produces logical junk. Only
the output holds the population count; source and junk must not be described as
preserved or clean at that point. `adjoint=True` consumes a correct forward
triple and one explicit measurement slot per junk wire, restoring the original
source and clearing both junk and count output. It does not accept arbitrary
independent dirty triples as a unitary inverse.

## Runnable user path

```sh
python -m examples.algorithms.register_hamming_weight
```

The user still writes `circuit.append(PopulationCount())`. The operation computes
a four-bit count, copies the result with shared XOR, and runs measured erasure.
The example verifies every input/count complex amplitude, original input
restoration and six temporary wires returning to zero. A separate three-bit
result remains populated. Neither Qualtran nor NumPy is required at runtime.

## Measured primitives and independent source estimates

Let `a = n - popcount(n)` and `p = popcount(n)`.

| Path | Measured CNOT | Measured T/TDG | Source QEC model T | Measurements / source model | Logical junk | Count width |
|---|---:|---:|---:|---:|---:|---:|
| Forward | 11a+p | 4a | 4a | 0 / 0 | a, populated | bit_length(n) |
| Forward + measured adjoint | 16a+2p | 4a | 4a | a / a | a, cleared | bit_length(n), cleared |
| n=8 forward | 78 | 28 | 28 | 0 / 0 | 7 | 4 |
| n=8 forward + adjoint | 114 | 28 | 28 | 7 / 7 | 7 | 4 |
| Four-bit example, including three result-copy CNOTs | 53 | 12 | 12 | 3 / 3 | 3, cleared | 3 temporary + 3 retained |

Primitive counts include the six CNOTs in each source 4-T And computation,
five additional CNOTs per compressor, and five per inverse compressor.
Measured erasure contains H, measurement, conditional CZ and reset. Source
coarse Clifford accounting is not substituted for expanded primitive totals.
Forward junk is not hidden extra scratch; full erasure needs no quantum wires
outside the stated source/junk/output registers.

## Verification and reference boundary

There are 58 bitwise tests: five widths, three integer dtypes and three source
operations, complete small coherent permutations including an entangled
spectator, self-inverse restoration and invalid-input rejection. Existing
comparison/negation tests also exercise the shared replacements.

The count suite has 16 cases. Nine source widths (1,2,3,4,5,7,8,9,16) compare
forward gate streams; widths above one also compare the original Bloq
compute/adjoint chain, feedback and independent source QEC T/measurement models.
Five small coherent count tests check complete input/source/junk/output
amplitudes against an independent full-adder truth oracle. They then enumerate
every measurement outcome, postselecting the actual primitive IR without
normalization. Each branch equals the original complex state divided by
`sqrt(2**a)`, proving both phase/cleanup and uniform branch probabilities in
those concrete cases. The runnable user path is also exercised by a test.

For n=1, the source forward generator and native CNOT pair work, but Qualtran's
Bloq adjoint conversion raises `KeyError` while binding empty `QAny(0)` junk.
The test preserves that exact source failure, compares forward gates and native
restoration, and does not claim successful upstream Bloq-adjoint parity for
that boundary. Symbolic widths, other bitwise dtype encodings and whole-file
acceptance remain open.

Batch acceptance covers 847 regression cases: the concentrated run passed 826
and initially skipped 21 native-reference cases because its import path omitted
the pinned Kickmix extension. Those exact 21 cases were then rerun with the
verified in-place extension and all passed. No skipped acceptance cases remain.
There are 74 new tests. Both new compiler modules have 100% statement coverage;
strict mypy passes on 941 source files. Minimum/current Black, Ruff,
architecture boundaries and multi-level IR contract checks pass locally.
Integration CI, GPU and final review remain separate gates.
