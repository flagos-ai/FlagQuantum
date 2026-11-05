# Controlled comparison through shared extension and addition

This batch ports the unmodified Qualtran v0.7.0 call chain at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`:
`SignExtend/SignTruncate -> OutOfPlaceAdder -> CLinearDepthGreaterThan`.
Google copyright and Apache-2.0 notices are retained. Each module has one
responsibility and consumes existing And/control primitives.

## Contracts and intermediate data

`sign_extension.sign_extend_ir(source,prefix)` copies the big-endian sign bit
into initially zero prefix wires. The logical extended register is prefix+source;
no physical permutation is inserted. Applying the same source fanout to a
correct redundant prefix implements truncation and clears it. Arbitrary high
bits cannot be discarded. The original fanout was extracted from subtraction,
whose executable stream is unchanged. Valid QInt and QIntOnesComp source pairs
are compared separately; upstream disallows a one-bit QIntOnesComp datatype.

`out_of_place_addition.out_of_place_add_ir` preserves two n-bit inputs and writes
a sum into a zero output. With `include_msb=True`, output width is n+1; otherwise
it is n and addition wraps. There is no additional quantum work outside this
logical output. The explicit `adjoint=True` path consumes a correct sum and
clears the output by measured And erasure. It is not an arbitrary-dirty-output
adder or a blanket unitary inverse. Caller measurement slots number n for full
sums or n-1 for truncated sums. Source `adjoint()` is the reference entry.

The full-sum loop has n And computations, hence 4n T. Truncated sums have n-1,
hence 4n-4 T. The source class prose's undifferentiated 4n-4 statement is not
used to override its executable loop or `QECGatesCost` model. Zero-T measured
uncomputation is reported separately.

`controlled_comparison.controlled_greater_ir(control,left,right,target,work,bits,
signed=False,control_value=1)` follows the source extension, complement, full
OutOfPlaceAdder, controlled sign-copy, and reverse arithmetic. Inputs are equal
positive width, big-endian. Signed mode is two's complement; unsigned uses zero
sign bits. Target is XOR-updated and may be nonzero or entangled. All work is
initially zero and returns to zero, including the intermediate logical sum.
`controlled_greater_requirements(n)` returns n+5 quantum wires and n+2 classical
slots. Both control polarities retain the source MultiControlX decomposition.
Symbolic or more general control interfaces remain pending.

## Runnable user example

```sh
python -m examples.algorithms.register_controlled_comparison
```

`circuit.append(ControlledGreater(signed,cv))` checks every two-bit operand,
control and target amplitude in both signed modes and both control polarities.
It clears seven quantum work wires using four measurement slots, and repeating
the comparison restores the input. Printed counts are for one comparison.
No reference package is needed.

## Measured primitive counts versus model

| Controlled comparison | Measured CNOT | Measured X | Measured T/TDG | Model T at 4 T/And | Measured / model measurements | Clean work |
|---|---:|---:|---:|---:|---:|---:|
| n=3, unsigned, control=1 | 71 | 18 | 20 | 20 | 5 / 5 | 8 |
| n=8, unsigned, control=1 | 151 | 38 | 40 | 40 | 10 / 10 | 13 |
| n=8, signed, control=1 | 155 | 38 | 40 | 40 | 10 / 10 | 13 |
| n=8, signed, control=0 | 155 | 42 | 40 | 40 | 10 / 10 | 13 |

| Eight-bit sum mode | Compute CNOT | Compute+erase CNOT | Compute T/TDG / model T | Erase T/TDG | Erase measurements / model | Logical output width | Extra quantum work |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full sum | 88 | 128 | 32 / 32 | 0 | 8 / 8 | 9 | 0 |
| Truncated sum | 79 | 116 | 28 / 28 | 0 | 7 / 7 | 8 | 0 |

The primitive set is X/H/S, CNOT, T/TDG, conditional CZ, measurement and reset.
Complete source and native gate streams agree in operands, per-wire dependencies
and feedback. Independent T/measurement models agree at explicit 4 T per And;
raw Clifford model counts are coarse and are not compared as primitive totals.
Sign fanout uses 2d-1 CNOT for d extension wires; its source model counts one
multi-target Clifford operation, kept distinct from that primitive expansion.

Reference binding preserves source register bit order: flattening the controlled
comparison's Bloq discovers sum output bits least-significant first, mapped
explicitly to the caller's big-endian sum word. This only assigns temporary
wire names; it does not patch source algorithms or alter gate counts. Sum
compute/erase pairs are also compared directly through original Cirq generators.

## Verification scope

The extension suite has 13 cases: valid source compute/truncate pairs, coherent
sign-copy and cleanup, datatype boundary rejection and alias validation. The sum
suite has 17 cases covering widths 1,2,3,5,8, both output modes, complete small
complex sums and measured cleanup. The controlled comparator has 41 cases:
30 source combinations (five widths, three integer dtypes, both polarities),
ten complete native signed/unsigned/control amplitude cases and validation.
Native checks include signed extrema, arbitrary XOR target, all sampled small
measurement results and double-comparison restoration. Exact branch probabilities
are not inferred from those finite samples. No upstream algorithm patches are
used in these comparisons.

Batch acceptance: all 773 combined regression cases pass, including 71 new
cases. The three new modules have 100% statement coverage. Strict mypy passes
on 939 source files; both the minimum and current Black checks, Ruff,
architecture boundaries and multi-level IR contract checks pass. These are
local results; final integration CI, GPU and review remain separate gates.
