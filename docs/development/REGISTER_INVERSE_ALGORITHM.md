# Complete modular inverse body

`compiler/registers/inverse_algorithm.py` ports the unchanged Qualtran v0.7.0
`_KaliskiModInverseImpl` call chain into shared native register IR. It composes
initialization, 2n complete iterations, final coefficient conversion and the
zero-input correction. It owns no execution backend or upstream runtime objects.

Source: [pinned modular division](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py),
Google LLC, Apache-2.0. The source file is unchanged; its Git blob remains
`947f40eb5d17101ad91ff4d4432ac5287f867bd5`.

## Inputs, histories and physical ownership

`inverse_algorithm_ir(u, v, r, s, markers, active, terminals, modulus, workspace,
classical_bits, adjoint=False)` takes four n-bit big-endian words, two 2n-bit
history words, one active flag, 2n+7 clean work wires and max(2n-1,n+1) reusable
classical slots. The forward input v contains x; every other logical input and
all work start zero. Valid inputs are zero or units modulo a fitting positive
odd modulus, with n>=2 and p+1<2**n.

The output r contains `x^-1 * 2^(2n) mod p`, with zero mapped to zero. It is a
Montgomery-scaled inverse, not an ordinary unscaled inverse. The other words and
active flag return zero. Marker and terminal words contain the original logical
history, needed by the inverse; they are not disposable clean scratch. Classical
slots retain recent measurement outcomes and are not claimed zero.

`inverse_algorithm_outputs(r, workspace)` follows every original iteration's
physical output aliases and returns the actual coefficient and clean-work roles.
The exact reverse consumes these roles and the full logical history. There are
no materializing output SWAPs or logical resets. During final conversion, the
now-clean iteration pool is reused for the original AddK's loaded constant and
carry wires.

The reverse inverts original components, rather than reversing their flattened
primitive lists. In particular, XorK and controlled XorK reverse component order
while retaining their original internal bit traversal. Gate-stream tests enforce
this distinction.

## Promises and unchanged source failures

The unchanged source cannot decompose its AddK(p+1) load when p=2**n-1. Tests
execute that original decomposition and resource evaluator at n=2,3,5, preserving
the actual too-large-value failure. The native entry rejects that boundary;
it does not normalize the constant to manufacture literal-source parity.

For a composite modulus, nonunits do not have the clean output promised by the
algorithm. At n=4,p=9,x=3 or 6, the original high-level model asserts and the
literal circuit leaves nonzero logical cleanup words. Tests preserve this
counterexample. It is not corrected with a reset or silently declared valid.
The previously documented single-iteration reachable-state promises also apply.

## Source, phase and erasure evidence

The oracle walks the unmodified complete source graph, retaining original
Split/Join, iteration, arithmetic, XOR and feedback calls. Physical scratch
lifetimes and every output role are bound explicitly. Forward and reverse
complete gate/operand/feedback/port streams match at n=2,3,4,8.

Every valid x is checked at n=2,p=1; n=3,p=3 and 5; n=4,p=9 and 13; and
n=8,p=251. Expected coefficients and both history words come from the original
public model, independently checked against the scaled modular inverse formula.
The full complex linear map, every measured erasure history, all clean work and
reverse restoration are certified by the existing primitive composition proof.
For the eight-bit case the event count is 1,111, so branch probability is kept
as **2^-1111**, rather than underflowing a floating-point number to zero.

Complete floating complex amplitudes are also executed on all valid inputs in a
single superposition at n=3,p=5; n=4,p=9; and n=8,p=251. Three seeded measurement
histories per direction run through the existing G1 sparse primitive test
executor. A fourth forward history substitutes the unchanged source-generated
stream without changing the consumer. Every support key is compared, including
floating cancellation residuals, at absolute tolerance 2e-12 and zero relative
tolerance; unexpected work amplitudes are not filtered out. Each measured
conditional probability is checked as one half. These are actual primitive
numerical checks, not a new production sparse simulator, and are not an
exhaustive numerical enumeration of all histories.

## Measured gates versus upstream models

Both directions have the same counts. T includes T-dagger. There is one S and
one conditional CZ per measured erasure, in addition to the columns below.

| n, p | Measured T | Measured CX | Measured X | Measured H | M / resets | Clean work | Total qubits | Logical history | Classical slots | Default model T | Exact-swap model T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2, 1 | 592 | 1,086 | 257 | 327 | 85 | 11 | 28 | 8 | 3 | 484 | 592 |
| 3, 5 | 1,292 | 2,383 | 458 | 696 | 176 | 13 | 38 | 12 | 5 | 1,040 | 1,292 |
| 4, 9 | 2,260 | 4,157 | 719 | 1,201 | 299 | 15 | 48 | 16 | 7 | 1,804 | 2,260 |
| 8, 251 | 8,812 | 16,308 | 2,181 | 4,581 | 1,111 | 23 | 88 | 32 | 15 | 6,940 | 8,812 |

Independent source models give default T=2n(52n+16)+4(n-1), literal seven-T swap
T=2n(67n+13)+4(n-1), M=2n(8n+5)+n-1, Q=10n+8. Default and exact-swap costs use
different CSwap assumptions; neither model is substituted for measured gates.

## Local acceptance

The module adds 31 tests. The concentrated dependency regression passes 674
cases without skips, followed by a final 31-case module run including two newly
added nonunit counterexamples. The body, iteration dependency and source reader
have 100% statement coverage; final body coverage remains 100%. Strict types
pass on 970 production files and the example; minimum/current Black, Ruff,
compiler scope, architecture and multi-level IR checks pass. This is a scoped
regression and does not claim every historical G2 test was rerun.

## Runnable append example

```bash
python -m examples.algorithms.register_inverse_algorithm
```

The example uses `circuit.append(InverseAlgorithm())` and its exact inverse,
prints actual IR gates and output/work roles, and separates model costs. It
constructs 38-wire IR without allocating a dense 38-qubit statevector. It needs
neither Qualtran nor NumPy. Complete execution evidence above is provided by the
sparse primitive tests and all-history certificates, rather than being claimed
as a dense production execution of that example.

This slice delivers the original full body. The public in-place wrapper's
concrete compute/uncompute path is accepted separately in
[REGISTER_MODULAR_INVERSE.md](REGISTER_MODULAR_INVERSE.md), including junk
allocation/free and physical output mapping. Symbolic interfaces and remaining
inventory are still open; this does not mark the whole source file or G2 complete.
