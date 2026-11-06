# Inverse-square-root Newton update

This compiler component implements the arithmetic data flow of Figure 5 in
Appendix C. Big-endian unsigned source and estimate words encode
`a=W/2**f` and `x=X/2**b`. The result uses the estimate's fractional
width `b` and an independently chosen number of integer bits. The
component XORs the truncated next estimate into any result word. The
source and estimate are preserved and the caller's scratch is restored
to zero. [Appendix C, Figure 5](https://arxiv.org/pdf/1805.12445#page=9)

The arithmetic domain has `0 <= a*x*x <= 3` and requires the truncated
result to fit. The polynomial update is defined at `a=0`; a complete
inverse square root will need a separate convention at that singularity.
The `x=0` state remains zero, including for `a=0`.

The factor word holds `3*2**(f+2*b) - W*X**2` without intermediate
truncation. The final product is formed in the shared column compressor
and only the required high bits are copied before the product columns
are erased. At equal fractional precision the copy drops exactly
`f+2*b+1` low bits. The addition and square are then uncomputed with
the same bindings and classical slots. With `W=1, f=1, X=2, b=2`, the
result word is 2. Rounding the inner product down to b fractional bits
before subtracting gives 3.

The tests check every input pair in the valid arithmetic domain across
six layouts, including zero, pure fractional and mixed inputs, and
results both narrower and wider than the live product slice. They compare
coherent complex amplitudes, certify every measurement history, run raw
primitive trajectories and verify the `circuit.append(op)` round trip.

## Native gate counts

| Source bits / estimate bits / source frac / estimate frac / output bits | T/T† | CX | Measurements | Work wires | Classical slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 / 0 / 0 / 1 | 44 | 85 | 11 | 10 | 2 |
| 1 / 2 / 1 / 2 / 2 | 180 | 388 | 45 | 38 | 6 |
| 2 / 3 / 0 / 2 / 3 | 408 | 955 | 102 | 61 | 7 |
| 3 / 3 / 1 / 2 / 3 | 532 | 1,277 | 133 | 68 | 8 |
| 4 / 4 / 4 / 2 / 8 | 968 | 2,415 | 242 | 115 | 11 |

These counts include one update with its internal cleanup. Each
measurement also emits one reset. No CCX is left in the emitted IR.

## Locked Qualtran model

The locked chemistry `NewtonRaphsonApproxInverseSquareRoot` declares
`QAny` words and models a fixed-point estimate and an integer source.
Its resource graph includes a square, integer scaling, two pure-fraction
multiplications and one addition. Its decomposition is unimplemented.
The following estimates come from the unmodified `QECGatesCost`; the
native circuit above follows the paper's widened arithmetic and includes
explicit erasure. [Locked chemistry source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/grid_ham/inverse_sqrt.py)

| Source / estimate / output bits | Model Toffoli | Model And | Model measurements |
| --- | ---: | ---: | ---: |
| 2 / 3 / 3 | 15 | 2 | 2 |
| 5 / 15 / 22 | 1,140 | 21 | 21 |
| 7 / 8 / 12 | 345 | 11 | 11 |
