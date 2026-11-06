# Fixed-point reciprocal

`reciprocal_ir` accepts an unsigned big-endian source with at least one
integer bit and XORs a pure fractional estimate into a caller-owned result.
The result can retain any nonempty prefix of the working precision. The
working precision must cover the source's integer width. Inputs below one
leave the result untouched, and one is approximated from below. The source
is preserved. Caller-owned clean history and scratch return to zero, and
classical slots can be reused after the call.

The circuit computes the Figure 4 seed, applies the widened Newton step
`ceil(log2(precision))` times into successive history words, copies the
requested prefix, reverses the updates and erases the seed. All updates
share one scratch bank. Each step subtracts at full width and truncates
once. [Reciprocal construction](https://arxiv.org/pdf/1511.08253#page=30)

For `a >= 1` the seed lies in `[1/(2*a), 1/a]`. Each exact
step squares the relative error. On the interval from the seed to `1/a`
the update has derivative between zero and one, so a truncation adds at
most one working ulp to the absolute error bound without amplifying earlier
errors. With `s` steps at `b` bits, the absolute error is bounded by
`2**(-2**s)/a + s*2**(-b)`, plus one output ulp if fewer than `b` bits
are retained.

The tests check every source value in six layouts against exact rational
arithmetic. A channel certificate checks every measurement history with
occupied and empty outputs. Primitive trajectory tests compare full complex
amplitudes, including a coherent spectator and zero scratch. The append
test runs through `circuit.append(op)` and verifies a second application
restores the input.

## Native gate counts

| Source bits / fraction bits / working bits / result bits | T or T† | CX | CCX | Measurements | Work wires | Classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 0 / 1 / 1 | 0 | 7 | 0 | 0 | 2 | 0 |
| 2 / 0 / 2 / 2 | 312 | 714 | 2 | 78 | 31 | 5 |
| 3 / 1 / 3 / 2 | 1,416 | 3,450 | 2 | 354 | 62 | 8 |
| 4 / 2 / 5 / 4 | 4,952 | 12,570 | 2 | 1,238 | 127 | 13 |

Counts include one forward application and its internal cleanup. Each
measurement also emits one reset. CCX remains a separate operation in this
IR count.

## Locked Qualtran model

Qualtran v0.7.0 `InvertRealNumber` exposes a resource graph with
`ceil(log2(source_width))` iterations and an output using the same mixed
format as the source. It has no executable gate decomposition. Its
iteration graph includes a pure-fraction multiplication that cannot accept
the mixed-format source directly. Its source-width one and two models fail
when constructing the fixed-point square.

| Source bits / fraction bits | Model Toffoli | Model And | Model measurements |
| --- | ---: | ---: | ---: |
| 3 / 1 | 21 | 5 | 5 |
| 4 / 2 | 47 | 7 | 7 |

These values come from the unmodified `QECGatesCost` model. The native
circuit uses explicit widened products and reverses its history, and the
4-bit native case above also uses higher working precision.
[Locked multiplication and reciprocal source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py)
