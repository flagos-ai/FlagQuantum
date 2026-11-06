# Contiguous triangular indexing

The register compiler preserves the unsigned, equal-width, big-endian
registers `mu` and `nu` and XORs `mu + nu*(nu+1)//2` into the result.
The caller owns the result and may supply an occupied word. A narrow word
receives the low bits and a wide word is zero-extended. The compiler
restores all borrowed clean work qubits and may overwrite classical slots.

The circuit computes the exact square, adds `nu`, and rebinds the
known-zero least significant bit as the high bit of the shifted word.
It accumulates `mu`, copies the result by XOR, and uncomputes in reverse.
The numerator and full index fit in twice the input width. Square and
add share the same work pool in disjoint stages.

## Reference and gate counts

The [locked Qualtran implementation](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/contiguous_index.py)
uses the plus sign in its zero-based classical method. Its circuit drawing
uses a minus sign. The [paper](https://arxiv.org/pdf/2011.03494#page=15)
uses one-based indices in Eq. 29. The locked implementation has no gate
decomposition. Its THRU result also prevents invoking the classical bloq
entry point with an occupied output. Tests compare the literal classical
method and cost model separately from the native XOR and cleanup contract.

| Input bits per word | Result bits | Work qubits | Classical slots | T + T† emitted | CX emitted | Measurements emitted | Qualtran model Toffoli |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 3 | 1 | 16 | 36 | 4 | 1 |
| 2 | 5 | 7 | 3 | 64 | 154 | 16 | 5 |
| 3 | 6 | 12 | 5 | 128 | 316 | 32 | 11 |
| 5 | 10 | 30 | 9 | 304 | 772 | 76 | 29 |
| 8 | 16 | 72 | 15 | 688 | 1,786 | 172 | 71 |

The source model is `n²+n−1` Toffoli. The emitted counts include compute,
result XOR and cleanup. Tests cover complete small input domains, occupied
results, every measurement history through a channel certificate, and
direct complex amplitudes in all 16 raw histories at width one. The
specialized construction in Appendix F is a separate optimization.
