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
direct complex amplitudes in all 16 raw histories at width one.


## Direct weighted columns

The second path uses [Appendix F, Eq. F6 and F7](https://arxiv.org/pdf/2011.03494#page=68).
For little-endian index i, it adds mu[i] at column i, nu[0] at column
zero, and each nu[i] for i ≥ 1 at columns i−1 and 2i−1. Each distinct
pair i > j contributes nu[i] AND nu[j] at column i+j. It makes one
temporary copy for each duplicated bit before the shared compressor
mutates any operand. After result XOR, the compressor uncomputes all
products and carries with a single reusable measurement slot. The
copies are then erased by XOR.

| Input bits per word | Result bits | Work qubits | T + T† emitted | CX emitted | Measurements emitted | Qualtran model Toffoli |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 1 | 4 | 10 | 1 | 1 |
| 2 | 5 | 6 | 20 | 60 | 5 | 5 |
| 3 | 6 | 13 | 44 | 132 | 11 | 11 |
| 5 | 10 | 33 | 116 | 342 | 29 | 29 |
| 8 | 16 | 78 | 284 | 822 | 71 | 71 |

The measured implementation uses one 4-T temporary AND per modeled
Toffoli, counting products and carries together. This comparison is in
the temporary-AND cost convention. The table reports the actual native
Clifford+T stream and the locked resource estimate separately. The
model supplies no source gate order. The direct path has the same result
contract as the square-and-add path and passes the same branch and
occupied-output tests.
