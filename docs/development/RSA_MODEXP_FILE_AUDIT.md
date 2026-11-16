# RSA modular exponentiation

The locked reference is Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch covers
`qualtran/bloqs/cryptography/rsa/rsa_mod_exp.py`,
`rsa_mod_exp_test.py` and `factoring-via-modexp.ipynb`.

## Binding and reference comparison

The native declaration preserves the big-endian through exponent, fresh
right-side result word, source initializer and symbolic child counts.
The circuit prepares 1 on the low bit of the initially clean result.
It visits exponent wires from low to high significance, using the
same squared constants, including repeated constants of value 1.
Every child keeps the same physical result binding and cleans the
shared work. The native factory also matches the source symbolic
expression and the seeded concrete rejection sampling.

Three complete gate comparisons bind the actual locked child Soquets
in their original order. They cover moduli 3, 15 and 6, each with
permuted physical wires, locally assigned measurement slots and
full primitive comparison of initialization, controlled multiply,
inverse accumulation and exchange. The channel certificate covers
all 32 basis exponents across the three cases and every balanced
measurement-erasure history in each circuit. Two independent full
sparse runs per case check complex amplitudes and all clean work.

| Base, modulus, exponent bits, result bits | Native emitted | Source model |
|---|---|---|
| 2, 3, 3, 2 | 618 T/T†, 144 measurements, 18 wires | 144 AND, 6 CSwap, 144 measurements, peak 18 wires |
| 4, 15, 4, 4 | 2,928 T/T†, 704 measurements, 27 wires | 704 AND, 16 CSwap, 704 measurements, peak 27 wires |
| 5, 6, 3, 3 | 1,287 T/T†, 306 measurements, 22 wires | 306 AND, 9 CSwap, 306 measurements, peak 22 wires |

The native exact exchange stream uses seven T/T† per controlled swap.
The source default T estimate discounts that leaf. The per-stage
comparison includes its underlying gates and feedback.

## Source boundary and test disposition

For modulus 8 the factoring factory produces three result bits. The
source high-level callback computes the value, and the lowered source
fails because a modular-add constant cannot fit. The native executable
binding rejects that configuration at admission. The user can supply an
explicit result width that accommodates the modulus. A zero-width
exponent has a mathematical result of 1 on the admitted domain. The
native binding emits only that initializer; the source composite
classical simulator fails in its empty-bit conversion.

The native callback matches the original 100 seeded factory cases.
The full frozen test file reports 14 passes, five source skips and one
expected failure. The frozen 18-cell tutorial passed in an independent
source kernel. The [native tutorial](../../examples/notebooks/register_factoring_via_modexp.ipynb)
repeats the twenty values of (8^emod 221), determines period 8,
recovers factors 13 and 17, and binds a small modular exponentiation
circuit. Its six code cells and two SVGs passed in a separate native
kernel. Q-0469 and Q-0472 close. Q-0471 retains the recorded source
boundary and the unverified full-width demonstration stream.
