# Kaliski modular inverse source-file audit

The reference is Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `mod_division.ipynb` | `fd34e136d02bd80d927945123f9a65f428c38e2a` | Q-0572 |
| `mod_division.py` | `947f40eb5d17101ad91ff4d4432ac5287f867bd5` | Q-0573 |
| `mod_division_test.py` | `2b8a7fbca15e4b5e7143f4b50d002334165b4c5d` | Q-0574 |

The [native tutorial](../../examples/notebooks/modular_inverse.ipynb) covers
the concrete and symbolic examples, including the adjoint signatures and
all eight direct operation identities. It takes the history width from the
source signature. The signature exposes four bits per input bit in the
`junk` output, while the original prose describes a two-bit-per-input-bit
register. The forward `circuit.append(op)` transforms the Montgomery word
in place and returns history on explicit output wires. The adjoint consumes
that history. Clean work is private to the emitter.

## Execution and resources

Native tests match the source's gate operands and measurement feedback in
both directions on width/modulus pairs (5, 7) and (6, 15). A width-five
iteration also exercises the source's standalone six-step call sequence.
Across widths five and six and moduli 3, 5, 7, 11, 13 and 15, the tests
compare every unit and the zero sentinel against locked source values
and retained history, then require exact restoration and clear work.
The feedback certificate composes the measurement branches of each block.
Full-support complex-state checks at modulus 15 exercise the actual
forward and inverse channels at both widths.

The independent source model is `26n² + 9n − 1` Toffoli equivalents.
Its default T estimate is four times that expression. The model with
exact controlled swaps is `2n(67n + 13) + 4(n − 1)`. The emitted stream
below is measured through native forward append.

| Width, modulus | Emitted T/T† | Emitted measurements | Clean work | Logical history | Default source T estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5, 7 | 3496 | 454 | 33 | 20 | 2776 |
| 6, 15 | 5000 | 641 | 38 | 24 | 3956 |

## Remaining source-file boundaries

Q-0572 and Q-0574 are accepted. The locked suite passed 55 cases including
its independent notebook, with one Montgomery modulus metadata skip and
one expected failure for the declared counts. Q-0573 retains the source
decomposition's full-width all-ones modulus limit when it loads `p + 1`.
It also retains the nonunit boundary and the divergence in steps five and
six on states outside the reachable iteration domain. Q-0568 remains open
pending the multiplication and subtraction consumers.
