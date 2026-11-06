# Duplicate detection on sorted register words

The register compiler ports the locked Qualtran v0.7.0 `HasDuplicates`
component sequence. It uses a strict less-than half comparison for each
adjacent pair, toggles a flag through an all-ones controlled X followed
by X or CX, and clears the adjacent comparisons in reverse order. With
the optional positive control, the flag toggles only on the active
branch. Each big-endian word represents an unsigned or two's-complement
integer. The input list is sorted and the flag may have either value.

The caller owns the clean workspace. Each adjacency retains an n+1-bit
difference and one comparison flag until erasure. The two extension
wires are reused across comparisons and share a pool with the
multi-controlled X ladder. Classical slots are reused between measured
cleanup blocks. The normal route preserves word order and returns every
work wire to zero.

The `source_aliases=True` diagnostic route follows the pinned graph's
LT adjoint output bindings. For two unsigned two-bit words [0, 1], the
original flattened graph returns [1, 0] through an alias exchange, with
the physical input state restored. For [0, 0, 1], the next adjoint
receives the wrong operand and its And cleanup fails. The original
one-layer graph gives the expected duplicate flag for both cases. An
uncontrolled singleton also fails in the original decomposition when
zero controls are passed to X. The normal route uses identity for that
singleton. The controlled singleton retains the source's two CX gates.

The normal route keeps the same half comparison, flag and cleanup gate
blocks, and binds each adjoint to its original physical operand pair.
This preserves the intended predicate while repairing cleanup at the
consumer. The reference remains unmodified. Literal graph parity applies
to the diagnostic route, and full clean-channel verification applies to
the normal route.

Ten source-graph cases compare gates, wire order, feedback and the model.
Numerical tests cover all sorted words in the tested one- and two-bit
layouts, both signednesses, both flag inputs, both control values and
a coherent spectator. Every tested input has an exact And-channel
certificate, including applying the predicate twice. Cases through
three words also execute all raw measurement histories on a complex
state. Four-word cases compare complete sparse complex trajectories.

| Words × bits, mode | Physical wires | Source peak estimate | Literal T-family | Literal CX | Literal measurements | Source model T / measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 × 1, unsigned | 8 | 7 | 4 | 23 | 1 | 4 / 1 |
| 2 × 2, controlled signed | 12 | 11 | 12 | 54 | 3 | 12 / 3 |
| 3 × 2, unsigned | 17 | 16 | 20 | 83 | 5 | 20 / 5 |
| 4 × 3, controlled signed | 32 | 32 | 48 | 206 | 12 | 48 / 12 |

Both routing modes have the same literal primitive counts on these
layouts. The physical allocation reserves two extension wires for the
comparison even where the source peak estimator reports one fewer wire.

Run `python -m examples.algorithms.register_sorted_duplicates` for a
coherent flag and clean-work check through `circuit.append(op)`.
