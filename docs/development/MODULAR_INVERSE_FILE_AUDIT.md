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
six on states outside the reachable iteration domain. Q-0568 package export
has its independent accepted ledger disposition.

## Current stored-graph batch interface — 2026-11-20

Proposal 108 was published at aa23ec1cb. The next complete source-file batch
contains the nine classes below and 48 definitions. Its inventory is retained
in `ftqc-reference-env/g2-modular-inverse-native-inventory.json`.


The complete source-file batch contains nine classes and 48 definitions.
All words and history words are MSB-first. Forward history is a declared
logical output. Temporary conjunction, comparison and arithmetic work returns
zero only on the original promised domain. Specialized inverse dependencies
consume matching forward outputs and preserve the original source constants.

| Frozen class | Native owner and port contract |
| --- | --- |
| `_KaliskiIterationStep1` | Termination owner preserves v/m/f/is_terminal, with zero-test conjunction work owned internally |
| `_KaliskiIterationStep2` | Parity owner preserves u/v/b/a/m/f, applies original two predicates and measured conjunction inverses |
| `_KaliskiIterationStep3` | Ordering owner preserves u/v/b/a/m/f and consumes owned half-comparison outputs after fanout |
| `_KaliskiIterationStep4` | Exchange owner preserves u/v/r/s/a and two exact controlled swaps |
| `_KaliskiIterationStep5` | Arithmetic owner preserves u/v/r/s/b/f with original complements and two controlled additions |
| `_KaliskiIterationStep6` | Scaling owner preserves u/v/r/s/b/a/m/f, original descending controlled shifts, ModDbl alias and parity cleanup |
| `_KaliskiIteration` | Six ordered calls, two owned zero branch flags, retained marker and termination outputs |
| `_KaliskiModInverseImpl` | Through four words, 2n marker/terminal histories and active flag, original initialization and zero correction |
| `KaliskiModInverse` | Through x, RIGHT forward or LEFT inverse 4n junk, inlined algorithm, source allocation/release order |

The existing step, algorithm and public physical emitters retain their accepted
source-stream, inverse, complex-amplitude and resource evidence. Stored graph
implementation and normal append replacement remain pending until independently
validated. Source all-ones p+1 loading, nonunits and unreachable step-five/six
callback disagreements remain open. The accepted package export Q-0568 has an
independent ledger disposition.

### Frozen definition inventory

| Class | Source members |
| --- | --- |
| `_KaliskiIterationStep1` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep2` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep3` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep4` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep5` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep6` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIteration` | `signature`, `build_composite_bloq`, `build_call_graph`, `on_classical_vals` |
| `_KaliskiModInverseImpl` | `signature`, `_kaliski_iteration`, `build_composite_bloq`, `build_call_graph` |
| `KaliskiModInverse` | `signature`, `build_composite_bloq`, `adjoint`, `build_call_graph`, `on_classical_vals` |

Original example functions are `_kaliskimodinverse_example`, `_kaliskimodinverse_symbolic`.

This inventory records scope and interfaces. It provides no new acceptance
decision for the pending stored graph batch.

### First stored owner in the current batch

`inverse_exchange_component.py` now composes Step4's two actual CSwap calls,
sharing one exact child definition. Forward and adjoint root order match the
source on widths 1, 2, 3, 5 and 8. All primitives, operands and output aliases
match, with no workspace events or classical slots. Complete complex states
and an entangled spectator are checked at widths 1 through 3, including exact
restoration through the append consumer.

Its emitted T count is 14n and CX count is 16n. The separate original default
model is 8n T equivalents, or 14n when the exact seven-T controlled-swap weight
is selected. Physical peak is 4n+1 with zero work and measurements. Existing
Step4 evidence is reused. The focused source/graph regression and strict typing,
format, scope, architecture, IR and API checks pass. Receipt prefix is
`g2-modular-inverse-native-exchange-*`. This owner is locally accepted; the
complete nine-class stored-graph batch remains in progress until the other
owners, normal root execution and concentrated regression close together.
