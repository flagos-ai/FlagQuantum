# Modular multiplication source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `mod_multiplication.ipynb` | `d8f7d6166fece00ea1ec07759ad8907bf9a41b35` | Q-0575 |
| `mod_multiplication.py` | `c0242f9735bf269564c9480c14c008d5f9621fd7` | Q-0576 |
| `mod_multiplication_test.py` | `d6a1caa6231a0dcf4769b3f8c2b747bd15e8212c` | Q-0577 |

The [native tutorial](../../examples/notebooks/modular_multiplication.ipynb)
covers all 29 original cells, including six example instances, three signature
diagrams and three call graphs. The added append section emits the five
concrete instances. Fixture tests compare every signature, direct child and
multiplicity, including Montgomery modulus metadata and the three
direction-changing ports on the dirty operation's adjoint.

## Executable contracts

- `ModDbl` consumes a big-endian residue word. The old high bit returns as
  clean work and the new low bit takes its place in the result's logical
  binding. `ModDbl†` uses the original inverse arithmetic ordering.
- `CModMulK` preserves the control on both branches. It computes and clears
  the shared second word through scaled additions and uses exact controlled
  swaps. The adjoint reverses the original component order.
- `SingleWindowModMul` takes low-order-first multiplier bits and big-endian
  accumulator wires. It retains the selected lookup index and rotates the
  accumulator binding. The whole multiplication processes windows from the
  lowest order upward and retains a big-endian index for each window.
  The adjoint consumes the result, retained indices and reduction bit in
  their original physical wire locations.

The concentrated regression reuses the unchanged-source component walkers,
including gate operands and measurement feedback in both directions. The
small complete-channel cases cover every valid basis row and measurement
history by composing their certified blocks. Complex-state tests include
spectator amplitudes and zero work. The blocked QROAM cases use complete
complex output comparisons on seeded sparse trajectories. The file audit also
checks every residue and control in the original doubling and controlled
prime-constant grids, using both original register types. The numerical
window and Montgomery suites retain array/integer order, result/history
values, and the distinct behaviors of the source callback and primitive graph.

## Counts on the original concrete examples

| Example | Emitted T/T† | Emitted measure | Emitted CX | Qubits | Source default T | Source measure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubling, width 4, modulus 13 | 36 | 9 | 109 | 17 | 36 | 9 |
| Doubling, width 32, modulus 1000000007 | 260 | 65 | 807 | 101 | 260 | 65 |
| Controlled multiply by 123, width 8, modulus 221 | 2744 | 672 | 7792 | 44 | 2720 | 672 |
| Dirty multiply, width 6, window 2, modulus 7 | 464 | 116 | 1125 | 42 | 440 | 110 |
| Dirty multiply, width 16, window 4, modulus 32767 | 2924 | 731 | 8394 | 108 | 2764 | 675 |

Using seven T per exact CSwap reconciles the constant-multiplier model with
the emitted T count. For each Montgomery window, the executable bank
cleanup reverses the forward lookup. Its cost is the forward lookup cost
where the source model counts the measurement-assisted lookup adjoint.
The independent source symbolic checks retain `2n + 1` CCZ equivalents for
doubling and `9n²/4 + 29n/4 - 1` for four-bit windows in both directions.

## Remaining source-file boundaries

Q-0575 and Q-0577 are accepted. The locked test suite passed 1,959 cases
including its independent notebook and both slow grids. Six source
example type checks skipped and two doubling example count checks were
expected failures. The native audit explicitly checks source signature and
direct-call equality on all six examples.

Q-0576 remains open for the recorded exterior-input counterexamples, the
window carry overflow disagreement, and the final gate-stage addition
versus callback subtraction when reduction fires. It retains the
concrete-width symbolic-modulus QROAM construction failure and the source
callback's Fermat-based assumptions. Q-0568 also depends on the modular
subtraction file family.
