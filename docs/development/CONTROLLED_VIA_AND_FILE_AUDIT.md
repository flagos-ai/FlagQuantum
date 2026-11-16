# ControlledViaAnd source file audit

The reference is Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The tests verify these Git blobs.

| Source | Blob | Scope |
| --- | --- | --- |
| `controlled_via_and.ipynb` | `825c66f74d650124a0db19f21b433d680b8d4072` | Q-0549, all 14 cells and both original fixtures, including nested control |
| `controlled_via_and.py` | `3947ddf86be37f4ff3ba58fcc316a8a64922f84f` | Q-0550, grouped signature, graphs, predicate execution, attachment and resource model |
| `controlled_via_and_test.py` | `9e60e3eccb50cfdbcac9054f9471f107062c6d7a` | Q-0551, original examples, three seeded tensor cases, nested integer groups and nested X |

The compiler prepends ordered control groups and derives names from the child
signature. The predicate computes into private work bits, calls the controlled
body, and erases the predicate with measurement and feedback. The controls and
target return through the original ports. Integer groups expand in source
big-endian order using two's-complement for signed words. The nested extension
puts the new groups first and computes a single predicate. Existing component
extension tests cover attachment, name collisions, and returned binding order.

The [notebook](../../examples/notebooks/controlled_via_and.ipynb) displays both
original signatures and call graphs and binds the nested example through
`circuit.append(op)`. Tests compare ordered primitive names, operands and
feedback against the locked source, including the 64-wire integer fixture.
The four- and six-control H cases cover all 8 and 32 histories with full complex
amplitudes and zero work bits. The typed nested X case checks all eight histories
with the same output and work-bit contract. The original source suite passes
all ten collected cases.

| H fixture | Emitted T/T† combined | Emitted H-controlled primitive | Erasure measurements | Modeled And | Modeled peak qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four bit controls | 12 | 1 | 3 | 3 | 8 |
| Grouped integer controls | 124 | 1 | 31 | 31 | 64 |
| Six nested bit controls | 20 | 1 | 5 | 5 | 12 |

The source QEC model counts the controlled-H child as four Cliffords and two
rotations. The nested X source model matches three And calls, four Cliffords,
and three measurements.

Q-0550 remains open for signed word admission through the typed Program binder
and for generic source attachment contracts outside the current directional
port rules. Signed groups are covered in metadata, direct gate emission, and
the original seeded tensor cases. The 64-wire fixture has exact gate, output
and resource comparisons. Exhaustive amplitude checks apply to the small
fixtures listed above.
