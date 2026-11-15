# First-quantization kinetic PREPARE audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/pbc/first_quantization`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0367 | `prepare_t.py` | `cb69fc70b19d834d9ecdad3da011b09cfac1f41a` |
| Q-0368 | `prepare_t_test.py` | `afa581df1f20b2eaafa46510c77296179229a448` |

## Contracts and call graph

The native kinetic module owns both signatures. Power-of-two preparation
uses the through word `r`. The parent has a two-bit coordinate word `w`
and two through words `r` and `s` of the requested momentum width.
The parent binds the two instances of the power-of-two leaf independently.
Both directions preserve the three logical words, and each word has one
incoming and one outgoing connection. The disjoint branches can appear
in different topological orders.

The source power-of-two leaf declares `bitsize - 2` Toffoli and has no
executable decomposition. The native model retains the negative
declaration at bitsize one. The tutorial's kinetic signature now comes
from the same owner as the model and the coordinate binding.

## Independent accounting

| Kinetic fixture at momentum width six | Evidence |
| --- | --- |
| Source declared parent | 21 Toffoli, peak 18 |
| Source decomposed parent model | Eight Toffoli, five And, five measurements, peak 18 |
| Executable coordinate child | 20 T/T†, five measurements, four work qubits |
| Original combined forward and adjoint i/j plus kinetic budget | 126 Toffoli |

The parent declaration assigns 13 Toffoli to the coordinate substep.
Its decomposition calls `PrepareUniformSuperposition(3)`. This difference
causes the original annotation/composite count check's expected failure.
Five original modes passed and one was marked expected failure. Both
native model views match their corresponding frozen queries in both
directions.

`prepare_kinetic_coordinate_on` binds the executable child through
`circuit.append(op)`. Compiler owns the uniform preparation and its
feedback. The binder's entire forward and adjoint instruction streams
match the shared primitive on a reordered target. The shared source
comparison checks the actual primitive dependencies, full complex
outputs and clean work in all 32 histories for each direction.

Q-0367 and Q-0368 close the source scope and its tests. Executing the
full kinetic parent still requires a physical power-of-two leaf.
