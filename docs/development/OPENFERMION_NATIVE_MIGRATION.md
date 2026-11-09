# Native fermionic science migration

The local branch `feat/native-fermion-migration` implements a selected scientific
scope, not the entire OpenFermion API. Its base is `2a073ab9d`. The upstream
reference is OpenFermion `34adf3196cd566f4ca53893f868753786797ae2a`.

## Delivery units

| Unit | Capability | Boundary |
| --- | --- | --- |
| 01 | Scope, provenance and isolation | Development and architecture |
| 02 | Existing canonical fermion algebra | Reuse public type without relocation |
| 03 | Complex Pauli algebra | Core values and explicit observable conversion |
| 04 | Integral and molecular problem data | Static native chemistry |
| 05 | JW, parity and Fenwick BK | Operator encoding |
| 06 | Encoded reference occupations | Shared energy consumer |
| 07 | Explicit diagonal parity sector reduction | Operator and state embedding |
| 08 | Active-space selection and frozen core | Native molecular transformation |
| 09 | Optional direct RHF PySCF driver | Ecosystem |
| 10 | Adjacent complex Givens decomposition | Static mathematical rotations |
| 11 | Slater determinant preparation | Native circuit synthesis |
| 12 | Canonical Gaussian vacuum preparation | Native circuit synthesis |
| 13 | Real chemist-integral low-rank decomposition | Static numerical factors |
| 14 | Fermionic swaps and final layout | Native circuit synthesis |
| 15 | Low-rank first/second-order evolution | Native differentiable time |
| 16 | Compatibility, checks and review draft | Local release preparation |

General non-diagonal stabilizer tapering, FFFT, file formats, complete upstream
API compatibility, geometry gradients, and distributed training remain outside
this selected batch. Unit 07 deliberately starts with diagonal parity
generators, an exact and independently verifiable subset.

## Existing workflow preserved

```python
import flagquantum as fq
from flagquantum.observables import jordan_wigner, number

circuit = fq.Circuit(2)
circuit.x(0)
observable = jordan_wigner(number(0), n_modes=2)
result = fq.run(circuit, outputs=[fq.expectation(observable)])
```

New examples and actual validation results are recorded as their implementations
land. This document does not assert that planned interfaces already exist.

## Review rules

Every implementation commit includes focused behavior tests and source metadata
where code is adapted. Core imports no execution or vendor objects. Optional
chemistry absence must not break native algebra or circuit imports. New domains
have a README with a shortest executable example and focused test command.

Integration-owned boundary changes precede their implementations in local
history. Keep machine-readable protected API snapshots unchanged. Stop only the
affected part if an unavoidable stable contract conflict is discovered and
present the concrete compatibility options.
