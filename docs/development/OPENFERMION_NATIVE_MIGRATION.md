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

## Implemented modules and support matrix

| Module | Owns | Verified scope |
| --- | --- | --- |
| core.algebra.pauli | Complex canonical Pauli values | Standard library, no circuit or tensor dependency |
| algorithms.fermion.encoding | General operator and occupation encoding | JW, prefix parity, Fenwick BK; explicit idle width |
| algorithms.fermion.observables | Hermitian measurement boundary | Complex values validated before native Observable conversion |
| algorithms.fermion.symmetry | Selected diagonal parity sectors | Independent encoded Z products; affine state embedding |
| algorithms.chemistry_domain | Static molecular data and active selections | Real spatial chemist ERIs, closed shell, interleaved spin, explicit offsets |
| ecosystem.chemistry.pyscf | Optional RHF integral driver | H2 STO-3G/6-31G, LiH STO-3G; native data returned immediately |
| algorithms.fermion.decomposition | Adjacent orbital unitary factors | Complex unitary reconstruction and determinant phase |
| algorithms.fermion.lowering | Small native orbital gates | JW occupation order; one- and two-qubit matrices |
| algorithms.fermion.preparation | Slater and canonical Gaussian vacua | Complex minors, canonical annihilator constraints, even/odd parity |
| algorithms.fermion.low_rank | Signed density-square factors | Real chemist ERIs, explicit rank, full one-body correction |
| algorithms.fermion.swap_network | Adjacent FSWAP and output ordering | Every logical pair once; explicit final layout and fermion signs |
| algorithms.fermion.evolution | First/second-order product formulas | JW full register, exact offset phase, differentiable scalar time |

All execution evidence here is local CPU complex128. Static decompositions own
CPU snapshots and reject autograd inputs. Time differentiation is verified with
finite differences; geometry, integrals, orbital frames, rank selection and
Gaussian block differentiation are unsupported. Encoding and reduction results
are separate contracts: preparation/evolution circuits do not accept encoded or
reduced values and always use the unreduced JW occupation register.

The swap network is independently usable and tested. Low-rank evolution uses
explicit density-pair gates, which may be nonadjacent, and restores its input
layout after each orbital basis change. It does not insert the swap network or
claim nearest-neighbor connectivity. This avoids an unnecessary dependency
between evolution and routing. A future connectivity pass can consume ordinary
IR and handle this at the compiler boundary.

The low-rank approximation keeps the full contraction correction. Consequently,
a truncated density-square model is not obtained by simply replacing the ERI
and recomputing its correction. discarded_bound bounds omitted density-square
operator norms; it is not a bound on time integration error. The default retains
all eigenpairs. Dense helpers are bounded inspection paths, not scalability
implementations.

## Executable workflow and checks

```bash
python -m examples.algorithms.fermion_chemistry
# All three encodings: HF energy -1.1166843871 Ha.
# The example also differentiates a native evolution amplitude with respect to time.
python -m pytest tests/unit/fermion -q
python -m pytest tests/integration/test_native_fermion_isolation.py -q
python -m pytest tests/integration/test_native_chemistry_driver.py -q
```

The isolation test runs in a fresh interpreter that refuses imports of
OpenFermion, openfermionpyscf, Cirq, PySCF, NumPy, SciPy and h5py. It covers chemistry
data, all encodings, measurement, Slater/Gaussian preparation and differentiable
evolution. The optional PySCF entry raises an actionable missing-extra error in
that environment. Independent references use Fock-space ladder action, complex
Slater minors, annihilator-vacuum residuals and exact quadratic/full Hamiltonian
matrix exponentials. They do not require OpenFermion or Cirq as test oracles.
