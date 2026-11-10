# Native fermionic science migration

## Current delivery status

Status checked on November 11, 2026, against main revision
`504dc460f34f745db67566597cd69a94e345c298`. This is a selected native scientific
workflow, not complete OpenFermion API compatibility. The initial PR01–PR16
units below describe internal responsibilities delivered by one initial PR;
they are not sixteen outstanding pull requests.

| Scope | Delivery | Status at the checked revision | Shortest workflow |
| --- | --- | --- | --- |
| Initial selected migration | [#1109](https://github.com/flagos-ai/FlagQuantum/pull/1109) | Merged | `python -m examples.algorithms.fermion_chemistry` |
| EXT02: real closed-shell FCIDUMP input | [#1110](https://github.com/flagos-ai/FlagQuantum/pull/1110) | Merged | `python -m examples.algorithms.fermion_fcidump` |
| EXT03: MolecularData HDF5 input | [#1115](https://github.com/flagos-ai/FlagQuantum/pull/1115) | Merged; optional h5py | `python -m examples.algorithms.fermion_hdf5` |
| EXT04: fermionic Fourier circuit | [#1118](https://github.com/flagos-ai/FlagQuantum/pull/1118) | Merged | `python -m examples.algorithms.fermion_fourier` |
| EXT06: explicit Pauli stabilizer sector | [#1119](https://github.com/flagos-ai/FlagQuantum/pull/1119) | Merged | `python -m examples.algorithms.fermion_tapering` |
| EXT07: controlled low-rank evolution | [#1121](https://github.com/flagos-ai/FlagQuantum/pull/1121) | Merged | `python -m examples.algorithms.fermion_controlled_evolution` |
| EXT08: real orbital-parameter gradients | [#1122](https://github.com/flagos-ai/FlagQuantum/pull/1122) | Open; absent from this main revision | See the PR's branch-specific example |

EXT01 (additional model Hamiltonians) and EXT05 (optional OpenFermion object
conversion) remain demand-driven extensions. Neither is a prerequisite for
these native workflows. A model extension must specify one physical model and
reuse existing algebra. Object conversion belongs at an optional adapter
boundary and must not become a production dependency.

The following initial-batch records and CI counts retain their original scope.
Later extensions and commits require their own evidence. The original branch
was `feat/native-fermion-migration`, based on `2a073ab9d`; its upstream reference
was OpenFermion `34adf3196cd566f4ca53893f868753786797ae2a`.

## Module boundaries and composition

| Responsibility | Owning module | Consumer boundary |
| --- | --- | --- |
| Canonical fermion algebra | `observables.fermion` | Existing public operator type is reused |
| Complex Pauli values | `core.algebra.pauli` | Values contain no execution objects |
| Static molecular data and active spaces | `algorithms.chemistry_domain` | `MolecularProblem` owns static snapshots |
| PySCF, FCIDUMP and HDF5 input | `ecosystem.chemistry` | External data becomes `MolecularProblem` immediately |
| Encodings and operator-sector reduction | `algorithms.fermion.encoding`, `symmetry`, `stabilizer` | Encoded/reduced operator contracts remain explicit |
| Static orbital mathematics | `algorithms.fermion.decomposition`, `fourier`, `low_rank` | Mathematical plans do not own execution |
| Preparation and evolution circuits | `algorithms.fermion.preparation`, `lowering`, `evolution` | Ordinary circuits are executed by existing `fq.run` |

File readers and the direct driver share the same active-space, encoding and
measurement consumers. Circuit preparation and evolution use full unreduced JW
occupation registers; an encoded or reduced operator does not automatically
provide a matching ansatz. Explicit state lifting is a bounded inspection path.
Routing and low-rank evolution retain separate responsibilities.

Core/native fermionic workflows require neither OpenFermion nor Cirq. The
repository still offers `cirq-core` through optional Cirq interoperability
extras. Optional PySCF and h5py dependencies stop at their input adapters; they
are not native algebra dependencies. See the local
[fermion domain README](../../flagquantum/algorithms/fermion/README.md),
[chemistry domain README](../../flagquantum/algorithms/chemistry_domain/README.md)
and [input adapter README](../../flagquantum/ecosystem/chemistry/README.md).

## Initial-batch delivery units

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

EXT06 and EXT07 are independent extensions tracked after the selected migration
batch. EXT07 reuses the existing evolution circuit and `Circuit.control`; its
contract and limits are in `FERMION_CONTROLLED_EVOLUTION.md`.

General non-diagonal stabilizer tapering, FFFT and file formats were outside
the initial batch and were delivered separately as shown above. Complete
upstream API compatibility, geometry gradients and distributed training remain
outside the delivered scope. Unit 07 deliberately starts with diagonal parity
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


## Local acceptance evidence

On November 9, 2026, the complete default smoke/unit tier passed with 18,548
passes and 494 skips using four pytest workers, Python 3.12.14 and PyTorch
2.14.1 on local CPU. Focused science, existing fermion, vocabulary and import
isolation checks passed 161 tests; chemistry, molecular and Trotter regressions
including the direct PySCF integrations passed 356 tests. Release configuration
checks passed 62 tests. These are separate runs with overlapping selections;
their counts are not additive.

Ruff and Black pass on the changed Python scope; strict mypy passes on all 18
Python files in the added domains. Architecture, dependency policy, stable API,
capability maturity, team scope and qubit vocabulary gates pass without changes
to the stable API snapshots. The source archive and arm64 wheel build with the
existing isolated PyTorch 2.10 build configuration; all added domain sources and
distributed attribution were checked in the wheel. The local PR description is
in docs/development/OPENFERMION_NATIVE_PR.md. This local acceptance run preceded
the later publication and CI acceptance recorded below.

## Follow-up scientific and compatibility audit

On November 9, 2026, an independent environment installed OpenFermion from the
fixed source revision `34adf3196cd566f4ca53893f868753786797ae2a`.
A reference-only process generated numeric fixtures; native comparison processes
loaded only the arrays and did not import OpenFermion or Cirq. With seed 20261109,
both Python 3.12.14 / PyTorch 2.14.1 and the installed wheel on Python 3.10.22 /
PyTorch 2.10.0 passed the same 347 cases:

| Scope | Cases | Maximum absolute error |
| --- | ---: | ---: |
| Complex operator JW/parity/BK coefficients, widths 1 through 8 | 230 | 4.58e-16 |
| Complex Slater states, all particle counts through 7 modes | 35 | 3.45e-15 |
| Paired Gaussian vacua, including explicit parity flips, 2 through 7 modes | 30 | 6.38e-15 |
| Signed low-rank reconstruction, full contraction and discarded bounds | 40 | 1.71e-13 |
| Frozen-core shifts and reordered active-space integrals | 12 | 1.78e-15 |

Upstream state vectors are compared after aligning their unspecified overall
phase; native determinant-minor and evolution tests continue to check exact
phase separately. Upstream `parity_code(1)` raises a decoder reshape error, so
its ten width-one comparisons are excluded. Native width-one parity remains
covered by the independent Fock-space tests. This audit covers selected native
functions, not all upstream APIs or production-sized systems.

The core wheel was installed outside the source checkout on Python 3.10.22 /
PyTorch 2.10.0. Its native CPU extension loaded and 454 focused science and
existing fermion/chemistry/Trotter tests passed, with no NumPy, SciPy, h5py,
PySCF, OpenFermion or Cirq installed. After adding PySCF 2.8.0, all three direct
RHF tests passed with SciPy 1.14.1 and NumPy 2.2.6. A resolved SciPy 1.15.3 wheel
initially failed dynamic loading on this local macOS host; the successful driver
result applies to the explicitly tested SciPy 1.14.1 combination, not every
transitive dependency combination. No global dependency pin was added for that
local wheel issue.

The direct-driver tests now skip only when PySCF is absent. An installed driver
that fails import raises a test error. Isolated fault-injection probes verify
both behaviors. Final chemistry and native-import integration checks pass four
tests on each tested Python/PyTorch environment. Architecture, dependency,
stable API, ownership and capability gates pass. Review confirmed that Core
contains no execution imports, mathematical decomposition remains independent
of circuits, optional SDK objects stop at the chemistry adapter, existing
chemistry formulas are reused, and routing remains independent of evolution.
The audit made no production source or stable API changes.

## Final scope reconciliation and CI review

The original delivery units describe responsibilities, not sixteen remote PRs
or a commitment to preserve every proposed intermediate type. The implemented
scope makes these deliberate adjustments:

| Units | Final decision | Deferred or excluded work |
| --- | --- | --- |
| 01 | Fixed revision, provenance, import isolation and separate numeric reference process | Complete upstream API catalog and compatibility |
| 02-03 | Reuse FermionOperator in place; add complex Pauli values and explicit measurement conversion | Relocating stable types or rewriting existing Hermitian encodings |
| 04-06 | Native molecular snapshots and three explicit encodings share existing assembly and measurement | Automatic encoded ansatz selection |
| 07 | Exact independent encoded Z-parity sectors and affine embedding | General stabilizer tapering or automatic symmetry discovery |
| 08-09 | Reordered active selections, existing frozen-core formulas and direct optional closed-shell RHF | Automatic active spaces, open shells and geometry gradients |
| 10-12 | Mathematical factors and separate lowering create Slater and canonical Gaussian vacuum circuits | Additional preparation-plan IR, stored decomposition residual field or excited Gaussian basis synthesis |
| 13-15 | Signed density-square factors, independent FSWAP network and first/second-order evolution | Mandatory coupling of evolution to FSWAP or nearest-neighbor routing |
| 16 | Local installation, compatibility, attribution, review and qualified Linux CI | Maintainer review and merge authorization remain outstanding |

A decomposition is verified by reconstruction; it does not carry a separately
stored residual field. This keeps the mathematical boundary small. Circuit
builders consume the mathematical results directly through the existing IR
rather than introducing a second general preparation-plan language. JW/parity
formulas are intentionally retained in the protected Hermitian API while the
new general-complex path has its own small ladder images. Neither becomes an
adapter to an external framework.

The `native-fermion` Ubuntu matrix checks both Python/PyTorch endpoints and the
real optional RHF driver independently from the offline images. Coverage plan
and shard preflight also import PySCF explicitly. The driver skip condition is
visible at module scope, where the existing lane-dependency audit reads it;
missing PySCF skips, while an installed broken driver still fails.
The new matrix depends on source preparation and is required by `ci-complete`.
Existing merge-gate tests verify that failed, cancelled or skipped dependencies
cannot produce a successful final check. The testing manual lists all 27 jobs.

All four immutable CPU images were rebuilt and published. The lock records the
verified registry and offline OCI identities; the offline wrapper still compares
both dependency manifests byte for byte. The 192 coverage worker caches were
qualified against those identities. No manifest comparison, coverage floor or
merge gate was relaxed.

## Historical initial-batch remote CI acceptance

The following results are bound to implementation revision
`901472c81d72f52fa966520de47166557c52ed79`, before the documentation reconciliation:

| Check | Evidence | Result |
| --- | --- | --- |
| Full Linux workflow | CI run `42595072278`, attempt 2 | 233 jobs successful; both native science endpoints passed |
| Runtime selection | Same CI run | 6,215 passed, 403 skipped; real distributed launches passed |
| Coverage | Same CI run | 192 of 192 shards and aggregate passed |
| GPU workflow | Run `42595072337` | Passed |
| Container publication | Run `42595072334` | Passed |

These CI results complement the separately scoped local scientific evidence;
they do not establish native fermionic GPU scalability or full upstream parity.
Linux validation is owned by CI, as requested by the user. Local Docker was
unavailable during the earlier attempt and is not cited as execution evidence.

At that recorded pre-merge point, #1109 was open and #1110 was a stacked
FCIDUMP follow-up. The then-required order was to merge #1109, retarget #1110
to main and validate that target. Both PRs have since merged, as recorded in
the current delivery table. Any later commit, including
documentation changes, needs its own applicable checks; the runs above remain
evidence for their recorded revision.

## Independent extension: fermionic Fourier transform

EXT04 adds `algorithms.fermion.fourier.fourier_decomposition` and
`algorithms.fermion.lowering.fermionic_fourier_transform`. The former returns the
existing OrbitalDecomposition and never imports circuits; the latter reuses the
existing determinant-preserving two-mode lift. The radix-2 Cooley-Tukey
factorization is implemented directly from its mathematical identity; no new
OpenFermion or Cirq source is copied and no production dependency is added.
Forward F[p,q] = exp(-2j*pi*p*q/n)/sqrt(n), inverse F dagger, vacuum phase one,
natural mode order, positive powers of two only. Adjacent fermionic routing
costs O(n**2) factors and storage. This is a static full-register JW workflow,
not a qubit QFT or a claim of OpenFermion API compatibility or FFT speedup.

Run `python -m examples.algorithms.fermion_fourier`: every output mode has
occupation 0.25 and the inverse amplitude residual is below 1e-12. Focused tests
compare one-particle plans with torch.fft, small full Fock matrices and selected
eight-mode particle sectors with Fourier minors, retaining absolute phase.
The fresh-interpreter dependency isolation test also composes the Fourier and
inverse circuits with external scientific SDKs blocked. No chemistry input,
Core, runtime, stable export or dependency manifest changes are needed.

## EXT06: explicit commuting Pauli stabilizer reduction

The next stage extends diagonal Z parity sectors to supplied independent signed
X/Y/Z Pauli words. Its contract and independent acceptance criteria are in
[FERMION_STABILIZER_REDUCTION.md](FERMION_STABILIZER_REDUCTION.md).
`reduce_stabilizers` returns a reduced operator, native Clifford instructions and
a diagonal-frame ReductionMap. The latter is reused rather than expanded to
pretend an affine bit map can express X/Y eigenstate superpositions. Original
state lifting composes the inverse Clifford with that map. Symbolic algebra
imports no circuit, runtime or simulation implementation; lowering owns native
circuit construction and the explicitly bounded statevector inspection path.

Exact CX conjugation has one implementation shared with the original Z-parity
path. New tests check native Clifford matrices, all sectors against independent
dense projectors, the intertwining identity H E = E H_reduced, complex amplitude
norm/phase, signed and randomly Clifford-rotated generators, diagonal replacement
and invalid inputs. The H2 example reduces four qubits to two, measures energy
and particle number with fq.run, and verifies the lifted eigenvector residual.
No upstream source is copied, no dependency or Stable Core schema is changed,
and no full OpenFermion compatibility or automatic symmetry search is claimed.

## Independent extension: controlled low-rank evolution

EXT07 composes the existing `low_rank_evolution` circuit with the existing
`Circuit.control` operation; it adds no public algorithm wrapper or Core
contract. The system remains a full, spin-interleaved JW register and gains one
control wire. The controlled branch retains the energy-offset phase, which is
observable through X/Y interference. The H2 workflow measures the complete
overlap defect against a bounded dense reference for one, two and four second-
order steps. It reports measured error, not an error bound. The feature has no
new dependency and makes no claims for tapered encodings, multiple controls,
hardware routing, GPU or distributed execution. See
`docs/development/FERMION_CONTROLLED_EVOLUTION.md` and
`tests/unit/fermion/test_controlled_evolution.py`.
