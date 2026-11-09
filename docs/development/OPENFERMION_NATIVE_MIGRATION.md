# Native fermionic science migration

The feature branch `feat/native-fermion-migration` implements a selected scientific
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

## Remote CI acceptance and review status

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

PR #1109 is open and ready for maintainer review; it has not been merged. The
FCIDUMP follow-up is reviewed separately in stacked PR #1110. Merge #1109 first,
then retarget #1110 to main and validate that target. Any later commit, including
documentation changes, needs its own applicable checks; the runs above remain
evidence for their recorded revision.
