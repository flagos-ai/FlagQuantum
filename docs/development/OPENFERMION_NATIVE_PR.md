# Native fermionic science without OpenFermion or Cirq

This is the archived review description for the initial migration
[#1109](https://github.com/flagos-ai/FlagQuantum/pull/1109), which has merged.
The review status and CI results below describe the recorded initial-batch
revision, not the current branch. For subsequent deliveries and runnable
entry points, see [current migration status](OPENFERMION_NATIVE_MIGRATION.md#current-delivery-status).

Add a native workflow from complex fermionic operators and molecular integrals
to encoded measurement, state preparation and low-rank evolution. Production
execution uses FlagQuantum and PyTorch. OpenFermion and Cirq are not required.
The existing FermionOperator, stable root exports and API snapshots are unchanged.

At the recorded review point, this PR delivered the selected native fermionic
workflow and was ready for maintainer review. Full Linux CI run `42595072278`
(attempt 2), GPU run
`42595072337` and container publication run `42595072334` passed for revision
`901472c81d72f52fa966520de47166557c52ed79`. All four fixed images and 192 coverage
worker caches were qualified. At that point, no merge had been performed.
Later commits require
their own applicable checks; these run IDs describe the recorded revision.

## User workflow

```bash
python -m examples.algorithms.fermion_chemistry
```

Expected output:

```text
jordan_wigner: HF energy -1.1166843871 Ha
parity: HF energy -1.1166843871 Ha
bravyi_kitaev: HF energy -1.1166843871 Ha
Real reference amplitude 0.9745154458; time derivative -0.2537232266
Integral reconstruction residual 5.660e-16
```

The example uses the native hydrogen input without the optional chemistry extra.
It prepares a Slater determinant and differentiates a two-step, second-order
low-rank evolution amplitude with respect to time.

## Module boundaries

- Core owns standard-library-only complex Pauli arithmetic. Conversion to a
  Hermitian Observable belongs to the fermionic algorithm domain.
- Encodings own JW, prefix parity, Fenwick BK and occupation maps with explicit
  width. Each encoding uses the same unchanged native measurement consumer.
- Chemistry owns static real spatial chemist integrals, spin interleaving,
  explicit energy offsets, occupied references and reordered active selections.
  Assembly and frozen-core formulas reuse the existing implementation.
- The optional `chemistry-pyscf` extra owns direct closed-shell RHF input. Its
  driver imports PySCF only when called and immediately returns native data.
- Mathematical decomposition returns small orbital matrices without importing
  Circuit. Lowering owns native one- and two-qubit gates. Preparation owns
  complex Slater determinants and even/odd canonical Gaussian vacua.
- Low-rank factorization owns signed density-square factors, full contraction
  correction and a conservative discarded-factor bound. Evolution composes
  these factors with differentiable time and exact energy-offset phase.
  FSWAP routing owns pair coverage and final layout independently.

The only new production dependency is the optional bounded PySCF extra. Existing
Cirq interoperability remains independently available. Source attribution is
fixed to OpenFermion `34adf3196cd566f4ca53893f868753786797ae2a` in
`third_party/openfermion/`. LICENSE and NOTICE ship in wheel and source archive;
the source archive also includes per-file adaptation provenance.

## Validation

The following local scientific results use macOS arm64 CPU, complex128; remote
CI results are recorded separately below:

| Environment or check | Result |
| --- | --- |
| Python 3.12.14 / PyTorch 2.14.1 | 18,548 passed, 494 skipped; final full default-suite rerun |
| Independent OpenFermion comparison on Python 3.12 / PyTorch 2.14 | 347 cases passed; maximum absolute error 1.71e-13 |
| Installed wheel on Python 3.10.22 / PyTorch 2.10.0 without optional science packages | 454 science and existing fermion/chemistry/Trotter tests passed; native extension loaded |
| Same wheel and minimum runtime against upstream fixtures | 347 cases passed |
| Direct PySCF 2.8.0 with SciPy 1.14.1 / PySCF 2.15.0 | H2 STO-3G, H2 6-31G and LiH STO-3G RHF cases passed |
| Driver absence and simulated installed-driver import failure | Absence skips; installed-driver failure produces errors |
| Architecture, dependency, stable API, ownership, capability and qubit vocabulary gates | Passed without stable snapshot changes |
| Wheel and source archive | Built; all 18 new domain source files and distributed attribution verified |

The 347 reference cases cover complex JW/parity/BK coefficients, all Slater
particle sectors through seven modes, paired Gaussian vacua with parity flips,
signed/truncated low-rank reconstruction and reordered frozen-core integrals.
Reference and native processes exchange arrays; native execution imports neither
OpenFermion nor Cirq. Upstream `parity_code(1)` cannot reshape its decoder, so its
ten width-one comparisons are excluded; independent Fock tests cover native
width-one parity. State comparisons align the upstream's unspecified overall
phase. Native minors and evolution tests check exact phase separately.

A clean native subprocess refuses OpenFermion, openfermionpyscf, Cirq, PySCF,
NumPy, SciPy and h5py imports while running encoding, chemistry, measurement,
state preparation and time gradients. Static inputs reject autograd. Existing
fermionic and chemistry behavior is covered by independent Fock action, minors,
annihilator residuals, exact matrix exponentials and finite differences.

A SciPy 1.15.3 wheel failed dynamic loading on the local minimum-version host.
The successful PySCF 2.8 result uses SciPy 1.14.1; it does not certify all
transitive dependency versions. No global pin was added for that local issue.

## CI acceptance and merge prerequisites

The new `native-fermion` Ubuntu matrix installs Python 3.10 / PyTorch 2.10 and
Python 3.12 / PyTorch 2.14, verifies a core environment without external science
packages, then installs the pinned PySCF driver and runs real RHF tests. Coverage
planning and execution now explicitly import PySCF before testing. The dependency
lane audit recognizes the chemistry module's skip condition, preventing the
suite from silently losing every installed-dependency consumer.
The matrix waits for source preparation and is included in `ci-complete`, so
either endpoint failing, being cancelled or being skipped blocks the merge gate.

The Linux matrix passed both Python 3.10 / PySCF 2.8 and Python 3.12 / PySCF 2.15
in CI run `42584600022`, separately from the local validation results above.

Both dependency manifests changed, so `docker/ci/images.lock` records four
rebuilt images from publisher run `42582666522`. Each row includes the actual
registry digest and the hash-verified OCI manifest digest from an offline export.
The wrapper continues to compare both manifests byte for byte. Cache provisioning,
qualification and the final CI checks use these immutable identities; no mutable
tag or relaxed manifest comparison is used.

The publisher now emits complete lock-row artifacts. Source restoration retries
truncated HTTP responses within its existing two-retry bound while retaining Git
object and commit verification. The focused CI infrastructure suite initially passed 135
tests; the API transport tests also passed all 18 cases on Python 3.10. The later
expanded CI policy selection passed 488 tests. These selections overlap and their
counts are not additive.

Final implementation-revision CI run `42595072278` (attempt 2) completed all 233
jobs successfully, including 192 coverage shards and the aggregate coverage gate.
The runtime selection passed 6,215 tests with 403 skips; real distributed launch
checks also passed. This is CI infrastructure evidence, not a native fermionic
scalability claim. Maintainer review and merge authorization remain outstanding.
The stacked FCIDUMP PR #1110 must follow #1109 and be retargeted to main after
its dependency merges.

## Supported scope

This is a selected migration, not complete OpenFermion API parity. Symmetry
reduction supports explicit independent encoded Z-parity generators. Preparation
and evolution use unreduced JW occupation registers; parity/BK or reduced-register
ansatz synthesis, general stabilizer tapering, excited Gaussian transformations,
FFFT, file formats, geometry/integral/decomposition gradients and controlled or
higher-order evolution are outside this batch. Evolution permits nonadjacent
density gates and preserves input layout; it does not compose the FSWAP network
or claim nearest-neighbor routing. Truncation and product-formula errors are
separate. No GPU, distributed, performance or release-certification claim is made.

See `docs/development/OPENFERMION_NATIVE_MIGRATION.md` for the unit-by-unit scope
and `docs/architecture/decisions/ARCH_OPENFERMION_NATIVE_BOUNDARIES.md` for the
allowed dependency direction.
