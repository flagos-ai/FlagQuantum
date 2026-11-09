# Native fermionic science without OpenFermion or Cirq

Add a selected native scientific workflow from fermionic operators and molecular
integrals through encoding, measurement, state preparation and low-rank time
evolution. The workflow runs without OpenFermion or Cirq. Existing FermionOperator
and stable root exports keep their identity and behavior; new interfaces live in
concrete submodules. This is one local PR draft against `2a073ab9d`, with no push,
remote PR or merge performed.

## User workflow

```bash
python -m examples.algorithms.fermion_chemistry
```

Expected energy output:

```text
jordan_wigner: HF energy -1.1166843871 Ha
parity: HF energy -1.1166843871 Ha
bravyi_kitaev: HF energy -1.1166843871 Ha
```

The same example then creates a native Slater determinant, composes a two-step
second-order low-rank evolution, and differentiates its real reference amplitude
with respect to time. At time 0.2, the amplitude is approximately 0.9745154458 and
its time derivative is -0.2537232266. No chemistry extra is required for this
native hydrogen example.

## Responsibilities and compatibility

- Core owns standard-library-only complex Pauli values. Hermitian conversion to
  the existing Observable belongs to the fermionic algorithm domain.
- Encodings own JW, prefix parity and Fenwick BK operator images and reference
  occupation maps, with explicit width. Multiple encodings use one unchanged
  measurement consumer.
- Chemistry owns real spatial chemist-integral snapshots, explicit energy offsets
  and reference occupancy. Active-space selection reuses existing frozen-core
  formulas and preserves arbitrary orbital order and original labels.
- The optional `chemistry-pyscf` extra owns direct closed-shell RHF input. PySCF
  objects are converted immediately to native MolecularProblem data. It adds no
  dependency to the core install.
- Static orbital decomposition returns small adjacent matrices. Circuit lowering
  owns their native one- and two-qubit gates. Slater determinants and canonical
  Gaussian vacua use these gates, with complex amplitudes and even/odd parity.
- Low-rank factorization owns signed density-square factors, their full one-body
  contraction correction and a conservative discarded-factor norm bound.
  Evolution owns first/second-order product formulas, exact energy-offset phase
  and differentiable scalar time. FSWAP routing owns pair coverage, fermion signs
  and final layout independently.

The optional existing Cirq interoperability adapter remains separately usable.
No Stable Core snapshot, root export, existing fermionic implementation, runtime
schema, backend registry or global encoding setting changes. Apache-2.0 source
attribution, NOTICE and fixed-revision provenance are retained in
`third_party/openfermion/`; LICENSE and NOTICE ship in the wheel and source
archive, and the source archive also includes PROVENANCE.toml.

## Validation

The complete default smoke/unit tier passed: **18,548 passed, 494 skipped**.
The existing optional/device skips remain visible; this is CPU verification.
The same complete selection ran with four pytest workers:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_ADDOPTS='-n 4 --dist loadscope' \
  python tools/ci_tier.py pr-default
```

The release-configuration checks also passed all 62 tests after adding the
optional chemistry coverage dependency, file ownership and license metadata.

Verified with Python 3.12.14 and PyTorch 2.14.1 on local macOS arm64 CPU. Optional
chemistry tests used PySCF 2.15.0. The source/wheel build used an isolated PyTorch
2.10 build environment, as required by the existing project configuration.

```bash
python -m pytest tests/unit/fermion tests/unit/test_observables_fermion.py \
  tests/unit/test_census_wire_vocabulary.py \
  tests/integration/test_native_fermion_isolation.py -q
# 161 passed.

python -m pytest tests/unit/test_algorithms_chemistry.py \
  tests/unit/test_algorithms_chemistry_integrals.py \
  tests/unit/test_algorithms_molecular.py tests/unit/test_trotter.py \
  tests/integration/test_native_chemistry_driver.py -q
# 356 passed.
```

Independent references cover direct Fock ladder action, every particle sector's
complex minors, canonical annihilator-vacuum residuals, signed factor
reconstruction, full-correction truncation bounds, exact matrix exponentials,
FSWAP signs and finite-difference time gradients. A fresh process refuses
OpenFermion, openfermionpyscf, Cirq, PySCF, NumPy, SciPy and h5py imports while
running the native workflow and checking the optional driver's missing-extra
error. The docstring gate executes the new Slater example.

Ruff, Black and strict mypy pass on the added scientific sources. Architecture,
dependency policy, stable API, capability maturity and qubit vocabulary gates
pass. Source archive and wheel build successfully; all 18 Python source files in
the added domains match the wheel. Mandatory metadata dependencies remain torch
and the Python 3.10 tomli backport.

## Support boundaries

This batch does not implement the entire OpenFermion API. Symmetry reduction
accepts explicitly selected independent encoded Z-parity generators, rather than
general stabilizer tapering. Preparation and evolution create unreduced JW
occupation-register circuits. Parity/BK and reduced-register ansatz synthesis,
excited Gaussian basis transformations, FFFT, file formats, geometry/integral
or decomposition gradients are excluded. Low-rank evolution uses potentially
nonadjacent density gates and preserves the input layout; it does not compose
the FSWAP network or claim nearest-neighbor connectivity. Truncation and
product-formula errors are separate. No distributed, GPU capacity, performance
or complete upstream parity claim is made.

See `docs/development/OPENFERMION_NATIVE_MIGRATION.md` for module entry points
and the verified support matrix, and
`docs/architecture/decisions/ARCH_OPENFERMION_NATIVE_BOUNDARIES.md` for dependency rules.

The existing offline coverage profile now includes chemistry-pyscf, so direct
RHF integration tests have a real CI consumer instead of skipping in every lane.
Third-party attribution files are classified as Integration-owned protected
metadata. The installation contract includes their distributed license paths.
