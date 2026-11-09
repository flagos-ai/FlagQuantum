# Read restricted FCIDUMP files into native chemistry workflows

This PR replaces the chemistry producer with a file reader: callers can load
real restricted FCIDUMP integrals without PySCF, OpenFermion or Cirq, then pass
the returned MolecularProblem to existing active-space, encoding and measurement
consumers. It depends on #1109 and is initially reviewed against
`feat/native-fermion-migration`, keeping its diff independent from that migration.
Retarget to main after the dependency merges; do not merge this branch first.

## User workflow

```bash
python -m examples.algorithms.fermion_fcidump
```

The example creates a two-orbital file, loads it and measures the occupied
reference with unchanged native circuit consumers. Expected output:

```text
jordan_wigner: reference energy -0.9000000000 Ha
parity: reference energy -0.9000000000 Ha
bravyi_kitaev: reference energy -0.9000000000 Ha
```

For an existing file, call
`flagquantum.ecosystem.chemistry.fcidump.read_fcidump("FCIDUMP")`.

## Ownership and supported scope

- Ecosystem owns the parser and terminates the format boundary immediately at
  MolecularProblem. Core, integral assembly, active-space formulas, encodings,
  runtime, stable root exports and dependency manifests are unchanged.
- Real restricted spatial integrals use chemist notation and eightfold ERI
  symmetry. NORB, NELEC and MS2 are required, with MS2=0 and even electron count.
  Optional IUHF must be zero. ORBSYM and ISYM are validated but do not perform
  symmetry reduction or determine the reference.
- `&END` and `/` header terminators, Fortran D exponents, blank lines and `!`
  comments are supported. Missing integral records are zero. Conflicting
  symmetry duplicates raise errors instead of overwriting values.
- The file constant becomes core_energy exactly once, with nuclear_repulsion=0.
  The file does not distinguish nuclear and frozen-core pieces. The default
  reference fills the first NELEC/2 orbitals and has no computed HF energy.
- max_orbitals defaults to 32 and is checked before dense allocation. ERI
  storage alone is 8*NORB**4 bytes; the native snapshot requires extra memory.
- Complex, unrestricted and open-shell files, orbital-energy records, unknown
  header extensions, symmetry-label conversion and writing are outside scope.
  This is original parser code; no new dependency or format registry is added.

## Validation

Focused tests exercise independent sparse symmetry records, real signed factors,
full native operator matrices, reordered frozen-core selection and all three
encodings; malformed headers, unsupported variants, conflicting duplicates and
allocation limits are rejected. Independent optional PySCF producers write H2
STO-3G and 6-31G files; native integrals and reference energies are compared to
the producer. A clean subprocess blocks NumPy, SciPy, h5py, PySCF, OpenFermion
and Cirq while executing the file-to-measurement example.

Executed locally on macOS arm64 CPU:

- Python 3.12.14 / PyTorch 2.14.1 / PySCF 2.15: 110 science and integration tests passed.
- Python 3.10.22 / PyTorch 2.10 / PySCF 2.8: 41 FCIDUMP and isolation tests passed.
- Installed wheel outside the repository: 40 FCIDUMP tests passed; parser source
  matches the wheel and the native extension loads.
- Installation and CI dependency-audit contracts: 29 passed.
- Parser coverage: 119 statements and 68 branches, 100%, using a pure-Python
  tracer started after preloading PyTorch. Two attempts to trace PyTorch import
  crashed locally and are retained as unsuccessful measurements.
- Ruff, Black, strict mypy on all five Python files, architecture, dependency,
  stable API and repository language checks passed.

The first complete default-suite attempt lacked the new worktree's native CPU
extension: 18,443 passed, 628 skipped and nine execution-path tests failed. After
building the unchanged extension, all nine passed. The final full rerun reported
18,585 passed, 494 skipped and one watchdog failure: the local sandbox denied
process-group signalling and `ps` during child cleanup. No watchdog code or
assertion was changed. This is not a clean full-suite result. CI capacity,
merge-gate, testing-manual and dependency-audit contracts passed 82 tests after
the Ubuntu test selections were extended.
The independent Ubuntu matrix runs FCIDUMP import isolation before optional
dependencies and the two PySCF producer cases after driver installation. Linux
verification belongs to CI. Dependency #1109 rebuilt and qualified all four fixed
images and all 192 coverage worker caches. This stacked PR inherits that lock;
its diff changes neither dependency manifests nor image locks.

## Remote CI acceptance and review status

The following results are bound to implementation revision
`ca9a978fc1da07071e67817b7d291750455a6249`, before documentation reconciliation:

| Check | Evidence | Result |
| --- | --- | --- |
| Full Linux workflow | Manual CI run `42605814883` | 233 jobs successful; both science endpoints and FCIDUMP producer/isolation tests passed |
| Runtime selection | Same CI run | 6,217 passed, 404 skipped; real distributed launches passed |
| Coverage | Same CI run | 192 of 192 shards and aggregate passed |
| GPU workflow | Run `42605825581` | Passed |

The full workflow was dispatched explicitly because the stacked base does not
trigger the main-target PR workflow. These remote results are separate from the
local sandbox failure recorded above. Later commits need their own applicable
checks; the run IDs above remain evidence for their recorded revision.

PR #1110 is open and ready for maintainer review, with no merge performed. Merge
#1109 first, then retarget this PR to main and run target-branch checks before
merging. Neither workflow success nor review preparation is maintainer approval.
