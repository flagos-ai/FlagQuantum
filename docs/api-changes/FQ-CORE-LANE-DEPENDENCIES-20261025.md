# Core-lane dependency discipline: guard optional imports and fall back from `tomllib`

- **Status:** implemented
- **Date:** 2026-10-25
- **Area:** development tooling, CI, dependency policy
- **Public API impact:** none

## Decision

The environment the `cpu-core` job builds holds PyTorch, the `dev` extra, and
nothing else. A module that the marker-selected suite collects or the gates
execute must therefore either not reference an optional root or reference it
where the absence is handled. That rule is now machine-checked by
`tools/check_dependency_policy.py`, `numpy` joins
`[import_policy.core_forbidden_imports]`, every `tomllib` reference uses the
`try`/`except ModuleNotFoundError` fallback, and the `cpu-core` job asserts both
absences instead of relying on them.

Nothing in this record changes an export, a signature, a default, an enum or
`Literal` value, a result field, a serialized schema, or documented exception
behavior. It adds no opcode, no IR field, and no serialized key.

## Why: one defect class with three measured forms

`cpu-core` installs `pip install -e '.[dev]'` on top of a CPU PyTorch wheel. No
optional extra is present, so `numpy`, `stim`, `qiskit`, `matplotlib`, `jax`,
and the rest are genuinely absent. The job's own log states it in as many words:

```text
torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize
NumPy: No module named 'numpy' (Triggered internally at
.../torch/csrc/utils/tensor_numpy.cpp:84.)
```

Three landed changes turned that absence into a red lane while every developer
machine stayed green, because those machines have the extras installed:

1. A test module imported an unavailable module outright.
   `tests/unit/test_circuit_power.py` imported `numpy`, which is in no extra.
   Collection failed on all three `cpu-core` legs with
   `ModuleNotFoundError: No module named 'numpy'`.
2. A test module imported `tomllib` with no fallback.
   `tests/unit/test_density_matrix_output_contract.py` had a bare
   `import tomllib`; the 3.11 and 3.12 legs passed and the 3.10 leg failed
   collection.
3. `pytest.approx` against a zero-dimensional tensor holding an inexact value
   returned `False` rather than comparing tolerantly, which is a numerical
   contract defect rather than an environment defect and is recorded with the
   density-matrix output change.

The correct resolution for the first two is to remove or guard the import. It is
not to lean on a transitive dependency of PyTorch — PyTorch declares no `numpy`
requirement at all — and it is not to widen a tolerance.

A single point measurement cannot find this class, because the failure is a
property of the environment rather than of the code path. Locally simulating the
absence is itself easy to get wrong; the measured comparison is:

| Absence simulation | `import` | `find_spec` | Verdict |
| --- | --- | --- | --- |
| raising `MetaPathFinder` | raises | **raises** | not equivalent; caused 122 spurious `torch.fx` failures, because `torch._dynamo` calls `find_spec` with no `try` |
| in-process finder plus `-n 2` | raises in the parent | — | silent no-op; xdist workers never see the blocker |
| interpreter without the package | raises | **`None`** | faithful |

The faithful instrument is an interpreter that lacks the package. Under one, the
delivered suite reports **6 failed, 8486 passed, 304 skipped**, and all six
failures reproduce with `numpy` installed, so no failure in the tree is caused by
its absence.

## What changes

### 1. The reference scan reads `tests` and `tools`

`reference_errors()` parses every Python source under `COLLECTION_TREES` and
reports each reference to a `core_forbidden_imports` root that is not handled.
It accepts exactly these forms:

- the import sits inside a `try` whose handlers absorb
  `ImportError`/`ModuleNotFoundError` (or a bare or `Exception`-wide handler);
- the enclosing module or function calls `pytest.importorskip("<root>")`;
- the enclosing module or function calls `find_spec("<root>")` **and** also
  skips — an inspection alone is not a guard, and the scan reports it;
- the node exists only under `if TYPE_CHECKING:`, so it never executes.

A guard on a root that *requires* the referenced root also counts. That is the
`guard_implications` table described below.

`benchmarks` is deliberately out of scope: it holds standalone comparison
scripts whose module-level imports are entry points, not modules a marker
selects. That is measured rather than assumed: `pytest --collect-only benchmarks`
reports `collected 0 items` because `pytest.ini` sets `testpaths = tests`, and the
CI lanes that do run those files (`local-gpu.yml`, `scheduled-hardware.yml`) run
them as scripts through `torchrun`, on hardware lanes that never touch `cpu-core`.

`flagquantum` is out of scope for the same rule for a different reason: it is not
a collection surface, and the gate that covers it is stronger. `check_import_time.py`
imports the package in a fresh interpreter and fails if any forbidden module got
loaded, so a module-level `import numpy` added to the package fails the probe at
the moment it loads rather than being read out of the source. The 28 `numpy`
imports inside the package are all function-local, inside
`flagquantum/runtime/executors/jax/**` and `flagquantum/simulation/jax/**`; that
backend runs only once JAX is importable, and JAX requires numpy, so those
references are reached only in an environment that has it.

### 2. `numpy` joins `core_forbidden_imports`

At the time of this core-lane change, `numpy` was on no extra. The later QBoson
adapter placed it in the isolated `kaiwu` remote-provider extra, without adding
it to core and without redistributing the Kaiwu SDK. `torch` still declares no
`numpy` requirement, and nothing on the core path imports it. Recording it as
forbidden makes three existing proofs cover it with no new machinery: the
`cpu-core` assertion, the subprocess check in
`tests/unit/test_dependency_policy.py`, and the `check_import_time.py` probe that
already reads this list and fails when a forbidden module is loaded.

### 3. `guard_implications` replaces per-file exemptions

Several test modules legitimately import `numpy` after
`pytest.importorskip("stim")`: `stim` requires `numpy`, so the guard holds when
`numpy` is absent, because a lane that has stim has numpy too. That fact comes
from stim's metadata and from no file in this repository, so it is declared once:

```toml
[import_policy.guard_implications]
numpy = ["jax", "matplotlib", "pymatching", "qiskit", "stim"]
```

Every entry is the distribution's own `requires_dist`: stim 1.16.0 requires
numpy, jax 0.11.2 requires `numpy>=2.1`, matplotlib 3.11.2 requires
`numpy>=1.25`, pymatching 2.4.0 requires numpy, and the pinned qiskit requires
`numpy<3,>=1.17`. The policy rejects an entry that names a root outside
`core_forbidden_imports` or that names itself, so the table cannot become a
laundering route for a reference the scan should report.

### 4. `tomllib` uses one fallback form everywhere

`tomllib` is the 3.11 standard library. Three of the fifty-four sites used
`if sys.version_info >= (3, 11):` and the rest already used `try`/`except
ModuleNotFoundError`. The version test is inferior on three counts: it states a
fact about the interpreter rather than providing a fallback, it cannot be
simulated on a supported interpreter the way a missing import can, and it does
not survive a stripped standard library. The three sites are normalized
(`flagquantum/simulation/numerics/conformance.py`,
`tools/check_interop_capability_gap_matrix.py`, `tools/check_qubit_vocabulary.py`),
and `stdlib_fallback_errors()` holds all four source trees to the single form.

### 5. The `cpu-core` job asserts the lane

| Assertion | Why |
| --- | --- |
| `find_spec('numpy') is None` | the absence the scan's rule assumes, recorded rather than inferred from a `UserWarning` |
| `find_spec('tomllib') is not None` iff `sys.version_info >= (3, 11)` | the 3.10 leg is the one that exercises the fallback |
| `find_spec('tomli') is not None` wherever `tomllib` is absent | the fallback the `try` form imports is installed where it is needed |

`numpy` is deliberately not added to the existing "external quantum frameworks
are absent" list: it is not a quantum framework, and folding it in would describe
the assertion as something it is not.

## Non-goals

- **No rewrite of `pytest.importorskip`.** It is a correct guard. A durable
  `ModuleNotFoundError`-style alternative would cost roughly 220 lines and change
  no measured fact.
- **No scan restriction to `flagquantum`.** The defect appears in `tests` and
  `tools`, which the lane collects and executes; a package-only scan would have
  reported both landed defects as clean.
- **No new scanner.** The rule extends the existing dependency gate rather than
  adding a second source of truth, per engineering decision principle 6 and
  maintainability guardrail 9.
- **No existing gate is weakened.** `--print-floor` still reads only the policy,
  and `policy_errors` keeps validating the manifest independently of the scan.

## Evidence

On the delivered tree, with the local interpreter (Python 3.12.14, torch 2.13.0):

```console
$ python tools/check_dependency_policy.py
dependency policy passed
$ python tools/check_dependency_policy.py --print-reference-census
scanned tests 811, tools 85; guarded references: jax 30, matplotlib 4, numpy 30, pymatching 1, qiskit 8, stim 2, torch 1
$ python tools/check_dependency_policy.py --print-floor
ruff==0.15.0 black==24.1.0 mypy==2.0.0
```

The `tomllib` census over all four source trees reads **54 `try`-form sites, 0
other**. `tests/unit/test_dependency_policy.py` passes 24 tests, including the
two that assert the scans return no error on the real tree and that the scan
covers every collected tree — a scan that silently reads nothing fails rather
than reports success. `tools/check_import_time.py` passes with `numpy` on the
forbidden list and reports no forbidden module loaded. `python -m pytest
tests/smoke/test_no_jax_core.py -q` passes.

### Falsification

The gate was driven in six directions on a scratch tree, then reverted:

| Input | Expected | Measured |
| --- | --- | --- |
| `import numpy as np` at module scope in `tests/` | fail | fail, naming line 1 and the three accepted guards |
| the same reference after `pytest.importorskip("stim")` | pass | pass |
| the same reference after `find_spec("stim") is None:` → `pytest.skip(..., allow_module_level=True)` | pass | pass |
| `find_spec("stim") is None:` → `pass`, then `import numpy` | fail | fail — inspection is not a guard |
| `if sys.version_info >= (3, 11): import tomllib` in `tools/` | fail | fail, naming the fallback form |
| scratch files removed | pass | pass |

## Compatibility

The `dependency-policy.toml` schema identifier stays
`flagquantum_dependency_policy_v2`. `[import_policy.core_forbidden_imports]`
gains one member and `[import_policy.guard_implications]` is added; both are
validated by the same `policy_errors` call, and `--print-floor` output is
unchanged, so the `dependency-bounds` lane is unaffected. No protected contract
changes shape.

## Authorization

This is an integration change: it touches `.github/**`, `tools/**`, and
`dependency-policy.toml`, all of which are protected paths, and it is
cross-cutting by construction rather than an ordinary single-domain change. It
is admitted under the current strategic priority's clause 2 because it
**replaces** an existing check — the `check_import_time.py` probe and the
dependency-policy subprocess test now cover `numpy` — instead of introducing a
parallel mechanism, and per engineering decision principle 10 the replacement is
demonstrated by the existing consumers passing unchanged. Maintainability
guardrail 4 is satisfied because no new contract type, identity, verdict,
registry, or intermediate representation is created: the rule is two functions
inside the gate that already owns dependency policy.
