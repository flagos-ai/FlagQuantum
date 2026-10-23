# QEC-RESTRUCTURE-002: code matrices and noise execution

Status: implemented for review after the user's 2026-10-23 approval of the
[001 plan and compatibility strategy](QEC_RESTRUCTURE_001.md). Scope is batch
002 only. Owner: Core; source-reference contract maintenance is a separate
Integration commits. Classification: `single_device_fast_path`.

The branch starts at fetched `origin/main`
`ff69b2526809772b7fadde5bcb420f9f5899dcf6` and fast-forwards the approved 001
commit `398765f7a886e6eea356eea00e830ca399834635`. Since audit PR #735 is still
open, this implementation is a separate stacked PR against its branch, with
only 002 differences. No PR is merged by this work. DynamicCircuit PR #731 and
its branch are not modified or incorporated.

## Implemented boundaries

- `qec/codes.py` becomes `qec/codes/__init__.py` atomically. Every definition,
  code registry, export and lazy CSS alias stays in module `flagquantum.qec.codes`;
  only relative imports deepen. There is no parallel flat module, second code
  table or code-family split.
- `codes/matrices.py` owns `CssCodeMatrices`, `_binary_matrix` and
  `css_code_matrices`. Their algorithms are moved intact. `dem_matrices` retains
  the noise-to-mechanism conversion, and re-exports the exact same matrix class,
  extractor and validator. `dem_construction` retains its existing re-exports.
- `css_code` imports the record directly from `codes.matrices`, removing its
  dependency on the DEM construction/execution module.
- `qec/noise.py` retains its noise records and validation. The original sweep
  loop moves to `execution/noise_sweep.py`; the old module resolves the public
  function lazily and preserves its historical experiment alias and `dir()`
  visibility. `execution/__init__.py` loads no experiment machinery.
- [Code ownership](../../flagquantum/qec/codes/README.md) and
  [execution ownership](../../flagquantum/qec/execution/README.md) document
  allowed dependencies, public entries and a short contributor workflow.

The existing QEC root facade still eagerly imports its public objects. This PR
removes execution dependencies from the data modules themselves; it does not
claim that a cold `import flagquantum.qec` avoids compiler/runtime. No new public
protocol, registry, serialized schema, dependency or numerical algorithm is added.
The independent surface classes are not moved or combined in 002.

## Historical identity and serialization

`CssCodeMatrices` and `css_code_matrices` are the same objects at the QEC root,
`codes.matrices`, `dem_matrices` and `dem_construction`. Their defining metadata
stays `flagquantum.qec.dem_matrices`; the old module re-exports those objects.
The sweep is likewise one object at the root, `noise` and `execution.noise_sweep`,
with its defining metadata kept at `flagquantum.qec.noise`.

The relocated class's own constructor, dataclass methods and property accessors
also retain the historical module. This deliberately scoped metadata assignment
prevents method pickles and caller-owned safe-global identifiers from changing.
It introduces no wrapper class, reducer, production allowlist or replacement
serialization format. Function bodies and dataclass fields are unchanged.
Physical implementation/source links follow the new files; Python import and
pickle identifiers remain historical.

The new trusted pickle fixture was produced from the unchanged 001 worktree,
before testing the moved implementations. It contains a tensor-backed Steane
matrix record, extractor and sweep functions, a noise profile, and seven class
methods. Tests load it, verify all alias identities and tensor contents,
round-trip with pickle/PyTorch, and load a data-only checkpoint with an explicit
caller-owned `weights_only=True` class allowlist. No allowlist is changed in the
library. A second fixture retains seeded pre-move sweep results for three
feedback modes and the noise-model identity.

The 001 compatibility test and its JSON fixture remain byte-for-byte unchanged.
They continue to protect all 108 ordered root exports, signatures, field
contracts, module surfaces, registry behavior, both rotated surface identities,
13 retained historical pickle cases and the seeded DEM baseline.

## Source-reference maintenance

The first commit atomically relocates the code module and updates only physical
source paths in three current check ledgers:

- `contracts/qec-cudaq-alignment-checklist.toml`;
- `contracts/qubit-vocabulary-contract.toml`;
- `contracts/cudaq-parity-matrix.toml` and its rendered reference document.

A follow-up metadata commit updates the maturity ledger's development-artifact
path and the QEC README/implementation source links. The doctest coverage gate
retains the existing tested class entry and verifies its physical source module
explicitly, because the class's historical `__module__` must remain unchanged.
It neither skips the example nor relaxes the coverage assertion.

These paths change from `qec/codes.py` to `qec/codes/__init__.py`. No schema,
retirement count, API identifier, capability status, tolerance or enforcement
logic changes. Historical audit inventory/proposal evidence and Stable Core API
snapshots are not rewritten. The path checks and complete vocabulary counts pass.

## Runnable user workflow

This uses existing public imports and requires no optional SDK, credential or
hardware:

```python
from flagquantum.qec import (
    CssCode, SteaneCode, css_code_matrices, run_repetition_memory_noise_sweep,
)

matrices = css_code_matrices(SteaneCode())
code = CssCode(matrices=matrices, distance=3)
print(code.num_data_qubits, matrices.num_checks, matrices.num_observables)
points = run_repetition_memory_noise_sweep(
    (0.0,), rounds=3, shots=8, seed=53,
)
print(points[0].logical_error_rate)
```

Expected output:

```text
7 6 2
0.0
```

The existing end-to-end example is also runnable:

```bash
PYTHONPATH=. python examples/qec/restructure_compatibility.py
```

Expected output:

```text
Correction observables: (0,)
Logical error rate: 0.0
Decoded data: (0, 0, 0)
```

## Verification and limits

The new six boundary scenarios cover old/new identities, pre-move tensor and
method pickles, PyTorch checkpoint loading, seeded feedback outcomes and data
construction with compiler/runtime/DEM/execution imports blocked in a fresh
process after bootstrapping the existing eager facade.

- Python 3.12.14 / PyTorch 2.13.0: 17 combined 001/002 cases pass.
- Python 3.10.21 / PyTorch 2.13.0: the same 17 cases pass.
- Isolated standard sdist build and wheel-from-sdist build use the declared
  PyTorch 2.10 build dependency. Both artifacts contain `codes/__init__.py`,
  `codes/matrices.py` and `execution/noise_sweep.py`, with no flat `codes.py`.
- A fresh Python 3.12 environment installs the final wheel with PyTorch 2.10.0;
  the 17 cases and golden-path example run outside the source checkout with
  source import insertion disabled.
- The moved matrix validation, record, extractor and noise-sweep ASTs match the
  pre-move definitions. Metadata assignments are explicit outside those bodies.
- QEC/CUDA-Q alignment: 493 checks pass. Parity rendering and the complete qubit
  vocabulary checks pass without relaxing their counts.
- Black, Ruff and ownership checks cover the changed production/test files.

The standalone Stable Core checker reports the same three failures as unchanged
001: `OutputRequest` fields, the `expectation` signature and `submit`. No affected
Stable Core implementation or API snapshot is changed here. The full default and final QEC results follow; a local failure is not reported
as a green gate.

Python 3.11, other operating systems and one-extra installation variants are not
certified by this local run. This is a package-structure change, not a new code
family, numerical/scalability claim, or authorization to proceed into batch 003.


Final full QEC/reference run: **1,818 passed, 1 skipped** in 239.35 s, including
Stim 1.16.0 and PyMatching 2.4.0. The skip is the existing gate-noise case with no
applicable `cx` fault. The final method-metadata and weights-only checks also pass
in both source interpreters and the clean installed wheel environment.

The full smoke/unit run completed with **14,721 passed, 28 failed, 6 errors,
278 skipped, 3,491 deselected** (753.74 s). Comparing every failure/error node ID
against the retained 001 inventory found the same 22 existing failures and six
existing errors, plus six migration-related gate failures. Those six were the
missing maturity-artifact path, broken local source links and moved-example
coverage. After fixing them, **all 53 cases** in the complete public-docstring,
capability-maturity, documentation-source and repository-hygiene files passed
(11.26 s). The docs source-of-truth CLI also passes with `PYTHONPATH=.`.

No unrelated native CPU, evidence-history, local-network or Gloo implementation
was changed, and no failing assertion was disabled. The remaining 22 failure
nodes and six error nodes are the previously reproduced 001 baseline. This is
not a claim that the repository-wide default gate is green; the full broad tier
was not repeated after the scoped documentation/metadata fixes.

CI currently triggers automatically only for PRs targeting `main`. After #735
is merged, this stacked PR needs its base changed to `main` before required CI
and merge review; no workflow or branch-protection rule is bypassed or changed.
