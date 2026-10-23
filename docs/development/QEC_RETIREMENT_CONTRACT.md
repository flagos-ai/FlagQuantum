# QEC 0.4.0 retirement contract and acceptance

## Approval and scope

The API owner approved [proposal PR #757](https://github.com/flagos-ai/FlagQuantum/pull/757)
and explicitly authorized a contract/test PR directly against `main` before
implementation. The approved proposal head is
`ed0b572657c8249c1d7fe605e8d0b05cdd74fc7b`. Approval does not authorize an agent to
merge either PR. The merged preparation PR #760 changed no library code or version. The
subsequent implementation activates the retired phase explicitly and targets
0.4.0; it keeps the original frozen fixtures and all CI thresholds unchanged.

A subsequent explicit owner decision on 2026-10-23 **supersedes the requirement
to preserve two independent `RotatedSurfaceCode` classes**: ultimately keep only
the original registered class. Both current definitions describe the same code;
the historical `surface` definition adds no independent frame state. Its removal
also removes a Python type identity. This is an additional accepted break, not
an inference from permission to remove import aliases.

The authoritative class remains
`flagquantum.qec.codes.surface.rotated.RotatedSurfaceCode`, also exported through
`flagquantum.qec`, `flagquantum.qec.codes` and `flagquantum.qec.codes.surface`,
and selected by `get_code("rotated_surface", ...)`. Preserve its existing
`flagquantum.qec.codes` pickle module. Do not implement the `rotated_frame`
public module proposed in #757. Delete the duplicate private implementation in
the implementation PR, without replacing the registered class, adding a
subclass or changing its constructor, checks or logical observables.

## Machine-readable contract

[`qec-historical-import-retirement-v0.4.json`](../../contracts/qec-historical-import-retirement-v0.4.json)
records the exact 23 module replacements, target release **0.4.0**, narrow
import/pickle exception, single-class amendment, retained API and SHA-256 of every
frozen restructuring fixture. It references the existing 001 root API fixture;
it does not regenerate or replace that baseline.

For `surface.RotatedSurfaceCode`, the migration is to the existing registered
class or root factory. For `surface.ZxxzSurfaceCode`, use the existing root export
or `codes.surface.zxxz`. The other 22 mappings retain the approved proposal's
owner modules. Shared leaves, root exports, numerical results, schemas,
registration keys, duplicate rejection and plugin authority remain protected.
Artifacts referring only to surviving module paths keep their compatibility
guarantees. Removed-path pickle lookup has no backward-read guarantee.

## Two explicit acceptance phases

The active phase is `retired`, recorded explicitly in the contract. From a core
installation in the repository root:

```bash
PYTHONPATH=. python tools/check_qec_retirement.py --phase retired
PYTHONPATH=. python -m pytest tests/api_contract/test_qec_retirement_contract.py -q
```

The first command must print:

```text
QEC retirement contract passed: retired, 23 modules, 108 exports
```

The former `prepare` phase remains available for diagnosis and now exits 1 with
exact missing-module errors. In merged PR #760 the opposite was required:
preparation passed and retirement refused. That historical result did not mean
removal had occurred. Tests now assert the explicit retired boundary, including
negative checks for accidental aliases and unrelated missing dependencies.
There is no activation inferred from files, broad skip or expected failure.

The tests protect fixture bytes, root/expert/factory identity, one registered
surface class, plugin registration and original seeded results. Of the 13
embedded 001 pickles, three retired-path payloads refuse lookup and ten
surviving-path payloads retain class identity and exact protocol-4 bytes.

The retired phase must pass in a fresh process. It checks:

- every retired import fails for that exact module, including cached aliases;
- replacement imports work, retired source files are absent, and surviving root
  objects retain their canonical identities;
- formerly retired-path root definitions and their own methods use canonical
  module metadata, while surviving-path metadata remains unchanged;
- new global pickles round-trip, surviving old payloads retain class identity
  and bytes, and the three explicitly retired-path old payloads fail lookup;
- `_DecoderT` and the returned decoder registration decorator have valid module
  metadata, without introducing another registry;
- exactly one `RotatedSurfaceCode` definition exists in the QEC source tree, the
  original registered factory selects it, and neither private nor proposed
  public duplicate module is importable;
- frozen fixture hashes and all 108 ordered root exports remain unchanged;
- the retirement does not ship under a 0.3.x patch version.

## Implementation and release acceptance

The implementation activates the retired phase explicitly and updates the
preparation-only assertions to their approved retirement equivalents. It must
not infer activation from whichever files happen to exist or skip incompatible
tests wholesale. The existing numerical, signature, state, registry and
optional-dependency tests continue to run.

Split affected assertions by their actual obligation:

| Existing test area | Authorized change | Required retained evidence |
| --- | --- | --- |
| `test_restructure_compatibility.py` | Replace the exact 23 historical import/metadata assertions and three affected embedded pickle lookups | All 108 exports, signatures/defaults/fields, surviving imports/pickles, names, seeded results and feedback |
| `test_restructure_code_families.py` | Remove duplicate type identity and old family-path requirements | Original registered class identity, family results, checks/logicals and surviving code pickle paths |
| `test_restructure_boundaries.py` | Replace removed matrix-path identifiers and retired imports | Tensor values, shared noise behavior, seeded sweeps and dependency boundaries |
| `test_restructure_execution.py`, `test_restructure_dem.py`, `test_restructure_decoders.py` | Classify removed globals/method metadata without altering frozen payloads | Numerical/state JSON evidence, surviving globals, canonical new pickles and streaming restoration |

Frozen compound pickle files can contain both surviving and retired globals.
An expected failure of a whole compound payload must not replace positive
coverage of its surviving objects or numerical/state payloads. Add specific
canonical reconstruction coverage and retain the immutable JSON oracles; do not
add an unpickling compatibility bridge, broad expected failures or regenerate
snapshots to conceal drift. The additional
[`retirement_fixtures`](../../tests/qec/retirement_fixtures/README.md) preserve
surviving compound-pickle objects and old tensor values exported under the
pre-retirement installation, without touching the 15 frozen artifacts.

Complete source CI, single-GPU and required two-GPU checks must pass at the
reviewed head. Before release, inspect source/wheel/sdist membership, test clean
installed distributions on Python 3.10-3.12 under core/Stim/PyMatching/both
profiles, and run the retired gate plus user examples outside the checkout.
The gate alone does not certify optional SDK behavior or installed packaging.
Current documentation/examples/plugins must stop using the removed imports;
historical audit documents and frozen fixtures retain their original evidence.
No change to or merge of #731 is included.

## Runnable user workflow, unchanged after retirement

This uses existing APIs and needs no optional SDK:

```python
import pickle

import flagquantum.qec as qec
from flagquantum.qec.codes.surface import RotatedSurfaceCode

code = qec.get_code("rotated_surface", distance=3)
print("Public exports:", len(qec.__all__))
print("Registered class:", type(code) is qec.RotatedSurfaceCode is RotatedSurfaceCode)
print("Code pickle path:", type(pickle.loads(pickle.dumps(code))).__module__)
model = qec.DetectorErrorModel.from_stim_text(
    "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
)
print("Correction:", qec.get_decoder(qec.AUTHORITY_NAME, model).decode((0,)).observables)
point = qec.run_repetition_memory_noise_sweep((0.0,), rounds=3, shots=8, seed=53)[0]
print("Logical error rate:", point.logical_error_rate)
```

Expected output:

```text
Public exports: 108
Registered class: True
Code pickle path: flagquantum.qec.codes
Correction: (0,)
Logical error rate: 0.0
```
