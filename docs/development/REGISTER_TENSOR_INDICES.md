# External register tensor index diagnostics

Simulation owns `register_tensor_indices.py`. The naming expressions and group
rules follow the unchanged Qualtran v0.7.0 `_get_outer_indices` and
`_get_outer_superindices` at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

`register_tensor_outer_indices` consumes the existing Core signature and the
same original, undoubled input/output word bindings as the register tensor
binder. It returns an integer-edge side table with either `(name, index, bit,
group)` values or friendly `name{index}_{bit}{group}` strings. Scalar indices
omit the empty tuple; array indices retain Python tuple punctuation and spaces.
Groups are `l`/`r`, or `lf`/`lb`/`rf`/`rb` on existing even/odd conjugate labels.
Quantum and classical bits both contribute indices; zero-width and empty-array
ports need no binding. The existing binder owns width and binding validation.

This finite slice supplies diagnostics and canonical connectivity inspection.
It does not rename execution labels, allocate tensor data, contract, discover
external topology, infer discarded ports, or create another graph type. The
caller provides the external bindings. Unlike the source's graph reindexing,
the side table cannot invalidate the dense grouping consumer. Display names can
collide (a register name can contain array-index punctuation), so they must
never replace the existing integer execution identity. Duplicate external labels
are rejected as ambiguous; this is an explicit native constraint rather than a
claim about source failure timing. Boolean modes require built-in booleans.

The existing node-connectivity test now consumes this production helper instead
of maintaining a second external-name loop. Complete phase-circuit graphs check
pure and doubled external names against the original graph conversion, then
contract the unchanged native graph before and after inspection. Resource
counts and `circuit.append(op)` execution are unchanged: no physical instructions
are emitted by index diagnostics. The existing binary-field example and measured
Add branch tests remain the user-workflow checks.

Run focused checks in the pinned reference environment, with torch preimported:

```python
import torch
import pytest

torch.set_num_threads(1)
raise SystemExit(pytest.main([
    "-q", "tests/unit/test_register_tensor_indices.py",
    "tests/unit/test_component_tensor_nodes.py",
]))
```

Automatic graph discovery, complete superoperator primitive suppliers, original
empty/zero initialization failures, and complete source-file acceptance remain
open. This module does not extend execution capacity or the public API.

## GRAPH65 verification

- 86 new cases include 72 side/array/pure-or-doubled configurations. Both
  structured and friendly modes are compared against the unchanged source
  helpers using actual source Connection/Soquet objects. Complete phase graphs
  compare both naming modes with original conversions and preserve the native
  contraction before and after the side table is replaced.
- 1,099 related tests passed (12.43 s, five existing warnings), 400 focused
  coverage cases passed (14.28 s, three existing warnings), and 64 integration
  cases passed (3.31 s). The new owner has 23 statements and 16 branches, all
  covered. Twelve related owners total 344 statements and 156 branches at 99%;
  the two pre-existing GRAPH60 refusal paths remain uncovered.
- Strict mypy across 24 files, Black, Ruff, team scope, architecture, Program IR
  and protected API checks passed. Fourteen original source blobs match the
  inventory Git objects. Sixteen unmodified source tests passed (0.67 s, two
  existing notebook marker warnings). Qualtran 0.7.0 and Kickmix 0.1 Simulator
  were present; the binary-field arithmetic append example passed.
- Resource evidence separates diagnostics (zero added physical instructions),
  numerical nodes (seven in the phase fixture), eight pure/sixteen doubled
  external names, and the source model. Phase primitives are H=3, S=2; the
  separate source model reports Clifford=5, T-equivalent=0, measurement=0,
  peak=4. The measured Add path rechecks all 16 branches, 256 inverse history
  pairs and eight append trajectories, including clean work and complex
  amplitudes, through the existing regression. Its primitives and QEC model
  remain separate in the resources artifact; the upstream DiscardQ QEC-model
  exception remains explicit.

Raw evidence is stored locally under `ftqc-reference-env/g2-tensor-indices-*`:
source inventory, regression, coverage JSON, integration, original tests,
environment, example, resources and quality-check logs. Default-suite results
and the independent baseline comparison are recorded separately; this focused
acceptance does not certify a complete upstream file or the entire G2 phase.
