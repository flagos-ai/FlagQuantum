# Open-system register tensor contraction

Simulation owns `register_superoperator_binding.py`. This finite local numerical
slice adapts Qualtran v0.7.0 `_dense.py::_group_outer_inds` at
`bcb17bb25360ed68ce3bc084645db3e929799389`; reference sources remain unchanged.
It consumes Core `Signature` and existing `TensorNetworkNode` objects, without
introducing a foreign graph, contract, backend, or public export.

Incoming/outgoing mappings contain original undoubled labels, one MSB-first word
per row-major register element. `Register.bitsize` already uses `dtype.num_bits`,
so classical bits contribute axes just like quantum bits. The existing binder
validates widths and connections. Empty arrays and zero-width ports contribute
no axes and need no binding. The result groups are nonempty `rf`, `rb`, `lf`,
`lb`, in that order. Labels use the existing even-forward/odd-backward encoding;
backward data is elementwise conjugated without transposition.

The caller supplies doubled nodes and explicit trace merges. The native greedy
contraction engine returns a scalar, a density/effect matrix, or a rank-four
channel. At least one tensor node is required, including an explicit scalar
node for a scalar-only network. This explicit native refusal is not attributed
to the source. No normalization, reset, discard discovery, or Quimb simplification
is performed. Dense channel storage for n bits per side is `element_size * 16**n`;
this is no large-system capacity claim. Automatic SSA discard discovery, complete
primitive suppliers and a complete superoperator runner remain open.

The existing user workflow remains `circuit.append(op)`. The shared Add example
in `examples/algorithms/binary_field_arithmetic.py` remains executable; internal
contraction is verified against its bound primitive path and measured Add tests.
Run the focused tests with the pinned reference environment and preimport torch:

```python
import torch
import pytest

torch.set_num_threads(1)
raise SystemExit(pytest.main([
    "-q", "tests/unit/test_register_superoperator_binding.py",
    "tests/unit/test_tensor_conjugate_pairs.py",
]))
```

Tests compare original index groups and dense results for 27 mixed quantum/classical
layouts, scalar/array/empty ports and both register sides. Original MeasureZ and
MeasureZ-to-Discard channels check classical output axes and mixed-state action.
Complex autograd and supplier replacement preserve coherence. Existing full
phase channels and initialized densities now consume the production helper;
16 measured Add branch densities use it before verifying work zero and branch
sums, with existing 256 inverse history pairs and eight append trajectories.
Contraction emits no gates: primitive instruction counts, numerical tensor counts,
and source resource estimates must be reported separately.

## GRAPH64 local evidence

- 34 new cases; 1,013 related tests passed (18.17 s, five existing warnings),
  314 focused coverage cases passed (20.80 s), and 64 integration cases passed
  (4.23 s). The new owner has 15 statements and two branches, all covered;
  eleven related owners total 321 statements and 140 branches at 99%, retaining
  the two previously uncovered GRAPH60 refusal paths.
- Strict mypy across 22 files, Black, Ruff, team scope, architecture, Program IR
  and protected API checks passed. Fourteen reference blobs match their pinned
  inventory Git objects; 16 unmodified source tests passed (1.30 s, two existing
  notebook marker warnings). Qualtran 0.7.0 and Kickmix 0.1 Simulator were present;
  the binary-field arithmetic example passed.
- The measured Add emits H=12, T=8, CX=40, Tdag=8, S=4, measurement=4, CZ=4,
  reset=4. The separate source model reports AND=4, Clifford=22, measurement=4,
  T-equivalent=16, peak=8; two clean work bits are checked independently.
  MeasureZ and MeasureZ+Discard each model one measurement and peak one bit.
  DiscardQ's source QEC estimate raises `DecomposeNotImplementedError`; its
  peak-two-bit estimate is available. No fabricated zero QEC estimate is used.

Raw local evidence resides in `ftqc-reference-env/g2-superoperator-binding-*`
(source inventory, regression, coverage JSON, integration, original tests,
environment, example, resources, format/type/architecture checks and default
baseline comparison). Default-suite results are recorded separately in the
integration ledger, including existing failures and any tests not reached.
