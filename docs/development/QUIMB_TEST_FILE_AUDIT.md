# Complete original composite tensor tests

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, Apache-2.0, Google LLC.
The complete `simulation/tensor/_quimb_test.py` blob is
`ceb9a347a024e85317bb11eb8befd43cf2cba5c3`.

This audit covers Q-0892, the complete test file, independently of Q-0891's
broader implementation scope and original defects. It does not waive any
empty-output, zero-width, direct-discard or arbitrary supplier-discovery boundary.

## Complete source-to-native map

| Source definition or scenario | Native owner and acceptance |
| --- | --- |
| `TensorAdderSimple.signature` | Existing typed `Signature` with one caller-owned, scalar, opaque one-bit `x` port. |
| `TensorAdderSimple.my_tensors` | Explicit test supplier returns the same binary identity tensor with input/output axes in the original order. No source callback enters production. |
| `test_cbloq_to_quimb` | Four calls to that typed function retain four distinct numerical tensors and the exact external `x` left/right labels. Full per-node data, canonical ordered edges and full complex operator match. |
| `test_cbloq_to_quimb_with_no_ops_on_register`, selection/target | Existing builder Split/Join and `register_bit_wiring_nodes` preserve the two-bit MSB order, two source bookkeeping calls, four numerical wiring tensors, and one idle-target identity. Complete three-bit operator is identity. |
| Same test, sole idle target | Direct return inserts exactly one existing native idle identity. Full one-bit operator and both external axes match. |

`tests/unit/test_quimb_original_file.py` executes both original test functions
without changing their code or assertions and compares all three source graphs
to native consumers. It verifies the complete frozen test-file blob, actual
Cirq profile, Qualtran 0.7.0 and Kickmix 0.1 with its real Simulator present.
There is no source file edit, source implementation replacement, new numerical
owner, duplicate graph representation or new public entry point.

Native graph wiring, contraction, binding and ordinary `circuit.append(op)` run
with Qualtran and Quimb imports blocked. Every complex matrix entry agrees,
and coherent input vectors preserve complex amplitudes with the additional
clean work bit zero. There are no measurements in these scenarios, so there
is one deterministic branch. Quantum words remain caller-owned. Numerical
edge labels are local tensor indices, not physical work or resource estimates.
The source idle-placeholder axis convention is explicitly reversed for the
comparison with the existing native output/input convention; the binary
identity data and canonical connectivity remain equal.

## Resources and reused evidence

Each bound native program emits zero primitive instructions, zero measurements
and zero allocated private work. Four custom numerical identity tensors are
not four identity gates. Split/Join and idle placeholders also emit no physical
gates. The source custom class declares no independent QEC cost; this audit
makes no unsupported model estimate. The unchanged larger measured-Add,
complex-branch, resource and work-cleanup evidence remains in
[the component tensor audit](COMPONENT_TENSOR_NODES.md).

The original implementation's separate defects remain in
[the Quimb source disposition](QUIMB_SOURCE_DISPOSITION.md). They are not
assertions in these two original test functions and do not block accepting
this complete test file. Q-0891 remains open.

## Reproduction

With the locked optional reference dependencies and preimported torch:

```bash
python -m pytest tests/unit/test_quimb_original_file.py -q
```

The batch receipt beside the worktree records both actual Cirq profiles,
concentrated regression and strict verification. No scientific dependencies
were installed. No CI, GPU, performance or whole-repository pass is claimed.

Local acceptance on 2026-11-26: 146 concentrated cases pass in each actual
Cirq 1.7.0 and 1.6.1 profile, without skips. Strict mypy checks the real locked
SDK definitions; Black, Ruff, verification scope, language and links pass.
Production files are unchanged from published `0a1f4cbe7`, so its completed
architecture, IR, API, dependency and 145-case local integration receipts are
reused rather than rerun. An initial pytest plugin-autoload environment error
was resolved with the established disabled-autoload profile. Initial strict
source reexport errors were corrected by importing actual defining SDK modules;
no suppression or source edit was introduced.

Dependency follow-up: the initial acceptance reused the production dependency
receipt too broadly. The new test file also requires its own optional NumPy
and Cirq collection guards. They are now explicit `pytest.importorskip` checks;
locked reference execution still verifies real versions and records no skips.
This fixes two new collection-policy findings without changing source scenarios
or weakening the mandatory reference lane. The subsequent external-adjoint
batch records the full dependency-policy comparison and this correction.
