# Lazy numerical validation for fixed two-qubit gates

Owner: Compiler. Classification: `single_device_fast_path`.

## Requirement and implementation

Repeated optimization rounds construct matrices for isolated fixed two-qubit
unitaries even though a known nonidentity singleton cannot shorten. Avoid that
work using the existing operator schema, without a cache or a dependency.
Only a declared fixed gate with no parameters and no matrix override qualifies.
Routing-phase guards remain first. Multiple-gate blocks, parameterized gates,
extra parameters, mutable supplied matrices and trainable values retain the
existing numerical path. No pass order, public API, routing policy, serialization
or target replay changes. This is an internal improvement, not a new upstream
algorithm port. Existing Qiskit/Cirq attribution remains applicable to their
separate reused implementations.

All fixed two-qubit opcodes and both operand orders are checked against the
independent runtime matrix reader for nonidentity. Tests also cover mutable
matrix overrides, parameter mutation, parameterized identities, batched and
trainable refusals, and routing evidence barriers. A 2,000-gate interfering
singleton circuit preserves every instruction without constructing matrices.
The focused 61 tests, 278 expanded compiler tests, 387 tests selected by
`smoke or unit or integration` in this sparse worktree, and strict mypy for both
complete changed modules pass. The sparse selection is not the full repository
tier; required complete CI remains outstanding.

## Prospective acceptance

Publish this candidate and protocol before observing complete-workflow timing.
The exact baseline is integrated main
`d8941cea0950447ef4e3bb73ad42a5d5efe4b2ae`. The control loads that module's
original two private functions; the runner verifies all other AST nodes match.
Both variants use the same complete adapter and existing verification pipeline.
The controlled-gate extension from draft PR #1304 is absent from both variants;
passing this control would not admit that extension or erase its failed results.

Freeze the source hashes, dependency versions, runner hashes, adapter options,
compiler seed 11 and unchanged SDKBench commit
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Use all 60 application inputs
(three widths, five families, four topologies), both ordinary-restored and target
artifact workflows, and both variants: 240 observations per draw. The first
nonce is `01` repeated 32 times. Generate exactly one fresh 32-byte nonce after
publication and freeze it before the second draw. Preserve all failures and
outputs; do not redraw, omit cases or tune using the fresh draw.

Every paired serialized operations/final-layout payload must be identical,
with equal audits and numerical acceptance. Independently accept every output
with unchanged topology/layout/capacity and 1e-8 semantic thresholds. Width six
uses full operators; widths eight/twelve use three limited probes, which do not
prove full-operator equality. Reconstruct both reports in a fresh process without
importing FlagQuantum, Qiskit or Cirq. No SDKBench adapter or gate is changed.

For inspected QFT6/grid, repetition6/ring and QFT12/grid, execute both complete
workflows with two warmups, nine retained alternating rounds and one separate
traced round per variant. Use fresh adapters, identical patch lifetimes and one
numerical thread. Accept and match the frozen quality output hash, metrics and
audit of all 144 warmup/timed/traced executions outside timing. Keep every sample.
Each of six candidate/baseline median ratios must be at most 1.05. At least two
of the three ordinary-workflow ratios must be at most 0.95 to establish a useful
benefit. Every Python allocation peak must be within both 1.10 times baseline
and 128 KiB above baseline. Python tracing does not measure native allocation,
process RSS or cgroup memory. These descriptive local ratios do not establish
statistical significance or cross-SDK performance rank.

Any failed gate rejects admission: retain the first outcome and do not rerun for
a favorable result, widen limits or select cases afterwards. A different future
candidate needs a separate prospective control. Source stays immutable during
observations; tests do not run concurrently with measurements. Each quality draw
has a 900-second process-group budget; resource/replay phases have 300 seconds;
each raw report has a 16 MiB ceiling. Output files are created exclusively.
Retain raw reports or byte-exact reconstructable hashed chunks below repository
artifact limits. Full required exact-head CI remains necessary before merge.
Formal Linux qualification remains deferred. Do not restart any machine/service,
upgrade a kernel or infer scalability, new hardware support or global superiority.

## Existing supported user workflow

```python
import flagquantum as fq
from flagquantum.compiler import compile

circuit = fq.Circuit(3).cx(0, 1).cx(0, 2)
result = compile(circuit)
assert [(gate.name, gate.wires) for gate in result] == [
    ("cx", (0, 1)), ("cx", (0, 2))
]
```

Compilation preserves the two interfering singleton gates. The implementation
avoids proving their known nonidentity through repeated matrix construction.
This does not claim fewer CZ gates or improved routing quality.

Frozen control driver: `fq-fixed-two-qubit-control-20261122.py`, SHA-256
`91e3ce2f9043966fcfd652b3adf145cff462dfd291000eea051fddf4acb8f5c6`.
Producer-free replay: `fq-fixed-two-qubit-replay-20261122.py`, SHA-256
`a3887c493abdae5716c71e60afedbf1daab80ec40bafb8310c1eea72e49dec73`. Both complete sources are embedded in their reports.
