# Rejected lazy validation for fixed two-qubit gates

Owner: Compiler. Classification: `single_device_fast_path`.

**Rejected for insufficient complete-workflow benefit.** The production module
is restored byte for byte to the declared main baseline. The candidate below is
a historical experiment retained at its published freeze, not an enabled path.

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


## First outcome and withdrawal

Published freeze: `00f73d4e26c74ad51986f5de989352401df1627e`.
Both complete draws accept all 480 observations. All 240 paired output payloads
and audits are identical. Fresh producer-free replay accepts all 480 again,
with maximum residuals 6.072657351995283e-14 and 2.8335272960520905e-14
against unchanged 1e-8. There are 160 full-operator observations and 320 limited
three-probe observations. No improved CZ count, dependency depth or routing
quality is established.

The first and only resource observation validates all 144 warmup/timed/traced
outputs, audits and metrics. All six nonregression and Python-memory gates pass,
but **zero** ordinary workflows meet the required ratio at most 0.95. Admission
therefore fails. No favorable rerun, changed threshold or case selection follows.
The numerical singleton shortcut does not establish a useful end-to-end benefit
on these declared workflows and does not resolve the draft controlled-gate
extension's separate failed resource gate.

| Input | Workflow | Baseline median ms | Candidate median ms | Candidate/baseline | Baseline Python peak bytes | Candidate Python peak bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 06/qft/grid | ordinary-restored | 7.140 | 7.247 | 1.014927 | 479867 | 478052 |
| 06/qft/grid | target-artifact | 91.444 | 92.243 | 1.008735 | 2199094 | 2199078 |
| 06/repetition/ring | ordinary-restored | 6.177 | 6.201 | 1.003906 | 565412 | 565254 |
| 06/repetition/ring | target-artifact | 118.049 | 117.135 | 0.992260 | 2567421 | 2567371 |
| 12/qft/grid | ordinary-restored | 28.438 | 28.504 | 1.002324 | 2136960 | 2130504 |
| 12/qft/grid | target-artifact | 562.677 | 579.800 | 1.030430 | 10118164 | 10118214 |

The production shortcut is withdrawn. The 36 retained behavioral tests exercise
the unchanged public optimization behavior, including every fixed gate's
nonidentity against independent runtime matrices, supplied-matrix mutation,
parameter mutation, trainable/batched refusal and routing evidence preservation.
The experiment-only assertion requiring zero internal matrix-reader calls is
removed with the shortcut; this does not weaken any numerical acceptance.

The [inspected index](evidence/fixed-two-qubit-inspected-screen.json) and
[fresh index](evidence/fixed-two-qubit-fresh-screen.json) preserve every row,
source/configuration/version/nonce pin, audit and original compile-wall value.
Their hashed payload files reconstruct both complete raw reports byte for byte,
preserving field order and original SHA-256. The
[resource report](evidence/fixed-two-qubit-resources.json) retains all individual
samples and checks. The [producer-free replay](evidence/fixed-two-qubit-replay.json)
records its complete source and residuals. The reports embed the frozen protocol
and runner source. Original raw reports remain distinct from this retrospective
decision; no measured value is rewritten.

Local sparse-checkout verification and exact-head CI must be reported separately.
Formal Linux qualification, native-memory measurement, statistical significance,
cross-SDK rank and global superiority remain unproven.

After withdrawal, the sparse-checkout `smoke or unit or integration` selection
passes all 387 tests. Strict Linux-targeted mypy passes for the complete
production optimization module and retained test module. Ruff, Black, changed
file language checks, Markdown links, revision-field vocabulary checks and
`git diff --check` pass. Production source is byte-identical to the declared
baseline. These checks are local evidence; full exact-head CI is outstanding.
