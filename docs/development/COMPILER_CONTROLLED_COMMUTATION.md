# Controlled-gate commutation

Owner: Compiler. Classification: `single_device_fast_path`.

## Requirement and implementation

Uncomputation, parity networks and repetition encoders often contain controlled
operations with a shared control or target. The existing structural checker
declines these even when their target operators commute, preventing the default
commuting-cancellation pass from removing redundant gates. Extend that existing
authority rather than add a second table, numerical oracle or compiler pass.

The admitted two-wire opcodes are the existing controlled X/Y families: `cx`,
`crx`, `cy` and `cry`. Neither target may be the other gate's control. Distinct
targets then commute, including distinct target axes. Shared targets additionally
require the same Pauli family. Existing diagonal rules remain separate; this
change makes no additional claim for arbitrary controlled matrices or gates with
more than two operands.

For a common control, write a gate as `P0 * I + P1 * U`: the orthogonal control
projectors eliminate cross terms, while operators on distinct targets commute.
For distinct controls and a common target, the control projectors commute and
each target branch is a function of the same Pauli matrix. A target that is the
other gate's control invalidates this reasoning and is declined. The proof holds
for every parameter value; implementation reads only opcodes and wire roles.

This extends FlagQuantum's existing structural analysis, whose block pattern is
documented against Qiskit's CommutationAnalysis. It is a mathematical extension,
not a newly copied Qiskit or Cirq implementation. No dependency, public signature,
default, routing policy, result field or serialized schema changes. Routing replay
and native-block evidence validation remain intact. This does not admit the
separate commuting-frontier experiment into production.

## Executable workflow and validation

```python
import flagquantum as fq
from flagquantum.compiler import compile

source = fq.Circuit(3).cx(0, 1).cx(0, 2).cx(0, 1)
compiled = compile(source)
assert [(gate.name, gate.wires) for gate in compiled] == [("cx", (0, 2))]
```

Expected: default optimization removes the two uncomputed CX gates, preserving
the complete operator. Replacing the middle operation with `cx(1, 2)` must keep
all three gates because the target/control dependency is real. The same-family
shared-target example `cx(0, 1); cx(2, 1); cx(0, 1)` also reduces to one gate.

The new tests check all 1,728 ordered X/Y-controlled opcode, overlapping wire
placement and angle combinations against complete three-qubit runtime matrices,
in both commutation directions. These samples validate the implementation of the
structural proof, not its universal quantifier. Separate tests preserve symbolic
and batched trainable parameter identity, decline matrix overrides and crossing
control/target roles, and verify complete operators through default compilation.
Before the implementation, 29 new tests fail and five negative cases pass.
The existing 8,100-pair commutation census remains unchanged because its two-wire
placements omit the newly admitted fan-out and fan-in configurations.

## Frozen application diagnostic

Publish the implementation and this protocol before observing application
quality. Use unchanged SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144`,
production baseline `46ba1523ed0c47fcf304d64d33b34d3f6c4b9629` and exactly this
rule extension. Compare the old and new `commute` functions with equal patch
lifetimes, fresh adapters and all other compiler code identical. Do not install
the commuting-frontier selector or any rejected native-cost selector.

Run all 60 application inputs: widths 6/8/12; QAOA, Ising, hopping, QFT and
repetition; line/ring/grid/sparse. Both ordinary-restored and target-artifact
workflows use baseline/candidate at compiler seed 11, giving 240 observations
per draw. First use the inspected nonce `01` repeated 32 times. Then generate
exactly one fresh 32-byte nonce after publication and freeze it before executing
the second complete draw. Do not tune, discard cases or redraw after observation.
Fresh generation excludes prior application-result inspection for that draw;
it does not establish an independently administered qualification campaign.

Freeze source hashes, both manifests, driver and exact participant configurations.
Retain every failure, unsupported outcome, output, layout, original compile-wall
acceptance input and adapter audit. Alternate baseline/candidate order by input.
Independent acceptance remains unchanged: topology and operation caps, 120-second
adapter deadlines, 1e-8 semantic threshold, complete operators at width six and
three probes at widths eight/twelve. Those probes are not universal equivalence.
Replay the archives in a fresh process without importing any producer SDK.

Use one numerical thread and no concurrent local test suite, with each draw
under a 900-second process-group supervisor and a 64 MiB raw-report disk cap.
Compare delivered CZ count and dependency depth separately, retaining all
regressions and tradeoffs. Compilation wall values in this diagnostic are
acceptance inputs, not a timing ranking. Any follow-up resource protocol must be
declared before its measurements. The shared macOS host cannot certify formal
Linux cgroup peak memory or exclusive-core qualification. No cross-SDK ranking,
global superiority, hardware or scalability claim follows from this study.
