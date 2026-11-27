# Bounded commuting-frontier routing experiment

Status: development experiment; no production policy is enabled. Owner:
Compiler. Classification: `single_device_fast_path`.

## Requirement and existing mechanisms

The application gap study identifies remaining delivered CZ/depth differences
on QFT and repetition circuits. SABRE currently serializes every shared-wire
instruction, even when the native structural checker proves two operations
commute for every parameter value. Reuse that existing checker rather than
introduce a numerical commutation oracle or a new opcode table.

`benchmarks/compiler_commuting_frontier.py` builds bounded consecutive runs of
pairwise commuting instructions on each wire. The scan follows the same
pairwise-block principle already documented in `optimization/commutation.py`
for Qiskit's CommutationAnalysis. The graph and its integration with the existing
SABRE private dependency injection are FlagQuantum-owned experimental code;
this is not a new Qiskit or Cirq source-code port. Existing upstream attribution
and the independent structural rule tests remain authoritative.

All operations in the previous run precede all operations in the next run on
that wire. Each run contains at most 20 operations. For N operations of arity
at most two, this gives at most 40 * N quantum edges and O(20 * N) pairwise
predicate work. Runs split at the bound even if a larger run would commute.
No all-circuit pairwise dependency graph or unbounded relation cache is built.
The graph is owned by one call; the trial copies its indegrees as usual.

The entire circuit retains the production path if it contains annotations,
custom matrices, unknown operations, nonunitaries, or unsupported arity. This
conservative first slice has no new classical-dependency implementation. Symbolic
and trainable parameters remain the original objects; the checker reads no
parameter value and compilation performs no sampling or detachment.

## Candidate selection and limits

Retain the complete existing production selection as the incumbent. If its
baseline has no SWAPs, or the dependency analysis offers no relaxation, do no
additional search. Otherwise attempt exactly one deterministic SABRE trial with
the commuting graph, pruning at the incumbent's delivered cost. Select it only
if forward plus final-restoration SWAP cost is strictly smaller; equal costs keep
the incumbent. Existing caller-supplied directed-cost callbacks remain in force.
The extra trial retains existing progress, cycle and topology limits.

This experiment adds a trial when eligible. It does not claim unchanged compile
time or memory, nor that smaller routing cost guarantees smaller delivered CZ
count or depth after native optimization. Keep both metrics, failures, and raw
resource observations. Existing default strategies, optimization levels,
restored final placement, public fields, and independent routing replay are
unchanged. Ordinary and target-artifact workflows remain separate.

## Frozen first diagnostic

Before observing quality results, freeze this protocol and implementation, then
run the unchanged SDKBench source at
`9aa9260920984e2ca05bcb293abb885fccd9c144` against integrated runtime
`46ba1523ed0c47fcf304d64d33b34d3f6c4b9629`. Record exact source/driver hashes in
an exclusive-create manifest. This is an inspected-input diagnostic using nonce
`01` repeated 32 times, not a new unseen draw or a qualification campaign.

Use all 60 application inputs: widths 6/8/12; QAOA, Ising, hopping, QFT and
repetition; line/ring/grid/sparse targets. Compare baseline and candidate for
both `flagquantum-ordinary-restored` and `flagquantum-target-artifact`, compiler
seed 11. Alternate complete baseline/candidate order across cases. This gives
240 attempted observations; retain every failure and unsupported result. Do
not select source cases, seeds, graph bounds or options after seeing results.
No participant source or adapter option is modified.

The baseline and candidate use equal patch lifetimes around the existing trial
selector. Preserve normalized outputs and layouts for independent replay, all
adapter audits, original compile-wall acceptance inputs and all returned metrics.
The unchanged evaluator checks topology, operation caps and its 1e-8 semantic
threshold: complete operators at width six, three probes at widths eight/twelve.
Three probes do not establish general unitary equivalence.

Run with one numerical thread, no concurrent local test suite, the adapter's
existing 120-second deadline and evaluator caps, and a 900-second process-group
supervisor. Persist after each input. The shared macOS host cannot establish
formal Linux cgroup peak memory, exclusive-core qualification or SDK rank. No
machine, service, driver or kernel may be restarted or upgraded. This diagnostic
can justify a next production proposal, not automatic production admission.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(4).h(0).cphase(0, 3, theta=0.2).cphase(0, 1, theta=0.3)
compiled = compile(source, coupling_map=CouplingMap.line(4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

Expected: equivalent legal compilation with restored placement. This workflow
continues using the production selector unless an explicit development harness
installs the experimental selector for the duration of its call.
