# Prospective application quality diagnosis

Status: freeze before the single input draw. Owner: Compiler. Classification:
`single_device_fast_path`. This local diagnostic identifies remaining native
compiler quality gaps; it changes no compiler implementation or SDKBench code.
It does not replace the resource-qualified campaign deferred by the user.

## Fixed sources, inputs and participants

Execute FlagQuantum integrated main
`2ea5b78b6449b837db151e0d29c90895f6b51a4e`, with the clean unchanged
SDKBench generator/adapters/evaluator at
`9aa9260920984e2ca05bcb293abb885fccd9c144`. The driver may live on a
separate published study branch, but its entire `flagquantum` tree must equal
that runtime pin. Require actual imported source paths, exact freeze checkout,
runner/protocol hashes and dependency versions to match before execution.
Use installed Qiskit 2.6.2 and Cirq 1.7.0; neither is constrained to disable a
competitor feature.

Freeze this protocol and `benchmarks/compiler_application_gaps.py`, publish the
exact commit, then draw exactly one `secrets.token_hex(32)` nonce into a new
exclusive-create manifest. Preserve that manifest before generating any selected
input. Record the published study commit, integrated runtime pin, harness pin,
versions, configurations, unique declared compiler seeds, protocol/runner hashes
and nonce. No redraw, source change, participant option change, input reduction,
favorable rerun or post-draw retuning is permitted.

Generate all SDKBench application recipes: widths 6/8/12; QAOA, Ising, hopping,
QFT and repetition; line/ring/grid/sparse targets: 60 source/target inputs.
Use all five distinct seeds from the existing execution profile, in their
declared first-occurrence order: 11/23/37/51/73. The formal profile repeats those
seeds across ten process blocks; this diagnostic performs one call per distinct
seed and does not claim independent process replication.

For every input/seed, execute these existing configurations unchanged:

- `flagquantum-ordinary-restored`;
- `flagquantum-target-artifact`;
- `qiskit-transpile-restored`;
- `cirq-route-cz-restored`.

This gives 1,200 attempted observations. Rotate the complete workflow order
between seed groups. Every adapter receives a fresh copy of the same declared
fixture; mutation is a failure and cannot affect a subsequent participant.
Do not patch SDKs, substitute algorithms or select the best output for a report.
FlagQuantum/Cirq currently do not support the declared compiler seed; retain
all outcomes and identify repeated normalized outputs rather than counting
those as independent stochastic searches.

## Acceptance and evidence

Use the unchanged routing evaluator with its existing per-compile time cap,
20,000-operation cap, legal target edges, normalization and `1e-8` semantic
threshold. Require identity delivered final placement for every workflow.
Width six compares complete operators up to one global phase; widths eight and
twelve use the unchanged three probes, which do not establish universal
operator equivalence. Preserve every failure and unsupported outcome, including
its exception and any returned artifact, rather than dropping the input.

Record the complete fixture digest and the digest of operations/target edges
separately. Compare the latter with inspected nonce `01` repeated 32 times.
QFT line/ring/grid programs can repeat despite new probes; count actual changed
source/target inputs and repeated controls explicitly. Once observed, all newly
drawn inputs are inspected development cases and cannot be reused as unseen
qualification.

Retain every row, adapter audit, numerical residual, CZ count, two-body dependency
depth, final layout and normalized output digest. Deduplicate full normalized
payloads by SHA-256, retaining their canonical JSON strings in `payloads` for
independent replay. Each row supplies its original compile-wall value so the
complete evaluator payload can be reconstructed without executing the compiler.
Do not summarize a failed or incomplete attempt as a completed study.

Report all five-seed outcomes for each input. For each FlagQuantum workflow
versus each other participant workflow, report CZ and dependency depth separately,
plus dominance/tradeoff counts; do not invent a weighted global score or global
SDK ranking. Seeds without native support are repeated controls. The two
FlagQuantum workflows have different native bases and artifact responsibilities;
keep their conclusions separate. Target-artifact compilation includes validation,
lineage, artifact construction and serialization. The ordinary workflow does not
invoke its target-CZ block stage. External restored workflows use SDKBench's
common token restoration; the target artifact uses its own native restoration.
These are delivered workflows, not identical internal algorithm boundaries.

Select the next inspected correctness/quality witness by the greatest positive
FlagQuantum CZ excess against Qiskit among accepted pairs on actually changed
source/target inputs. Break ties by greatest depth excess, then case/workflow/seed
in ascending lexical order. Report all tied and opposite-direction cases in the
complete data. This is a diagnostic selection, not evidence that the witness
represents the population or that a future fix generalizes.

## Resources, failure and limits

Run locally with one numerical thread and no simultaneous local test suite.
Keep the adapter's own compile deadline and original evaluator caps unchanged.
A separate owned process group has a 900-second outer budget, with TERM/KILL
cleanup and retained supervision; never install an outer signal alarm over
SDKBench's stage timer. Persist observations after every completed seed group.
Do not restart any machine or service or change a kernel.

The shared macOS host is not reserved, lacks the formal Linux resource executor,
and cannot establish aggregate cgroup peak memory, exclusive physical cores or
qualified latency. Record compile-wall values only as original acceptance
inputs; do not publish timing ratios, process-memory comparisons, statistical
significance, hardware behavior, scalability or a global rank. A completed
quality diagnostic is not formal SDKBench publication eligibility.

The existing user workflow supported by this diagnosis is:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(6).h(0).cx(0, 5).ry(2, theta=0.2).cx(2, 4)
compiled = compile(source, coupling_map=CouplingMap.line(6))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

Expected: legal equivalent compilation with restored output placement. The study
asks where delivered application cost remains higher, using unchanged independent
acceptance, before choosing the next production implementation.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.compiler_application_gaps \
  --sdkbench-root /path/to/frozen/sdkbench \
  --manifest /tmp/application-gap-freeze.json \
  --output /tmp/application-gaps.json
```
