# PF-01 Runtime IR audit: candidate not adopted

This audit follows revision 9 sections 13.5–13.6 and
[#1023](https://github.com/flagos-ai/FlagQuantum/pull/1023), merged as
`92ad9bf6de0a644cde81423ed601d5a0475695c4` after 57 successful checks and
three configured skips. It tests whether reusing the circuit decoded during
complete plan validation materially improves the existing synthesis workflow.

**The experimental fast path is not enabled.** Final production files under
`flagquantum/` match that merged baseline exactly. The retained delivery consists
of semantic regression tests, raw measurements and this decision record.
Fewer reconstructed objects did not establish a repeatable benefit at the
revision-9 review line. No cache, public API/schema/identity change, Core IR
change or PF-02/Rust migration is introduced.

## Existing user workflow

Users continue writing the same composable quantum function:

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x*x+y


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


program = synth.synthesize(main, memory_limit_bytes=64 * 1024**2)
result = fq.run(program)
outcomes = dict(program.probabilities(result, registers=("x", "y", "result")))
assert outcomes.keys() == {(x, y, x*x+y) for x in range(4) for y in range(4)}
assert all(abs(p - 1/16) < 1e-12 for p in outcomes.values())
```

The matching Qmod and independent physical Circuit are retained in the
[first Services delivery](SYNTH_PYTHON_PERFORMANCE_PF01_DELIVERY.md#existing-user-workflow)
and the PR description. The native/Qmod circuit remains 16 qubits, eight
workspace qubits, 81 operations and depth 59. The two-bit hand derivation remains
nine qubits, one workspace, 15 operations and depth 11. This audit does not
close that resource gap or ship a new latency claim.

## Experiment and contract

`plan_program` currently obtains a fully validated canonical payload and then
deserializes its program. Validation already deserializes that program to check
the IR and fingerprint. Experimental commit
`c44d7a7807f553786a53308e3d8ebbf2ebbe5b95` retained the locally decoded IR,
normalized payload and canonical text in one private Runtime helper.

All schema/options/environment/decision/extension/compiler/identity checks and
canonical JSON validation were retained on every call. The fast path applied
only when the original internal text equaled the validated canonical text.
Noncanonical internal JSON kept the original normalized reconstruction path,
preserving metadata key order. Native-plan transport remained separate from
CircuitIR reading. There was no persistent or object-ID cache.

The experiment passed initial transport, execution, mutation-isolation and
error-diagnostic checks. Reversion commit
`d40336f50ffd3f099ffe09ed4638bb7c1a450c49` then restored production Runtime
code while retaining those useful semantic tests. The experimental source
remains reachable for reproduction; it is not a second production default.

The tests protect canonical/noncanonical JSON, complete complex metadata and
Tensor transport, fresh ownership after callers modify a returned circuit,
Bell-state execution, unchanged validation errors for six corrupted fields,
and native-plan transport/refusal separation. Existing archive and native
version tests accompany these cases. Tests assert behavior rather than requiring
a particular reconstruction count in ordinary CI.

## Measurement and independent confirmation

Before: `92ad9bf6de0a644cde81423ed601d5a0475695c4`.
Experimental after: `c44d7a7807f553786a53308e3d8ebbf2ebbe5b95`.
Both primary cohorts use clean source, the unchanged PF-00 driver and the same
ten configurations as the
[preceding Services comparison](SYNTH_PYTHON_DECISION_FIELDS_PF01_DELIVERY.md#reproducible-comparison).
They cover three logical entrances, native large energy/shared-square, constant
product, native basis/line, Qmod line and a native budget refusal.

Each side uses five fresh timing processes per configuration, two identical
warmups, three separate unprofiled RSS processes and one intrusive diagnostic
process, plus five cold starts. Each primary side records 81 verified samples
and nine expected refusals, with no omitted failure/timeout. E0 includes checked
synthesis; E1 includes the first run and named readout for bounded small states.
Large-state exclusions and independent bounded XOR/cleanup interpretation remain
explicit. All circuit hashes, resources and refusal codes match.

| Observation | Before / experimental median | Review result |
| --- | --- | --- |
| Large native energy E0 | 572.580 / 557.750 ms | 2.59%; below line |
| Large shared-square E0 | 989.847 / 936.575 ms | 5.38%; below line |
| Native basis E1, original cohort | 344.682 / 305.904 ms | 11.25%, 38.78 ms; qualifies once |
| Native basis E1, independent confirmation | 329.041 / 307.520 ms | 6.54%, 21.52 ms; below line |

No primary E0 group reaches 10% and 20 ms. Only the basis E1 group initially
does. A separately recorded pair of fresh-process cohorts repeated that exact
request, using five timing samples, two warmups and three separate RSS samples
per source. Its reduction falls below 10%. The confirmation does not replace or
average away the original favorable result. The evidence supports a small
improvement, not a stable improvement at the agreed review line.

The primary basis RSS median increases by 30.73 MiB/17.38%, an unfavorable
observation retained in the raw archive. Independent confirmation has sampled
RSS medians 207.781 / 207.750 MiB. Neither direction establishes a memory
benefit. Lifetime high-water, compiler-window sampled RSS and planned state bytes
remain separate. The nonexclusive Apple M5/Python 3.12.14/Torch 2.13.0 host has
unknown frequency/thermal state; no precise allocation-peak, hardware, release
or cross-host claim follows.

Diagnostic counts explain why fewer objects alone are insufficient. Complete
identity-validation calls remain intact, while ordinary canonical plan readers
avoid their second IR reconstruction. The cost of target transforms, repeated
canonical serialization and other validation remains. Intrusive diagnostic
times include observer overhead and are not natural stage-time percentages.

## Reproduction and decision

The hash-bound
[`pf01-runtime-manifest.json`](../../benchmarks/synthesis/baselines/pf01-runtime-manifest.json)
lists all original and confirmation samples, comparisons and the experimental
and reversion commits. No original AO/PF-00/#1021/#1023 data or oracle is changed.
Fetch the exact commits before validating in a shallow checkout.

```bash
python -m benchmarks.synthesis.performance_compare \
  --before benchmarks/synthesis/baselines/pf01-runtime-before.json.gz \
  --after benchmarks/synthesis/baselines/pf01-runtime-after.json.gz \
  --output /tmp/runtime-primary.json
python -m benchmarks.synthesis.performance_compare \
  --before benchmarks/synthesis/baselines/pf01-runtime-confirm-before.json.gz \
  --after benchmarks/synthesis/baselines/pf01-runtime-confirm-after.json.gz \
  --output /tmp/runtime-confirmation.json
```

For fresh measurements, use the preceding comparison's exact ten-case command
at each source pin. For confirmation, select only
`energy-2x2:qfunc:inline:basis:success`, keeping five samples, two warmups and
three memory trials. GC, thread limits, budgets and oracles stay enabled.

Decision: retain existing production materialization. The new semantic tests
are useful independently of the rejected fast path. Remaining Core serialization
work has its own protected ownership/contract boundary; a Rust rewrite or PF-02
cache is not justified by these constructor counts. AO-03's bounded Boolean
network/resource work remains independent and should not wait on this audit.
The nine-qubit/one-workspace/24-operation/depth-20 targets are unchanged.

Final test/quality and broad local default results are recorded separately in
the PR and delivery ledger, together with exact-head remote CI. Scope acceptance
is not a green full-suite or release claim.
