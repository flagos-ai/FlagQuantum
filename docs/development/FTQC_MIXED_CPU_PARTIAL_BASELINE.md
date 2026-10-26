# FTS-10A: Partial Exact CPU Mixed-Circuit Observations

Status: the admitted campaign stopped at its first monitoring failure. This is
partial measurement evidence, not a completed matrix, algorithm selection,
performance improvement or high-performance FTQC qualification. No retry ran.

## Admission and fixed producer

The user explicitly approved the [finite study](FTQC_MIXED_STRATEGY_STUDY_PROPOSAL.md)
on 2026-10-25, with observer review before collection. Observer #848 was reviewed,
merged and accepted on actual main `5cd642a69a0142b7c7e42013851c38dee701a3d5`:
[complete CI, all 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40163173134),
[source/single-GPU/required two-GPU validation](https://github.com/flagos-ai/FlagQuantum/actions/runs/40163175164)
and [main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40162583184)
passed. Three original optional pools were unselected. Package job 121078838495
passed real QEC optional installation and evidence retention. A fresh isolated
sdist/wheel preserves all 832 package Python members exactly; Python 3.12.14 /
Torch 2.14.1 core installation, dependencies, native Bell/master/HTH/CLI and
installed three-route interference plus observer examples passed outside the
checkout without PYTHONPATH or SDK overlays. Focused source tests passed 127/127,
no skips, using the separately declared development Torch 2.13/SDK overlay.
That source test environment is not the study environment or a fresh profile.

The study used the accepted installed wheel, Apple M5 CPU (read-only host sysctl
query), Python 3.12.14, Torch 2.14.1, one CPU thread, no PYTHONPATH/optional SDK
layer, and the reviewed watchdog from the same main revision. Source, wheel,
controller and watchdog identities are retained. Latest report base main
`87751dbe694269d8801d0c9af11c96e749357207` adds independent QEC parity, Quafu SDK
and documentation work; those changes were audited, are not this study's
producer and are not credited here. Existing observer and watchdog are unchanged.

The local controller passed eight fake-process boundary tests before launch:
fixed matrix/roles, low-disk refusal, first-failure stop and unattempted ledger,
explicit refusals, whole-time reservation, no output replacement, RSS-gap
refusal and cleanup on external timeout. These tests did not execute study
circuits. Developer smoke runs from #848 are not pooled with this campaign.

## Complete outcome ledger, including the missing evidence

The immutable [public raw records and summaries](evidence/ftqc_mixed_cpu_partial_20261025.json)
retain all 192 planned process slots across 48 points, original-file digests,
controller/test source with local paths redacted, full plans and native metadata.
The index links three digest-bound record parts (64 slots each), each below the
unchanged repository single-file limit. Reassembling their `records` lists
reproduces the original JSON object exactly; no observation or numerical value
is removed. There are **164 completed processes**: 41 host warm-ups and 123 measured fresh
processes, covering 41 complete points. The next host warm-up failed; **27 process
slots remain not attempted**, covering that point's three measurements and the
six n=8/t=8 route/topology points. No capability refusal or automatic retry ran.
The n=4 submatrix is complete; the n=8 submatrix is partial.

The campaign elapsed 225.8824166249833 seconds and retained 4,236,736 raw bytes.
These are observations, not speed or storage guarantees. The completed cases
passed the specifically approved probability/normalization/normalized-density
comparisons: maximum recorded absolute error 4.218847493575595e-15 against
atol=1e-12, rtol=0. Their largest sampled group RSS was 237,240,320 bytes and
largest successful sample gap 0.05218137500924058 seconds. These values do not
qualify the failed or unattempted cases.

### First failure: monitoring admission, not a numerical result

`n8-t4-ring-tensor_network-0` stopped before a producer JSON existed. Its watchdog
reported `reason="rss_monitor_unavailable"`: `ps -axo pgid=,rss=` exceeded the
unchanged 0.05-second query timeout. The watchdog terminated its owned command
with SIGTERM (`command_returncode=-15`, watchdog exit 124). Its diagnostics record
120 positive RSS observations, a last observed group maximum 219,316,224 bytes,
and 0.004393542010802776 seconds of termination work. The last successful sample
gap does not describe the interval after monitoring was lost. Root cause of the
slow host query is not established; no environment-repair claim follows.

The controller then encountered `FileNotFoundError` when it tried to read the
absent producer JSON. That is a secondary diagnostic-order issue in the local
controller: it preserved the failed status and stopped, but its top-level message
obscures the watchdog cause. The raw manifest and watchdog diagnostic are both
retained, not rewritten. The failure is not evidence of TN numerical accuracy,
capability refusal, OOM, 2-GiB capacity exhaustion or a computational speed limit.

No threshold, 50-ms query timeout, 100-ms maximum sampling gap, checker, fixture,
engine, skip or workflow was changed. No failed point or missing measurement was
rerun or replaced. A continuation needs a separate reviewable monitoring and
resource admission; this report does not authorize automatic resumption or
relaxing the policy. The original campaign directory remains immutable.

## Per-point measured medians and explicit gaps

Each complete point has one retained host warm-up and three fresh measured
processes. Only those three measurements form the median. Times below are
seconds. `phase` is the sum of construction/planning/first execution intervals;
`external` spans watchdog launch through completion and includes startup and
cleanup. Peak is the largest sampled process-group RSS across the four processes,
in MiB. Missing points are listed, not pooled, imputed or omitted.

| n | T | Pattern | Mode | Status | Plan | Execute | Phase | External | Peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0 | chain | statevector | completed | 0.080589 | 0.025332 | 0.114821 | 0.685449 | 214.359 |
| 4 | 0 | chain | mps | completed | 0.082703 | 0.025331 | 0.117387 | 0.714327 | 214.031 |
| 4 | 0 | chain | tensor_network | completed | 0.084699 | 0.153179 | 0.246929 | 0.968253 | 217.859 |
| 4 | 0 | ring | statevector | completed | 0.083432 | 0.026324 | 0.118743 | 0.716972 | 214.172 |
| 4 | 0 | ring | mps | completed | 0.084215 | 0.028031 | 0.118505 | 0.714222 | 214.203 |
| 4 | 0 | ring | tensor_network | completed | 0.083236 | 0.195406 | 0.287342 | 1.098482 | 217.875 |
| 4 | 2 | chain | statevector | completed | 0.080850 | 0.025331 | 0.115864 | 0.677590 | 213.984 |
| 4 | 2 | chain | mps | completed | 0.081760 | 0.024901 | 0.115513 | 0.684429 | 214.188 |
| 4 | 2 | chain | tensor_network | completed | 0.081499 | 0.170888 | 0.261858 | 1.034594 | 217.609 |
| 4 | 2 | ring | statevector | completed | 0.081312 | 0.026028 | 0.116146 | 0.677870 | 214.250 |
| 4 | 2 | ring | mps | completed | 0.085197 | 0.026233 | 0.120088 | 0.673447 | 214.297 |
| 4 | 2 | ring | tensor_network | completed | 0.082388 | 0.226847 | 0.317306 | 1.191860 | 217.766 |
| 4 | 4 | chain | statevector | completed | 0.084313 | 0.025732 | 0.119173 | 0.676318 | 214.016 |
| 4 | 4 | chain | mps | completed | 0.084653 | 0.026326 | 0.120456 | 0.722394 | 214.359 |
| 4 | 4 | chain | tensor_network | completed | 0.079662 | 0.191569 | 0.280071 | 1.036710 | 216.672 |
| 4 | 4 | ring | statevector | completed | 0.084035 | 0.026157 | 0.119496 | 0.696601 | 214.141 |
| 4 | 4 | ring | mps | completed | 0.083399 | 0.025414 | 0.118030 | 0.682363 | 214.281 |
| 4 | 4 | ring | tensor_network | completed | 0.081054 | 0.242560 | 0.332634 | 1.255386 | 217.328 |
| 4 | 8 | chain | statevector | completed | 0.083113 | 0.025839 | 0.117897 | 0.676526 | 213.422 |
| 4 | 8 | chain | mps | completed | 0.082426 | 0.025597 | 0.116231 | 0.712907 | 214.172 |
| 4 | 8 | chain | tensor_network | completed | 0.082111 | 0.224936 | 0.316585 | 1.210798 | 217.703 |
| 4 | 8 | ring | statevector | completed | 0.083250 | 0.026068 | 0.118662 | 0.701472 | 214.141 |
| 4 | 8 | ring | mps | completed | 0.083580 | 0.027359 | 0.119714 | 0.728752 | 214.188 |
| 4 | 8 | ring | tensor_network | completed | 0.083178 | 0.294756 | 0.386828 | 1.408046 | 217.688 |
| 8 | 0 | chain | statevector | completed | 0.085605 | 0.027878 | 0.122754 | 0.738747 | 219.250 |
| 8 | 0 | chain | mps | completed | 0.084952 | 0.136620 | 0.229874 | 0.837229 | 219.547 |
| 8 | 0 | chain | tensor_network | completed | 0.085918 | 2.384460 | 2.480722 | 4.757297 | 224.094 |
| 8 | 0 | ring | statevector | completed | 0.085805 | 0.027610 | 0.123017 | 0.728570 | 219.141 |
| 8 | 0 | ring | mps | completed | 0.084730 | 0.141092 | 0.234917 | 0.845993 | 219.547 |
| 8 | 0 | ring | tensor_network | completed | 0.079376 | 2.655779 | 2.749525 | 5.278336 | 224.453 |
| 8 | 2 | chain | statevector | completed | 0.080036 | 0.025417 | 0.114334 | 0.689859 | 218.797 |
| 8 | 2 | chain | mps | completed | 0.081097 | 0.132173 | 0.221184 | 0.816862 | 219.672 |
| 8 | 2 | chain | tensor_network | completed | 0.080885 | 2.385034 | 2.473368 | 4.732538 | 224.188 |
| 8 | 2 | ring | statevector | completed | 0.080622 | 0.025455 | 0.114855 | 0.666225 | 219.125 |
| 8 | 2 | ring | mps | completed | 0.079919 | 0.131910 | 0.220385 | 0.771378 | 219.547 |
| 8 | 2 | ring | tensor_network | completed | 0.080356 | 2.757155 | 2.848197 | 5.510944 | 226.250 |
| 8 | 4 | chain | statevector | completed | 0.082738 | 0.025853 | 0.117758 | 0.691621 | 218.875 |
| 8 | 4 | chain | mps | completed | 0.080435 | 0.130272 | 0.219560 | 0.774029 | 219.469 |
| 8 | 4 | chain | tensor_network | completed | 0.081313 | 2.455338 | 2.545567 | 4.933565 | 224.094 |
| 8 | 4 | ring | statevector | completed | 0.080987 | 0.025767 | 0.115846 | 0.667838 | 218.922 |
| 8 | 4 | ring | mps | completed | 0.080745 | 0.131254 | 0.220797 | 0.796680 | 219.500 |
| 8 | 4 | ring | tensor_network | failed | — | — | — | — | — |
| 8 | 8 | chain | statevector | not_attempted | — | — | — | — | — |
| 8 | 8 | chain | mps | not_attempted | — | — | — | — | — |
| 8 | 8 | chain | tensor_network | not_attempted | — | — | — | — | — |
| 8 | 8 | ring | statevector | not_attempted | — | — | — | — | — |
| 8 | 8 | ring | mps | not_attempted | — | — | — | — | — |
| 8 | 8 | ring | tensor_network | not_attempted | — | — | — | — | — |

Complete raw precision and all repeats are in JSON; table rounding does not
replace the raw values. Three repeats are a small pilot, not a tail-latency
estimate. There is no overall ranking across unlike n/T/topology points.

## Interpretation and timing/memory boundaries

Every process used exactly one construct/plan/run on the same authored circuit
and requested full ordered probabilities with forced exact CPU complex128 and
fallback disabled. MPS zero cutoff/unbounded bond plus recorded truncation,
actual route/precision, and TN contraction/path metadata were retained. Coherent
Clifford decomposition was not implemented or measured.

The scalar reference is independent of production gates/compiler/state helpers.
Density checks observe relative phase and ignore only global phase. Initial
imports, plan/IR serialization, reference evaluation, error reduction, native
state conversion for validation and report IO are outside phase intervals;
lazy imports inside called APIs remain included. Dense validation adds two
n<=8 density arrays and may materialize an MPS/TN state. Whole-process RSS
includes imports and that validation; it is neither exclusive native workspace
nor a simulator-capacity certificate. Shared pages may be counted twice by group
RSS. Probability/density byte counts are explicit storage, not measured peaks.

Independent offline validation reconciles each completed point's original
producer/watchdog SHA-256, return codes, exact matrix/roles, actual runtime/thread
profile, resource values, all four error fields and medians. Failed diagnostics
and not-attempted entries are separately checked. Only hostname/local absolute
path metadata is redacted in the public copy; every numeric value is preserved.
Original digests refer to unredacted files. The redacted controller source is
provided for inspection, not advertised as a portable controller: adapt local
paths and complete a fresh admission before executing it elsewhere.

## Runnable current user example

Requirements: supported Python, core FlagQuantum and CPU; no optional SDK,
credentials or dataset. Save as `/tmp/ftqc_interference_example.py` and run
`python /tmp/ftqc_interference_example.py` from an installed package, or
`PYTHONPATH=. python /tmp/ftqc_interference_example.py` from a source checkout.

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1).t(0).h(0).h(1)
for mode in ("statevector", "mps", "tensor_network"):
    options = fq.ExecutionOptions(
        mode=mode, device="cpu", precision="complex128",
        allow_approximate=False, allow_backend_fallback=False,
    )
    probabilities = fq.run(
        circuit, outputs=fq.probabilities([0, 1]), options=options,
    ).probabilities
    assert probabilities is not None and probabilities.shape == (1, 4)
    print(mode, *[f"{float(p):.8f}" for p in probabilities[0]])
```

Expected stdout:

```text
statevector 0.42677670 0.07322330 0.07322330 0.42677670
mps 0.42677670 0.07322330 0.07322330 0.42677670
tensor_network 0.42677670 0.07322330 0.07322330 0.42677670
```

This two-qubit public sanity example is not a rerun of a failed study point or
one of the 48 matrix points. See the admitted observer command in the design
for a single explicit developer invocation; do not treat it as permission to
repeat this stopped campaign.

## Requirement checklist and next decisions

- [x] R03/R04: finite existing-route observer reviewed and actual main accepted.
- [x] Preserve one admitted campaign's 164 valid observations, failed monitor
  diagnostics and 27 unattempted slots without retry or threshold changes.
- [ ] Complete the 48-point study: seven points lack complete measured evidence.
- [ ] Review the controller's diagnostic order and investigate monitoring loss
  before a separate continuation proposal. Do not substitute a relaxed timeout.
- [ ] FTS-10 final method selection and representative code-derived W4 workload
  evidence remain open; FTS-02/09 are not waived by this pilot.
- [ ] R07/R09/R10 scale/performance and R11/R12/R15 domestic/experimental evidence
  remain open. No new API, approximation, FTQC package or 1.5x/2x target is enabled.

## Proposed separate continuation admission

The [seven-point continuation proposal](FTQC_MIXED_CPU_CONTINUATION_PROPOSAL.md)
states the additional 28-process budget, unchanged monitoring policy and required
diagnostic-order repair. It is not approved or executed; all original records
remain immutable. Post-859 main acceptance is recorded there.

## Separately approved continuation

The [completed bounded pilot report](FTQC_MIXED_CPU_COMPLETED_PILOT.md) adds a
separately admitted seven-point campaign after controller review/main acceptance.
This original failed campaign and its raw manifest remain unchanged. Combined
evidence has 48 complete points, 192 successful processes plus this original
monitor failure, 193 attempts total; it is not a method-selection or speedup claim.
