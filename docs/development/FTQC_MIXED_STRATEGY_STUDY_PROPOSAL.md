# Bounded Clifford / Non-Clifford Strategy Study

Status: Proposed experiment design, FTS-10A. No algorithm selection, new engine,
approximation policy, public contract or measurement campaign is activated.

Baseline: main `773b55387c735f86afe3caa8d7bcaf64285e1f0f`. The
[master plan](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md#14-immediate-next-steps-and-decisions-required)
requests early FTS-10 experiment design to expose algorithm risks. FTS-02/09
and the complete FTS-10 selection remain open; this preparatory slice does not
waive their dependencies. The accepted five-qubit feedback conformance in #837
qualifies only its declared pure-X, two-round domain, not adaptive non-Clifford
execution.

## Question and deliverable

Which existing representation is an appropriate baseline for static,
Clifford-dominated circuits with a bounded number of T gates, and what would
have to be added before a coherent Clifford decomposition could be a credible
competitor? Freeze the experiment before collecting comparative timings.

The next approved measurement PR would reuse public `fq.Circuit`, `fq.plan`,
`fq.run`, existing execution options and the benchmark registry/writer. It would
report correctness, supported/refused requests, phase times and observed memory
without selecting a winner by assumption. No production kernel, planner default,
fourth numerical core, FTQC package or approximation is introduced here.

## Candidates and existing boundaries

| Candidate | Evidence and reuse | Required restriction or unresolved question |
| --- | --- | --- |
| Existing statevector | Runtime mode `statevector`; small dense reference and full probability output already supported. | It is a baseline, not evidence of a new capacity breakthrough. State storage grows with Hilbert-space size. |
| Existing MPS | Runtime mode `mps`; existing `MPSConfig`, truncation records and numerical implementation. | Exact study requires no discarded information: inspect the actual plan/configuration, disable approximation and fallback, and record bond dimensions. Never label a truncated run exact. |
| Existing tensor network | Runtime mode `tensor_network`; existing contraction planning and executor. | Record contraction policy and planning/workspace cost. A two-qubit success does not establish every future topology, output or precision combination. |
| Coherent Clifford decomposition | Research candidate only; existing stabilizer sampling and Clifford fault propagation are reusable foundations, not this algorithm. | Demonstrate branch phase and overlap semantics before any production implementation. An incoherent mixture of Clifford sample distributions is not equivalent to a coherent sum. |

The stabilizer engine explicitly supports a Clifford subset and refuses gates
outside it. Its declared-DEM sampling, measurement sampler and Pauli-response
primitive must not be relabeled a T-gate state-evolution engine. The independent
Ising sampled-charge and AI decoder work on this main revision serves different
domains and is not a general Clifford+T strategy result.

### Research basis, not imported performance claims

[Bravyi and Gosset](https://arxiv.org/abs/1601.07601) motivate treating
non-Clifford resources separately from total circuit size. This proposal does
not import their implementation, approximation guarantee or performance numbers.
[Vidal](https://arxiv.org/abs/quant-ph/0301063) motivates examining entanglement
as a simulation resource, rather than predicting MPS cost from T count alone.
[Markov and Shi](https://arxiv.org/abs/quant-ph/0511069) motivate recording circuit
structure and contraction complexity for tensor-network candidates. These
primary papers justify candidate dimensions; none certifies FlagQuantum's
current implementation or the proposed workloads.

For example, algebraically `T = a I + b S`, with
`b = (exp(i*pi/4)-1)/(i-1)` and `a = 1-b`. Expanding several T gates introduces
complex branch coefficients. The output probability involves interference
between branches, not only individual branch probabilities. A feasibility
review must specify relative phase, overlap evaluation, cancellation, precision
and resource exhaustion before choosing a representation or public surface.
Do not implement a dense sum of branches and present it as a new large-scale
stabilizer algorithm. Approximate decompositions remain outside admission.

## Proposed finite CPU matrix

The following is a request for a later bounded measurement stage, not work
already executed or resource approval inherited from Stage A.

| Dimension | Proposed values |
| --- | --- |
| Physical qubits | n=4 and n=8. |
| T count | t=0,2,4,8, with the literal authored count retained alongside any optimized count. |
| Entangling pattern | Open chain and ring, defined below. |
| Executable routes | Existing statevector, MPS and tensor-network modes, explicitly selected. |
| Precision and outputs | CPU complex128; full ordered n-wire probabilities, no samples, gradients, channels, reset or feedback. |
| Number of points | 16 circuit descriptions, 48 route/workload points. A refused route remains a point, not an omitted observation. |

Construct each description deterministically as follows. Start at zero and apply
H on each wire in ascending order. For each of four layers l=0,1,2,3, apply
S to wire `(l+1) mod n`, then CX edges in ascending control order: `(i,i+1)`
for i=0..n-2, followed by `(n-1,0)` only for the ring. In that layer apply T
to wire l mod n when `l < t`, and additionally to wire `(l+1) mod n` when
`l+4 < t`. Finally apply H on every wire and request probabilities in wire
order 0..n-1. This defines exactly t authored T instructions for all four t
values and keeps Clifford layers fixed. Freeze circuit/IR hashes, gate ordering
and output order before timings; preserve any phase-relevant compiler rewrite.

The small n values deliberately bound the first comparison and permit independent
references. They are not the W4 n=32/64 target, a logical protocol, a magic-state
cost estimate or evidence of a simulation-capacity improvement.

## Correctness and refusal protocol proposed for approval

Before timing a route, compare with an independently authored small dense
reference that does not call the numerical gate/evolution implementation under
test. Reuse existing reference infrastructure where it already meets that
independence boundary; do not create a public state engine. Include the analytic
two-qubit interference example below and the scalar T/I/S identity as anchors.

Proposed complex128 comparisons are absolute probability error, normalization
residual and normalized density-matrix error <=1e-12, with relative tolerance
zero. This is a **new study-domain proposal**, not automatically approved by the
same number in the earlier five-qubit conformance contract. A global phase is
irrelevant to density matrices; relative phase and interference must remain
observable. Hold actual dtype, outputs, circuit semantics and noise-free scope
equal across routes. No statistical interval or shot-frequency tolerance is
needed for these full-probability requests.

Require `allow_approximate=False` and `allow_backend_fallback=False`. Inspect
the actual plan/result and MPS truncation metadata/configuration rather than
inferring exactness from a mode name. If an existing route cannot execute the
requested exact profile, retain its diagnostic and mark it refused. Do not
silently add a bond cutoff, use another backend, change requested outputs or
relax an error target. Any newly exposed defect is reported before a production
change; an API/serialization/approximation change needs its own approval.

## Timing, memory and proposed resource admission

Reuse the reviewed local RSS watchdog, process ownership and existing benchmark
metadata/atomic writer. This proposal does not change those tools or extend
their hardware qualification.

| Resource or procedure | Proposed first-study limit |
| --- | --- |
| Repetitions | One retained host warm-up and three fresh measured processes per route/workload point; at most 192 processes. No concurrent measured processes. |
| Per process / whole campaign | 120 seconds per process, 45 minutes total including startup/controller work. |
| Workload process-group RSS | 2 GiB sampled stop threshold with actual gaps <=100 ms, using the admitted watchdog. Sampling can overshoot; this is not an allocator/cgroup limit. |
| Storage and free space | Retained evidence <=128 MiB; require >=512 MiB free before starting. |
| Failure handling | Stop the campaign on the first correctness, budget or monitoring failure; preserve failed and unattempted cases. No automatic retry or fastest-subset selection. A capability refusal is recorded explicitly. |

Each process constructs the circuit, plans once and executes that plan once;
there are no inner warm-ups or repeated timed executions. Reused-plan timing is
deferred to a separately declared study, so it cannot silently change this
192-process ceiling. Separate construction, planning and first execution.
Record actual end-to-end process wall time including
imports and cleanup separately. Keep warm-ups, fresh-process timings, cache
state, contraction planning and error-reduction/report-I/O boundaries explicit;
do not compare one route's warm cache with another's cold startup.

Retain raw repeats and report per-point summaries without pooling unlike
topologies or n/t values. Three repetitions are a small pilot, not a reliable
tail-latency distribution. Supervisor sampled group RSS, producer high-water RSS,
workspace estimates, bond dimensions and output bytes remain separate measures;
shared pages can be counted more than once in group RSS. No exclusive-memory,
n=64 capacity, speedup, 1.5x-throughput or 2x-capacity claim follows from this
pilot. The user's deferred performance targets remain disabled.

Explicit approval must cover this exact workload matrix, comparison values,
repetitions and resource limits before a new measurement producer or collection
is started. The previous Surface/shot Stage A approval does not cover this study.

## Decision and subsequent PR sequence

1. Review this FTS-10A design and the existing route/refusal boundaries. Keep
   FTS-02/09 gaps explicit; do not claim final FTS-10 method selection.
2. After explicit bounded-study approval, add a narrow observer/reference slice
   to existing benchmark infrastructure and review it before collection. Its PR
   must include an actual public-user workflow and expected output.
3. Collect the approved matrix once, preserving failures/refusals and unattempted
   points. Report correctness and measured resource tradeoffs without changing
   a runtime default or promoting a winner by one small benchmark.
4. Review whether larger, code-derived workloads or coherent decomposition are
   justified. Any new representation, phase/overlap contract, approximation,
   device scope or larger budget is a separate decision before implementation.

## Runnable current user example

Requirements: supported Python, core FlagQuantum and CPU. No optional quantum
SDK, credentials, physical device or dataset is required for this small example.
Save as `/tmp/ftqc_interference_example.py`. From an installed package run
`python /tmp/ftqc_interference_example.py`; from a source checkout run
`PYTHONPATH=. python /tmp/ftqc_interference_example.py`.

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

The two equal large probabilities are `(2+sqrt(2))/8`, and the two smaller ones
are `(2-sqrt(2))/8`. T's relative phase becomes observable after the final H
gates. Formatting this tiny example to eight decimals is a runnable sanity
check, not acceptance of the new study's tolerance or a performance comparison.
This example is not one of the proposed 48 study points and initiates no timing
campaign.

## Requirement checklist and current blockers

- [x] R04: identify existing routes, a decomposition feasibility gap, independent
  reference needs and phase-sensitive current user code. No new algorithm is
  implemented or selected.
- [ ] Approve the finite CPU matrix, accuracy comparisons and resource procedure
  before implementing/collecting the proposed study.
- [ ] R03/R04: independently validate and measure the admitted comparison, then
  review method selection on representative code-derived workloads.
- [ ] R07/R09/R10: validate actual complex protocols and capacity/performance
  benefits under separately approved equivalent workloads and budgets.
- [ ] R05/R06: adaptive non-Clifford frames/branches require another concrete
  contract; #837's pure-X profile does not authorize them.
- [ ] R11/R12/R15: domestic device and experimental-data ownership/access remain
  external decisions. This document contains no device or experimental result.
