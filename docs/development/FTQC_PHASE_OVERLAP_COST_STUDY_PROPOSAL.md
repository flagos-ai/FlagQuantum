# Bounded Phase and Overlap Cost Study

Status: the finite study and observer implementation were specifically approved
on 2026-10-26 after proposal #931 passed full CI and required GPUs. Actual main
`354676710840672737ed620bee0501c1a93ebaf6` then passed all 54 CI jobs, required
source/single/two-GPU checks, smoke, real QEC optional installation/evidence,
fresh isolated 847-member package/core and installed public examples.
Observer/controller implementation is now under separate review; no new probe
or collection has run. Implementation review and actual-main admission must
complete before the one admitted collection can start. The old 193 attempts
remain immutable and do not supply this new allowance.

Baseline: main `4ec20e1adf4c1b92ef878c49d42977f25ace3318`. Reuse the
[accepted M-P representation](FTQC_PHASE_OVERLAP_FEASIBILITY_PROPOSAL.md#approved-implementation-rules-and-ownership),
[coherent reference descriptions](FTQC_COHERENT_BRANCH_CONFORMANCE_PROPOSAL.md),
[completed pilot](FTQC_MIXED_CPU_COMPLETED_PILOT.md) and
[master-plan FTS-10 requirements](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md#fts-10c-phaseoverlap-feasibility-decision).
Domain numerical ownership and existing public execution contracts remain
authoritative; this study uses the single-device CPU fast path.

## Decision to inform

The finite M-P prototype supplies canonical phase and complex-overlap
correctness on ten descriptions, not measured algorithm cost. This study asks
what the complete current small-domain full-probability workflow costs for the
private candidate and the existing statevector, MPS and tensor-network routes.
It would inform whether to propose a larger-domain feasibility stage. It would
not select a general mixed-circuit algorithm or close FTS-02/09/10, FTS-12/14,
R04 or M2. The target W4 scale and code-derived tasks remain outside this slice.

Preserve the old n=4/8 pilot and continuation as historical evidence with their
original producer versions, monitor failure, unattempted slots and missing
continuation outer-wall field. Do not rerun or relabel those points, splice old
fast timings into a new comparison, invoke an old collector or overwrite either
raw directory. This proposal needs a distinct producer/controller and directory.

## Fixed workload and route matrix

Reuse exactly the ten A-D static descriptions qualified by M-P: the Cartesian
product n in {2,3}, authored T count in {0,1,2,4}, plus the one-qubit HTH and
two-qubit Bell-T-HH anchors. Preserve literal gate/wire order, coefficients,
wire-zero-most-significant bit order and description/IR hashes. No ring, t=8,
n>=4, extra gate, noise, gradient, reset, measurement trajectory or dynamic record
is included. The private prototype retains its one-to-three-wire and 22-word
bound. A wider prototype requires another proposal, not removal of its guards.

There are 40 route/workload points: ten descriptions times four routes:

| Route | Execution and required output |
| --- | --- |
| Private M-P candidate | Expand literal T into the existing aI+bS terms, construct CH branch and ordered basis states, evaluate the within-case Gram and basis-bra/branch-ket overlaps, check coherent norm and reconstruct full ordered probabilities. No public mode or planner route. |
| Statevector | Current public fq.plan/fq.run, explicitly selected statevector mode. |
| MPS | Same public entries, explicitly selected MPS mode; inspect actual truncation/configuration evidence. |
| Tensor network | Same public entries, explicitly selected tensor-network mode; retain contraction/planning policy evidence. |

All routes request the same noise-free full ordered computational probabilities,
CPU complex128 evolution and float64 probabilities, with approximation and
backend fallback disabled. Proposed probability/normalization and private
complex-overlap comparisons use atol=1e-12 and rtol=0. This new observation
boundary was separately approved even though its numbers equal M-P. No external
SDK or optional overlay is used by the measured worker.

Reuse the existing independent direct-T scalar reference for the eight matrix
cases and the two analytic anchors. The observer's literal builder and reference
adapter must be checked against the preserved A-D fixtures before collection;
the worker must not import test/pytest packages or production gate evolution as
its sole oracle. Structural review can compare frozen literal words, coefficient
identities and reference source before collection without executing a new
workload. Every new numerical qualification run belongs to the 160-start budget;
there are no hidden trial, preflight or observer self-test executions. Controller
tests may use synthetic payloads and mocked clocks/process readers; any proposed
additional live numerical run or monitor probe needs another explicit budget. Do not move or rewrite frozen fixtures. Require finite dtype,
shape, probability norm and independent probability agreement without repairing
bad values by normalization. Candidate complex phases remain private; this does
not add a public canonical-phase guarantee.

## One execution, explicit phase boundaries

The proposed observer is a finite internal research tool, not a new registered
benchmark, public API, result schema or production execution entry. It may import
the specifically approved private Simulation prototype only inside its research
worker. The ordinary installed package keeps the prototype disconnected from
production dispatch. Review the observer and controller before any collection.

Freeze one accepted observer-main revision and its built package, dependency
profile, wheel/source hashes and supervisor identity for the whole matrix.
Invoke the worker outside the checkout with no PYTHONPATH/SDK overlay; assert
that numerical modules come from that installed package. Keep tool/reference
provenance explicit and do not change the package midway through the campaign.

Each fresh worker constructs one literal circuit and produces one output:

- Record process import/setup separately, then description/IR construction.
- Candidate phases are expansion and CH state construction including updates,
  within-case Gram overlaps, basis overlaps, and coherent reconstruction/norm.
  Existing from_word combines allocation and updates; report that combined
  boundary rather than inventing an isolated gate-update timing.
- Public phases are circuit construction, one plan and one execution of that
  plan. Do not time fq.run after a separate plan while letting it plan again.
- Record validation and report I/O separately. Independent reference work is
  outside the numerical phase timers and is still included in process wall time.
- The controller independently retains workload process-group outer wall through
  exit/cleanup for every attempt, including failure. Record worker high-water
  memory and sampled group RSS separately, with their scopes and units.

The candidate includes its explicit Gram/norm work; do not hide it when
reporting that implementation's total. Equal requested outputs do not imply
identical internal work. Report phase definitions beside the numbers so later
optimizations cannot silently drop work from one route's total. No kernel-only
ratio is an end-to-end speedup. Do not clear shared machine caches or change
clocks, CPU quotas, runner labels or MAX_JOBS=2 for this study.

## Proposed repetition and execution budget

Use one retained host warm-up followed by three fresh measured processes per
point. Each worker plans/executes once and has no inner warm-up or timing loop.
Retain every raw repeat. Fresh processes do not imply cold filesystem caches;
record that distinction. Predeclare four passes: warm-up, then repeats 1-3. In
each pass visit descriptions in the existing A-D/anchor order. Start from route
order statevector/MPS/tensor-network/candidate and rotate it by
(case_index + pass_index) modulo four. Do not randomize or search for a seed.

| Budget | Proposed maximum |
| --- | --- |
| Workload processes | 40 points x 4 retained processes = 160 starts. |
| Monitoring admission | One new one-second owned idle probe, no numerical workload. Its process start is retained and consumes a separate attempt. |
| Total new process attempts | At most 161, including a failed probe or worker. No retry or automatic continuation. |
| Candidate construction | At most 24 retained states in any worker; 416 constructions across four complete candidate passes. |
| Candidate queries | At most 384 successful overlaps in any worker; 3400 across four complete candidate passes. |
| Existing public executions | At most 120 total, one per public-route worker. Candidate workers use scalar/analytic references and make no extra public reference call. |
| Parallelism | One worker at a time, one numerical CPU thread, restored where applicable. |
| Per worker / complete campaign | 30 seconds per worker; 20 minutes total including admission, startup, supervision and cleanup. |
| Workload group RSS | 2 GiB sampled stop threshold; retain actual intervals/gaps, query deadline 50 ms and maximum gap 100 ms. This is not an allocator/cgroup guarantee and may overshoot. |
| Evidence/free space | At most 128 MiB retained collection evidence; require at least 512 MiB free before starting. Package/install/cache assets are pre-existing prerequisites with separate provenance, not hidden report bytes. |

The candidate counts follow from the frozen descriptions. Branches total 50 per
pass. Separate fresh candidate workers build 54 basis states per pass (four
n=2 cases times four, four n=3 cases times eight, HTH two and Bell four), hence
104 states per pass, 416 overall. Gram and basis queries total 562+288=850 per
pass, 3400 overall. The largest n=3,t=4 point uses 16+8=24 states and
16**2+8*16=384 queries. This is a new proposed measurement allowance, not M-P's
64-state module-verification allowance silently multiplied.

Use reviewed supervision primitives under a new finite controller identity and
new destination, with no invocation or mutation of the old campaign collectors
or probes. The new probe must establish monitor availability before the first
worker. Preserve failed admission and start no workload if it fails. A failed
monitor query, excessive sample gap, timeout, RSS/storage/free-space violation,
correctness error or unexpected exception stops the campaign immediately.
Retain exit/signal, elapsed wall, last owned process identities, monitoring data
and all unattempted slots. Cleanup may target only process groups owned by that
controller. Do not relabel a failure as a refusal, normalize it away or rerun it.

A named capability refusal from unchanged public planning is retained as a
route/workload outcome, not a zero-time success. Its later repetition slots stay
explicitly unattempted for that declared reason; other points may continue.
Only that predeclared planning refusal is nonfatal. A prototype refusal on an
admitted literal word, generic exception or numerical mismatch is a failure.
No pytest skip or CI threshold changes are part of this protocol.

## Evidence and interpretation rule

The new internal evidence record must retain immutable attempt/point identity,
source/wheel/dependency hashes, circuit/IR hashes, literal/compiled T counts,
actual plan/backend/mode/dtype/outputs and exactness metadata, route-specific
work counters, all named phase timings, worker and controller outer wall, memory
scopes, monitor query deadlines/actual gaps, outcome/diagnostic and raw reports.
Use the repository's atomic evidence-writing patterns; any added internal
fields and their meanings belong to the observer review under this approval.
Do not alter public serialization or reuse an old schema field for another
measurement.

Keep warm-ups separate from fresh repeats. For each comparable point, report all
three fresh observations, their median and range. Three repeats do not support
p95/p99 estimates or a strong uncertainty claim. Do not pool different n/T/topology
points, trim outliers or select fastest repeats. A refusal or failure leaves a
comparison gap. Report cost regressions as well as any favorable observation.

The decision is limited to whether the measured small-domain prototype is
correct, bounded and worth proposing a wider feasibility stage. Any missing
correctness/resource evidence blocks promotion. Even favorable small-domain
costs do not select a production algorithm, establish a capacity benefit or
activate 1.5x/2x targets. A larger-domain study needs new representation limits,
target W4 tasks, a selection criterion and another explicit budget decision.
No hardware, experimental, approximation or general sampling qualification is
inferred.

## Runnable current public example

Requires a supported Python installation, core FlagQuantum, PyTorch and CPU;
no optional SDK or GPU. Python 3.12.14/Torch 2.14.1 is the verified example
profile. Save this code as `/tmp/ftqc_cost_example.py`.

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1).t(0).h(0).h(1)
for mode in ("statevector", "mps", "tensor_network"):
    probabilities = fq.run(
        circuit,
        outputs=fq.probabilities([0, 1]),
        options=fq.ExecutionOptions(
            mode=mode, device="cpu", precision="complex128",
            allow_approximate=False, allow_backend_fallback=False,
        ),
    ).probabilities
    assert probabilities is not None and probabilities.shape == (1, 4)
    print(mode, *[f"{float(p):.8f}" for p in probabilities[0]])
```

With an installed package, run `python /tmp/ftqc_cost_example.py`. From the source root, run `PYTHONPATH=. python /tmp/ftqc_cost_example.py`.

Expected stdout:

```text
statevector 0.42677670 0.07322330 0.07322330 0.42677670
mps 0.42677670 0.07322330 0.07322330 0.42677670
tensor_network 0.42677670 0.07322330 0.07322330 0.42677670
```

The existing example passed on accepted main `82143b09` with Python 3.12.14/Torch 2.14.1 outside the checkout with no PYTHONPATH or SDK overlay (`pr922-main-public-example.json`). Every proposal PR must execute and record this example on its own final reviewed head. This engineering smoke check is not a point in the proposed study.

## Observer implementation under review

The finite worker is `tools/observe_ftqc_phase_cost.py`; the serial controller
is `tools/collect_ftqc_phase_cost.py`. Neither adds a registry entry or public
mode. The worker imports only the admitted installed package, verifies its
retained-wheel member hashes, and reuses the private Simulation CH prototype.
The independent direct-T reference and handwritten Clifford scalar oracle are
validation work outside numerical phase timers. The four complex/probability
comparisons use the approved values without renormalizing results. Literal
words, coefficients, complex captures, plan/exactness, full probabilities and
phase times are retained. Report serialization/write time has its own sidecar;
controller outer wall includes process startup, all I/O and owned-group cleanup.

The controller reuses the existing 50 ms POSIX process-group RSS reader and
owned-group cleanup helpers, without invoking old collectors or old probes.
A single new one-second idle command precedes the 160 slots. Each slot reserves
16 seconds of its 30-second outer budget for TERM/KILL/wait cleanup, so workload
execution stops early enough to leave that reserve. This is an upper resource
bound, not a promise of 30 seconds of numerical work. Retained output files have
a derived 512 KiB ceiling and the controller reserves 4 MiB of storage headroom;
no numerical or CI threshold is relaxed. Raw RSS traces stay in immutable
attempt reports instead of being duplicated into the evolving ledger. The
installed interpreter disables bytecode writes during observation.

Before starting, require the admitted observer-main delivery receipt, a clean
matching checkout, its accepted installation report and byte-identical retained
wheel/package members. Freeze dependency metadata, interpreter/configuration,
controller/worker/monitor hashes and CPU identity. Run the controller from that
checkout with its accepted interpreter and absolute report paths:

```bash
python -m tools.collect_ftqc_phase_cost --source-commit "$OBSERVER_MAIN_SHA" \
  --install-report "$INSTALL_REPORT" --delivery-report "$DELIVERY_REPORT" \
  --output "$NEW_COLLECTION_DIRECTORY"
```

These variables must identify the reviewed, main-admitted observer revision and
a new directory. This command is a research procedure, not the public user
example above; do not execute it during PR validation. A completed collection
would retain one probe and up to 160 workload attempts. Any earlier failure
retains its diagnostic, supervision, elapsed/cleanup times and unattempted
slots. There is no resume, automatic retry or existing-directory overwrite.
Only a named public planning CapabilityError can be a nonfatal refusal.

Implementation tests use synthetic worker payloads, fake processes/clocks and
structural comparison with the preserved A-D descriptions. They do not call
from_word, overlap, public circuit execution or a live RSS probe. Full PR/main CI
and the current public example remain engineering admission, not formal cost
observations. Collection evidence will name the actual observer-main commit,
not this proposal's earlier prerequisite revision.

After collection, the controller derives all three fresh observations and their
median/minimum/maximum for each complete point, retaining warm-up status and
refusal/unattempted gaps. The derived summary does not select fastest repeats,
pool different cases, estimate tail percentiles or choose a production method.
No additional workload or query is needed for this derivation.
