# External scientific-program compiler observations

This study follows the historical Linux routing controls with a prospectively
published population of external QASMBench programs. It evaluates delivered
workflows against unchanged SDKBench acceptance, rather than constructing an
aggregate SDK score. Results remain development evidence with
`publication_eligible=false` and `scalability_claim_allowed=false`.

The complete population contains seven original programs, two targets each and
four registered workflows: 56 intended rows. The frozen converter admitted DNN
(eight wires), QAOA (six wires) and Simon (six wires), yielding six tasks and 24
executable rows. Four sources account for 32 input-not-admitted rows: CC and BB84
contain quantum operations after measurement; both VQE inputs fail source syntax
admission because `q` is undefined. Those are preparation refusals, not SDK
failures. They remain in the denominator; no input was repaired or replaced.
There is no admitted twelve-wire external program.

The source census, converter, exposure audit, participant wheel/configuration
identities, independent references, limits and final driver were published before
participant outcomes. An audit of 8662 historical files found no exact matches but
retained 242 representation/access gaps. Full Git history, private campaigns and
developer knowledge were not certified. External origin and prospective freezing
do not supply a complete novelty certificate. These inputs are now inspected;
subsequent compiler tuning must treat them as development data.

## Boundaries and independent acceptance

The four workflows retain the previously measured FlagQuantum wheel from source
`f88ebc88f8584cd580802cd219ba16d935fa6715`, Qiskit 2.6.2 and Cirq Core 1.7.0.
The unchanged installed dependency inventory includes Python 3.12.14 and CPU
Torch 2.14.0. Each block uses one CPU/thread, complex128/float64 numerical data,
a 16-GiB aggregate cgroup memory cap, no swap and denied external sockets.
This is static exact compilation within numerical tolerances, with no gradient,
QPU, accelerator, distributed workload or scalability claim.

All workflows receive the same canonical operation list and undirected line or
grid target, arbitrary input states, identity initial layout, no ancillas,
arbitrary one-qubit unitaries and full CZ entanglers. Output logical order is
restored. The ordinary workflows include the same timed deterministic layout
restoration; the FlagQuantum artifact workflow includes its native restoration,
validation, provenance and executable serialization. Different delivery
obligations prevent attributing total runtime differences to one compiler pass.
The declared upstream options include Qiskit level 2 with exact synthesis and
arbitrary-input semantics, and Cirq RouteCQC/full-CZ target synthesis.

The existing independent evaluator checks topology, output layout, complete CZ
count and two-body dependency depth, with semantic absolute error at most 1e-8,
zero relative tolerance and normalization error at most 1e-10. Six-wire checks
use the complete operator up to one common global phase. The eight-wire DNN
uses three complete output-state probes, which do not prove universal operator
equivalence. Every case retains ten fresh process blocks, two warmups and three
measured positions per block; compiler seed requests and participant order
rotation are unchanged. Failed cases contribute no successful timing statistic.

## Artifact admission failure

DNN and QAOA fail the FlagQuantum artifact workflow during
`ProgramArtifactV2.from_circuit_ir`, before target compilation or routing. The
existing `_strict_v2_value` counts all mapping and list entries, not just quantum
operations. The approved artifact-v2 contract caps these entries at 4096. The
abstract target's 20000-operation allowance does not override this source
serialization boundary. Report the retained `error` verdict rather than
reclassifying it as an independently accepted output or hiding the workflow.

A separate post-observation local source-envelope diagnostic used the exact
measured `_artifacts.py` implementation. DNN's 2640 operations require 21143
mapping/list entries and 232657 canonical payload bytes. QAOA's 702 operations
require 5637 entries and 61926 bytes. Simon's 104 operations require 853 entries
and 8992 bytes, and its input envelope constructs successfully. These byte counts
are below the separate 16-MiB payload cap. This isolates an input-envelope
constraint; it does not measure routing, synthesis or compiler speed. Nine
unchanged artifact-v2 candidate tests pass, including the approved bounds.

Changing this protected bound requires a separate proposal, compatibility and
resource analysis, conformance tests and explicit authorization. No bound,
snapshot, evaluator, participant option or protocol budget was changed to make
this campaign pass.

## Operational failures retained

The first host-admission script could count its own activity on measured CPU 96
and refused admission before creating a campaign. Its original after-sample did
not survive, so exact busy-tick attribution is unavailable. The next driver moved
the observer to CPU 0 but propagated that affinity to the campaign supervisor.
All 240 blocks of campaign attempt 1 were correctly refused as `access-blocked`
before participant worker execution. All 720 placeholder records, zero
supervised executions, original hashes and independent replay are retained.
Neither failure is an SDK measurement.

A separately published attempt-2 driver retained the observer on CPU 0 and
explicitly launched the unchanged candidate CLI on CPU 96. A launch-only control
confirmed the split, and all fifteen unchanged resource controls passed with
this invocation. The new attempt retained its own directory and fresh idle
admission, unchanged acceptance, original phase budgets and original reservation
end. This was recovery from a demonstrated execution prerequisite failure,
not selection or repetition of compiler outcomes.

## Existing workflow supported by this report

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(4).h(0).cx(0, 3)
compiled = compile(source, coupling_map=CouplingMap.line(4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

This existing CPU workflow is unchanged. The report concerns the cost and quality
of native compilation and separate artifact packaging for application programs;
it introduces no public API, support boundary or maturity promotion.


## Complete measured outcomes

Twenty of 24 executable cases were accepted; four retained `error`. Across all
56 intended rows this is 20 accepted, four participant errors and 32
input-not-admitted. The 240 blocks retain 720 measured positions (600 accepted,
120 error) and 480 warmup positions. Maximum accepted numerical residual is
1.1072e-14. Producer-free replay verified 240 blocks and 825 campaign files.
The campaign returned 1 in 1019.991 seconds because of the four failed cases;
independent replay returned 0 in 163.212 seconds; the unchanged report tool's
second replay and summary returned 0 in 208.111 seconds. None exceeded its
prospective phase ceiling. These evaluation costs are separate from compile time.

The table uses independently recomputed CZ/dependency-depth ranges over accepted
records. Timing is the median of ten block medians, each based on three accepted
measurements after two warmups. The machine-readable summary retains every
sample, inclusive quartiles and block-median range. Memory is the maximum
aggregate cgroup peak for a complete block including imports, warmups, measured
work, export and descendants. Memory for a failed case is diagnostic only and
not evidence of a faster or smaller successful compilation.

| Instance | Workflow | Status | Accepted / 30 | CZ range | Two-body depth | Median ms | Maximum block MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXT-02-LINE-08 | `cirq-release-restored` | accepted | 30 | 197 | 84 | 457.591 | 153.04 |
| EXT-02-GRID-08 | `cirq-release-restored` | accepted | 30 | 103 | 47 | 317.330 | 149.36 |
| EXT-03-LINE-06 | `cirq-release-restored` | accepted | 30 | 100 | 73 | 170.090 | 145.73 |
| EXT-03-GRID-06 | `cirq-release-restored` | accepted | 30 | 72 | 51 | 131.151 | 143.77 |
| EXT-04-LINE-06 | `cirq-release-restored` | accepted | 30 | 32 | 32 | 51.043 | 141.36 |
| EXT-04-GRID-06 | `cirq-release-restored` | accepted | 30 | 50 | 45 | 69.463 | 142.33 |
| EXT-02-LINE-08 | `fq-integrated-artifact` | error | 0 | — | — | — | 158.33 |
| EXT-02-GRID-08 | `fq-integrated-artifact` | error | 0 | — | — | — | 158.17 |
| EXT-03-LINE-06 | `fq-integrated-artifact` | error | 0 | — | — | — | 153.43 |
| EXT-03-GRID-06 | `fq-integrated-artifact` | error | 0 | — | — | — | 153.46 |
| EXT-04-LINE-06 | `fq-integrated-artifact` | accepted | 30 | 34 | 31 | 174.505 | 154.52 |
| EXT-04-GRID-06 | `fq-integrated-artifact` | accepted | 30 | 36 | 33 | 178.035 | 154.78 |
| EXT-02-LINE-08 | `fq-integrated-ordinary` | accepted | 30 | 270 | 114 | 318.118 | 164.11 |
| EXT-02-GRID-08 | `fq-integrated-ordinary` | accepted | 30 | 264 | 111 | 317.957 | 164.44 |
| EXT-03-LINE-06 | `fq-integrated-ordinary` | accepted | 30 | 108 | 72 | 80.168 | 156.25 |
| EXT-03-GRID-06 | `fq-integrated-ordinary` | accepted | 30 | 84 | 51 | 88.907 | 155.82 |
| EXT-04-LINE-06 | `fq-integrated-ordinary` | accepted | 30 | 32 | 30 | 16.342 | 152.76 |
| EXT-04-GRID-06 | `fq-integrated-ordinary` | accepted | 30 | 44 | 43 | 17.276 | 152.98 |
| EXT-02-LINE-08 | `qiskit-release-restored` | accepted | 30 | 136–142 | 47–65 | 19.952 | 81.28 |
| EXT-02-GRID-08 | `qiskit-release-restored` | accepted | 30 | 124–130 | 52–72 | 19.079 | 83.17 |
| EXT-03-LINE-06 | `qiskit-release-restored` | accepted | 30 | 94–104 | 61–77 | 11.424 | 70.13 |
| EXT-03-GRID-06 | `qiskit-release-restored` | accepted | 30 | 70–82 | 50–57 | 9.985 | 66.57 |
| EXT-04-LINE-06 | `qiskit-release-restored` | accepted | 30 | 28–32 | 26–32 | 6.295 | 64.54 |
| EXT-04-GRID-06 | `qiskit-release-restored` | accepted | 30 | 40–50 | 34–44 | 6.637 | 65.03 |

FlagQuantum's ordinary workflow was slower than Qiskit on every admitted task:
15.94/16.67 times on DNN line/grid, 7.02/8.90 on QAOA and 2.60/2.60 on Simon.
It used more CZs than every Qiskit observed seed on DNN and QAOA. On Simon its
counts overlap Qiskit's ranges. Its maximum block-memory peak was larger than
Qiskit's in all six cases. It was faster than Cirq in five tasks and slightly
slower on DNN grid (317.957 versus 317.330 ms); these short observations do not
establish statistical significance or an SDK-wide ranking.

The artifact workflow succeeded only on Simon. On the line it emitted 34 CZs
and depth 31, versus the ordinary workflow's 32 and 30: a quality regression,
in addition to higher latency. On the grid it reduced those measures to 36/33
from 44/43, but cost 178.035 ms versus 17.276 ms. Both artifact cases were slower
than the compared Qiskit and Cirq workflows and had larger block-memory peaks.
The native block optimizer's local strict-reduction rule does not guarantee an
improvement over a separately optimized ordinary whole workflow.

These observations supersede any suggestion from the three historical controls
that artifact gate counts are uniformly better or ordinary performance is close
to Qiskit. No universal superiority, confidence interval, optimality, device
fidelity, dynamic-circuit, symbolic-gradient or twelve-wire external claim is made.

## Host closure and next engineering decision

Fresh admission used CPU 96 with idle SMT sibling 288; the observer stayed on
CPU 0. Pre/post OS package and stable-file baselines matched; no foreign CI worker
was observed in the retained heartbeats, and no worker cgroup remained. The
original sixteen scheduling label sets were restored after both the blocked
first attempt and the completed second attempt. No machine/service restart,
kernel/driver change or unrelated process termination occurred. Directly
administered snapshots are not independent immutable-host attestation.

Keep production code and the comparison protocol unchanged by this report.
Priorities are the protected source-artifact capacity inconsistency, ordinary
whole-workflow latency and DNN/QAOA target-CZ quality. Profile development inputs
and inspect existing/upstream passes before selecting an implementation. Source
normalization, routing, native synthesis and evidence/serialization costs require
separate attribution. This population is now development data for subsequent
changes; a new prospective population is needed for new unseen-program claims.

## Public evidence and complete ledger

The [SDKBench report](https://github.com/sdkbench-org/sdkbench/blob/docs/compiler-external-program-results/docs/analysis/COMPILER_EXTERNAL_RESULTS_20261123.md)
and its adjacent JSON retain all 56 intended rows and the independent summary.
The [public evidence release](https://github.com/FlagQuantum/FlagQuantum-evidence/releases/tag/evidence-compiler-external-2026-11-23.1)
contains the complete campaign, operational failures, sources, LICENSE and NOTICE,
per-file manifest and credential-scan receipt. Detailed private host/process
snapshots are excluded.

The explicitly approved archive is 21,904,135 bytes, SHA-256
`2062cab0857de6088abe402d05ea7722dad8d37cc5cb13206a478f79e07428d8`.
Its manifest SHA-256 is
`596c3e69fe89f46b12730a07a1cadc4c4982f3f037597c078b445d30f519e201`.
Downloaded draft assets matched both identities before public release on
November 23, 2026. Public accessibility does not promote these development
observations to a certified unseen-program comparison.
