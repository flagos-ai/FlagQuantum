# Target-CZ-aware SABRE trial experiment

This experiment finds real downstream gate-count opportunities inside existing
SABRE trials, but the inspected implementation is too expensive for the default
compiler. It remains a benchmark-only policy. No production module or default
changes here.

## Existing algorithms and scope

The experiment reuses FlagQuantum's existing Qiskit-derived KAK/Weyl engine:
`_prepared_basis("cz").entangler_count(_weyl_decomposition(...))` estimates the
local entangler cost before one-qubit synthesis. It also reuses the existing
bounded block collector, basis translator, synthesis proposals, residual checks
and resource rejection rules. The upstream provenance and Apache notices remain
in their owning implementation; see
[the reuse audit](COMPILER_UPSTREAM_REUSE.md). This does not import an external IR
or add a production dependency.

The diagnostic constructs physical native previews for the incumbent and the
existing four additional deterministic trials. It retains at most one alternative
for actual bounded block synthesis. Including the incumbent, at most two native
previews are realized; executable artifacts are not built for every trial.
Nevertheless the surrounding target pipeline can invoke selection repeatedly
while pricing, emitting and independently replaying routing evidence. The
experiment does not bypass those checks or introduce a cross-call cache.

Scope is an unannotated numeric one/two-qubit source, identity input placement,
undirected connectivity and the abstract `rx/rz/cz` target. The existing explicit
restoration emits the final permutation. Annotations, custom matrices, symbolic
or batched angles and unmodeled input placement are refused. Directed-cost
callbacks retain the original selector. This is not target-independent routing.

## Why a Weyl count is not delivered cost

The initial calibration predicts the delivered CZ count on all 60 inspected
incumbent application/topology pairs. That agreement is insufficient admission.
A real twelve-operation, two-wire circuit with four CZ gates has a Weyl estimate
of two CZs, but with `max_added_operations=0` the existing synthesis rejects its
instruction growth and retains four CZs. The regression test compares both zero
and nonzero budgets against the authoritative native stage and checks full
operators independently with the runtime.

An unguarded trial selector reduces CZ count on five inspected pairs, but three
12-wire QAOA cases increase delivered dependency depth:

| Topology | Baseline CZ/depth | Unguarded CZ/depth |
| --- | ---: | ---: |
| Line | 270 / 101 | 262 / 113 |
| Grid | 150 / 76 | 146 / 80 |
| Sparse | 122 / 51 | 118 / 55 |

The maintained experiment therefore uses the existing synthesis acceptance and
actual operation budget for both shortlisted previews. It keeps the incumbent
unless the alternative reduces realized CZ count and does not increase realized
two-body dependency depth. These are explicit experiment admission conditions,
not a newly declared universal hardware cost function. Real devices can have
other objectives; this experiment does not assign their tradeoffs implicitly.

## Complete inspected results

The final source driver retains 60 paired target-artifact applications and 120
independent SDKBench acceptances. Two pairs improve both objectives:

| Inspected pair | Baseline CZ/depth | Guarded CZ/depth |
| --- | ---: | ---: |
| 6-wire QFT, grid | 65 / 48 | 63 / 38 |
| 6-wire repetition, ring | 82 / 57 | 76 / 55 |

The other 58 pairs are unchanged. The set spans widths 6/8/12, five application
families and line/ring/grid/sparse topologies. These are already-inspected
nonce-`01` development controls, not unseen qualification. Six-wire checks use
full operators up to global phase; larger checks use the existing three probes.
The generator, adapter, evaluator, native/routing budgets and acceptance
thresholds are unchanged. The driver reads the native operation budget from the
pinned SDKBench workflow configuration rather than substituting one.

[Raw stages, exact historical drivers and counterexample](evidence/sabre-native-cost.json)
retain the calibration, unguarded selector, guarded result, resource witness and
measurements. Production source is main
`10962eaa3189457e46a89469050060eebdd47fb4`; external SDKBench source is
`9aa9260920984e2ca05bcb293abb885fccd9c144`. The report source identities are
historical measurements and are not relabeled as measurements of later merges.
Revision field names in archived exploratory reports are normalized to the
existing repository vocabulary, retaining the original document hashes.

## Resource decision

With two warmups and seven alternating retained timing samples per variant,
separate from independent acceptance and from Python allocation observations:

| Workflow | Baseline median | Guarded median | Python traced peak, baseline / guarded |
| --- | ---: | ---: | ---: |
| 6-wire QFT, grid | 102.293 ms | 473.892 ms | 1,899,608 / 2,504,966 bytes |
| 6-wire repetition, ring | 132.689 ms | 460.840 ms | 2,578,435 / 2,602,511 bytes |
| 12-wire QFT, grid | 670.028 ms | 1,383.251 ms | 10,107,653 / 10,118,143 bytes |

The observations are roughly 4.63x, 3.47x and 2.06x baseline compiler time.
They are shared-host development observations, not resource-qualified latency,
statistical significance or process-RSS measurements. All three Python peak
observations increase. The raw timing driver checks stable output within each
variant and independently accepts its separate memory observations too.

**Do not enable this implementation in production.** Quality improvement in two
inputs does not offset this measured work amplification for a default policy.
No fresh nonce is drawn to seek a favorable result. A successor must reduce
repeated trial/preview work before independent routing replay and artifact
construction, preserve the incumbent and actual synthesis/resource admission,
and then freeze a new protocol before prospective inputs. A lower-cost proxy
can propose a candidate; it cannot certify its realized benefit.

## Verification and reproduction

The focused suite passes 92 tests, including full-operator reversed-control and
restoration scenarios, the real budget counterexample, and refusal of annotations,
symbolic parameters and unmodeled input permutations. A separate injected-failure
scenario attempts all 60 cases: one deliberate exception and one deliberate
metric regression remain in the report, 59 workflows finish real independent
acceptance, and the experiment exits nonzero. Those injected metrics are not
reported as measured routing quality.

```bash
python -m pytest -q tests/unit/test_compiler_sabre_native_cost.py \
  tests/unit/test_native_block_optimization.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m benchmarks.compiler_sabre_native_cost \
  --sdkbench-root /path/to/unchanged-sdkbench \
  --output /tmp/sabre-native-cost.json
```

Use an explicit outer process-group timeout; the retained run used 240 seconds.
The report checkpoints after every attempted case and exits nonzero on any
execution failure or delivered CZ/depth regression. Black, Ruff, focused mypy and architecture checks pass. The complete Git-object
walk of 994 tracked evidence JSON documents plus the new archive uses only
the already-declared revision field names. Full repository CI remains
required. Formal Linux SDKBench qualification remains deferred. No ranking,
hardware, scalability or superiority claim follows, and no reboot is involved.

Existing users continue the unchanged workflow:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

program = compile(fq.Circuit(4).h(0).cx(0, 3), coupling_map=CouplingMap.line(4))
assert program.metadata["routing"]["mapping_restored"] is True
```

Expected: legal device edges and restored logical output placement. The
experimental selector is not installed by this example or by importing its module.

## Archive serialization

The report is serialized without indentation to respect the unchanged 1,800,000-byte repository file limit. All parsed values, historical drivers and measurements are preserved. The original indented document SHA-256 was `fca49bbaa7f8699a0dfdd159efac5d3860b26a3504ebc66745d3cce923bd2c08`; its size was 2,703,662 bytes. The compact document is 683,439 bytes. Parsed-document equality was checked before replacing the representation. No resource or acceptance threshold was relaxed.
