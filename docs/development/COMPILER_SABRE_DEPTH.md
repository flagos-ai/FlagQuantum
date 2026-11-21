# Equal-cost SABRE depth experiment

Status: rejected development candidate on 2026-11-22. Compiler owns this
`single_device_fast_path` investigation. The production selector, public API,
defaults, target contracts and capability maturity are unchanged.

## Question and candidate

The current bounded trial selector retains the incumbent whenever the complete
restored SWAP count ties. Could those ties hide useful parallel routes?
The baseline is main `2d78133a1f5a7d234acbb128724ccf2674e2eeb8`, including the
existing deterministic route, three seeded trials and layer-lookahead trial.

The explicit diagnostic replaces only trial selection. It retains that same
candidate set and ranks by `(restored_swap_count, estimated_depth)`. The depth
estimate assigns three dependency levels to each inserted SWAP, one to each
two-wire source operation and zero to other operations, while propagating wire
dependencies. It includes identity restoration. This is an inexpensive routing
heuristic, **not** exact native CZ depth, duration or a classical dependency
schedule. In particular, a source RZZ operation can lower to more than one CZ.
The experiment does not apply this estimate to directed-cost callbacks.

The committed-forward-SWAP pruning bound becomes `best_cost + 1`, allowing an
equal-count candidate to finish. An equal count and equal estimated depth retain
the incumbent. No extra seed, layout freedom, target ancilla or competitor
configuration is introduced. The existing upstream adaptations remain as
documented in [compiler upstream reuse](COMPILER_UPSTREAM_REUSE.md); this tie
policy is an FQ experiment, not a new Qiskit or Cirq source port.

## Complete workflow results

The [generic observations](evidence/sabre-depth-generic.json) contain 80 cases:
widths 6/8/12/16, line and two-row grid targets, seeds 0–9, and `4 * width` CX
operations. Both variants pass exact complete GF(2) operator checks, every
coupling edge check and identity-output restoration. Total SWAPs remain 6694;
the sum of logical CX/SWAP dependency depths changes from 4576 to 4546, with no
observed depth regression. That reported depth counts each logical CX and SWAP
as one, separately from the weighted heuristic used for selection.

The [application observations](evidence/sabre-depth-applications.json) contain
120 paired cases using unchanged SDKBench
`9aa9260920984e2ca05bcb293abb885fccd9c144`: QAOA, Ising, fermionic hopping,
QFT and coherent repetition recovery; widths 6/8/12; and line/ring/grid/sparse
targets. The public development nonce is `01` repeated 32 times.

| Workflow | Cases | Sum of delivered CZ counts | Sum of CZ dependency depths | Changed cases |
| --- | ---: | ---: | ---: | --- |
| Ordinary restored | 60 | 8368 → 8368 | 5187 → 5165 | 4 depth improvements; no CZ changes |
| Target artifact | 60 | 7564 → 7566 | 4701 → 4688 | 2 depth improvements; 1 CZ regression |

Both variants pass the existing independent evaluator on every application.
Six-qubit checks compare complete operators up to a global phase; widths 8 and
12 check three output probe states and do not prove universal equivalence.
The maximum reported semantic error is `6.072657351995283e-14`, below the
unchanged `1e-8` threshold. Both workflows restore identity output layout and
use no ancillas. The ordinary workflow and target-artifact workflow retain
their distinct delivered boundaries; totals are not cross-SDK rankings.

## Counterexample and decision

Six-qubit grid QFT illustrates why a routing-only depth win is insufficient:

| Target-artifact observation | Baseline | Candidate |
| --- | ---: | ---: |
| Native baseline CZ count before block optimization | 87 | 87 |
| Accepted block rewrites | 5 | 5 |
| Delivered CZ count | 65 | 67 |
| Delivered CZ dependency depth | 48 | 38 |

Changing the route changes the instruction order, operand placements and numeric
blocks available to later synthesis. Equal pre-block CZ counts and equal rewrite
counts do not ensure equal final CZ costs. The retained audit establishes this
workflow interaction; it does not isolate which block or synthesis formula causes
the two-CZ difference. The corresponding ordinary output retains 75 CZs and
improves depth from 52 to 42, so evaluating only ordinary compilation would miss
the target-artifact regression.

Do not enable this candidate in the default pipeline. Its observed depth
benefits include a delivered CZ tradeoff, and this study does not establish its
latency or memory cost. The next admission study should evaluate candidate routes
against the downstream target objective under a bounded budget, preserving the
incumbent and reporting count/depth/time tradeoffs. Do not special-case this QFT
fixture or choose different acceptance rules for FlagQuantum.

## Reproduction and limits

Use a disposable FlagQuantum checkout at the baseline above with this change's
benchmark module, and a clean SDKBench checkout at its pinned revision. The
recorded runner and SABRE SHA-256 digests bind the observations to their inspected
files. The runner checkpoints every successful case and records an exception
before propagating it. Production modules are patched only within the explicit
diagnostic process and restored on context exit.

On Linux with the existing development dependencies, run each suite with a
600-second process budget:

```bash
timeout --signal=TERM --kill-after=10s 600s env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.compiler_sabre_depth --suite generic \
  --output /tmp/sabre-depth-generic.json
timeout --signal=TERM --kill-after=10s 600s env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.compiler_sabre_depth --suite applications \
  --sdkbench-root /path/to/unchanged/sdkbench \
  --output /tmp/sabre-depth-applications.json
```

The retained execution used a shared macOS CPU host, Python 3.12 and an external
process-group watchdog with the same 600-second limit and termination cleanup.
The runner does not implement SDKBench's formal resource executor. These are
inspected development inputs, not prospective held-out draws. The fixed
baseline-first order, shared host and absence of independent memory enforcement
preclude latency, memory, significance or qualification claims. No hardware,
fidelity, scalability, global SDK ranking or superiority claim is made.

The initial scratch probe selected a nonexistent `ROUTE-LINE-08` case and stopped
after width-six ordinary cases. It produced no accepted complete report. The
committed runner uses the existing `ROUTE-GRID-08` acceptance record at width
eight and reproduced both full suites against main, with the evaluator and
thresholds unchanged. Earlier scratch runs against unmerged scalar-encoding code
are not the archived main-baseline observations. Formal host qualification
remains deferred; no host configuration change or reboot is required.

## Existing user workflow

This study supports the existing ordinary compiler workflow; it introduces no
new user option. The same circuit still compiles to legal device edges and
restores its logical output order:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile, final_layout

circuit = fq.Circuit(6)
circuit.gate("h", (0,))
circuit.gate("cx", (0, 5))
coupling = CouplingMap.line(6)
program = compile(circuit, coupling_map=coupling, config=circuit.runtime_config)
assert final_layout(program).logical_to_physical == tuple(range(6))
assert all(
    len(instruction.wires) != 2 or coupling.has_edge(*instruction.wires)
    for instruction in program.instructions
)
```
