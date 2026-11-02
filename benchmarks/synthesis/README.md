# AO-00 arithmetic resource baseline

This is a bounded Compiler baseline, not a new synthesis engine. It changes no
production lowering, selection, default, export, dependency or public contract.
The primary user workflow is a composable Python `qfunc` program: allocate its
ports, prepare Hadamards and call an arithmetic helper in the same program, then
use `synth.synthesize(main)`, `fq.run(program)` and named
`program.probabilities(...)`. See the runnable
[example](../../examples/synthesis_arithmetic_baseline.py). The frozen measured
corpus retains its original pure-function/InputState harness; it is historical
benchmark machinery, not a reversal of the unified program-entry design.

Run from the repository root in an installed FlagQuantum environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.synthesis.arithmetic_baseline --output /tmp/ao00.json
python -m benchmarks.synthesis.arithmetic_baseline \
  --compare benchmarks/synthesis/baselines/ao00.json /tmp/ao00.json
```

The comparison exits 1 for changed circuit/resources/completion status and refuses
incompatible configurations or malformed cost records. Timing and host metadata
are observational and excluded from equality. A timeout may vary across hosts;
it is an incomplete observation, not evidence of infeasibility. Every compiled
case has a circuit hash. Every artifact records the measured code revision,
source-file SHA256 hashes, Python/Torch/platform, thread count and fixed seed.
The code revision can precede the evidence commit; source hashes identify the
actual scripts independently of that evidence commit.

## Fixed corpus

`expressions.py` contains retained Python source for ten families:

| Family | Expression |
| --- | --- |
| Addition | `x+y` |
| Repeated operand | `x+x` |
| Square | `x*x` |
| Product | `x*y` |
| Square and add | `x*x+y` |
| Multiply and add | `x*y+z` |
| Repeated subexpression | `(x+y)*(x+y)` |
| Overlapping products | `x*x+x*y` |
| Constant linear combination | `3*x+5*y` |
| Overlapping factors | `(x+y)*(x+z)` |

There are 90 cases: 60 equal-width cases (each family at 1/2/3/4/8/16 bits),
16 unequal-width cases (`(1,3,2)` and `(2,4,1)`, truncated to the argument count
for the eight multi-input families), and 14 edge cases (at 1/3 bits: multiplication
by zero/one/eight/three, addition of zero/256, Python literal power `x**2`).
The power case's Qmod equivalent is multiplication, not a claim of Qmod power
syntax. Integration tests additionally check helper-expanded and exchanged
operand spellings and all 256 manual-reference input/target combinations.

Domains are full unsigned ranges. Zero and uniform initialization never narrow
them. Output width comes from an independent monotone integer specification:
all admitted operations have nonnegative operands and are coordinatewise
monotone, so the upper corner attains the maximum. Zero still has one output bit.
This argument does not authorize extending that bound formula to subtraction,
negative constants, signed arithmetic or nonmonotone operators.

## Independent verification

`oracle.py` implements mathematical specifications separately from the captured
functions. It never calls production arithmetic kernels, evaluates source, or
uses synthesized circuits to generate expected results. `verification.py`
interprets exact X/CX/CCX permutations independently:

- Up to six input bits: exhaust all input combinations. Above that: check zero,
  the upper corner and sixteen fixed-seed samples; this is not a whole-domain
  proof. Check zero, one and all-one target values. The two-bit square-and-add
  sentinel checks all sixteen target values, hence 256 combinations.
- Compare the entire output permutation, including unchanged inputs and zero
  workspace. An extra dirty scratch bit fails the check.
- Up to six input bits **and** twenty total circuit qubits: execute two complete
  CPU complex128 states, uniform and a deterministic nontrivial relative-phase
  product state. Compare every complex amplitude (absolute tolerance 1e-12)
  and workspace nonzero probability (at most 1e-12). No phase alignment,
  postselection, renormalization or probability-only substitute is used.
- Larger cases never execute a quantum state. They report zero coherent checks
  and null amplitude/leakage metrics, not a simulated correctness claim.

This is AO-00 baseline evidence, not a formal proof for large sampled domains.
Later AO rules need their own equivalence/cleanup evidence before default use.

## Cost and budget contract

Schema: `flagquantum.synthesis.arithmetic_baseline.v1`. `rows` preserve exact
corpus order, expression, widths, full-domain output width, status, diagnostic
codes and verification scope. `compiled`, `refused` and `timeout` are distinct.
The reader rejects missing/reordered cases, altered domains, invalid histograms,
wrong state sizes, incomplete results claiming costs and static-only rows
claiming amplitude checks. No refused or timed-out case counts as passed.

All `resources` use the existing `estimate_resources` over the generated
CircuitIR. Preparation and complete cleanup are included. Allocated width,
used width, peak scheduler workspace, opcode histogram, operation total, ASAP
depth and two-qubit operation count have separate fields. The complex128 buffer
is `16 * 2**allocated_qubits` bytes, a planned dense state size, **not peak RSS**.
A three-qubit CCX contributes one logical operation and zero two-qubit operations
until decomposed. Do not interpret that logical two-qubit count as device cost.

The 90-case matrix uses the existing logical CPU width objective, both current
retain/recompute candidates, 256-qubit/100000-operation construction ceilings
and the existing default target optimization. Planning allows a theoretical
`16 * 2**256`-byte state ceiling only to inspect static circuits without allocating
them. Actual quantum execution is capped at twenty qubits (16 MiB state buffer,
excluding temporary verification tensors). This is not a claim that a 256-qubit
state can be simulated locally.

Cases above twelve input bits run in a child process with a 30-second wall limit,
including child startup and verification. Timeout rows have no synthesis timing,
resource or verification claim; `worker_wall_seconds` records that external wall
interval. Smaller cases use the compiler's existing structural budgets.

`synthesis_and_planning_seconds` measures one complete `synth.synthesize` call:
source capture, analysis, both candidates, target compilation and plan validation.
It excludes oracle verification, host startup and worker startup. Candidate
construction times and the selected target compilation time are existing nested
Compiler measurements; they overlap the total and **must not be added to it**.
No warmed median, speedup, isolated compiler-stage or performance claim is made.

## Target sentinel and independent Circuit

The two-bit `x*x+y` has Python, generated Qmod and independent hand-derived
Circuit references. The baseline compares nine rows: three implementations times
logical, the existing exact phase-capable named basis (`PHASE_BASIS`), and that
basis on a fixed-width line coupling graph. These are target conversions of
fixed source circuits, **not** a fresh target-aware synthesis search. Other
corpus cases do not claim target-basis or line-topology verification.

All target costs use the final executable from the existing target pipeline,
including decomposition, optimization, routing and layout restoration. The
existing basis includes `u3` to admit optimizer output; a strict Clifford+T-only
request currently refuses some post-optimization `u3` and is not relabeled as
successful Clifford+T synthesis. The target pipeline validates its normal
contracts; the runner does not claim full-state simulation of every converted
large sentinel. The generic Python/Qmod and manual logical sentinel states are
checked independently against mathematics.

The hand-derived reference uses `x=2a+b`, `y=2c+d`; low-to-high result bits are
`b XOR d`, `c XOR bd`, `a(1-b) XOR bcd`, `ab`. Its scratch temporarily stores `bc`
and is cleaned. It is a benchmark reference, never a production implementation
selected by recognizing this expression. See `manual_reference()` and the
[delivery record](../../docs/roadmap/SYNTH_ARITHMETIC_AO00_DELIVERY.md).

This evidence is local CPU static-resource/correctness evidence. It establishes
neither optimality, runtime speed, process peak memory, hardware execution,
release readiness, independent user trials nor distributed scalability.

The checked AO-00 JSON is the historical starting point for later optimization
PRs. Preserve it; record new optimizer results in a new artifact with their own
source revision. Review any deliberate update to current-generator cost tests
alongside equivalence evidence and comparison to this frozen baseline. Do not
regenerate the old baseline merely to turn a resource regression green.
