# Bounded native preview matrix reuse

## Requirement and ownership

Compiler owns this `single_device_fast_path` experiment. The target-CZ route
selector repeatedly reads native RX/RZ/CZ matrices while collecting blocks,
estimating their costs and realizing two routes. The unchanged numerical reader
is authoritative; this optimization changes neither the selector's objectives
nor production compiler policy. Qiskit-derived KAK/Weyl and existing Cirq-related
compiler mechanisms retain their [upstream attribution](COMPILER_UPSTREAM_REUSE.md).
This is a native repeated-work optimization, not a new upstream code port.

[Diagnostic profiles](evidence/native-preview-hotspots.json) include profiler
overhead and overlapping totals; they are not latency measurements. The
[untimed census](evidence/native-synthesis-census.json) executes every original
synthesis. It finds repeated exact matrices but establishes no speedup. These
observations motivate examining numerical work rather than adopting the rejected
translation-template experiment. Their production/experimental source bytes
match this base; their original revision identities remain intact.

## Implementation boundaries

One private matrix reader lives for one route selection. It retains at most 32
matrices with least-recently-used eviction. Keys contain native opcode, the
operand's side in the ordered two-qubit basis, and the exact IEEE-754 bytes of
a finite Python float angle. Signed zero is distinct. A fixed parameter-free CZ
is symmetric in its two operands. No rounded key is used.

Only unannotated RX/RZ with exactly one float `theta`, and fixed CZ without
parameters or matrix overrides, qualify. All tensors, symbolic values, batches,
trainable values, extra parameters, annotations, custom matrices, unrelated
wires and other opcodes pass to the original reader on every request. A parameter
dictionary changed between reads changes its key. No instruction, tensor,
program, plan, executable or serialized artifact is retained in the table.
Matrices are read-only inputs to the existing multiplication routines. No cache
survives selection, exception or another source/target call.

Source admission and native translation still run. Source matrices are reused
only inside the bounded collector, estimate and realization. Each synthesis
proposal still passes the existing actual operation-growth, delivered CZ/depth
and independently reconstructed residual checks. Production routing replay,
artifact creation and downstream verification remain operative. No production
module, API, default, schema, dependency or participant adapter changes.

## Prospective protocol, November 23, 2026

Publish the implementation and protocol before application or resource
observations. Base: `959f25462c5e869a05b1957cc0cd68ef8900680c`.
Load its complete benchmark module alongside the new one. Require AST equality
outside the reader and the four helpers that receive it. The same completed-trial
selector calls the old or new route-choice function; all production source stays
identical. Freeze revisions, source and driver hashes, dependencies, configuration,
compiler seed and input nonce. Both source trees stay clean during measurement.

Use unchanged SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144`,
`flagquantum-target-artifact`, seed 11 and its 20,000-operation native growth
budget. Execute all 60 width/family/topology combinations for both variants,
alternating order. First use the inspected nonce `01` repeated 32 times, then
one fresh 32-byte nonce frozen after publication. The fresh draw is not an
independently blinded held-out test. Preserve every failure and attempt.

Require every output to pass the unchanged evaluator, coupling legality and
identity final layout. Require byte-identical operation/layout payloads, audits
and metrics for every old/new pair. Retain compile times outside semantic hashes.
Width six checks complete operators; widths eight/twelve check three probes at
the existing `1e-8` threshold. Replay all 240 outputs without importing any
producer SDK in a fresh process. Any mismatch blocks admission.

Only after both quality draws and replay pass, measure complete adapter execution
for inspected 6-wire QFT/grid, 6-wire repetition/ring and 12-wire QFT/grid. Include
production, old experiment and new experiment contemporaneously. Two warmups
per variant precede nine rotating three-variant rounds and one separate traced
Python allocation round per variant after garbage collection. Verify every
output and audit. Old/new outputs must match their quality rows. Production's
first validated warmup establishes its reference; subsequent production outputs
must match it. Retain all 36 observations per control, including warmups and
traced runs. No other local tests run concurrently.

Admit this experimental implementation only if all three new/old median ratios
are at most `1.05`, all new Python peaks are at most `1.10` times old and at most
128 KiB above old, and at least two controls have new/old ratios at most `0.90`.
Report production comparisons separately, including any time or memory deficit;
the experimental implementation's admission does not admit a production default.
No rerun for favorable results, case omission or retrospective threshold change.

Bound each quality draw to 600 seconds, replay to 180 seconds, resources to
300 seconds and reports to 16 MiB, with process-group cleanup on timeout.
This shared macOS host cannot certify formal Linux resource limits, process RSS,
statistical significance, SDK rankings or scalability. Formal benchmarking remains
deferred. No reboot, service restart or kernel upgrade is authorized.

## Verification and existing user workflow

Tests compare matrices for different physical labels, both ordered basis sides,
RX/RZ distinction and reversed CZ operands. They cover mutable angles, signed
zero, tensor/symbolic/annotated/custom inputs, bounded eviction, new-reader
isolation, complete small operators and zero-growth synthesis rejection.

```sh
python -m pytest tests/unit/test_compilation_sabre_trials.py \
  tests/unit/test_compilation_sabre_dependencies.py \
  tests/unit/test_compiler_sabre_native_cost.py \
  tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

The existing supported workflow remains:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4)
circuit.h(0)
circuit.cx(0, 3)
compiled = compile(circuit, coupling_map=CouplingMap.line(4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

Expected: topology-legal compilation with restored logical output order. This
experiment does not enable target-native-cost selection in that default workflow.

Frozen control driver SHA-256: `a6059bcd89af8077b8811124f05a2c21093fc7b15e6462a93c4e928330dc2b4b`.
The eventual evidence archive must retain its exact source.

Before application measurement, all 243 focused unit/integration tests pass.
Ruff, Black and strict mypy pass both changed Python modules. Complete final-head
CI remains required before integration; this is not full-suite certification.
