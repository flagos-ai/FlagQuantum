# PF-01: reuse validated plan fields within a Services check

This Services slice implements revision 9 sections 13.5–13.6, after PF-00
[#1018](https://github.com/flagos-ai/FlagQuantum/pull/1018), merged as
`6bb1b7d9fa99719f326dd2106ff88ecdfc909f72`. It removes repeated local
conversion and reconstruction without introducing a cache or changing the
validation trust model. It is not PF-02, Rust migration or AO-03 arithmetic work.

## Existing user workflow

Users continue to prepare, compute, compose and run one native program:

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
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
```

All sixteen `(x,y,x*x+y)` outcomes have probability `1/16`. The matching Qmod
can be passed to `synth.synthesize(text, source_format="qmod", ...)`:

```qmod
qfunc energy(const x:qnum<2>,const y:qnum<2>,output result:qnum) {
    result=x*x+y;
}
qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {
    allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
    energy(x,y,result);
}
```

An independently derived, two-bit-only Circuit comparison requires physical
wires and manually designed reversible logic:

```python
import torch
import flagquantum as fq

circuit = fq.Circuit(9, dtype=torch.complex128)
for wire in range(4):
    circuit.h(wire)
circuit.cx(1, 7).cx(3, 7).cx(2, 6).ccx(1, 3, 6)
circuit.x(1).ccx(0, 1, 5).x(1)
circuit.ccx(1, 2, 8).ccx(8, 3, 5).ccx(1, 2, 8).ccx(0, 1, 4)
result = fq.run(circuit)
state = result.statevector()
```

Here x uses wires `(0,1)`, y `(2,3)`, result `(4,5,6,7)`, and wire 8 is
clean workspace. The generic synthesized circuit remains 16 allocated/used
qubits, eight workspace qubits, 81 operations and depth 59; the independent
Circuit remains nine qubits, one workspace, 15 operations and depth 11.
PF-01 reduces Python delivery overhead; it does not close that circuit-resource
gap. The runnable native/Qmod example is
[`examples.synthesis_arithmetic_implementations.py`](../../examples/synthesis_arithmetic_implementations.py);
the independent reference is `manual_reference` in
[`arithmetic_baseline.py`](../../benchmarks/synthesis/arithmetic_baseline.py).

## Responsibility and validation boundary

Only `services/_synthesis_execution.py` changes production behavior internally.
Previously each `_program` check obtained the decision, extensions, recorded IR
and actual IR through separate validated Runtime conversions. It now obtains
one validated `ExecutionPlan.to_dict()` payload for the decision, extensions
and program fingerprint. Runtime's serializer recomputes the sealed program's
content hash before accepting that fingerprint; Services does not trust an
unvalidated supplied hash.

The existing `plan_execution_program(plan)` still validates and reconstructs
the actual instruction layers through Runtime's authoritative implementation.
Services compares their content hash with the freshly validated fingerprint.
It neither duplicates Runtime's canonicalizer nor adds a new Runtime/Core API.
Public schemas, identities, signatures and compiler candidate selection remain
unchanged. Plan/result admission, raw controls, layout, basis, budgets and
actual state shape/dtype checks remain at their existing boundaries. Local
payloads live only within one call; nested mutable data is never cached.

New negative cases alter the sealed program metadata, program fingerprint,
decision or extensions after successful inspection/execution. Inspection,
resource access and real-result association must all reject the altered plan;
the original remains usable. Existing tests additionally cover actual layer
changes, raw RuntimeConfig mutation, topology/routing evidence, constraints,
result shape/precision and nonidentity physical layouts with relative phase.

## Reproduction and measurement scope

The clean comparison pins are:

- Before: `6bb1b7d9fa99719f326dd2106ff88ecdfc909f72`.
- After: `a1c6308123c938a711fc036f54f1d3393f1aa07c`.

At each pin, run the unchanged PF-00 driver with that pin supplied explicitly:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m benchmarks.synthesis.performance \
  --production-revision <full-before-or-after-SHA> --output /tmp/pf01.json \
  --case energy-2x2 \
  --case energy-8x8:qfunc:inline:logical:success \
  --case shared_square-8x8:qfunc:inline:logical:success \
  --case constant_product-5:qfunc:inline:logical:success \
  --case square-3:qfunc:inline:logical:success \
  --samples 5 --warmups 2 --memory-trials 3 --deadline 30
```

The twenty-two configurations cover the small expression's scalar, native
qfunc and Qmod entrances, helper/exchanged spellings, logical/basis/line targets,
three budget refusals, and larger energy/shared-square, constant product and
square expressions. Each side uses five fresh timing workers per configuration,
two warmups, three separate unprofiled RSS workers and one intrusive diagnostic
worker. E0 includes checked Program delivery. E1 includes E0, first `fq.run` and
named readout where bounded state simulation is permitted. Larger expressions
use independent bounded basis/nonzero-XOR/cleanup interpretation; they do not
claim a simulated large state. GC stays enabled, worker threads are one and all
workers must be reaped. OS sampled RSS and lifetime high-water marks retain their
distinct scopes from PF-00. Intrusive spans/call counts are mechanism evidence,
not natural stage-time percentages.

Both cohorts use the same Apple M5/Python 3.12.14/Torch 2.13.0 host and serial
configuration order. This is development evidence on a nonexclusive host;
frequency/thermal state is unknown and sequential cohorts are not randomized
paired trials. No cross-host, hardware, scalability or release claim follows.
The original AO/PF-00 artifacts, oracle and measurement implementations are
unchanged. New receipts separately pin each production tree. The unchanged
driver's individual receipts retain its P0-observation label; the new comparison
is the place where a scoped before/after conclusion is made.

## Results and acceptance

The paired measurement and broader validation are in progress. Numerical
conclusions will be filled from validated raw receipts before submitting this PR.

The benefit review line is a median reduction of at least 10% and 20 ms for
E0/E1, or at least 10% and 16 MiB for sampled compile-window RSS. Smaller changes
are reported without claiming that gate. Missing/failed primary samples must
not disappear into a successful-subset median. A material regression needs
separate review. CI does not assert noisy wall-clock thresholds.

PF-01 here is limited to Services-local conversion reuse. Remaining conversion
work in Runtime/Core/Compiler needs its own owner-scoped change if the residual
profile warrants one. PF-02 requires a separate invalidation/trust-boundary
argument, and Rust remains conditional on Python-first evidence.
