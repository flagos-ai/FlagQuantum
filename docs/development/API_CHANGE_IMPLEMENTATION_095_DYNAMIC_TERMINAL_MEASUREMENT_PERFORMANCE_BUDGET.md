# API Change Implementation 095: terminal-measurement performance budget

## Status

**Measured after the API Change 095 implementation merged in PR #892 on
2026-10-26.**

This closure records only local CPU costs for semantic normalization and result
projection. It does not import a provider SDK, read credentials, access a
network, submit a task, consume quota, or make a QPU performance claim.

## User workflow

Users continue to use the stable `fq.run` entry:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).cx(0, 1)
circuit.measure(1, classical_bit=0)
circuit.measure(0, classical_bit=1)

result = fq.run(circuit, target="quafu:Baihua", shots=1024)
```

The terminal measurements are removed from the submitted unitary prefix, and
the returned counts are projected into the declared classical-bit order. This
path uses Quafu service compilation and does not claim mid-circuit measurement
or feed-forward support.

Maintainers can reproduce the local performance evidence with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run dynamic_terminal_measurement \
  --widths 2 32 128 --count-entries 64 \
  --warmup 5 --iterations 101 \
  --json-output ./dynamic-terminal-measurement.json
```

The command exits successfully only when every measured case passes its budget.
The JSON retains raw timing samples, p50, p95, dispersion, normalization
identity, correctness checks, budget inputs, derived verdicts, and explicit
no-remote-activity metadata.

## Measurement design

Each circuit is built before timing. The measured normalization validates a
full terminal measurement suffix, constructs the static unitary prefix, binds
the reverse qubit-to-classical-bit permutation, and calculates the immutable
normalization identity. The projection measurement applies that permutation to
up to 64 distinct count entries and verifies preservation of register width and
shot total.

The runner measures widths 2, 32, and 128. A two-qubit register has only four
distinct bit strings, so that case uses four count entries; the larger cases use
64. No provider construction or remote facade call occurs in the timed region.

## Budget

The budget is a linear envelope over the work each operation must perform:

```text
normalization p95 limit = 5 ms + width × 0.2 ms
projection p95 limit = 5 ms + width × count_entries × 0.005 ms
```

The envelope is intentionally tolerant of shared-runner scheduling noise while
still rejecting a large fixed regression or unexpectedly superlinear mapping
work. Required CI checks the payload, arithmetic, correctness invariants, and
short local runs; reference timings remain comparison-only evidence.

## Reference observation

The reference macOS arm64 observation used Python 3.12.14, PyTorch 2.14.1,
FlagQuantum 0.4.0, one process, and one PyTorch thread.

| Width | Count entries | Normalize p50 | Normalize p95 | Project p50 | Project p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 33.500 us | 36.084 us | 1.041 us | 1.125 us |
| 32 | 64 | 188.375 us | 315.583 us | 48.625 us | 55.292 us |
| 128 | 64 | 715.375 us | 734.792 us | 143.417 us | 144.375 us |

All cases passed. These values characterize this machine and run only; the
versioned runner and budget arithmetic are the durable regression mechanism.

## Compatibility boundary

- No stable signature, public export, result schema, or serialized IR changes.
- Only full-register terminal `counts` with a bijective mapping are measured.
- `compiler="qsteed"`, partial measurement, sparse classical registers,
  samples, repeated measurement, reset, and conditions remain outside V1.
- The runner is local non-release evidence and cannot support provider latency,
  scalability, throughput, or hardware claims.
- No raw benchmark artifact is committed; users choose an output path.

## Verification

- benchmark runner discovery and CLI contract;
- reverse-permutation correctness and deterministic normalization identity;
- source retention, measurement-suffix removal, shot-total preservation, and
  classical-width preservation;
- derived budget arithmetic and invalid-argument refusal;
- explicit no-provider/no-network metadata; and
- Black, Ruff, strict mypy, documentation, ownership, and diff checks.
