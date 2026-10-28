# Bind-time angle validation on CPU

A compiled `fq.Module` builder is traced, so its gates are constructed from
tracers and the canonical finite-angle rule in `flagquantum/core/ir` defers the
magnitude check: a tracer has no value yet. Nothing re-applied that check when
the reusable circuit was rebound to concrete parameters, so the program
completed with `tensor([nan])` while the same circuit built directly refused with
`IRValidationError`. This artifact measures what restoring the check at the bind
step costs a public forward pass.

Payloads `run-1.json`, `run-2.json` and `run-3.json` were produced by
`benchmarks/module_bind_validation.py` from source commit
`2b5d4646b0fd499e1a1c263819e3d8da524d40e9` with the producer committed. Three
fresh processes, Python 3.12.14, PyTorch 2.14.1, macOS 27.0 arm64, CPU,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.

Reproduce:

```bash
for run in 1 2 3; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python benchmarks/module_bind_validation.py \
    --output artifacts/development/module_bind_validation_cpu_20261027/run-$run.json
done
python -m pytest tests/unit/test_module_bound_angle_validation.py -q
```

## Method

The cost is measured against the same code with the check removed, in the same
process, alternating whole blocks of repeats and comparing medians of block
medians. A single ordered pair of arms put warm-up drift into the difference: the
first version reported `-5.7%` for one shape, which is a measurement of the
process rather than of the check.

The shapes are a depth-1, width-2 ladder (5 gates), a depth-4, width-8 ladder
(92 gates) and a depth-4, width-12 ladder (140 gates). Angles are ordinary
finite values; the check has nothing to refuse in any of them.

## Result

| Shape | Gates | With validation | Without | Overhead | Fraction | Three processes |
| --- | --- | --- | --- | --- | --- | --- |
| 2x1 | 5 | 0.165 ms | 0.162 ms | 0.0030 ms | 2.24% | 2.24%, 3.03%, 1.84% |
| 8x4 | 92 | 1.325 ms | 1.284 ms | 0.0417 ms | 3.12% | 3.25%, 2.76%, 3.12% |
| 12x4 | 140 | 1.832 ms | 1.763 ms | 0.0690 ms | 3.84% | 3.92%, 3.84%, 3.37% |

Cost per gate is 436-482 ns for the two larger shapes and 738 ns for the
smallest, where a fixed per-call cost dominates. These are single-process wall
times on one machine, not a cross-machine or accelerator comparison.

## What was tried and rejected

Three ways of asking the same question were measured on 184 scalar parameters
before choosing:

| Strategy | Cost |
| --- | --- |
| `isfinite().all()` per parameter | 279 µs |
| one `torch.cat` of all values, then one check | 66.5 µs |
| `float(value)` and `math.isfinite` per parameter | 15.2 µs |
| iterating the parameters alone | 1.6 µs |

Per-parameter tensor calls cost about 1.5 µs each on this host, so grouping looks
like the obvious fix and is not: at this size the grouping costs four times more
than the check it replaces, and the `cat` version measured 5.5% overhead in the
end-to-end harness against 3.1% for the scalar path. That is why the shipped
check takes the cheapest form each value allows rather than one uniform form.

## Scope and remaining gaps

- The refusal itself is verified by
  `tests/unit/test_module_bound_angle_validation.py`: 20 tests, 14 of which failed
  before the change, across every mode, the JAX backend, cold and warmed modules,
  the `execute` path, a feature input, a zero and a nonzero Hamiltonian
  coefficient, and an optimizer that must not be stepped through a poisoned
  value.
- The check reuses `flagquantum.core.ir._normalize_angle` for anything that is
  not a real scalar tensor, and every refusal is raised by that rule, so the
  domain and the error contract are Core's. It does not introduce a second
  finite check per simulator.
- Cost scales with the number of instructions, so a circuit with far more gates
  than 140 will pay proportionally more. That was not measured beyond 140.
- Only the locally bound parameter path is covered. A value that arrives through
  a distribution, a noise parameter or a checkpoint is validated where those
  paths already validate, not here.
- Single-process CPU only. No accelerator, distributed, or time-to-target claim,
  and no capability maturity change: this restores behaviour that was documented
  and that the direct-construction path already had.
