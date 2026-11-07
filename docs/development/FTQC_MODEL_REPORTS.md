# Dimension-only resource reports

The existing component cost graph can estimate resources before constructing a
circuit. This report entry reuses that graph and its symbolic register formulas;
it does not introduce a second estimator or depend on Qualtran at runtime.

From the repository root:

```sh
python -m examples.algorithms.ftqc_resource_estimation --add 8 8 --json
python -m examples.algorithms.ftqc_resource_estimation --qrom 17 8 --qrom-controls 1 --json
python -m examples.algorithms.ftqc_resource_estimation --thc 2 4 8 1 --json
python -m examples.algorithms.ftqc_resource_estimation --single-factorization 10 50 12 --json
```

The JSON gives all logical gate categories, peak logical qubits, separate T and
CCZ magic-state budgets, and T equivalents. The conversion policy is explicit:
11 T states per rotation and four T equivalents per CCZ-like operation. This is
a declared model, not a synthesized primitive count or an executed trajectory.
`emitted_ir` and `execution_measurements` remain null. An explicit dimension-only
model option is required with `--json`, so the command cannot accidentally take
the circuit-building default example path.

For the eight-bit THC example the locked model gives 107 Toffoli, 11 controlled
swaps, one Clifford, zero modeled measurements, 118 CCZ states, 472 T equivalents
and 24 peak logical qubits. These values are not the emitted circuit's T count.
The QROM shape model does not allocate its table and does not assume
content-dependent lookup optimizations. Original source formula anomalies remain
visible; JSON export does not repair or reinterpret the model.

The Python entry is already available at
`flagquantum.qec.resources.estimation.estimate_ftqc_resources`: pass a
`ComponentCost` or an operation with a zero-argument `resource_model()` method.
Use `estimate_symbolic_ftqc_resources` for a `SymbolicResourceFormula`. Neither
requires execution. A caller-paired `emitted_ir` can be supplied separately to
the concrete entry for syntactic instruction counts. These internal integration
entries do not change the protected root API or `circuit.append(op)` workflow.

## Validation

Seven report tests cover JSON-only operation, rejection without dimensions, and
single-factorization report reuse. Five of them compare every nonzero category,
magic-state budget and peak width to the unmodified Qualtran v0.7.0 source at
`bcb17bb25360ed68ce3bc084645db3e929799389`, including a million-row QROM and two
THC erasure configurations. The report, THC, cost-query, concrete-model and
symbolic-register suites pass together: 62 tests, no skips. Strict mypy for the
example, Black, Ruff and architecture checks pass.
