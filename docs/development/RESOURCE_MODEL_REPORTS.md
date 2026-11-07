# Logical resource model reports

`flagquantum.compiler.resource_reporting.resource_model_report` reuses the
existing recursive component estimator, symbolic formulas and magic-state budget.
It does not construct or execute a circuit. This reporting adapter is native
FlagQuantum code; the underlying model modules retain their pinned upstream
copyright, provenance and known boundaries.

```python
import json
from flagquantum.compiler.resource_reporting import resource_model_report
from flagquantum.compiler.registers.symbolic_resource_model import source_add_resource_formula

report = resource_model_report(source_add_resource_formula(8, 8))
print(json.dumps(report, indent=2))
assert report["t_equivalents"] == 28
assert report["peak_logical_qubits"] == 23
assert report["emitted_ir"] is None
```

The executable user example uses the same function:

```sh
python -m examples.algorithms.ftqc_resource_estimation --add 8 8 --json
python -m examples.algorithms.ftqc_resource_estimation --qrom 1000000 8 --json
python -m examples.algorithms.ftqc_resource_estimation --thc 2 4 8 1 --json
```

Symbolic dimensions remain expression strings in JSON, with an explicit encoding
label. The in-memory formula remains available for substitution; JSON is a report,
not an executable symbolic-expression interchange format. No source formula is
rounded, repaired, or inferred from an emitted circuit. A concrete `ComponentCost`
recursively sums gates and success probabilities, while sequential child calls
contribute their maximum live width, including declared bystanders.

Pass an already emitted `CircuitIR` with `emitted_ir=...` to add a separate
instruction-count column. User circuit construction still uses `circuit.append(op)`.
Conditional instructions are counted once syntactically. Neither this column nor
the model is a measurement of an executed branch, elapsed time or hardware memory.
Pairing a circuit and model does not prove their semantic equivalence.

Raw And, Toffoli, controlled swap, Clifford, rotation, measurement and T counts
remain separate. The report states the existing model conversion explicitly:
11 T per arbitrary rotation and 4 T equivalents per CCZ-like resource. These are
budget assumptions, not a claim that arbitrary-angle synthesis emitted 11 gates.
Physical error-correction scenarios remain owned by `qec/resources/estimation.py`.

Acceptance: 12 new tests compare concrete and symbolic Add reports to unmodified
Qualtran v0.7.0 at bcb17bb25360ed68ce3bc084645db3e929799389, preserve the one-bit
source anomaly and both QEC profiles, check recursive lifetimes and success, and
keep dynamic emitted counts independent. Run with the pinned reference environment:

```sh
pytest -q tests/unit/test_resource_model_report.py
```
