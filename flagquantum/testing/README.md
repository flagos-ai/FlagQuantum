# Testing

This package contains reusable correctness, conformance, watchdog, and evidence
validation helpers. It lets repository tests and extension authors apply the
same acceptance rules without copying them into each test suite.

## Ownership

Testing owns:

- deterministic certification cases and reproducible failure artifacts;
- versioned tolerance policies;
- watchdog progress records and stall diagnosis;
- validation of MPS numerical, capacity, stability, communication, scaling,
  and portability evidence.

Testing does not execute production workloads, select devices, schedule jobs,
implement simulator mathematics, or define capability claims. Production code
must not import this package to make an execution decision.

## Ten-minute path

Run one real local certification case:

```python
from flagquantum.testing import certification_matrix, execute_certification_case

case = next(
    item
    for item in certification_matrix()
    if item.backend == "pytorch" and item.operator == "x"
)
result = execute_certification_case(case)
assert result.executed and result.passed
```

Then run the scenario test that demonstrates the same workflow:

```bash
python -m pytest tests/unit/test_correctness_properties.py -q
```

For MPS evidence validation, start with the matching
`mps_*_certification.py` module and its `tests/unit/test_mps_*_certification.py`
scenario.
Keep schema-specific validation beside its existing validator; do not add a
generic certification manager or a second representation of execution evidence.

`_finite.py` holds the one rule every validator shares, because it is a property
of the transport rather than of any schema: artifacts arrive as JSON, which
carries `NaN` and `Infinity`, and `NaN` compares false against every bound. Any
number a gate compares against passes through it first. Schema rules stay in
their own validator.

## Changing this domain

When adding a rule, include one passing artifact and one focused injected-fault
test. Keep numerical reference calculations in Simulation and execution or
evidence production in Runtime/Benchmarking. Changes to public exports or an
accepted evidence schema require the repository's API and contract review
process.

## Component example metadata

`component_examples.py` owns lazy example factories and documentation metadata.
It records the inferred name, declared return class, original docstring and
optional resource generalizers. `ComponentDocumentation` captures an iterable
of examples once and selects the first graph example by default. An explicit
`call_graph_example=None` omits the graph. This metadata never decides execution
semantics and does not replace the Compiler or Drawer.

Run `python -m examples.algorithms.component_example_catalog` to create a
documented three-bit swap and verify its nine CX gates and complete complex
state through `circuit.append(op)`. Run its source-differential checks with
`python -m pytest tests/unit/test_component_examples.py -q`.

## Register example checks

`example_checks.py` owns the existing five register-example checks and their
status records. `RegisterExample` explicitly supplies its program factory,
physical lowerer, independent instruction counts and optional register-type
edges. Tools and tests consume the same `check_example` function. Pytest
skip/failure policy and classical Cartesian assertions remain in
`tests/unit/register_example_checks.py`, which re-exports the shared records for
its existing callers. Production execution never depends on these results.

Start with `tests/unit/test_reference_testing_file.py::parallel_example` and
`lower_fixture`, then run all five `CHECKS` through `check_example`. The report
consumer in `tests/unit/test_component_report.py` demonstrates this workflow
without importing pytest into the shared implementation.

## Notebook execution

`notebook_execution.execute_notebook_node` owns the existing optional Jupyter
execution path shared by SDK notebook helpers and documentation exports. It
returns the executed NotebookNode and conversion resources. Its optional file
lock surrounds actual kernel/client startup, and the Jupyter setup context owns
shutdown on success and failure. Numerical execution remains with the ordinary
FlagQuantum circuit backend. Run the real complex-state append example in
`tests/unit/test_notebook_execution.py` with Jupyter installed.

## Physical circuit fuzzing

`_circuit_fuzzer.py` composes the existing Compiler builder and Simulation packed
basis executor for callback-driven conformance checks. Input and expected-output
samples contain Core immutable fixed-width integers. Assign an updated value
back to `sample[name]`. `old(name)` retains the input, and expected phases use
Compiler `PhaseAngle128`. Context management or explicit `close()` owns the
unused-sampler warning. The helper stays private and is independent of production
execution decisions.

Run `tests/unit/test_circuit_fuzzer.py` for a three-bit increment, seeded callback
sampling, measurement feedback and first-failure diagnostics. Full source-file
closure, native ownership adaptations and resource evidence are recorded in
`docs/development/CIRCUIT_FUZZER_AUDIT.md`.

`_physical_operand_cases` supplies frozen-source scalar, array and broadcast
families plus 64-lane truth-table masks. It reuses Compiler operand identities
and the existing physical numerical session. It adds no production gate logic.
Run `tests/unit/test_physical_operand_cases.py` for full family ordering and
source mask read/write comparisons. `PHYSICAL_TESTING_HELPERS_AUDIT.md` records
the separate original register-fuzz batching defect and its pending replacement.
