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
`mps_*_certification.py` module and its `tests/unit/test_issue09*.py` scenario.
Keep schema-specific validation beside its existing validator; do not add a
generic certification manager or a second representation of execution evidence.

## Changing this domain

When adding a rule, include one passing artifact and one focused injected-fault
test. Keep numerical reference calculations in Simulation and execution or
evidence production in Runtime/Benchmarking. Changes to public exports or an
accepted evidence schema require the repository's API and contract review
process.
