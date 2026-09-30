# Continuous-Time Lindblad Evolution

This domain is the plan-and-run facade for Markovian open-system evolution. It
owns request validation, the sealed `LindbladPlan`, option interpretation, and
the result envelope. It does not own the numerics: the density-matrix
integrator is `flagquantum/simulation/lindblad.py`, which Simulation owns.

`plan` seals a request without running it. `run` accepts either a sealed plan or
the same arguments and returns an `EvolutionResult` whose `plan` field carries
the sealed request. A sealed plan is closed to semantic overrides, so a caller
cannot change a Hamiltonian, a collapse operator, or the time grid after the
plan has been reviewed.

Public entries are `plan`, `run`, `LindbladPlan`, `CollapseOperator`,
`EvolutionResult`, and `amplitude_damping`. The domain depends on Core error and
observable types, `ExecutionOptions`, and the Simulation integrator; `_plan.py`
adds only `hashlib` and `json` for the plan identity.

```python
import flagquantum.lindblad as fql

# A 0.5 * X drive on one qubit, damped from |1> once per unit time.
result = fql.run(
    [{"operator": "X", "wires": [0], "coefficient": 0.5}],
    "1",
    [0.0, 1.0, 2.0],
    collapse_operators=[fql.amplitude_damping(rate=1.0, qubit=0)],
)
print(result.populations)
```

Three boundaries are enforced rather than documented, and a change that widens
one of them is a capability change, not a refactor:

- Execution is CPU-only. A non-CPU `device` is rejected with
  `unsupported_device` instead of being moved to a backend.
- The Hamiltonian is a dense `dim x dim` matrix built once from an `Observable`,
  a sequence of operator-mapping terms, or a matrix. It is not a callable, so an
  arbitrary time-dependent generator is unsupported.
- The integrator is explicit fixed-step fourth-order Runge--Kutta, taking two
  steps between each pair of output times. The output grid therefore sets the
  integration step size and the accuracy, and there is no adaptive tolerance.

Gradients are unsupported, and independent requests are not batched: one call
evolves one state on one grid. Reference the capability registry entry
`continuous_time_lindblad` before describing a support boundary in user-facing
text.

For a typical change, edit the request validation or the result envelope and
verify against `tests/unit/test_lindblad_evolution.py`; a change to the
integrator itself belongs to `flagquantum/simulation/` and its own tests. Run
`examples/lindblad_evolution.py` for the shortest end-to-end path.
