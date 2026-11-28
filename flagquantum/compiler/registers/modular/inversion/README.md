# Modular inverse steps

This package owns arithmetic, ordering, parity, exchange and termination steps of the inverse algorithm.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [inverse_algorithm_component.py](inverse_algorithm_component.py) for
the stored call graph and [inverse_algorithm.py](inverse_algorithm.py) for the
explicit physical emitter. Step owners remain separate by responsibility.
Preserve MSB register order, retained logical history, feedback and actual returned
physical aliases. Shared arithmetic/comparison/conjunction owners supply measured
exact adjoints. The public stored shell lives in
[modular_inverse_component.py](../modular_inverse_component.py).

From the repository root, run the focused scenario tests and normal append example.

```bash
python -m pytest tests/unit/test_inverse_algorithm_component.py tests/unit/test_modular_inverse_component.py -q
python -m examples.algorithms.register_modular_inverse
```

The [register package guide](../../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

The iteration and algorithm `*_component_program` templates accept explicit
step and constant owners; `modular_inverse_component_program` keeps caller-owned
history and inlines the algorithm into the source public wrapper. Fixed widths
with unresolved modulus retain ten iteration calls, `2*n+14` algorithm calls and
`2*n+27` public wrapper calls in both directions. The actual sixth-step refusal
and four symbolic constant definitions remain unbound. Existing finite programs
and first-five-step owners retain their complete definitions. Run
`tests/unit/test_symbolic_kaliski_calls.py` for new parent boundaries, full
reachable closure certificates and source-codec metadata bridges. Numerical
execution still requires all source leaves to be concrete and executable.
