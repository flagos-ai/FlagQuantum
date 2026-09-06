# Algorithms

Algorithms assemble user-facing quantum and hybrid workflows from FlagQuantum
circuits, observables, differentiation, and optimization. This package owns
Hamiltonian helpers, reusable ansatz construction, VQE/QAOA-style workflows,
and staged optimization recipes whose scientific behavior is tested end to
end.

Algorithms do not define circuit or operator semantics, compiler passes,
runtime selection, provider lifecycle, numerical kernels, or benchmark claims.
New workflows should compose supported FlagQuantum APIs and must not introduce
external-framework objects or a second execution path.

## Where to start

- `core.py`: Hamiltonians, ansatz builders, losses, and complete algorithm
  workflows.
- `optimization.py`: reusable classical and quantum-aware optimization stages.
- `__init__.py`: the intentionally small public algorithms surface.

`core.py` accepts existing Circuit, MPS, statevector, and density-matrix inputs,
but dense Pauli-product mathematics lives in `simulation/pauli.py`. Keep new
numerical kernels in Simulation and preserve Algorithms as their workflow
composition layer.

## Ten-minute change path

For a small algorithm change, modify one owning module, add a scenario-level
test that demonstrates convergence or the expected failure, and run:

```bash
python -m pytest tests/unit/test_algorithms_package.py \
  tests/test_hybrid_optimization.py \
  tests/integration/test_cpu_vertical_slice.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Keep defaults deterministic, preserve requested precision and autograd, and
avoid embedding device selection or performance assertions in an algorithm.
Changes to root exports or protected result types require the public API change
process rather than an algorithms-local compatibility wrapper.
