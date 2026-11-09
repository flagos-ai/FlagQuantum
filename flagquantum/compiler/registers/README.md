# Register decompositions

This internal Compiler package owns register-level decompositions into native
FlagQuantum IR, compile-time wire binding and component cost declarations.
User workflows continue through `circuit.append(op)`.

## Package ownership

| Package | Responsibility | Entry module |
| --- | --- | --- |
| [primitives/](primitives/README.md) | Boolean/classical/quantum gate dispatch, temporary AND, controlled swaps, measurement cleanup | [and_compute.py](primitives/and_compute.py) |
| [bits/](bits/README.md) | XOR, bitwise transforms, population count, signed encodings and register rotations | [bitwise.py](bits/bitwise.py) |
| [layout/](layout/README.md) | Partition, cast and bit views, clean and borrowed workspace ownership | [partition.py](layout/partition.py) |
| [control/](control/README.md) | Control predicates, multi-control, indexed and controlled component composition | [controlled_component.py](control/controlled_component.py) |
| [arithmetic/](arithmetic/README.md) | Addition, subtraction, integer products, squares and their component costs | [addition.py](arithmetic/addition.py) |
| [modular/](modular/README.md) | Controlled modular arithmetic, multiplication, windowed methods and inverse composition | [modular_addition.py](modular/modular_addition.py) |
| [modular/inversion/](modular/inversion/README.md) | Arithmetic, ordering, parity, exchange and termination steps of the inverse algorithm | [inverse_algorithm.py](modular/inversion/inverse_algorithm.py) |
| [finite_field/](finite_field/README.md) | Polynomial-basis field operations, linear maps, product and inverse chains and erasure | [coefficients.py](finite_field/coefficients.py) |
| [comparison/](comparison/README.md) | Equality and ordering with coherent, measured and mixed operands | [less_than.py](comparison/less_than.py) |
| [lookup/](lookup/README.md) | QROM, QROAM, selected swaps, unary traversal and measured unlookup | [qrom_table.py](lookup/qrom_table.py) |
| [phase/](phase/README.md) | Phase gradients, programmable rotations, phase costs and measured QFT | [phase_gradient.py](phase/phase_gradient.py) |
| [fixed_point/](fixed_point/README.md) | Polynomial approximations, reciprocal and inverse-square-root steps | [fixed_point_polynomial.py](fixed_point/fixed_point_polynomial.py) |
| [sorting/](sorting/README.md) | Compare/exchange networks, permutations, duplicate detection and set combinations | [bitonic_sort.py](sorting/bitonic_sort.py) |
| [preparation/](preparation/README.md) | Uniform and rank-aware preparation and reflections | [uniform_preparation.py](preparation/uniform_preparation.py) |
| [resources/](resources/README.md) | Cross-component symbolic resource formulas consumed by domain cost modules | [symbolic_resource_model.py](resources/symbolic_resource_model.py) |

## Dependency boundaries

- Core owns register types, ordered signatures, symbolic parameter semantics and IR.
- Register components own decomposition, register bit order and workspace cleanup.
- `_component_trace/` owns call attribution, resource traversal and query policy.
- `algorithms/` composes components into algorithm-level operations.
- `lowering/`, `artifacts/` and `formats/` own target transformations and output.
- Runtime organizes execution. Simulation supplies numerical kernels.

Each subpackage initializer contains documentation only. Import a component from
its owning module. Source attribution stays in the implementation headers.

`component_bindings.py` groups Core indexed bindings into scalar/object-array
collections and visits them in declared port order. A caller supplies existing
value identities and owns connection/consumption effects. The shared Core dtype
policy runs after each callback, retaining the pinned source's failure effects.
The module provides the binding prerequisites for the future Program component
builder. General calls, returns and numerical graph traversal remain pending.

## Contributor path

Run an existing lookup consumer and the resource comparison example from the
repository root. These use concrete and symbolic tables and separate model
estimates from emitted instructions.

```bash
python -m examples.algorithms.register_qrom_table
python -m examples.algorithms.resource_model_comparison
python -m pytest tests/unit/test_qrom_table.py tests/unit/test_register_signature.py -q
```

Choose the owning package above, change the component, then run its focused tests
and the applicable source-parity regression. Preserve logical ports, measurement
branches, complex amplitudes and workspace ownership. Public APIs and serialized
IR contracts follow their existing change-proposal process.

## Relocation map

The [complete module map](../../../docs/development/REGISTER_PACKAGE_LAYOUT.md)
records all 191 moves. The internal flat module paths have been replaced at all
checked-in executable consumers. Test filenames retain their existing scenario
and upstream-reference identities.

Internal emitters use functional names. The callable map in the module guide
records 40 renamed functions and their owners. Source schedules, register
contracts and cost formulas retain their existing behavior.
General typed component graph execution remains separate integration work.
