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
The module supplies binding collections and traversal for the finite Program
component builder below. Direct numerical graph traversal remains pending.

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


## Finite gate-call expansion

`program_calls.inline_program_calls` expands verified single-block calls in
the existing Core ProgramRecord. Each invocation has its own SSA mapping. An
explicit stack handles deep finite call chains, and parameter/measurement
feedback bindings follow that same mapping. DNF conditions are canonicalized
after substitution, including classical aliases. Core independently validates
the result and attests any retained public-circuit payload.

The focused path is `tests/unit/test_program_call_lowering.py`. Its existing
Add/QROM append consumer is reused with hierarchical emitter replacement,
checking unchanged source gate dependencies, all four measurement histories,
complex amplitudes, workspace cleanup and separate primitive/model resources.
Workspace remains explicit in function inputs and outputs.

## Typed register expansion

`register_programs.lower_register_program` validates the complete Core module,
flattens declared words in port/row-major/MSB order and reuses the finite call
inliner above. Owned zero-width words have no wires after verification.
Split/join preserve ordered tuples and emit no gates. A reordered return remains
explicit, and Core payload restoration refuses a hidden output permutation.
Gate-only `inline_program_calls` refuses unlowered register functions.

Run `pytest tests/unit/test_register_program_lowering.py` for the shortest
contributor path. The existing Add/QROM append consumer runs unchanged with
three-level packed hierarchy replacement, primitive equality, all measurement
histories and independent resource checks. Finite mutable construction and
rebindings are implemented below. Automatic clean/dirty allocation and direct
numerical graph traversal retain separate acceptance work.

## Finite register construction

`component_builder.ComponentProgramBuilder` receives an ordered Signature and
real immutable typed function definitions. `inputs`, `call_dict`, `call_tuple`
and `call` preserve named/ordered port collection conventions. The shared
binding collection helper allocates one fresh SSA value per element. The shared
binding visitor preserves callback-before-dtype checks and consumption prefixes.
Calls and returns are recorded directly as existing Program records. Core
verifies every function body at finalization, including retained failed calls.
A failed zero-port call can retain a valid empty instance, matching the source.

Run `pytest tests/unit/test_component_program_builder.py` for identity, arrays,
zero-width ownership, failed calls/returns and the unchanged real Add/QROM append
consumer. These retain exact primitive instructions, all measurement histories,
workspace cleanup and separately reported models. Only the concrete quantum
bit/opaque/unsigned profile is admitted. Automatic clean or dirty workspace,
additional descriptors and numerical traversal remain separate.


## Builder word and bit rebindings

`component_rebindings.py` supplies real typed helper functions consumed by the
same builder's `split` and `join` methods. Each logical bit retains an owned word
handle. Helpers convert those words through the existing gate SSA rebindings.
The extra logical split/join operations lower to zero physical primitives and
zero workspace. These helpers carry executable bodies, not signature stubs.

Run `pytest tests/unit/test_component_rebindings.py`. Fifteen original ownership
and failed-join scenarios accept the native observation adapter. Four remaining
source scenarios compare connection order and source model cost against native
zero-primitive lowering. Complex spectator/permutation checks include a real X
call. Existing Add/QROM consumers retain all measurement histories and resources
after explicit split/join around every returned word, including width zero.

## Explicit workspace construction

`layout/component_workspace.py` supplies real typed allocation and release
function bodies. `ComponentProgramBuilder.allocate` and `free` use ordinary
finite calls and existing per-element SSA ownership. Explicit dtype overrides
`n`, zero-width scalar words retain ownership, and stale free calls reserve an
instance before failing. Both clean and dirty flags survive in the leaf record.
The admitted profile requires concrete quantum bit/opaque/uint descriptors and
built-in boolean flags. Source metadata construction has a wider domain.

Run `pytest tests/unit/test_component_workspace.py tests/unit/test_program_workspace.py`.
Sixteen unchanged source builder scenarios accept the native observation adapter.
The tests also verify real helper bodies, dtype changes after split/join, call
forwarding, source resource models and independent Core rejection.

`lower_register_program` rejects allocation/release anywhere in the module until
physical scheduling is admitted. The existing `layout/workspace.py` pool remains
the physical lease authority for that next slice. Graph acceptance establishes
typed consumption. Clean zero-state and dirty original-joint-state restoration
need separate numerical evidence. Source tensor/classical allocation uses zero
even for dirty allocation, while its Cirq path borrows unknown state. Keep those
entry points distinct when extending execution.
