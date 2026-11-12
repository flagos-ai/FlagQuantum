# Explicit factorized addition tensor composition

## Ownership and interface

Simulation owns binary bit-wiring nodes in `register_identity.py`. They connect
caller-declared output/input bit groups in signature order (row-major words,
MSB-first bits), require equal bit counts and distinct labels, and create
no physical workspace. Existing through-identity placeholders reuse this owner.

`examples/algorithms/factorized_addition_tensors.py` composes existing native
suppliers with `ComponentTensorProviders` for the existing factorized Add graph.
The example explicitly binds CNOT, AND, its logical conjugate transpose, supplied
basis vectors and the graph's split/join leaves. It does not add a registry,
Compiler transform or general automatic provider discovery. Missing numerical
recipes fail through the existing exact-name mapping. Compiler still selects and
orders calls; Simulation still wires and contracts them. Physical measured
AND erasure and its branch normalization remain unchanged.

The Add graph's work belongs to Compiler's existing predicate lifecycle. The
logical adjoint tensor is its valid-work projection, not a replacement physical
measurement algorithm. Dense matrices are enumerated only for two-/three-bit
primitives; word wiring and integer vectors stay factorized.

## Executable composition

```python
import torch

from examples.algorithms.factorized_addition_tensors import addition_tensor_providers
from flagquantum.compiler.registers.arithmetic.addition_component import addition_component_program
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import expand_components_for_tensor
from flagquantum.simulation.tensor_network.component_dense import contract_component_tensors

record = addition_component_program(2)
like = torch.empty((), dtype=torch.complex128)
providers = addition_tensor_providers(record, {}, like)
selected, _ = expand_components_for_tensor(
    record, providers.has_custom_tensor, full_flatten=False
)
order = tuple(position for position, _, _ in iter_component_calls(selected))
matrix = contract_component_tensors(selected, order, providers.supply, like)
assert matrix.shape == (16, 16)
```

For flanked graphs, pass the exact basis function/value/direction mapping to
`addition_tensor_providers`; generated names are never interpreted as numerical
values. Word-rebinding definitions must equal the existing Compiler helpers,
including their SSA ordering. A permuted return cannot silently become identity.
These definitions retain arbitrary physical binding and the existing
`circuit.append(op)` path; no Compiler code or protected API changes are needed.

## Evidence and limits

The previous test-only Add dispatcher is removed. Existing complete small Add
matrices, complex states entangled with a spectator, exhaustive small forward
measurement branches, zero workspace, physical source gate/feedback/resource
comparisons and append trajectories now consume the explicit recipe mapping.
Native 5-/32-bit notebook closed cases retain correct overlap one and incorrect
overlap zero, with input tensors at most 32 elements and one unsliced existing
cotengra path. This does not certify built-in greedy planner speed parity.

New Split/Join checks compare node counts, per-node data, exact relabeled axes
and complete unchanged-source tensors at widths 1, 2,
3 and 5 for float64/complex128, along with complex gradients. Both implementations use one binary identity per bit for the checked
Split/Join cases. This does not certify source-wide tensor storage parity. Presentation counts, numerical
storage, actual primitive gates and source resource models remain separate.
Provider replacement preserves the existing consumer, phase and gradient;
missing bindings, basis-name collisions and permuted rebindings fail explicitly.
This example does not implement source-wide automatic provider discovery or
close the original notebook parent file. G3 remains unstarted.

Local evidence: 23 new cases; 8514 concentrated regression cases pass with one
existing Graphviz rendering skip; 109 coverage-run and 84 CPU/TN integration
cases pass. Final node-data/axis assertions additionally pass in all 23 new
cases. Strict typing of four files, Black, Ruff, Simulation scope, architecture,
IR and protected API checks pass. Shared wiring has 100% coverage (20 statements,
8 branches); example composition has 96% (38 statements, 8 branches), with its
untyped-function refusal unexercised by that coverage run. Five pinned source
blobs match the fixed inventory. The documented composition executes successfully.
Evidence prefix: `g2-addition-providers-`.

The bounded default smoke/unit run reached about 82% and timed out after 600
seconds. It did not produce a complete result or failure set, so this is not a
full-suite pass or a verified baseline-only failure claim. The prior seven
recorded baseline failures remain unresolved. The timeout was not rerun.

## Read basis recipes from canonical definitions

The explicit Add composition now accepts `vectors=None`. Compiler's
`component_basis_vectors.integer_basis_function_data` reads the unsigned value
and state/effect direction from the existing function definition. It recognizes
only an exact `integer_basis_function` body, including the scalar `val` port,
clean allocation/release, split/join order, X operands, SSA returns and all other
function fields. It compares the complete definition with the constructor's
canonical result. Names do not encode values. Changed phases, dirty lifetimes,
permutations, extra operations and noncanonical SSA are not silently accepted.

Simulation continues to own all numerical vector data and contractions. This
Compiler query neither executes numerical code nor imports Simulation. The
example composes its results with the existing immutable provider mapping;
`vectors={...}` remains an explicit caller-owned choice, and `{}` does not enable
discovery. No process-global registry or new Core metadata is introduced.

```python
from flagquantum.compiler.registers.component_classical_vectors import flank_component_with_classical_vectors

closed = flank_component_with_classical_vectors(
    record, {"a": 1, "b": 2}, {"a": 1, "b": 3}
)
providers = addition_tensor_providers(closed, None, like)
selected, _ = expand_components_for_tensor(
    closed, providers.has_custom_tensor, "classical_vectors", full_flatten=False
)
order = tuple(i for i, _, _ in iter_component_calls(selected, "classical_vectors"))
result = contract_component_tensors(
    selected, order, providers.supply, like, "classical_vectors"
)
assert abs(result.item() - 1) < 2e-12
```

The original native 5-/32-bit closed Add checks now call the real flank factory
and read their recipes directly, with no constructor interception or external
value table. Their correct and incorrect output overlaps remain one and zero.
This closes that concrete composition gap, not source-wide provider discovery:
other primitives still require declared suppliers, equivalent noncanonical
functions are not recognized, and the notebook's open-port presentation and
planner-speed differences remain open. Actual primitive counts, feedback,
workspace and source resource estimates are unchanged. G3 is not entered.

Local acceptance for this extension: 145 new cases; 8,657 concentrated cases
pass with one existing Graphviz rendering skip. Two final refusal cases were
added afterward and pass in the final 264-case coverage run; the concentrated
suite was not repeated for those test-only additions. All integer values at
widths 1 through 5 are compared against unchanged upstream IntState/IntEffect
complex tensors in both directions. Eighteen noncanonical definition mutations,
renaming, explicit-mapping precedence and missing/untyped supplier refusal are
covered. Existing complete Add matrices, entangled amplitudes, measured branches,
zero work, gate/feedback/resource comparisons and append trajectories pass.
The 84-case CPU/TN integration run passes. Strict typing, Black and Ruff pass
all five changed Python files; scope, architecture, internal IR, protected API
and both executable documentation blocks pass. Six fixed source blobs match the
immutable inventory. Evidence prefix: `g2-basis-recipes-`.
