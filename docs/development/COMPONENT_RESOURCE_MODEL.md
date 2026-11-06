# Component resource estimates

The component resource model can estimate an operation before generating its
IR. Each node declares exclusive logical gate counts and its local peak
width. Each child edge declares a call multiplicity and the bystander qubits
live during that call. Counts add across calls. Width takes the maximum
scheduled width, so sequential reuse does not multiply the qubit count.
A leaf can declare a cost without an executable decomposition.

The model retains T, Toffoli, controlled swap, And, Clifford, rotation and
measurement counts separately. The default T-equivalent conversion uses
four T per Toffoli, controlled swap or And, and eleven per generic rotation.
These conversion factors are configurable. The result describes logical
resources and has no hardware timing, physical width or fidelity estimate.

The dense and sparse rotation preparations supply concrete model builders.
This example needs no Qualtran installation or circuit generation.

```python
from flagquantum.algorithms.primitives.rotation_resource_model import (
    sparse_rotation_preparation_cost,
)
from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

preparation = SparseRotationPreparation((2, 1), (1 + 1j, 1 - 1j), 3, 8)
estimate = estimate_component_resources(
    sparse_rotation_preparation_cost(preparation, target_width=4)
)
assert estimate.gates.total_t_count() == 92
assert estimate.gates.measurement == 11
assert estimate.peak_logical_qubits == 22
```

For the same sparse preparation, the emitted native IR has 100 T/T†,
25 measurements and 28 declared qubits (12 data and 16 clean work qubits).
The model uses the original abstract gradient addition. The measured adder
in native IR has a different resource profile. The existing
`estimate_resources` entry point continues to describe static instructions
and basis conversion, with its existing data-dependence rules.

The present model assumes concrete counts and explicit sequential
bystanders. It covers the concrete rotation preparations and leaves
symbolic aggregation, automatic graph scheduling, additional component
models, and physical error-correction models for separate work.

- [Qualtran v0.7.0 gate costs](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_bloq_counts.py)
- [Qualtran v0.7.0 logical width](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py)
