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

For a generated circuit, call
`compare_component_resources(model, circuit_ir)` from
`flagquantum.compiler._component_trace.resource_comparison`. Its JSON report
separates the model from the emitted instructions. The IR section counts each
instruction once, including conditional instructions, without resolving a
measurement history. It reports the declared width, the number of qubits
touched by instructions, and output requests separately.

For example, continue with the preparation above:

```python
from flagquantum.compiler._component_trace.resource_comparison import (
    compare_component_resources,
)
from flagquantum.core.ir import CircuitIR

work, slots = preparation.requirements(4)
ir = CircuitIR(
    12 + work,
    preparation.emit(
        tuple(range(4)), tuple(range(4, 12)),
        tuple(range(12, 12 + work)), tuple(range(slots)),
    ),
)
report = compare_component_resources(
    sparse_rotation_preparation_cost(preparation, target_width=4), ir
)
assert report.emitted.t_count == 100
assert report.emitted.unconditional["measure"] == 25
assert report.emitted.conditional == {"cz": 25}
assert report.emitted.declared_qubits == 28
```

Reusable register models cover concrete unary QROM, Add,
AddIntoPhaseGrad, equality-driven basis permutations, QROAMClean and
SelectSwapQROM with clean or borrowed banks. The QROM model
handles bounded multidimensional axes and shaped target words. Its width
comes from the locked static formula, including the full selection pool
even when a uniform table uses less work. The phase-add model retains
the upstream abstract Toffoli count. The one-bit source Add reports
−3 Clifford gates and one-bit AddIntoPhaseGrad can report a negative
Toffoli count. These values reproduce source formulas for comparison.
The one-bit native Add emits one CX.

The QROAM measurement-erasure wrapper estimates its Toffolis from the
original table extent. The underlying leaf uses the padded extent. For
five rows, two-word forward banks, one control and one-word erasure blocks,
the wrapper estimates three Toffolis and the padded leaf estimates four.
Both report zero measurements in the locked cost declarations. The source
measurement-erasure leaf has no executable decomposition. These counts are
separate from the emitted reverse-lookup circuit.

The model assumes concrete counts and explicit sequential bystanders.
Symbolic aggregation, automatic graph scheduling, and physical
error-correction models remain separate.

- [Qualtran v0.7.0 gate costs](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_bloq_counts.py)
- [Qualtran v0.7.0 logical width](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py)
