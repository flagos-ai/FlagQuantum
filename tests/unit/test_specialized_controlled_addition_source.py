"""Forward specializations and generic inverse controls follow distinct source paths."""

from collections import Counter
from importlib import import_module
from itertools import product

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    for dependency in ("numpy", "cirq", "kickmix"):
        import_module(dependency)
    from qualtran import DecomposeTypeError, QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.registers.arithmetic.additional_inverse_control_component import (
    additional_inverse_controlled_addition_program,
    controlled_inverse_addition_program,
)
from flagquantum.compiler.registers.arithmetic.controlled_addition_component import (
    controlled_addition_program,
)
from flagquantum.compiler.registers.arithmetic.specialized_controlled_addition_component import (
    specialized_controlled_addition_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import unsigned_register_type
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_symbolic_modular_addition_calls import atomic_views

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_one_bit_generic_inverse_control_complete_operator_and_append() -> None:
    import numpy as np
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch

    record = controlled_inverse_addition_program(unsigned_register_type(1))
    source = Add(QUInt(1)).adjoint().controlled()
    bound = bind_register_program(record, ((0,), (1,), (2,)))
    assert Counter(op.name for op in bound.circuit.instructions) == {"ccx": 1}
    matrix = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    numbers = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(numbers, 7 - numbers).reshape(8, 2)
    amplitudes /= amplitudes.norm()
    initial = amplitudes.reshape(-1)
    expected = (matrix @ amplitudes).reshape(-1)
    torch.testing.assert_close(
        forced_branch(initial, bound.circuit.instructions, ()),
        expected,
        rtol=0,
        atol=2e-12,
    )
    circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(4, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=2, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=2e-12
    )


@pytest.mark.parametrize("a,b", ((1, 1), (1, 2), (3, 4), (4, 4)))
@pytest.mark.parametrize("inner", (None, 0, 1))
def test_generic_inverse_control_retains_every_available_source_layer(
    a: int, b: int, inner: int | None
) -> None:
    left, right = unsigned_register_type(a), unsigned_register_type(b)
    record = (
        controlled_inverse_addition_program(left, right)
        if inner is None
        else additional_inverse_controlled_addition_program(
            left, right, control_value=inner
        )
    )
    source = (
        (
            Add(QUInt(a), QUInt(b))
            if inner is None
            else CAdd(QUInt(a), QUInt(b), cv=inner)
        )
        .adjoint()
        .controlled()
    )
    pairs = compare_layers(record, source, atomic_views(record))
    functions = {function.name: function for function in record.functions}
    declarations = [name for name in pairs if not functions[name].blocks]
    assert bool(declarations) == (b > 1 or inner is not None)
    for name in declarations:
        with pytest.raises(DecomposeTypeError, match="non-thru registers"):
            pairs[name].decompose_bloq()
    if declarations:
        offset = 0
        inputs = []
        for register in record.functions[0].registers or ():
            size = int(register.bitsize)
            inputs.append(tuple(range(offset, offset + size)))
            offset += size
        with pytest.raises(ValueError, match="component bodies"):
            bind_register_program(
                record, tuple(inputs), clean=tuple(range(offset, offset + 20))
            )


@pytest.mark.parametrize(
    "values,a,b", (((1, 0), 1, 1), ((1, 1), 2, 3), ((0, 1, 0), 1, 1))
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_specialized_control_calls_primitives_all_complex_branches_and_append(
    values: tuple[int, ...], a: int, b: int, adjoint: bool
) -> None:
    import numpy as np
    import torch
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    record = specialized_controlled_addition_program(
        unsigned_register_type(a),
        unsigned_register_type(b),
        values=values,
        adjoint=adjoint,
    )
    source = _MultiControlledFromSinglyControlled(
        values, CAdd(QUInt(a), QUInt(b)), "ctrl"
    )
    source = source.adjoint() if adjoint else source
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 3
    terminals = atomic_views(record)
    target = calls[1].callee
    assert target is not None
    assert matches_component_definition(
        record,
        target,
        controlled_addition_program(
            unsigned_register_type(a), unsigned_register_type(b), adjoint=adjoint
        ),
    )
    terminals.add(target)
    compare_layers(record, source, terminals)
    k = len(values)
    logical = k + a + b
    ports = {
        "ctrl": tuple(range(k)),
        "a": tuple(range(k, k + a)),
        "b": tuple(range(k + a, logical)),
    }
    inputs = (*((wire,) for wire in range(k)), ports["a"], ports["b"])
    bound = bind_register_program(
        record, inputs, clean=tuple(range(logical, logical + k - 1 + a + b - 1))
    )
    allocations = tuple(
        w for event in bound.events if event.kind == "allocate" for w in event.wires
    )
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            source,
            ports,
            allocations,
            MeasurementSlots(bound.circuit.instructions),
            reuse_allocations=False,
            preserve_components=True,
        ),
    )
    counts = Counter(op.name for op in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == model.measurement
    assert counts["t"] + counts["tdg"] + 4 * counts["ccx"] == model.total_t_count()
    matrix = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    numbers = torch.arange(1, 2 * (1 << logical) + 1, dtype=torch.float64)
    amplitudes = torch.complex(numbers, 7 - 2 * numbers).reshape(1 << logical, 2)
    amplitudes /= amplitudes.norm()
    physical = bound.circuit.n_wires + 1
    initial = torch.zeros(1 << physical, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    after = matrix @ amplitudes
    for word in range(1 << logical):
        for spectator in (0, 1):
            index = (word << (physical - logical)) | spectator
            initial[index] = amplitudes[word, spectator]
            expected[index] = after[word, spectator]
    for history in product((0, 1), repeat=counts["measure"]):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** (counts["measure"] / 2),
            rtol=0,
            atol=4e-12,
        )
    circuit = DynamicCircuit(physical, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(physical, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=2, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=4e-12
    )
