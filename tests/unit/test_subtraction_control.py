"""Scalar subtraction controls retain source calls, phases and clean work."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.arithmetic.subtraction_control import (
    controlled_subtraction_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_typed_addition_component import integer_types

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 3), (2, 3), (3, 5), (5, 8)))
@pytest.mark.parametrize("cv", (0, 1))
@pytest.mark.parametrize("reverse", (False, True))
def test_complete_controlled_calls_operands_feedback_and_resources(
    kind: str, left: int, right: int, cv: int, reverse: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    if reverse:
        left = right
    a, b, sa, sb = integer_types(kind, left, right)
    if reverse:
        b = a
    graph = controlled_subtraction_program(
        a, b, subtract_from=reverse, control_value=cv
    )
    source = (SubtractFrom(sa) if reverse else Subtract(sa, sb)).controlled(
        CtrlSpec(cvs=cv)
    )
    compare_layers(graph, source, atomic_leaves(graph))
    logical = 1 + left + right
    ports = {
        "ctrl": (0,),
        "a": tuple(range(1, 1 + left)),
        "b": tuple(range(1 + left, logical)),
    }
    work = tuple(
        range(logical, logical + 2 * right - 1 + (0 if reverse else right - left))
    )
    bound = bind_register_program(graph, tuple(ports.values()), clean=work)
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    reference = reference_arithmetic_gates(
        source.decompose_bloq().flatten(),
        ports,
        work,
        slots,
        preserve_components=True,
        keep_exchange_leaves=True,
    )
    assert_source(bound.circuit.instructions, reference)
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in reference
    )
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["measure"] == counts["reset"] == model.measurement == 2 * right - 1
    # Atomic Toffoli is charged separately from the measured primitive T stream.
    assert counts["t"] + counts["tdg"] + 4 * counts["ccx"] == model.total_t_count()
    assert bound.outputs == tuple(ports.values())


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3)))
@pytest.mark.parametrize("cv", (0, 1))
@pytest.mark.parametrize("reverse", (False, True))
def test_appended_full_complex_tensor_all_raw_histories_and_clean_work(
    kind: str, left: int, right: int, cv: int, reverse: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    if reverse:
        left = right
    a, b, sa, sb = integer_types(kind, left, right)
    source = (SubtractFrom(sa) if reverse else Subtract(sa, sb)).controlled(
        CtrlSpec(cvs=cv)
    )
    graph = controlled_subtraction_program(
        a, a if reverse else b, subtract_from=reverse, control_value=cv
    )
    logical = 1 + left + right
    scratch = 2 * right - 1 + (0 if reverse else right - left)
    ports = ((0,), tuple(range(1, 1 + left)), tuple(range(1 + left, logical)))
    bound = bind_register_program(
        graph, ports, clean=tuple(range(logical + 1, logical + 1 + scratch))
    )
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 5 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + scratch), dtype=torch.complex128)
    initial[:: 1 << scratch] = amplitudes
    expected = torch.zeros_like(initial)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected[:: 1 << scratch] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    circuit = DynamicCircuit(
        logical + 1 + scratch, inputs=initial, dtype=torch.complex128
    )
    circuit.append(FieldOperation(logical + 1 + scratch, bound.circuit.instructions))
    measurements = 2 * right - 1
    for history in product((0, 1), repeat=measurements):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, expected / 2 ** (measurements / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(branch.reshape(-1, 1 << scratch)[:, 1:]) == 0


@pytest.mark.parametrize("cv", (-1, 2, True, 0.5))
def test_control_value_validation(cv: int) -> None:
    with pytest.raises(ValueError, match="zero or one"):
        controlled_subtraction_program(unsigned_register_type(2), control_value=cv)


def test_subtract_from_requires_equal_descriptors() -> None:
    with pytest.raises(ValueError, match="equal operand"):
        controlled_subtraction_program(
            unsigned_register_type(2), unsigned_register_type(3), subtract_from=True
        )


@pytest.mark.parametrize("reverse", (False, True))
def test_controlled_entry_reuses_original_integer_family_validation(
    reverse: bool,
) -> None:
    from flagquantum.core.ir.program.register_types import (
        bounded_unsigned_register_type,
    )

    with pytest.raises(ValueError, match="quantum signed"):
        controlled_subtraction_program(
            bounded_unsigned_register_type(3, 5), subtract_from=reverse
        )
