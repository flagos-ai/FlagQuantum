"""Complete subtraction graphs preserve source calls, prefix ownership and phase."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.arithmetic.subtraction_component import (
    subtract_from_program,
    subtraction_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    source_add_resource_formula,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_example_checks import (
    CHECKS,
    CheckKind,
    CheckStatus,
    RegisterExample,
    check_example,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_typed_addition_component import integer_types

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3), (3, 5), (5, 8)))
def test_complete_subtract_root_and_source_gate_feedback(
    kind: str, left: int, right: int
) -> None:
    from qualtran.bloqs.arithmetic.subtraction import Subtract
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, sa, sb = integer_types(kind, left, right)
    graph = subtraction_program(a, b)
    source = Subtract(sa, sb)
    assert verify_program_record(graph) == ()
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    ports = {"a": tuple(range(left)), "b": tuple(range(left, left + right))}
    work = tuple(range(left + right, 3 * right - 1))
    bound = bind_register_program(graph, tuple(ports.values()), clean=work)
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(source, ports, work, slots),
    )
    assert bound.outputs == tuple(ports.values())
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (right - 1)
    assert counts["measure"] == counts["reset"] == model.measurement == right - 1
    assert get_cost_value(source, QubitCount()) == 3 * right - 1
    add_model = source_add_resource_formula(right, right)
    for category in ("t", "toffoli", "cswap", "and_bloq", "rotation", "measurement"):
        assert getattr(model, category) == getattr(add_model, category)
    # Whole-word fanout is one Clifford in the source analytical model.
    assert model.clifford == add_model.clifford + 3 * right + 2 * int(
        kind == "signed" and left < right
    )
    events = Counter(event.kind for event in bound.events)
    assert events["allocate"] == events["free"] == right - 1 + int(left < right)


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3)))
@pytest.mark.parametrize("subtract_from", (False, True))
def test_source_full_tensor_all_histories_and_restoration(
    kind: str, left: int, right: int, subtract_from: bool
) -> None:
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    if subtract_from:
        left = right
    a, b, sa, sb = integer_types(kind, left, right)
    graph = subtract_from_program(a) if subtract_from else subtraction_program(a, b)
    source = SubtractFrom(sa) if subtract_from else Subtract(sa, sb)
    logical, count = left + right, right - 1 + (0 if subtract_from else right - left)
    inputs = (tuple(range(left)), tuple(range(left, logical)))
    work = tuple(range(logical + 1, logical + 1 + count))
    bound = bind_register_program(graph, inputs, clean=work)
    inverse = bind_register_program(
        (
            addition_component_program(left, right, left_dtype=a, right_dtype=b)
            if subtract_from
            else graph
        ),
        inputs,
        clean=work,
    )
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 5 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + count), dtype=torch.complex128)
    initial[:: 1 << count] = amplitudes
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << count] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    for first in product((0, 1), repeat=right - 1):
        result = forced_branch(initial, bound.circuit.instructions, first)
        torch.testing.assert_close(
            result, expected / 2 ** ((right - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(result.reshape(-1, 1 << count)[:, 1:]) == 0
        for second in product((0, 1), repeat=right - 1):
            restored = forced_branch(result, inverse.circuit.instructions, second)
            torch.testing.assert_close(
                restored, initial / 2 ** (right - 1), rtol=0, atol=3e-12
            )


@pytest.mark.parametrize(
    "factory",
    (
        "_sub_small",
        "_sub_large",
        "_sub_diff_size_regs",
        "_sub_from_small",
        "_sub_from_large",
    ),
)
@pytest.mark.parametrize("kind", CHECKS)
def test_all_actual_concrete_factories_existing_check_modes(
    factory: str, kind: CheckKind
) -> None:
    from qualtran.bloqs.arithmetic import subtraction

    source = getattr(subtraction, factory).make()
    left = int(source.signature.get_left("a").bitsize)
    right = int(source.signature.get_left("b").bitsize)
    a, b = RegisterType("signed", left), RegisterType("signed", right)
    is_from = factory.startswith("_sub_from")
    inputs = (tuple(range(left)), tuple(range(left, left + right)))
    work = tuple(range(left + right, 3 * right - 1))
    graph = subtract_from_program(a) if is_from else subtraction_program(a, b)
    bound = bind_register_program(graph, inputs, clean=work)
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    original = reference_arithmetic_gates(
        source, dict(zip(("a", "b"), inputs, strict=True)), work, slots
    )

    assert_source(bound.circuit.instructions, original)
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(record, inputs, clean=work).circuit

    result = check_example(
        RegisterExample(
            lambda: graph, lower, lambda: Counter(g.name for g in original)
        ),
        kind,
    )
    assert result.status is CheckStatus.PASS, result.message


def test_invalid_integer_word_admission() -> None:
    with pytest.raises(ValueError, match="integer"):
        subtraction_program(RegisterType("opaque", 2))
    with pytest.raises(ValueError, match="left <= right"):
        subtraction_program(RegisterType("uint", 3), RegisterType("uint", 2))
