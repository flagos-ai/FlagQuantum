"""Complete repeated constructor chains preserve phases, feedback and ownership."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_power_factories import power_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_on_each_factories import _catalogue
from tests.unit.test_legacy_word_factories import _library
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KEY = "qualtran.bloqs.basic_gates.power.Power"


def _resolve(source: Any, depth: int = 0) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, depth), 0, {**_catalogue(), **power_factories()}
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("repetitions", (1, 2, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_power_su2_closure_exact_repetition_full_complex_state_and_resources(
    repetitions: int, adjoint: bool
) -> None:
    from qualtran.bloqs.basic_gates import Power, SU2RotationGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = Power(SU2RotationGate(0.2, 0.3, -0.4, 0.6), repetitions)
    if adjoint:
        source = source.adjoint()
    program = _resolve(source)
    assert len(program.functions[0].blocks[0].operations) == repetitions
    assert source.build_call_graph(None) == {source.bloq: repetitions}
    bound = bind_register_program(program, ((0,),), clean=(2,))
    assert bound == bind_register_program(_resolve(source, 1), ((0,),), clean=(2,))
    child = bind_register_program(_resolve(source.bloq), ((0,),), clean=(2,))
    assert bound.circuit.instructions == child.circuit.instructions * repetitions
    assert [event.kind for event in bound.events] == ["allocate", "free"] * repetitions
    assert Counter(get_cost_value(source, QECGatesCost()).asdict()) == Counter(
        {
            key: repetitions * value
            for key, value in get_cost_value(source.bloq, QECGatesCost())
            .asdict()
            .items()
        }
    )
    for precision in (torch.complex64, torch.complex128):
        amplitudes = torch.tensor([1 + 2j, -2j, 3 - 1j, -2 + 1j], dtype=precision)
        amplitudes /= amplitudes.norm()
        initial = torch.zeros(8, dtype=precision)
        initial[::2] = amplitudes
        expected = torch.zeros_like(initial)
        matrix = torch.as_tensor(source.tensor_contract(), dtype=precision)
        expected[::2] = (matrix @ amplitudes.reshape(2, 2)).flatten()
        actual = (
            Circuit(3, inputs=initial, dtype=precision)
            .append(FieldOperation(3, bound.circuit.instructions))
            .state()
            .flatten()
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=4e-6 if precision == torch.complex64 else 4e-12,
        )
        assert not torch.count_nonzero(actual[1::2])


@pytest.mark.parametrize("repetitions", (2, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_power_measured_child_exact_feedback_all_histories_and_zero_work(
    repetitions: int, adjoint: bool
) -> None:
    from qualtran.bloqs.basic_gates import CZPowGate, Power
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = Power(CZPowGate(exponent=-0.3), repetitions)
    if adjoint:
        source = source.adjoint()
    bound = bind_register_program(_resolve(source), ((1,), (0,)), clean=(3,))
    expected_gates: list[Instruction] = []
    for index, (instance, _, _) in enumerate(source.decompose_bloq().iter_bloqsoqs()):
        gates, _ = source_gradient_gates(
            instance.bloq, {"q": (1, 0)}, {"q": (1, 0)}, work=3, bit=index
        )
        expected_gates.extend(gates)
    assert_source(bound.circuit.instructions, tuple(expected_gates))
    actual = Counter(gate.name for gate in bound.circuit.instructions)
    assert actual["measure"] == actual["reset"] == repetitions
    assert actual["t"] + actual["tdg"] == 4 * repetitions
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == repetitions and model.and_bloq == repetitions
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = torch.tensor([complex(i + 1, i - 2) for i in range(8)])
    initial /= initial.norm()
    matrix = torch.as_tensor(
        source.tensor_contract(), dtype=torch.complex128
    ).contiguous()
    expected = torch.kron(matrix, torch.eye(2, dtype=torch.complex128)) @ initial[::2]
    for history in product((0, 1), repeat=repetitions):
        branch = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            branch[::2], expected * 2 ** (-repetitions / 2), rtol=0, atol=4e-12
        )
        assert not torch.count_nonzero(branch[1::2])


def test_symbolic_and_source_numpy_boundaries_and_constructor_validation() -> None:
    from qualtran.bloqs.basic_gates import Power, XGate

    np = pytest.importorskip("numpy")
    for count in (sympy.Symbol("n", positive=True, integer=True), sympy.Integer(2)):
        source = Power(XGate(), count)
        program = _resolve(source)
        with pytest.raises(ValueError, match="Symbolic power"):
            source.decompose_bloq()
        if isinstance(count, sympy.Integer):
            # The original wire codec normalizes a SymPy integer into a Python int.
            from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

            restored = bloqs_from_proto(bloqs_to_proto(source, max_depth=0))[0]
            assert type(restored.power) is int
            assert len(restored.decompose_bloq().bloq_instances) == 2
            assert (
                len(bind_register_program(program, ((0,),)).circuit.instructions) == 2
            )
        else:
            assert not program.functions[0].blocks
            with pytest.raises(ValueError):
                bind_register_program(program, ((0,),))
    numpy_source = Power(XGate(), np.int64(2))
    with pytest.raises(ValueError, match="Symbolic power"):
        numpy_source.decompose_bloq()
    with pytest.raises(ValueError, match="Cannot serialize"):
        _library(numpy_source, 0)
    assert (
        bind_register_program(_resolve(Power(XGate(), True)), ((0,),))
        .circuit.instructions[0]
        .name
        == "x"
    )
    factory = power_factories()[KEY]
    child = _resolve(XGate()).functions[0]
    for args in (
        {},
        {"bloq": 1, "power": 2},
        {"bloq": child, "power": 0},
        {"bloq": child, "power": 1.5},
    ):
        with pytest.raises(SerializationError):
            factory("main", None, args)
    with pytest.raises(SerializationError, match="THRU"):
        factory("main", None, {"bloq": FunctionRecord("child", ()), "power": 2})
    with pytest.raises(SerializationError, match="signature"):
        factory("main", (), {"bloq": child, "power": 2})
    with pytest.raises(ValueError, match="exceeds"):
        factory("main", None, {"bloq": child, "power": 4097})


def test_zero_port_scalar_child_retains_every_execution_occurrence() -> None:
    from qualtran.bloqs.basic_gates import GlobalPhase, Power

    source = Power(GlobalPhase(exponent=0.3), 3)
    program = _resolve(source)
    assert program.functions[0].registers == ()
    assert len(program.functions[0].blocks[0].operations) == 3
    bound = bind_register_program(program, (), clean=(1,))
    assert Counter(gate.name for gate in bound.circuit.instructions) == {
        "x": 6,
        "phase": 6,
    }
    initial = torch.tensor([1 + 1j, 0, -2j, 0], dtype=torch.complex128) / 6**0.5
    actual = (
        Circuit(2, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(2, bound.circuit.instructions))
        .state()
        .flatten()
    )
    expected = initial * torch.as_tensor(
        source.tensor_contract(), dtype=torch.complex128
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=4e-12)
    assert not torch.count_nonzero(actual[1::2])


def test_nested_power_and_word_child_keep_the_same_real_function_closure() -> None:
    from qualtran.bloqs.basic_gates import OnEach, Power, SGate

    source = Power(Power(OnEach(3, SGate()), 2), 3)
    bound = bind_register_program(_resolve(source), ((2, 0, 1),))
    assert Counter(gate.name for gate in bound.circuit.instructions) == {"s": 18}
    assert not bound.events
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    initial = torch.arange(1, 9, dtype=torch.float64).to(torch.complex128)
    initial /= initial.norm()
    actual = (
        Circuit(3, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(3, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, matrix @ initial, rtol=0, atol=4e-12)
