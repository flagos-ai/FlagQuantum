"""Conditional arithmetic constructor closure retains source controlled complements."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_arithmetic_factories import (
    arithmetic_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import ResolvedArgument
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _resolve
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_typed_addition_component import integer_types

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3), (3, 5), (8, 8)))
@pytest.mark.parametrize("cv", (0, 1))
def test_named_and_direct_decomposition_match_source_gates_feedback_and_model(
    kind: str, left: int, right: int, cv: int
) -> None:
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    _, _, a, b = integer_types(kind, left, right)
    source = ControlledAddOrSubtract(a, b, add_when_ctrl_is_on=bool(cv))
    logical = 1 + left + right
    order = tuple(reversed(range(logical)))
    ports = {"ctrl": order[:1], "a": order[1 : left + 1], "b": order[left + 1 :]}
    work = tuple(range(logical, logical + right - 1))
    for original in (source, source.decompose_bloq()):
        bound = bind_register_program(
            _resolve(original), tuple(ports.values()), clean=work
        )
        slots = {
            g.wires[0]: int(g.metadata["classical_bit"])
            for g in bound.circuit.instructions
            if g.name == "measure"
        }
        expected = reference_arithmetic_gates(
            source, ports, work, slots, cirq_style=True
        )
        assert_source(bound.circuit.instructions, expected)
        assert bound.outputs == tuple(ports.values())
        counts = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(source, QECGatesCost())
        assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (right - 1)
        assert counts["measure"] == counts["reset"] == model.measurement == right - 1
        assert counts["x"] == 2 * cv
        assert get_cost_value(source, QubitCount()) == logical + len(work)


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("cv", (0, 1))
def test_all_raw_branches_full_complex_spectator_clean_work_and_append(
    kind: str, left: int, right: int, cv: int
) -> None:
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )

    _, _, a, b = integer_types(kind, left, right)
    source = ControlledAddOrSubtract(a, b, add_when_ctrl_is_on=bool(cv))
    logical, scratch = 1 + left + right, right - 1
    ports = ((0,), tuple(range(1, left + 1)), tuple(range(left + 1, logical)))
    work = tuple(range(logical + 1, logical + 1 + scratch))
    bound = bind_register_program(_resolve(source), ports, clean=work)
    coeff = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coeff, 7 - 3 * coeff)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + scratch), dtype=torch.complex128)
    initial[:: 1 << scratch] = amplitudes
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << scratch] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    for history in product((0, 1), repeat=scratch):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (scratch / 2), atol=3e-12, rtol=0
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << scratch)[:, 1:]) == 0
    for precision in (torch.complex64, torch.complex128):
        circuit = DynamicCircuit(
            logical + 1 + scratch, inputs=initial.to(precision), dtype=precision
        )
        circuit.append(FieldOperation(circuit.n_qubits, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=4, seed=37, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(4, -1),
            expected.to(precision).expand(4, -1),
            atol=2e-6 if precision == torch.complex64 else 3e-12,
            rtol=0,
        )


def test_defaults_symbolic_examples_and_constructor_rejections() -> None:
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        _ctrl_add_or_sub_signed,
        _ctrl_add_or_sub_signed_symb,
        _ctrl_add_or_sub_unsigned,
    )

    factory = arithmetic_factories(encode_constant=lambda dtype, value: ())[
        "qualtran.bloqs.arithmetic.controlled_add_or_subtract.ControlledAddOrSubtract"
    ]
    dtype = unsigned_register_type(2)
    result = factory("custom", None, {"a_dtype": dtype})
    assert isinstance(result, ProgramRecord)
    assert result.functions[0].registers is not None
    assert (
        factory("custom", result.functions[0].registers, {"a_dtype": dtype}) == result
    )
    for maker in (
        _ctrl_add_or_sub_unsigned,
        _ctrl_add_or_sub_signed,
        _ctrl_add_or_sub_signed_symb,
    ):
        for depth in (0, 1, 2):
            program = _resolve(maker.make(), depth)
            if maker is _ctrl_add_or_sub_signed_symb:
                assert program.functions[0].blocks == ()
    invalid: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"a_dtype": dtype, "extra": 1},
        {"a_dtype": opaque_register_type(2)},
        {"a_dtype": dtype, "b_dtype": "invalid"},
    )
    for args in invalid:
        with pytest.raises(SerializationError):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"a_dtype": dtype})


@pytest.mark.parametrize("polarity", (2, -1, 0.5, 0.0, "", "truthy"))
def test_source_unvalidated_polarity_retains_python_truth_value(
    polarity: int | float | str,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )

    source = ControlledAddOrSubtract(QUInt(2), add_when_ctrl_is_on=polarity)
    normalized = ControlledAddOrSubtract(QUInt(2), add_when_ctrl_is_on=bool(polarity))
    for original in (source, source.decompose_bloq()):
        actual = bind_register_program(
            _resolve(original), ((0,), (1, 2), (3, 4)), clean=(5,)
        )
        expected = bind_register_program(
            _resolve(normalized), ((0,), (1, 2), (3, 4)), clean=(5,)
        )
        assert_source(actual.circuit.instructions, expected.circuit.instructions)
