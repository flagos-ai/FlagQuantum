"""Native encoded constants retain source loading, Always and measured Add."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticEncoding
from flagquantum.compiler.registers.arithmetic.constant_add_resource_model import (
    source_constant_add_resource_formula,
    source_controlled_constant_add_resource_formula,
)
from flagquantum.compiler.registers.arithmetic.constant_addition_component import (
    constant_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide, RegisterType
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


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("value", (0, 1, -1))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("control", (None, 0, 1))
def test_source_calls_operands_feedback_and_separate_resource_counts(
    kind: str,
    width: int,
    value: int,
    adjoint: bool,
    control: int | None,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.arithmetic.addition import AddK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype, _, original_dtype, _ = integer_types(kind, width, width)
    # Signed width-one cannot encode +1. Caller encoding remains explicit.
    value = -1 if kind == "signed" and width == 1 and value == 1 else value
    graph = constant_addition_program(
        dtype, value % (1 << width), adjoint=adjoint, control_value=control
    )
    assert verify_program_record(graph) == ()
    source = AddK(original_dtype, value)
    if adjoint:
        source = source.adjoint()
    if control is not None:
        source = source.controlled(CtrlSpec(cvs=control))
    terminals = {
        f.name
        for f in graph.functions
        if f.name == "constant_add"
        or f.name.endswith(("bitwise_x", "bitwise_controlled_x", "__fq_predicate_0_x"))
        or any(r.side != RegisterSide.THRU for r in f.registers or ())
    }
    compare_layers(graph, source, terminals)
    offset = int(control is not None)
    ports = {"x": tuple(reversed(range(offset, width + offset)))}
    if control is not None:
        ports = {"ctrl": (0,), **ports}
    work = tuple(range(width + offset, 3 * width + offset - 1))
    bound = bind_register_program(graph, tuple(ports.values()), clean=work)
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    original = reference_arithmetic_gates(source, ports, work, slots)
    assert_source(bound.circuit.instructions, original)
    assert bound.outputs == tuple(ports.values())
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == counts["reset"] == model.measurement == width - 1
    encodings: dict[str, ArithmeticEncoding] = {
        "unsigned": "unsigned",
        "signed": "twos_complement",
        "montgomery": "montgomery",
    }
    formula = (
        source_constant_add_resource_formula(
            width, value, encoding=encodings[kind], adjoint=adjoint
        )
        if control is None
        else source_controlled_constant_add_resource_formula(
            width,
            value,
            encoding=encodings[kind],
            adjoint=adjoint,
            control_value=control,
        )
    )
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(formula, category) == getattr(model, category)
    assert (
        formula.logical_qubits
        == get_cost_value(source, QubitCount())
        == 3 * width + offset - 1
    )
    events = Counter(event.kind for event in bound.events)
    assert events["allocate"] == events["free"] == width


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_source_complex_tensor_all_raw_histories_and_exact_inverse(
    kind: str,
    width: int,
    control: int | None,
    adjoint: bool,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.arithmetic.addition import AddK

    dtype, _, original_dtype, _ = integer_types(kind, width, width)
    logical = width + int(control is not None)
    work_width = 2 * width - 1
    ports = (
        (tuple(range(width)),) if control is None else ((0,), tuple(range(1, logical)))
    )
    work = tuple(range(logical + 1, logical + 1 + work_width))
    forward = bind_register_program(
        constant_addition_program(
            dtype, (1 << width) - 1, adjoint=adjoint, control_value=control
        ),
        ports,
        clean=work,
    )
    backward = bind_register_program(
        constant_addition_program(
            dtype, (1 << width) - 1, adjoint=not adjoint, control_value=control
        ),
        ports,
        clean=work,
    )
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + work_width), dtype=torch.complex128)
    initial[:: 1 << work_width] = amplitudes
    source = AddK(original_dtype, -1)
    if adjoint:
        source = source.adjoint()
    if control is not None:
        source = source.controlled(CtrlSpec(cvs=control))
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << work_width] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    restored = CircuitIR.from_json(forward.circuit.to_json())
    for first in product((0, 1), repeat=width - 1):
        actual = forced_branch(initial, restored.instructions, first)
        torch.testing.assert_close(
            actual, expected / 2 ** ((width - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << work_width)[:, 1:]) == 0
        for second in product((0, 1), repeat=width - 1):
            recovered = forced_branch(actual, backward.circuit.instructions, second)
            torch.testing.assert_close(
                recovered, initial / 2 ** (width - 1), rtol=0, atol=3e-12
            )


@pytest.mark.parametrize("large", (False, True))
@pytest.mark.parametrize("kind", CHECKS)
def test_actual_concrete_factories_all_existing_checks(
    large: bool, kind: CheckKind
) -> None:
    from qualtran.bloqs.arithmetic.addition import _add_k_large, _add_k_small

    source = (_add_k_large if large else _add_k_small).make()
    width, value = (64, -23) if large else (4, 2)
    dtype = RegisterType("signed" if large else "uint", width)
    work = tuple(range(width, 3 * width - 1))
    original = reference_arithmetic_gates(
        source,
        {"x": tuple(range(width))},
        work,
        dict(zip(work[width:], range(width - 1), strict=True)),
    )

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(record, (tuple(range(width)),), clean=work).circuit

    result = check_example(
        RegisterExample(
            lambda: constant_addition_program(dtype, value % (1 << width)),
            lower,
            lambda: Counter(g.name for g in original),
        ),
        kind,
    )
    assert result.status is CheckStatus.PASS, result.message


def test_encoded_pattern_and_control_admission() -> None:
    dtype = RegisterType("uint", 3)
    for value in (-1, 8):
        with pytest.raises(ValueError):
            constant_addition_program(dtype, value)
    with pytest.raises(ValueError, match="control"):
        constant_addition_program(dtype, 1, control_value=True)
    with pytest.raises(ValueError, match="boolean"):
        constant_addition_program(dtype, 1, adjoint=1)  # type: ignore[arg-type]
