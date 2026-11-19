"""Native negation retains literal source calls, exact adjoints and predicates."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.constant_add_resource_model import (
    source_constant_add_resource_formula,
    source_controlled_constant_add_resource_formula,
)
from flagquantum.compiler.registers.arithmetic.negation_component import (
    controlled_negation_program,
    negation_program,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterType
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_example_checks import (
    CHECKS,
    CheckKind,
    CheckStatus,
    RegisterExample,
    check_example,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_register_bindings import SourceContract
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_typed_addition_component import integer_types

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("control", (None, 0, 1))
def test_source_root_primitives_feedback_and_separate_model(
    kind: str,
    width: int,
    adjoint: bool,
    control: int | None,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.arithmetic.negate import Negate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype, _, source_dtype, _ = integer_types(kind, width, width)
    graph = negation_program(dtype, adjoint=adjoint, control_value=control)
    assert verify_program_record(graph) == ()
    source = Negate(source_dtype)
    if adjoint:
        source = source.adjoint()
    if control is not None:
        source = source.controlled(CtrlSpec(cvs=control))
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    offset = int(control is not None)
    ports = {"x": tuple(range(offset, offset + width))}
    if control is not None:
        ports = {"ctrl": (0,), **ports}
    work = tuple(range(width + offset, 3 * width + offset - 1))
    bound = bind_register_program(graph, tuple(ports.values()), clean=work)
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    if kind == "signed" and width == 1:
        with pytest.raises(ValueError):
            reference_arithmetic_gates(source, ports, work, slots)
        # Encoded native one-bit negation is the identity in both sectors.
        initial = (
            torch.tensor([1, 2j, 3j, 4], dtype=torch.complex128)
            if control is not None
            else torch.tensor([1, 2j], dtype=torch.complex128)
        )
        initial /= initial.norm()
        state = torch.zeros(initial.numel() * 2, dtype=torch.complex128)
        state[::2] = initial
        torch.testing.assert_close(
            forced_branch(state, bound.circuit.instructions, ()),
            state,
            rtol=0,
            atol=1e-12,
        )
        return
    original = reference_arithmetic_gates(source, ports, work, slots)
    assert_source(bound.circuit.instructions, original)
    assert bound.outputs == tuple(ports.values())
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == counts["reset"] == model.measurement == width - 1
    formula = (
        source_constant_add_resource_formula(width, 1, adjoint=adjoint)
        if control is None
        else source_controlled_constant_add_resource_formula(
            width, 1, control_value=control, adjoint=adjoint
        )
    )
    for category in ("t", "toffoli", "cswap", "and_bloq", "rotation", "measurement"):
        assert getattr(model, category) == getattr(formula, category)
    assert model.clifford == formula.clifford + width
    assert (
        get_cost_value(source, QubitCount())
        == formula.logical_qubits
        == 3 * width - 1 + offset
    )


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_full_source_complex_tensor_all_history_pairs_and_forward_involution(
    width: int,
    control: int | None,
    adjoint: bool,
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate

    dtype = RegisterType("uint", width)
    logical, work_count = width + int(control is not None), 2 * width - 1
    inputs = (
        (tuple(range(width)),) if control is None else ((0,), tuple(range(1, logical)))
    )
    work = tuple(range(logical + 1, logical + 1 + work_count))
    forward = bind_register_program(
        negation_program(dtype, adjoint=adjoint, control_value=control),
        inputs,
        clean=work,
    )
    backward = bind_register_program(
        negation_program(dtype, adjoint=not adjoint, control_value=control),
        inputs,
        clean=work,
    )
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 5 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + work_count), dtype=torch.complex128)
    initial[:: 1 << work_count] = amplitudes
    source = Negate(QUInt(width))
    if adjoint:
        source = source.adjoint()
    if control is not None:
        source = source.controlled(CtrlSpec(cvs=control))
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << work_count] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    circuit = CircuitIR.from_json(forward.circuit.to_json())
    for first in product((0, 1), repeat=width - 1):
        result = forced_branch(initial, circuit.instructions, first)
        torch.testing.assert_close(
            result, expected / 2 ** ((width - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(result.reshape(-1, 1 << work_count)[:, 1:]) == 0
        for second in product((0, 1), repeat=width - 1):
            for inverse in (forward, backward):
                recovered = forced_branch(result, inverse.circuit.instructions, second)
                torch.testing.assert_close(
                    recovered, initial / 2 ** (width - 1), rtol=0, atol=3e-12
                )


def grouped_controls(
    case: str,
) -> tuple[tuple[Register, ...], tuple[ControlRegister, ...], Any]:
    from qualtran import CtrlSpec

    np = pytest.importorskip("numpy")
    registers: tuple[Register, ...]
    patterns: tuple[ControlRegister, ...]
    if case == "bits":
        registers = (Register("ctrl", quantum_bit_type(), (2,)),)
        patterns = (ControlRegister(1, (0, 1)),)
    elif case == "words":
        registers = (
            Register("ctrl1", RegisterType("uint", 2), (1,)),
            Register("ctrl2", quantum_bit_type()),
        )
        patterns = (ControlRegister(2, (1,)), ControlRegister(1, (0,)))
    else:
        registers = (Register("ctrl", RegisterType("signed", 2)),)
        patterns = (ControlRegister(2, (-1,), "twos_complement"),)
    converter: Any = SourceContract
    source = CtrlSpec(
        qdtypes=tuple(converter.register(r).dtype for r in registers),
        cvs=tuple(
            np.array(p.values).reshape(r.shape)
            for p, r in zip(patterns, registers, strict=True)
        ),
    )
    return registers, patterns, source


@pytest.mark.parametrize("case", ("bits", "words", "signed"))
@pytest.mark.parametrize("adjoint", (False, True))
def test_multiple_register_control_root_feedback_all_raw_branches_and_clean_work(
    case: str, adjoint: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.negate import Negate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    registers, patterns, spec = grouped_controls(case)
    source = Negate(QUInt(2))
    if adjoint:
        source = source.adjoint()
    source = source.controlled(spec)
    graph = controlled_negation_program(
        RegisterType("uint", 2), registers, patterns, adjoint=adjoint
    )
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    ports: dict[str, tuple[int, ...]] = {}
    inputs: list[tuple[int, ...]] = []
    offset = 0
    for reg, pattern in zip(registers, patterns, strict=True):
        flat = tuple(range(offset, offset + len(pattern.bits())))
        ports[reg.name] = flat
        inputs.extend(
            tuple(flat[i : i + pattern.width])
            for i in range(0, len(flat), pattern.width)
        )
        offset += len(flat)
    ports["x"] = tuple(range(offset, offset + 2))
    inputs.append(ports["x"])
    logical = offset + 2
    work_count = offset - 1 + 3
    work = tuple(range(logical + 1, logical + 1 + work_count))
    bound = bind_register_program(graph, tuple(inputs), clean=work)
    backward = bind_register_program(
        controlled_negation_program(
            RegisterType("uint", 2), registers, patterns, adjoint=not adjoint
        ),
        tuple(inputs),
        clean=work,
    )
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(source, ports, work, slots),
    )
    count = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert count["t"] + count["tdg"] == model.total_t_count() == 4 * offset
    assert count["measure"] == count["reset"] == model.measurement == offset
    assert get_cost_value(source, QubitCount()) == logical + work_count
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + work_count), dtype=torch.complex128)
    initial[:: 1 << work_count] = amplitudes
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << work_count] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    for first in product((0, 1), repeat=offset):
        result = forced_branch(initial, bound.circuit.instructions, first)
        torch.testing.assert_close(
            result, expected / 2 ** (offset / 2), rtol=0, atol=4e-12
        )
        assert torch.count_nonzero(result.reshape(-1, 1 << work_count)[:, 1:]) == 0
        for second in product((0, 1), repeat=offset):
            restored = forced_branch(result, backward.circuit.instructions, second)
            torch.testing.assert_close(
                restored, initial / 2**offset, rtol=0, atol=4e-12
            )


@pytest.mark.parametrize("kind", CHECKS)
def test_actual_eight_bit_factory_uses_all_existing_checks(kind: CheckKind) -> None:
    from qualtran.bloqs.arithmetic.negate import _negate

    source = _negate.make()
    work = tuple(range(8, 23))
    original = reference_arithmetic_gates(
        source, {"x": tuple(range(8))}, work, dict(zip(work[8:], range(7), strict=True))
    )

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(record, (tuple(range(8)),), clean=work).circuit

    result = check_example(
        RegisterExample(
            lambda: negation_program(RegisterType("signed", 8)),
            lower,
            lambda: Counter(g.name for g in original),
        ),
        kind,
    )
    assert result.status is CheckStatus.PASS, result.message


def test_flags_and_integer_admission() -> None:
    with pytest.raises(ValueError, match="control"):
        negation_program(RegisterType("uint", 2), control_value=True)
    with pytest.raises(ValueError, match="boolean"):
        negation_program(RegisterType("uint", 2), adjoint=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        negation_program(RegisterType("opaque", 2))


@pytest.mark.integration
def test_grouped_predicate_negation_runs_through_append_and_exact_inverse() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.negate import Negate

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    registers, patterns, spec = grouped_controls("bits")
    dtype = RegisterType("uint", 2)
    inputs, work = ((0,), (1,), (2, 3)), (5, 6, 7, 8)
    forward = bind_register_program(
        controlled_negation_program(dtype, registers, patterns), inputs, clean=work
    )
    backward = bind_register_program(
        controlled_negation_program(dtype, registers, patterns, adjoint=True),
        inputs,
        clean=work,
    )
    raw = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(raw, 5 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    initial[::16] = amplitudes
    matrix = torch.as_tensor(
        Negate(QUInt(2)).controlled(spec).tensor_contract(), dtype=torch.complex128
    ).reshape(16, 16)
    expected = torch.zeros_like(initial)
    expected[::16] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    circuit = DynamicCircuit(9, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(9, forward.circuit.instructions))
    result = run_dynamic(circuit, shots=16, seed=31, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=3e-12
    )
    circuit.append(FieldOperation(9, backward.circuit.instructions))
    restored = run_dynamic(circuit, shots=16, seed=31, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(16, -1),
        initial.expand(16, -1),
        rtol=0,
        atol=3e-12,
    )


@pytest.mark.parametrize("count", (2, 3))
def test_unknown_predicates_preserve_source_conjunction_boundary(count: int) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate
    from qualtran.symbolics import Shaped

    controls = (Register("ctrl", quantum_bit_type(), (count,)),)
    record = controlled_negation_program(RegisterType("uint", 2), controls, None)
    source = Negate(QUInt(2)).controlled(CtrlSpec(cvs=Shaped((count,))))
    compare_layers(
        record, source, {f.name for f in record.functions if f.name != "main"}
    )
    inputs = tuple((wire,) for wire in range(count)) + ((count, count + 1),)
    work = tuple(range(count + 2, count + 2 + 2 * count - 1))
    if count == 2:
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_register_program(record, inputs, clean=work)
        return
    positive = controlled_negation_program(
        RegisterType("uint", 2), controls, (ControlRegister(1, (1,) * count),)
    )
    assert record == positive
    bound = bind_register_program(record, inputs, clean=work)
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    original = reference_arithmetic_gates(
        source, {"ctrl": tuple(range(count)), "x": (count, count + 1)}, work, slots
    )
    assert_source(bound.circuit.instructions, original)
