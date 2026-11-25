"""Stored QVR constructors retain named ports, exact source graphs and boundaries."""

from itertools import product

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.qvr_catalog import QvrExample
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_qvr_factories import qvr_factories
from flagquantum.compiler.registers.phase.qvr_component import (
    qvr_phase_gradient_program,
    qvr_zpow_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "gradient,format_,gamma",
    tuple(
        product(
            (False, True),
            ((1, -1), (1, -2), (2, 0), (2, 2), (3, 2), (3, -1), (3, -3)),
            (0.375, -0.375),
        )
    ),
)
def test_named_fixed_cost_graph_and_source_operator(
    gradient: bool, format_: tuple[int, int], gamma: float
) -> None:
    from qualtran import BloqError, QFxp
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.rotations.quantum_variable_rotation import (
        QvrPhaseGradient,
        QvrZPow,
    )

    width, fraction = format_
    source = (QvrPhaseGradient if gradient else QvrZPow)(
        SourceRegister("energy", QFxp(width, fraction)), gamma, 0.2
    )
    library = _library(source, 0)
    record = resolve_component_program(library, 0, qvr_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == library.entries[0].declaration.registers
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    assert resolve_component_program(_library(source, 1), 0, qvr_factories()) == record
    dtype = fixed_point_register_type(width, fraction)
    accepted = (qvr_phase_gradient_program if gradient else qvr_zpow_program)(
        Register("energy", dtype), gamma, 0.2
    )
    if any(not function.blocks for function in record.functions):
        with pytest.raises((BloqError, ValueError), match="(dtypes|shape)"):
            next(iter(source.decompose_bloq().bloq_instances)).bloq.decompose_bloq()
        with pytest.raises((BloqError, ValueError), match="(dtypes|shape)"):
            _library(source, 2)
        return
    assert resolve_component_program(_library(source, 2), 0, qvr_factories()) == record
    example = QvrExample(
        width, gamma, 0.2, fraction_bits=fraction, via_gradient=gradient
    )
    logical = width + example.gradient_width
    wires = tuple(reversed(range(logical)))
    ports = (wires[:width], wires[width:]) if gradient else (wires,)
    clean = tuple(range(logical, example.qubit_count))
    bound = bind_register_program(record, ports, clean=clean)
    reference = bind_register_program(accepted, ports, clean=clean)
    assert bound.circuit == reference.circuit
    assert bound.events == reference.events
    assert bound.outputs == reference.outputs == ports
    contiguous = (
        (tuple(range(width)), tuple(range(width, logical)))
        if gradient
        else (tuple(range(width)),)
    )
    actual = bind_register_program(record, contiguous, clean=clean)
    torch.testing.assert_close(
        logical_operator(actual.circuit, logical),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )


@pytest.mark.parametrize("gradient", (False, True))
def test_original_examples_and_defaults(gradient: bool) -> None:
    from qualtran.bloqs.rotations.quantum_variable_rotation import (
        _qvr_phase_gradient,
        _qvr_zpow,
    )

    source = (_qvr_phase_gradient if gradient else _qvr_zpow).make()
    record = resolve_component_program(_library(source, 0), 0, qvr_factories())
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    factory = tuple(qvr_factories().values())[gradient]
    cost = Register("cost", fixed_point_register_type(3, 3))
    assert factory("default", None, {"cost_reg": cost}) == factory(
        "default", None, {"cost_reg": cost, "gamma": 1.0, "eps": 1e-9}
    )


@pytest.mark.parametrize("width", (3, sympy.Symbol("n")))
def test_symbolic_gradient_coefficient_retains_original_one_call_root(
    width: int | sympy.Expr,
) -> None:
    from qualtran import QFxp
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient

    source = QvrPhaseGradient(
        SourceRegister("energy", QFxp(width, width)), sympy.Symbol("gamma"), 0.2
    )
    record = resolve_component_program(_library(source, 0), 0, qvr_factories())
    assert verify_program_record(record) == ()
    assert len(record.functions[0].blocks[0].operations) == 1
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    with pytest.raises(ValueError):
        bind_register_program(
            record, ((0, 1, 2), tuple(range(3, 10))), clean=tuple(range(10, 30))
        )


@pytest.mark.parametrize("symbolic_precision", (False, True))
def test_direct_symbolic_index_refusal_remains_a_declaration(
    symbolic_precision: bool,
) -> None:
    from qualtran import QFxp
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow

    source = QvrZPow(
        SourceRegister("cost", QFxp(3, 2)),
        0.375 if symbolic_precision else sympy.Symbol("gamma"),
        sympy.Symbol("eps") if symbolic_precision else 0.2,
    )
    with pytest.raises(IndexError):
        source.decompose_bloq()
    record = resolve_component_program(_library(source, 0), 0, qvr_factories())
    assert not record.functions[0].blocks


@pytest.mark.parametrize("gamma", (0.375, 1e-5))
@pytest.mark.parametrize("width,fraction", ((2, 0), (2, 1), (3, 2), (4, -2)))
def test_signed_source_index_and_physical_storage_boundaries(
    gamma: float, width: int, fraction: int
) -> None:
    from qualtran import QFxp
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = QvrZPow(
        SourceRegister("cost", QFxp(width, fraction, signed=True)), gamma, 0.2
    )
    record = resolve_component_program(_library(source, 0), 0, qvr_factories())
    wires = tuple(reversed(range(width)))
    if gamma == 0.375:
        with pytest.raises(IndexError):
            source.decompose_bloq()
        assert not record.functions[0].blocks
        with pytest.raises(ValueError):
            bind_register_program(record, (wires,))
        return
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    for depth in (1, 2):
        assert (
            resolve_component_program(_library(source, depth), 0, qvr_factories())
            == record
        )
    bound = bind_register_program(record, (wires,))
    assert bound.outputs == (wires,)
    assert bound.events == ()
    assert bound.circuit.n_wires == width
    assert len(bound.circuit.instructions) == 1
    gate = bound.circuit.instructions[0]
    assert gate.name == "phase" and gate.wires == (wires[0],)
    assert gate.params["theta"] == torch.pi
    assert source.num_rotations < 0
    model = get_cost_value(source, QECGatesCost())
    assert model.rotation == source.num_rotations
    assert model.clifford == 0
    assert get_cost_value(source, QubitCount()) == width
    # Preserve the source's negative precision/count metadata; the actual body
    # retains one sign Z, which is not a negative physical gate inventory.
    source_matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    native = bind_register_program(record, (tuple(range(width)),))
    torch.testing.assert_close(
        logical_operator(native.circuit, width), source_matrix, rtol=0, atol=2e-15
    )
    cost = Register("cost", fixed_point_register_type(width, fraction, signed=True))
    inverse = bind_register_program(
        qvr_zpow_program(cost, gamma, 0.2, adjoint=True), (wires,)
    )
    labels = torch.arange(1 << (width + 1), dtype=torch.float64)
    state = torch.complex(labels + 1, labels.remainder(7) - 3)
    state /= state.norm()
    signs = 1 - 2 * (
        labels.to(torch.int64).bitwise_right_shift(width - wires[0]).bitwise_and(1)
    )
    circuit = Circuit(width + 1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 1, bound.circuit.instructions))
    torch.testing.assert_close(
        circuit.state().flatten(), state * signs, rtol=0, atol=2e-15
    )
    circuit.append(FieldOperation(width + 1, inverse.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-15)


def test_metadata_validation_and_source_signature_refusals() -> None:
    factories = tuple(qvr_factories().values())
    cost = Register("cost", fixed_point_register_type(3, 3))
    cases: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"cost_reg": 3},
        {"cost_reg": cost, "other": 0},
        {"cost_reg": cost, "gamma": True},
    )
    for factory in factories:
        for arguments in cases:
            with pytest.raises(SerializationError):
                factory("bad", None, arguments)
        with pytest.raises(SerializationError, match="signature"):
            factory("bad", (), {"cost_reg": cost})
    with pytest.raises(SerializationError, match="unsigned"):
        factories[1](
            "bad",
            None,
            {
                "cost_reg": Register(
                    "cost", fixed_point_register_type(3, 2, signed=True)
                )
            },
        )
    with pytest.raises(SerializationError, match="precision"):
        factories[1]("bad", None, {"cost_reg": cost, "eps": sympy.Symbol("eps")})


@pytest.mark.parametrize("fraction,gamma", tuple(product((-1, -2), (0.375, -0.375))))
def test_negative_fraction_all_histories_and_zero_work(
    fraction: int, gamma: float
) -> None:
    from qualtran import QFxp
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient

    source = QvrPhaseGradient(SourceRegister("energy", QFxp(1, fraction)), gamma, 0.2)
    record = resolve_component_program(_library(source, 0), 0, qvr_factories())
    example = QvrExample(1, gamma, 0.2, fraction_bits=fraction, via_gradient=True)
    logical = 1 + example.gradient_width
    total = example.qubit_count
    bound = bind_register_program(
        record, ((0,), tuple(range(1, logical))), clean=tuple(range(logical, total))
    )
    values = torch.arange(1, 2**logical + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, values.flip(0))
    amplitudes /= amplitudes.norm()
    work_dimension = 2 ** (total - logical)
    initial = torch.zeros((2**logical, work_dimension), dtype=torch.complex128)
    initial[:, 0] = amplitudes
    expected = torch.zeros_like(initial)
    expected[:, 0] = torch.as_tensor(source.tensor_contract()) @ amplitudes
    gates = bound.circuit.instructions
    measurements = sum(gate.name == "measure" for gate in gates)
    for history in product((0, 1), repeat=measurements):
        branch = forced_branch(initial.flatten(), gates, history)
        torch.testing.assert_close(
            branch, expected.flatten() / 2 ** (measurements / 2), atol=8e-12, rtol=0
        )
        work = branch.reshape(-1, work_dimension)[:, 1:]
        torch.testing.assert_close(work, torch.zeros_like(work), atol=8e-12, rtol=0)
