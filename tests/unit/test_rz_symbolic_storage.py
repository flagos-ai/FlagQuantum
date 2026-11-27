"""Rz casts interpret concrete storage as unsigned words before arithmetic."""

from collections import Counter
from importlib import import_module
from itertools import product
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import sympy
    import torch
    from qualtran._infra.data_types import BQUInt, QDType, QFxp, QMontgomeryUInt, QUInt
    from qualtran.bloqs.rotations.rz_via_phase_gradient import RzViaPhaseGradient
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_phase_rotation_factories import (
    phase_rotation_factories,
)
from flagquantum.compiler.registers.phase.rz_phase_gradient_component import (
    rz_via_phase_gradient_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
    montgomery_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def storage(width: int, kind: str) -> tuple[QDType[int], RegisterType]:
    parameter = sympy.Symbol("interpretation", integer=True)
    if kind in ("fixed", "signed"):
        fraction = parameter + 1
        return (
            QFxp(width, fraction, signed=kind == "signed"),
            fixed_point_register_type(width, fraction, signed=kind == "signed"),
        )
    if kind == "bounded":
        return BQUInt(width, parameter), bounded_unsigned_register_type(
            width, parameter
        )
    assert kind == "montgomery"
    return QMontgomeryUInt(width, parameter), montgomery_register_type(width, parameter)


@pytest.mark.parametrize(
    "width,kind,side",
    tuple(
        product(
            (2, 3),
            ("fixed", "signed", "bounded", "montgomery"),
            ("angle", "gradient", "both"),
        )
    ),
)
def test_source_defined_rz_metadata_views_full_channel_and_exact_erasure(
    width: int, kind: str, side: str
) -> None:
    foreign, neutral = storage(width, kind)
    source = RzViaPhaseGradient(
        foreign if side != "gradient" else QUInt(width),
        foreign if side != "angle" else QUInt(width),
    )
    raw_type = unsigned_register_type(width)
    angle = neutral if side != "gradient" else raw_type
    gradient = neutral if side != "angle" else raw_type
    factory = phase_rotation_factories()[
        "qualtran.bloqs.rotations.rz_via_phase_gradient.RzViaPhaseGradient"
    ]
    program = factory("main", None, {"angle_dtype": angle, "phasegrad_dtype": gradient})
    assert isinstance(program, ProgramRecord) and verify_program_record(program) == ()
    assert _native_trace(program) == _source_trace(SourceCalls, source.decompose_bloq())
    assert program.functions[0].registers is not None
    assert tuple(r.dtype for r in program.functions[0].registers)[1:] == (
        angle,
        gradient,
    )
    logical = 1 + 2 * width
    ports = ((0,), tuple(range(1, width + 1)), tuple(range(width + 1, logical)))
    clean = tuple(range(logical + 1, logical + width))
    bound = bind_register_program(program, ports, clean=clean)
    reference = bind_register_program(
        rz_via_phase_gradient_program(raw_type, raw_type), ports, clean=clean
    )
    assert bound.circuit == reference.circuit and bound.events == reference.events
    assert bound.outputs == ports
    if kind != "montgomery":
        # The original Montgomery codec drops its modulus. Other original
        # constructor wires can be checked without claiming that lossy roundtrip.
        for depth in (0, 1, 2):
            restored = resolve_component_program(
                _library(source, depth), 0, phase_rotation_factories()
            )
            restored_bound = bind_register_program(restored, ports, clean=clean)
            assert restored_bound.circuit == bound.circuit
            assert (
                restored_bound.events == bound.events
                and restored_bound.outputs == ports
            )
    matrix = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    raw_source = RzViaPhaseGradient(QUInt(width), QUInt(width))
    torch.testing.assert_close(
        matrix, torch.tensor(raw_source.tensor_contract()), atol=1e-12, rtol=0
    )
    contiguous = bind_register_program(
        program, ports, clean=tuple(range(logical, logical + width - 1))
    )
    torch.testing.assert_close(
        logical_operator(contiguous.circuit, logical), matrix, atol=8e-12, rtol=0
    )
    assert get_cost_value(source, QECGatesCost()) == get_cost_value(
        raw_source, QECGatesCost()
    )
    assert get_cost_value(source, QubitCount()) == get_cost_value(
        raw_source, QubitCount()
    )
    # All raw patterns are compared as storage, including bounded padding words.
    # This grants no numerical value admission for unresolved interpretation.
    total = logical + width
    values = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, 1 - 2 * values)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    stride = 1 << (width - 1)
    initial[::stride] = amplitudes
    expected[::stride] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    forward = (
        DynamicCircuit(total)
        .append(FieldOperation(total, bound.circuit.instructions))
        .to_ir()
    )
    assert forward.instructions == bound.circuit.instructions
    assert CircuitIR.from_json(forward.to_json()) == forward
    inverse = bind_register_program(
        rz_via_phase_gradient_program(angle, gradient, adjoint=True), ports, clean=clean
    )
    counts = Counter(g.name for g in forward.instructions)
    assert counts["measure"] == counts["reset"] == width - 1
    assert counts["t"] + counts["tdg"] == 4 * (width - 1)
    for gates, start, target in (
        (forward.instructions, initial, expected),
        (inverse.circuit.instructions, expected, initial),
    ):
        for history in product((0, 1), repeat=width - 1):
            result = forced_branch(start, gates, history)
            torch.testing.assert_close(
                result, target / 2 ** ((width - 1) / 2), atol=8e-12, rtol=0
            )
            assert torch.count_nonzero(result.reshape(-1, stride)[:, 1:]) == 0


def test_native_metadata_views_execute_without_reference_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError("Source SDK imported during native Rz execution")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    dtype = fixed_point_register_type(2, sympy.Symbol("fraction"))
    program = rz_via_phase_gradient_program(dtype, dtype)
    bound = bind_register_program(program, ((0,), (1, 2), (3, 4)), clean=(5,))
    assert (
        DynamicCircuit(6)
        .append(FieldOperation(6, bound.circuit.instructions))
        .to_ir()
        .instructions
        == bound.circuit.instructions
    )


def test_unresolved_dimensions_remain_logical_declarations() -> None:
    dtype = fixed_point_register_type(sympy.Symbol("n"), sympy.Symbol("fraction"))
    program = rz_via_phase_gradient_program(dtype, dtype)
    assert verify_program_record(program) == ()
    with pytest.raises(ValueError):
        bind_register_program(program, ((0,), (1, 2), (3, 4)), clean=(5,))
