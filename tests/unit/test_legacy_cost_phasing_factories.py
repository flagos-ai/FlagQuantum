"""Cost-phasing constructors retain source calls and explicit exact erasure."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_cost_phasing_factories import (
    cost_phasing_factories,
)
from flagquantum.compiler.registers.arithmetic.integer_square_component import (
    integer_square_program,
)
from flagquantum.compiler.registers.phase.cost_function_component import (
    cost_function_phase_program,
)
from flagquantum.compiler.registers.phase.qvr_component import (
    qvr_phase_gradient_program,
    qvr_zpow_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_cost_phasing_file import source_module as source_module
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_SQUARE = "qualtran.bloqs.arithmetic.multiplication.Square"
_PHASING = "qualtran.bloqs.rotations.phasing_via_cost_function.PhasingViaCostFunction"


@pytest.mark.parametrize("gradient", (False, True))
def test_original_constructor_root_and_atomic_square(
    source_module: Any, gradient: bool
) -> None:
    source = (
        source_module._square_via_phase_gradient
        if gradient
        else source_module._square_via_zpow_phasing
    ).make()
    record = resolve_component_program(_library(source, 0), 0, cost_phasing_factories())
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    assert (
        resolve_component_program(_library(source, 1), 0, cost_phasing_factories())
        == record
    )
    calls = record.functions[0].blocks[0].operations
    definitions = {f.name: f for f in record.functions}
    assert len(calls) == 3
    assert not definitions[calls[0].callee or ""].blocks
    assert not definitions[calls[2].callee or ""].blocks


@pytest.mark.parametrize("gradient,width", ((False, 1), (False, 2), (True, 1)))
def test_injected_square_owner_reuses_accepted_exact_execution(
    source_module: Any, gradient: bool, width: int
) -> None:
    from qualtran import QFxp
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.arithmetic import Square
    from qualtran.bloqs.rotations.quantum_variable_rotation import (
        QvrPhaseGradient,
        QvrZPow,
    )

    phase = (QvrPhaseGradient if gradient else QvrZPow)(
        SourceRegister("result", QFxp(2 * width, 2 * width)), 0.5, 1.0
    )
    source = source_module.PhasingViaCostFunction(Square(width), phase)
    evaluation = integer_square_program(width)
    inverse = integer_square_program(width, adjoint=True)

    def square(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        assert arguments == {"bitsize": width, "uncompute": 0}
        assert registers == evaluation.functions[0].registers
        return replace(
            evaluation,
            functions=(
                replace(evaluation.functions[0], name=name),
                *evaluation.functions[1:],
            ),
        )

    def erase(child: FunctionRecord) -> ProgramRecord:
        assert child.registers == evaluation.functions[0].registers
        return inverse

    factories = cost_phasing_factories(exact_adjoint=erase)
    factories[_SQUARE] = square
    record = resolve_component_program(_library(source, 0), 0, factories)
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    cost = next(
        r
        for f in record.functions
        for r in (f.registers or ())
        if r.name == "result" and r.dtype.encoding == "fixed_point"
    )
    phase_record = (qvr_phase_gradient_program if gradient else qvr_zpow_program)(
        cost, 0.5, 1.0
    )
    accepted = cost_function_phase_program(
        evaluation, phase_record, inverse, cost_names=("result",)
    )
    logical = width + (int(phase.b_grad) if gradient else 0)
    ports = (
        (tuple(range(width)), tuple(range(width, logical)))
        if gradient
        else (tuple(range(width)),)
    )
    clean = tuple(range(logical, logical + 20))
    actual = bind_register_program(record, ports, clean=clean)
    expected = bind_register_program(accepted, ports, clean=clean)
    # Equality covers every primitive operand, feedback slot, allocation and release.
    assert actual == expected
    from tests.unit.test_multiplexed_source_binding import logical_operator

    torch.testing.assert_close(
        logical_operator(actual.circuit, logical),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )


def test_constructor_refuses_unowned_oracles() -> None:
    from flagquantum.core.ir.program.register_types import fixed_point_register_type

    factory = cost_phasing_factories()[_PHASING]
    cost = Register("cost", fixed_point_register_type(2, 2))
    child = FunctionRecord("unknown", (), registers=(cost,))
    for arguments in ({}, {"cost_eval_oracle": 3, "phase_oracle": child}):
        with pytest.raises(SerializationError, match="(requires|resolved)"):
            factory("main", None, arguments)
    with pytest.raises(SerializationError, match="constructor context"):
        factory("main", None, {"cost_eval_oracle": child, "phase_oracle": child})


def test_external_phase_metadata_and_exact_evaluator_ownership() -> None:
    from flagquantum.core.ir.program.register_types import fixed_point_register_type

    evaluate = replace(integer_square_program(2).functions[0], name="evaluation")
    phase = replace(
        qvr_zpow_program(
            Register("result", fixed_point_register_type(4, 4)), 0.5, 1.0
        ).functions[0],
        name="phase",
    )
    arguments: dict[str, ResolvedArgument] = {
        "cost_eval_oracle": evaluate,
        "phase_oracle": phase,
    }
    factories = cost_phasing_factories(
        exact_adjoint=lambda child: integer_square_program(2, adjoint=True),
        phase_cost_registers=lambda child: ("result",),
    )
    record = factories[_PHASING]("main", None, arguments)
    assert isinstance(record, ProgramRecord)
    # Enclosing resolution supplies these independently resolved child closures.
    assert {evaluate.name, phase.name}.isdisjoint(f.name for f in record.functions)
    assert len(record.functions[0].blocks[0].operations) == 3
    with pytest.raises(SerializationError, match="signature"):
        factories[_PHASING]("main", (), arguments)
    without_erasure = cost_phasing_factories(
        phase_cost_registers=lambda child: ("result",)
    )
    with pytest.raises(SerializationError, match="exact-adjoint owner"):
        without_erasure[_PHASING]("main", None, arguments)
