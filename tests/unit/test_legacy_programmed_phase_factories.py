"""Complete programmed-resource constructors preserve source and channel boundaries."""

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
from flagquantum.compiler.formats.legacy_programmed_phase_factories import (
    programmed_phase_factories,
)
from flagquantum.compiler.registers.phase.programmed_phase import programmed_phase_ir
from flagquantum.compiler.registers.phase.programmed_phase_resource_model import (
    ProgrammedRotationModel,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_programmed_phase_file import (
    assert_model_matches,
)
from tests.unit.test_programmed_phase_file import (
    source_module as source_module,
)
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "example",
    (
        "_zpow_programmed_ancilla",
        "_zpow_programmed_ancilla_symb",
        "_zpow_using_programmed_ancilla",
        "_zpow_using_programmed_ancilla_symb",
        "_zpow_using_programmed_ancilla_symb_rounds",
    ),
)
def test_all_original_example_constructors(source_module: Any, example: str) -> None:
    source = getattr(source_module, example).make()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, programmed_phase_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == library.entries[0].declaration.registers


@pytest.mark.parametrize("exponent", (0.0, 0.375, -0.73, 2.0))
def test_preparation_graph_complex_output_and_owned_port(
    source_module: Any, exponent: float
) -> None:
    source = source_module.ZPowProgrammedAncilla(exponent)
    record = resolve_component_program(
        _library(source, 0), 0, programmed_phase_factories()
    )
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    assert (
        resolve_component_program(_library(source, 1), 0, programmed_phase_factories())
        == record
    )
    with pytest.raises(ValueError, match="ownership transfer"):
        bind_register_program(record, (), clean=(2,))
    bound = bind_register_program(record, (), clean=(2,), allow_workspace_outputs=True)
    assert bound.outputs == ((2,),)
    assert len(bound.events) == 1 and bound.events[0].kind == "allocate"
    assert bound.circuit.instructions == programmed_phase_ir(2, exponent=exponent)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    spectator = torch.tensor([1, 2j, -3, 4j], dtype=torch.complex128)
    spectator /= spectator.norm()
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = spectator
    expected = torch.kron(
        spectator, torch.tensor(source.tensor_contract(), dtype=initial.dtype)
    )
    torch.testing.assert_close(
        forced_branch(initial, bound.circuit.instructions, ()),
        expected,
        atol=2e-12,
        rtol=0,
    )
    circuit = Circuit(3).append(FieldOperation(3, bound.circuit.instructions))
    assert circuit.to_ir().instructions == bound.circuit.instructions


def test_symbolic_preparation_retains_source_root_and_nonlinear_binding(
    source_module: Any,
) -> None:
    theta = sympy.Symbol("theta")
    expression = sympy.sin(theta) + theta**2 / 3
    source = source_module.ZPowProgrammedAncilla(expression, sympy.Symbol("epsilon"))
    record = resolve_component_program(
        _library(source, 0), 0, programmed_phase_factories()
    )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    bound = bind_register_program(record, (), clean=(0,), allow_workspace_outputs=True)
    ir = CircuitIR.from_json(bound.circuit.to_json())
    circuit = Circuit(1, dtype=torch.complex128).append(
        FieldOperation(1, ir.instructions)
    )
    numeric = 0.37
    original = source_module.ZPowProgrammedAncilla(
        float(expression.subs(theta, numeric))
    )
    actual = get_unitary(circuit.bind_parameters({"theta": numeric}))[:, 0]
    torch.testing.assert_close(
        actual,
        torch.tensor(original.tensor_contract(), dtype=actual.dtype),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "rounds,correction",
    tuple(
        product(
            (0, 1, 3, sympy.Symbol("n", integer=True, positive=True)), (False, True)
        )
    ),
)
def test_adaptive_declaration_and_independent_worst_case_model(
    source_module: Any, rounds: Any, correction: bool
) -> None:
    source = source_module.ZPowUsingProgrammedAncilla(
        0.375, n_rounds=rounds, apply_final_correction=correction
    )
    record = resolve_component_program(
        _library(source, 0), 0, programmed_phase_factories()
    )
    assert verify_program_record(record) == () and not record.functions[0].blocks
    assert (
        record.functions[0].registers
        == _library(source, 0).entries[0].declaration.registers
    )
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, ((0,),))
    with pytest.raises(AttributeError, match="bitsize"):
        _library(source, 1)
    model = ProgrammedRotationModel(
        0.375, rounds=rounds, apply_final_correction=correction
    )
    assert_model_matches(model, source)


def test_constructor_rejects_invalid_metadata_and_ports() -> None:
    prepare, adaptive = programmed_phase_factories().values()
    for factory in (prepare, adaptive):
        with pytest.raises(SerializationError, match="arguments"):
            factory("main", None, {})
        with pytest.raises(SerializationError, match="signature"):
            factory("main", (), {"exponent": 0.5})
    with pytest.raises(SerializationError, match="boolean"):
        adaptive("main", None, {"exponent": 0.5, "apply_final_correction": 2})
