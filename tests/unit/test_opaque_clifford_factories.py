"""Opaque Clifford serialization preserves resource declarations without guessed gates."""

from dataclasses import asdict
from importlib import import_module
from typing import Any

import pytest

from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    arbitrary_clifford_cost_key,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_legacy_arithmetic_factories import _factories, _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (0, 1, 3, "positive", "unconstrained"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_opaque_clifford_metadata_models_and_explicit_execution_refusal(
    width: int | str,
    depth: int,
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    source_module = import_module("qualtran.bloqs.bookkeeping.arbitrary_clifford")
    resource = import_module("qualtran.resource_counting")
    n = (
        width
        if isinstance(width, int)
        else sympy.Symbol(
            "n", **({"positive": True, "integer": True} if width == "positive" else {})
        )
    )
    source = source_module.ArbitraryClifford(n)
    library = _library(source, depth)
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        program = resolve_component_program(library, 0, _factories())
    assert verify_program_record(program) == ()
    root = program.functions[0]
    assert (
        root.blocks == () and root.registers == library.entries[0].declaration.registers
    )
    assert (
        root.registers is not None
        and root.registers[0].dtype.num_qubits
        == library.entries[0].declaration.arguments[0].value
    )
    key = arbitrary_clifford_cost_key(n)
    native = {
        name: value
        for name, value in asdict(query_qec_gate_cost(key)).items()
        if value != 0
    }
    assert (
        native
        == resource.get_cost_value(source, resource.QECGatesCost()).asdict()
        == {"clifford": 1}
    )
    assert resource.get_cost_value(source, resource.QubitCount()) == n
    with pytest.raises(ValueError):
        bind_register_program(
            program, (tuple(range(n)) if isinstance(n, int) else (0,),)
        )
