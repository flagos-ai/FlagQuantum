"""Frozen modular addition constructors reuse the complete accepted graph owners."""

from collections import Counter
from importlib import import_module
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_addition_factories import (
    modular_addition_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_modular_addition_component import bind_case, fixture, source_stream
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _resolve(source: Any) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, 0), 0, modular_addition_factories()
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize(
    "kind", ("mod", "constant", "scaled", "unsigned", "montgomery")
)
@pytest.mark.parametrize("cv", (0, 1))
def test_complete_named_graph_dependency_stream_operands_feedback_and_inventory(
    kind: str, cv: int
) -> None:
    direct, source = fixture(kind, 2, 3, 1, cv)
    if kind == "mod":
        with pytest.raises(SerializationError, match="signature"):
            _resolve(source)
        # The source wire drops the constructor's Montgomery modulus metadata.
        resolved = modular_addition_factories()[
            "qualtran.bloqs.mod_arithmetic.mod_addition.ModAdd"
        ]("main", None, {"bitsize": 2, "mod": 3})
        assert isinstance(resolved, ProgramRecord)
    else:
        resolved = _resolve(source)
    bound, ports, _ = bind_case(resolved)
    expected, _, _ = bind_case(direct)
    assert bound == expected
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq
    assert counts["measure"] == counts["reset"] == model.measurement


@pytest.mark.parametrize(
    "name",
    (
        "_mod_add",
        "_mod_add_k",
        "_mod_add_k_small",
        "_mod_add_k_large",
        "_cmod_add_k",
        "_cmod_add_k_small",
        "_ctrl_scale_mod_add",
        "_ctrl_scale_mod_add_small",
        "_cmodadd_symbolic",
        "_cmodadd_example",
    ),
)
def test_all_original_examples_keep_declared_signatures_and_symbolic_boundaries(
    name: str,
) -> None:
    source = getattr(
        import_module("qualtran.bloqs.mod_arithmetic.mod_addition"), name
    )()
    if name == "_mod_add":
        with pytest.raises(SerializationError, match="signature"):
            _resolve(source)
        return
    program = _resolve(source)
    assert program.functions[0].registers is not None
    assert tuple(register.name for register in program.functions[0].registers) == tuple(
        register.name for register in source.signature
    )
    if (
        "symbolic" in name
        or any(
            isinstance(register.bitsize, sympy.Expr) for register in source.signature
        )
        or type(source).__name__ == "ModAddK"
    ):
        assert not program.functions[0].blocks


def test_atomic_constant_mapping_retains_the_source_model_only_body_boundary() -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.mod_arithmetic import ModAddK

    source = ModAddK(3, 5, -2, cvs=(0, 1))
    program = _resolve(source)
    assert program.functions[0].registers is not None
    assert program.functions[0].registers[0].shape == (2,)
    assert not program.functions[0].blocks
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(ValueError):
        bind_register_program(program, ((0,), (1,), (2, 3, 4)))


def test_modular_constructor_errors_keep_invalid_metadata_outside_execution() -> None:
    factories = modular_addition_factories()
    key = "qualtran.bloqs.mod_arithmetic.mod_addition."
    for kind in ("ModAdd", "ModAddK", "CModAdd", "CModAddK", "CtrlScaleModAdd"):
        with pytest.raises(SerializationError):
            factories[key + kind]("main", None, {})
    for arguments in ({"bitsize": "bad", "mod": 3}, {"bitsize": 2, "mod": 1.5}):
        with pytest.raises(SerializationError):
            factories[key + "ModAdd"]("main", None, arguments)
    with pytest.raises(SerializationError, match="signature"):
        factories[key + "ModAdd"]("main", (), {"bitsize": 2, "mod": 3})
    with pytest.raises(SerializationError, match="fit"):
        factories[key + "ModAddK"]("main", None, {"bitsize": 2, "mod": 5})
    declaration = factories[key + "ModAdd"](
        "main", None, {"bitsize": 2, "mod": sympy.Symbol("p")}
    )
    assert isinstance(declaration, FunctionRecord) and not declaration.blocks


def test_original_montgomery_signature_loss_stays_an_explicit_wire_boundary() -> None:
    from qualtran.bloqs.mod_arithmetic import ModAdd
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = ModAdd(2, 3)
    message = bloqs_to_proto(source, max_depth=0)
    restored = bloqs_from_proto(message)[0]
    assert restored.signature[0].dtype.modulus == 3
    library = _library(source, 0)
    registers = library.entries[0].declaration.registers
    assert registers is not None
    assert all(register.dtype.parameters == () for register in registers)
    with pytest.raises(SerializationError, match="signature"):
        resolve_component_program(library, 0, modular_addition_factories())
