"""Modular negation/subtraction constructors retain source gates and typed ownership."""

from collections import Counter
from importlib import import_module
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_subtraction_factories import (
    modular_subtraction_factories,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_modular_addition_component import bind_case
from tests.unit.test_modular_subtraction_component import fixture, source_stream
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _resolve(source: Any) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, 0), 0, modular_subtraction_factories()
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize(
    "kind,cv",
    (("neg", 1), ("cneg", 0), ("cneg", 1), ("sub", 1), ("csub", 0), ("csub", 1)),
)
@pytest.mark.parametrize("montgomery", (False, True))
def test_complete_constructor_source_stream_feedback_and_resource_scan(
    kind: str, cv: int, montgomery: bool
) -> None:
    direct, source = fixture(kind, 2, 2, montgomery, cv)
    bound, ports, _ = bind_case(_resolve(source))
    expected, _, _ = bind_case(direct)
    assert bound == expected
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq
    assert counts["measure"] == counts["reset"] == model.measurement


@pytest.mark.parametrize(
    "name", ("_mod_neg", "_cmod_neg", "_modsub_symb", "_cmodsub_symb")
)
def test_complete_original_examples_retain_symbolic_declarations(name: str) -> None:
    source = getattr(
        import_module("qualtran.bloqs.mod_arithmetic.mod_subtraction"), name
    )()
    program = _resolve(source)
    if name == "_mod_neg":
        assert (
            len(program.functions[0].blocks[0].operations)
            == len(source.decompose_bloq().bloq_instances)
            == 8
        )
    else:
        assert not program.functions[0].blocks
    assert program.functions[0].registers is not None
    assert tuple(register.name for register in program.functions[0].registers) == tuple(
        register.name for register in source.signature
    )


@pytest.mark.parametrize("kind", ("neg", "cneg", "sub", "csub"))
def test_original_all_ones_constant_loader_refusal_is_preserved(kind: str) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic import mod_subtraction

    constructors = {
        "neg": mod_subtraction.ModNeg,
        "cneg": mod_subtraction.CModNeg,
        "sub": mod_subtraction.ModSub,
        "csub": mod_subtraction.CModSub,
    }
    source = constructors[kind](QUInt(2), 3)
    with pytest.raises(SerializationError, match="modulus"):
        _resolve(source)


def test_constructor_invalid_fields_dtype_control_and_signature_fail_early() -> None:
    factories = modular_subtraction_factories()
    prefix = "qualtran.bloqs.mod_arithmetic.mod_subtraction."
    for kind in ("ModNeg", "CModNeg", "ModSub", "CModSub"):
        with pytest.raises(SerializationError):
            factories[prefix + kind]("main", None, {})
    for args in (
        {"dtype": 1, "mod": 2},
        {"dtype": unsigned_register_type(2), "mod": 2, "cv": 2},
    ):
        with pytest.raises(SerializationError):
            factories[prefix + "CModSub"]("main", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factories[prefix + "ModNeg"](
            "main", (), {"dtype": unsigned_register_type(2), "mod": 2}
        )
