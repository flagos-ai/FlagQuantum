"""The full Kaliski source chain resolves to its existing native graph owners."""

from importlib import import_module
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_inverse_factories import (
    modular_inverse_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.modular.inversion.inverse_algorithm_component import (
    inverse_algorithm_program,
)
from flagquantum.compiler.registers.modular.inversion.inverse_arithmetic_component import (
    inverse_arithmetic_program,
)
from flagquantum.compiler.registers.modular.inversion.inverse_exchange_component import (
    inverse_exchange_program,
)
from flagquantum.compiler.registers.modular.inversion.inverse_iteration_component import (
    inverse_iteration_program,
)
from flagquantum.compiler.registers.modular.inversion.inverse_ordering_component import (
    inverse_ordering_program,
)
from flagquantum.compiler.registers.modular.inversion.inverse_parity_component import (
    inverse_parity_program,
)
from flagquantum.compiler.registers.modular.inversion.inverse_scaling_component import (
    inverse_scaling_program,
)
from flagquantum.compiler.registers.modular.inversion.inverse_termination_component import (
    inverse_termination_program,
)
from flagquantum.compiler.registers.modular.modular_inverse_component import (
    modular_inverse_program,
)
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.errors import SerializationError
from tests.unit.inverse_component_checks import (
    bind_inverse_component,
    inverse_source_stream,
)
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PREFIX = "qualtran.bloqs.mod_arithmetic.mod_division."
_KINDS = tuple(f"_KaliskiIterationStep{i}" for i in range(1, 7)) + (
    "_KaliskiIteration",
    "_KaliskiModInverseImpl",
    "KaliskiModInverse",
)


def _direct(kind: str, inverse: bool = False) -> ProgramRecord:
    builders = (
        inverse_termination_program,
        inverse_parity_program,
        inverse_ordering_program,
        inverse_exchange_program,
        inverse_arithmetic_program,
    )
    if kind in _KINDS[:5]:
        return builders[_KINDS.index(kind)](3)
    if kind == "_KaliskiIterationStep6":
        return inverse_scaling_program(3, 5)
    if kind == "_KaliskiIteration":
        return inverse_iteration_program(3, 5)
    if kind == "_KaliskiModInverseImpl":
        return inverse_algorithm_program(3, 5)
    return modular_inverse_program(3, 5, adjoint=inverse)


@pytest.mark.parametrize("kind", _KINDS)
def test_every_constructor_exact_owner_program_and_original_wire_boundary(
    kind: str,
) -> None:
    arguments: dict[str, Any] = {"bitsize": 3}
    if kind not in _KINDS[:5]:
        arguments["mod"] = 5
    original = getattr(import_module(_PREFIX[:-1]), kind)(**arguments)
    program = modular_inverse_factories()[_PREFIX + kind](
        "inverse_component", None, arguments
    )
    assert program == _rename_program(_direct(kind), "inverse_component")
    assert isinstance(program, ProgramRecord) and verify_program_record(program) == ()
    if kind in _KINDS[:5]:
        restored = resolve_component_program(
            _library(original, 0), 0, modular_inverse_factories()
        )
        bound, ports, _ = bind_inverse_component(restored)
        expected, _, _ = bind_inverse_component(_direct(kind))
        assert bound == expected
        assert_source(
            bound.circuit.instructions, inverse_source_stream(original, bound, ports)
        )
    else:
        with pytest.raises(SerializationError, match="signature"):
            resolve_component_program(
                _library(original, 0), 0, modular_inverse_factories()
            )


@pytest.mark.parametrize("inverse", (False, True, 0, 1))
def test_wrapper_direction_and_source_history_ownership(inverse: bool | int) -> None:
    factory = modular_inverse_factories()[_PREFIX + "KaliskiModInverse"]
    program = factory(
        "inverse_component", None, {"bitsize": 3, "mod": 5, "uncompute": inverse}
    )
    assert program == _rename_program(
        _direct("KaliskiModInverse", bool(inverse)), "inverse_component"
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_complete_symbolic_constructor_chain_keeps_bodyless_signatures(
    kind: str,
) -> None:
    n, p = sympy.symbols("n p", positive=True, integer=True)
    arguments: dict[str, Any] = {"bitsize": n}
    if kind not in _KINDS[:5]:
        arguments["mod"] = p
    factory = modular_inverse_factories()[_PREFIX + kind]
    result = factory("symbolic", None, arguments)
    assert isinstance(result, FunctionRecord) and not result.blocks
    assert result.registers is not None and result.registers[0].dtype.num_qubits == n
    assert factory("symbolic", result.registers, arguments) == result
    if kind == "KaliskiModInverse":
        assert result.registers[-1].dtype.num_qubits == 4 * n
        assert result.registers[-1].side is RegisterSide.RIGHT
        arguments["uncompute"] = True
        reverse = factory("symbolic", None, arguments)
        assert isinstance(reverse, FunctionRecord) and reverse.registers is not None
        assert reverse.registers[-1].side is RegisterSide.LEFT


def test_invalid_metadata_and_original_all_ones_loader_refuse() -> None:
    factory = modular_inverse_factories()[_PREFIX + "KaliskiModInverse"]
    for arguments in (
        {},
        {"bitsize": "bad", "mod": 5},
        {"bitsize": 3, "mod": 5, "extra": 1},
        {"bitsize": 3, "mod": 5, "uncompute": 2},
    ):
        with pytest.raises(SerializationError):
            factory("bad", None, arguments)
    with pytest.raises(ValueError):
        factory("bad", None, {"bitsize": 2, "mod": 3})
