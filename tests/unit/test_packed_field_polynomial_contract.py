"""Keep Proposal 109's opaque default exactly compatible with its baseline."""

import json
from pathlib import Path

import pytest

from flagquantum.compiler.registers.finite_field.field_polynomial_component import (
    field_polynomial_component_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.finite_field.polynomial_addition_component import (
    field_polynomial_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import RegisterType

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["split", "join", "add", "constant"])
def test_opaque_program_bytes_hash_and_aliases(mode: str) -> None:
    fixture = json.loads(
        (ROOT / "tests/fixtures/packed_field_polynomial_compatibility.json").read_text()
    )[mode]
    layout = FieldPolynomialLayout(2, 2)
    field = RegisterType(
        "finite_field",
        2,
        parameters=(("characteristic", 2), ("degree", 2), ("order", 4)),
    )
    inputs: tuple[tuple[int, ...], ...]
    if mode in {"split", "join"}:
        program = field_polynomial_component_program(
            layout, field, reverse=mode == "join"
        )
        inputs = ((5, 1), (4, 0), (3, 2)) if mode == "join" else ((5, 1, 4, 0, 3, 2),)
    else:
        program = field_polynomial_addition_program(
            layout, field, constants=(1, 3) if mode == "constant" else None
        )
        inputs = ((5, 1, 4, 0, 3, 2),)
        if mode == "add":
            inputs += ((6, 7, 8, 9, 10, 11),)
    bound = bind_register_program(program, inputs)
    assert repr(program) == fixture["program_repr"]
    assert bound.circuit.to_json() == fixture["circuit_json"]
    assert bound.circuit.content_hash == fixture["content_hash"]
    assert [list(word) for word in bound.outputs] == fixture["outputs"]
    assert not bound.events and not bound.classical_outputs


def test_approved_contract_keeps_owned_operations() -> None:
    contract = tomllib.loads(
        (ROOT / "contracts/packed-field-polynomial-contract.toml").read_text()
    )
    assert contract["approval"] == "explicit_user_approval_2026-11-20"
    assert contract["public_methods_added"] == []
    assert contract["binding"]["view_primitives"] == 0
    assert contract["binding"]["view_work"] == 0
    assert contract["binding"]["view_classical_slots"] == 0
    assert contract["compatibility"]["operation_fields"] == "unchanged"
