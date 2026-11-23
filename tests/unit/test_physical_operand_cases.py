"""Replace complete frozen operation-test helper values with native ownership."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest

from flagquantum.compiler.artifacts.operand_array import OperandArray
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession
from flagquantum.testing import _physical_operand_cases as native

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "physical_operand_cases.json").read_text()
)


def _encode(value: object) -> list[object]:
    if isinstance(value, bool):
        return ["bool", int(value)]
    if isinstance(value, OperandIdentity):
        names = {
            OperandKind.QUANTUM: "q",
            OperandKind.CLASSICAL: "b",
            OperandKind.X_CLASSICAL: "xb",
            OperandKind.X_BOOLEAN: "xbool",
            OperandKind.BOOLEAN: "bool",
        }
        return [names[value.kind], value.value]
    return [
        "array" if isinstance(value, OperandArray) else "list",
        [_encode(item) for item in cast(Iterable[object], value)],
    ]


@pytest.mark.parametrize("case", _DATA["cases"])
def test_all_scalar_list_array_and_hash_selected_families(
    case: dict[str, object],
) -> None:
    index, size = int(cast(int, case["index"])), int(cast(int, case["size"]))
    assert [
        _encode(v)
        for v in native.x_list_cases(
            index, size, include_x_bits=bool(case["include_x_bits"])
        )
    ] == case["x"]
    assert [_encode(v) for v in native.z_list_cases(index, size)] == case["z"]
    assert [_encode(v) for v in native.x_cases_q(index)] == case["scalar_xq"]
    assert [_encode(v) for v in native.x_cases_bc(index)] == case["scalar_xbc"]
    assert [_encode(v) for v in native.z_cases(index)] == case["scalar_z"]


@pytest.mark.parametrize("case", _DATA["read_write"])
def test_original_constant_and_x_basis_store_read(case: dict[str, object]) -> None:
    descriptor = case["index"]
    assert isinstance(descriptor, list)
    kind, value = descriptor
    names = {
        "q": OperandKind.QUANTUM,
        "b": OperandKind.CLASSICAL,
        "xb": OperandKind.X_CLASSICAL,
        "xbool": OperandKind.X_BOOLEAN,
    }
    index = bool(value) if kind == "bool" else OperandIdentity(names[kind], value)
    sim = PhysicalBasisSession(64, debug_print=False)
    pattern = cast(int, case["pattern"])
    assert native.store64(sim, index, pattern) == case["written"]
    assert native.read64(sim, index) == case["read"]


@pytest.mark.parametrize("case", _DATA["broadcasts"])
def test_complete_source_broadcast_expansion(case: dict[str, object]) -> None:
    if case.get("args") == "all_scalar":
        args: tuple[object, ...] = (
            OperandIdentity(OperandKind.QUANTUM, 5),
            False,
            OperandIdentity(OperandKind.X_CLASSICAL, 2),
        )
    elif case.get("args") == "empty":
        args = ()
    else:
        size = cast(int, case["size"])
        literal = bool(case["literal"])
        args = (
            OperandIdentity(OperandKind.QUANTUM, 5),
            [OperandIdentity(OperandKind.CLASSICAL, k) for k in range(size)],
            literal,
            OperandIdentity(OperandKind.X_BOOLEAN, int(literal)),
        )
    assert _encode(native.broadcast_logic(*args)) == case["result"]


def test_constants_consumer_and_invalid_broadcast_lengths() -> None:
    for name, expected in _DATA["constants"].items():
        assert getattr(native, name) == expected
    with pytest.raises(AssertionError):
        native.broadcast_logic([True], [False, True])
    with pytest.raises(TypeError):
        native.broadcast_logic(None)
    sim = PhysicalBasisSession(64, debug_print=False)
    first = OperandIdentity(OperandKind.QUANTUM, 0)
    second = OperandIdentity(OperandKind.QUANTUM, 1)
    for left in (native.U64_BIT0, native.U64_BIT3):
        for right in (native.U64_BIT1, native.U64_BIT5):
            native.store64(sim, first, left)
            native.store64(sim, second, right)
            sim.do(read_kmx_circuit("CX q0 q1"))
            assert native.read64(sim, first) == left
            assert native.read64(sim, second) == left ^ right
    for enabled in (0, 1):
        index = OperandIdentity(OperandKind.BOOLEAN, enabled)
        assert native.store64(sim, index, 17) == (native.U64_TRUE if enabled else 0)
        assert native.read64(sim, index) == (native.U64_TRUE if enabled else 0)
