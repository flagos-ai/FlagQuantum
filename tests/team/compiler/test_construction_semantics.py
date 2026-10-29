"""Revision-six boundaries of the closed integer profile, not Qmod support."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.bindings import verify_bindings
from flagquantum.compiler._construction.profile import uint_type, verify_construction
from flagquantum.compiler._hybrid.model import (
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError

pytestmark = pytest.mark.unit


def identity() -> HybridProgram:
    x = Value(ValueId("identity", 0), uint_type(3))
    return HybridProgram(
        "identity",
        Region((Block((x,), (Operation("qint.return", (x,)),)),)),
    )


def test_binding_rejects_unverified_undefined_return() -> None:
    function = identity()
    x = function.body.blocks[0].arguments[0]
    undefined = Value(ValueId("identity", 1), uint_type(3))
    broken = replace(
        function,
        body=Region((Block((x,), (Operation("qint.return", (undefined,)),)),)),
    )
    with pytest.raises(HybridVerificationError, match="construction.ssa.use"):
        verify_bindings(
            (broken,), {x.id: (0, 1, 2)}, (3, 4, 5), entry="identity", n_wires=6
        )


@pytest.mark.parametrize(
    "opcode",
    [
        "qint.assign",
        "qint.inplace_xor",
        "qint.inplace_add",
        "qint.bind",
        "qint.free",
        "qint.drop",
        "quantum.phase",
        "quantum.measure",
        "quantum.reset",
        "quantum.control",
        "quantum.within_apply",
    ],
)
def test_source_effects_are_not_silently_reinterpreted(opcode: str) -> None:
    function = identity()
    block = function.body.blocks[0]
    x = block.arguments[0]
    effect = Operation(opcode, (x,), location=SourceLocation("source.qmod", 4, 2))
    unsupported = replace(
        function, body=Region((Block((x,), (effect, *block.operations)),))
    )
    with pytest.raises(
        HybridVerificationError,
        match=r"operation.unsupported: source.qmod:4:2",
    ):
        verify_construction((unsupported,), entry="identity")
    with pytest.raises(HybridVerificationError, match="operation.unsupported"):
        verify_bindings(
            (unsupported,), {x.id: (0, 1, 2)}, (3, 4, 5), entry="identity", n_wires=6
        )


@pytest.mark.parametrize(
    "attribute", ["disable_perm_check", "disable_const_checks", "input_role", "const"]
)
def test_unsupported_role_and_trust_attributes_are_not_ignored(attribute: str) -> None:
    callee = identity()
    x = Value(ValueId("main", 0), uint_type(3))
    result = Value(ValueId("main", 1), uint_type(3))
    call = Operation(
        "qint.call",
        (x,),
        (result,),
        {"callee": "identity", attribute: True},
        location=SourceLocation("roles.qmod", 9),
    )
    main = HybridProgram(
        "main", Region((Block((x,), (call, Operation("qint.return", (result,)))),))
    )
    with pytest.raises(HybridVerificationError, match="operation.attributes") as error:
        verify_construction((callee, main), entry="main")
    assert attribute in str(error.value)
    assert "roles.qmod:9:0" in str(error.value)


@pytest.mark.parametrize(
    "kind,parameters,linear",
    [
        ("quantum.uint", (3,), True),
        ("quantum.uint", (3, "signed"), False),
        ("quantum.fixed", (3, 1), False),
        ("quantum.array", (3,), False),
        ("quantum.struct", (), False),
    ],
)
def test_consumed_and_non_unsigned_types_need_their_own_profile(
    kind: str,
    parameters: tuple[object, ...],
    linear: bool,
) -> None:
    x = Value(ValueId("main", 0), IRType(kind, parameters, linear))
    function = HybridProgram(
        "main", Region((Block((x,), (Operation("qint.return", (x,)),)),))
    )
    with pytest.raises(HybridVerificationError, match="construction.type"):
        verify_construction((function,), entry="main")


def test_unknown_workspace_assertion_is_not_a_proof() -> None:
    function = identity()
    x = function.body.blocks[0].arguments[0]
    with pytest.raises(HybridVerificationError, match="workspace.state"):
        verify_bindings(
            (function,),
            {x.id: (0, 1, 2)},
            (3, 4, 5),
            entry="identity",
            n_wires=7,
            workspace=(6,),
            workspace_state="measured_zero",
        )
