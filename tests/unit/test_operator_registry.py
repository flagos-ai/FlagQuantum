"""The operator registration table, and what a registration does and does not buy.

Three properties are pinned here, and each is the one a caller would be misled by
if it were only true in prose.

Registration is a value, not a write. It returns a new
:class:`flagquantum.operators.OperatorRegistry` and leaves the built-in
:data:`flagquantum.core.operator_schema.OPERATOR_SCHEMAS` byte-identical, so a
registration cannot leak into an already imported module, cannot make the
checked-in ``docs/operator_manifest.json`` depend on which tests ran first, and
cannot be undone by a later registration.

Registration is a declaration, not an extension of the IR's vocabulary. The IR
resolves names through the built-in table, so an instruction naming a registered
opcode is still an unknown opcode and still has to carry its matrix. The user
journey at the bottom of this module is the whole path: declare once, emit with a
matrix, run, and get the amplitude the matrix implies.

Registration buys nothing at a backend until that backend is given a capability.
A freshly registered opcode is reported unsupported by all eight backends, and
the refusal names the opcode and the missing declaration, which is the
degradation contract rather than a fallback.
"""

from __future__ import annotations

import pytest

import flagquantum as fq
import flagquantum.operators as operators
from flagquantum.compiler.operator_lowering import (
    BACKENDS,
    DEFAULT_LOWERING_REGISTRY,
    UnsupportedLoweringError,
)
from flagquantum.core.ir import IRValidationError
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, get_operator_schema
from flagquantum.operators import OperatorRegistry, register_operator

pytestmark = pytest.mark.unit


def _amplitudes(circuit: fq.Circuit) -> list[complex]:
    return [
        complex(round(float(value.real), 6), round(float(value.imag), 6))
        for value in circuit.state().reshape(-1)
    ]


def _cphase_quarter_matrix() -> object:
    import torch

    return torch.diag(torch.tensor([1, 1, 1, 1j], dtype=torch.complex128))


def test_a_registration_returns_a_new_table_and_leaves_the_built_in_one_alone() -> None:
    before = dict(OPERATOR_SCHEMAS)

    registry = register_operator("cphase_quarter", arity=2)

    assert len(OPERATOR_SCHEMAS) == len(before)
    assert set(OPERATOR_SCHEMAS) == set(before)
    assert "cphase_quarter" not in OPERATOR_SCHEMAS
    assert registry.schemas["cphase_quarter"].opcode == "cphase_quarter"
    assert registry.registered() == ("cphase_quarter",)
    assert get_operator_schema("cphase_quarter") is None


def test_the_built_in_table_refuses_a_write_rather_than_accepting_one_quietly() -> None:
    with pytest.raises(TypeError):
        OPERATOR_SCHEMAS["cphase_quarter"] = OPERATOR_SCHEMAS["h"]  # type: ignore[index]


def test_a_second_registration_does_not_appear_in_the_first_table() -> None:
    first = register_operator("cphase_quarter", arity=2)
    second = register_operator("cphase_ninth", arity=2, registry=first)

    assert first.registered() == ("cphase_quarter",)
    assert second.registered() == ("cphase_ninth", "cphase_quarter")
    assert "cphase_ninth" not in first.schemas


def test_the_declaration_is_readable_and_keeps_the_names_it_was_given() -> None:
    registry = register_operator(
        "CPhase_Pi2",
        arity=2,
        parameters=("theta",),
        aliases=("cp2",),
        adjoint="self_inverse",
        qubit_convention=("control", "target"),
    )

    schema = registry.schemas["cphase_pi2"]
    assert schema.arity == 2
    assert schema.parameters == ("theta",)
    assert schema.aliases == ("cp2",)
    assert schema.qubit_convention == ("control", "target")
    assert schema.unitary
    assert not schema.differentiable
    assert registry.canonical("CP2") == "cphase_pi2"
    assert registry.canonical("cphase_pi2") == "cphase_pi2"
    assert registry.canonical("h") == "h"


def test_an_omitted_qubit_convention_states_positions_and_nothing_more() -> None:
    registry = register_operator("my_op", arity=3)

    assert registry.schemas["my_op"].qubit_convention == (
        "qubit_0",
        "qubit_1",
        "qubit_2",
    )


@pytest.mark.parametrize(
    ("opcode", "arity", "aliases", "message"),
    [
        ("h", 2, (), "operator schema 'h' is already registered"),
        (
            "x2",
            1,
            ("hadamard",),
            "operator schema 'x2' declares alias 'hadamard', which this table "
            "already resolves",
        ),
    ],
)
def test_a_name_the_table_already_resolves_is_refused(
    opcode: str, arity: int, aliases: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError) as caught:
        register_operator(opcode, arity=arity, aliases=aliases)

    assert str(caught.value) == message


def test_every_backend_reports_a_registered_operator_unsupported_until_declared() -> (
    None
):
    registry = register_operator("cphase_quarter", arity=2)
    lowerings = DEFAULT_LOWERING_REGISTRY.with_operator(
        registry.schemas["cphase_quarter"]
    )

    manifest = lowerings.manifest()["backends"]
    assert set(manifest) == set(BACKENDS)
    for backend in BACKENDS:
        assert "cphase_quarter" in manifest[backend]["unsupported"]
        assert "cphase_quarter" not in manifest[backend]["supported"]

    for backend in BACKENDS:
        with pytest.raises(UnsupportedLoweringError) as caught:
            lowerings.validate(backend, ["cphase_quarter"])
        assert str(caught.value) == (
            f"backend {backend!r} cannot lower: cphase_quarter (not registered)"
        )


def test_only_a_declared_capability_admits_the_operator_to_one_backend() -> None:
    registry = register_operator("cphase_quarter", arity=2)
    lowerings = DEFAULT_LOWERING_REGISTRY.with_operator(
        registry.schemas["cphase_quarter"]
    ).with_capability(
        backend="pytorch",
        opcode="cphase_quarter",
        strategy="lowering",
        implementation="flagquantum.simulation.gate_matrix",
    )

    assert lowerings.validate("pytorch", ["cphase_quarter"])[0].strategy == "lowering"
    for backend in BACKENDS:
        if backend == "pytorch":
            continue
        with pytest.raises(UnsupportedLoweringError):
            lowerings.validate(backend, ["cphase_quarter"])


def test_a_capability_declared_unsupported_keeps_the_operator_out_with_its_reason() -> (
    None
):
    registry = register_operator("cphase_quarter", arity=2)
    lowerings = DEFAULT_LOWERING_REGISTRY.with_operator(
        registry.schemas["cphase_quarter"]
    ).with_capability(
        backend="qasm",
        opcode="cphase_quarter",
        strategy="decomposition",
        implementation="flagquantum.compiler.openqasm",
        supported=False,
        reason="no OpenQASM 2 decomposition",
    )

    with pytest.raises(UnsupportedLoweringError) as caught:
        lowerings.validate("qasm", ["cphase_quarter"])

    assert str(caught.value) == (
        "backend 'qasm' cannot lower: cphase_quarter (no OpenQASM 2 decomposition)"
    )


def test_the_shared_registry_and_its_capability_document_do_not_move() -> None:
    before = DEFAULT_LOWERING_REGISTRY.manifest()

    register_operator("cphase_quarter", arity=2)

    assert DEFAULT_LOWERING_REGISTRY.manifest() == before
    assert "cphase_quarter" not in DEFAULT_LOWERING_REGISTRY.schemas


def test_an_unknown_backend_is_refused_rather_than_added() -> None:
    registry = register_operator("cphase_quarter", arity=2)
    lowerings = DEFAULT_LOWERING_REGISTRY.with_operator(
        registry.schemas["cphase_quarter"]
    )

    with pytest.raises(ValueError) as caught:
        lowerings.with_capability(
            backend="qpu",
            opcode="cphase_quarter",
            strategy="native",
            implementation="flagquantum.remote.qpu",
        )

    assert str(caught.value) == "unknown lowering backend 'qpu'"


def test_a_registered_opcode_still_needs_its_matrix_in_the_ir() -> None:
    register_operator("cphase_quarter", arity=2)

    with pytest.raises(IRValidationError) as caught:
        fq.Circuit(2).gate("cphase_quarter", (0, 1))

    assert str(caught.value) == (
        "unknown opcode 'cphase_quarter'; custom operations require an explicit matrix"
    )


def test_the_registration_and_the_emission_are_two_steps_a_user_can_see() -> None:
    registry = register_operator(
        "cphase_quarter",
        arity=2,
        adjoint="self_inverse",
        qubit_convention=("control", "target"),
    )
    lowerings = DEFAULT_LOWERING_REGISTRY.with_operator(
        registry.schemas["cphase_quarter"]
    ).with_capability(
        backend="pytorch",
        opcode="cphase_quarter",
        strategy="lowering",
        implementation="flagquantum.simulation.gate_matrix",
    )

    with pytest.raises(UnsupportedLoweringError):
        lowerings.validate("qasm", ["cphase_quarter"])

    circuit = (
        fq.Circuit(2)
        .h(0)
        .h(1)
        .gate("cphase_quarter", (0, 1), matrix=_cphase_quarter_matrix())
    )

    assert _amplitudes(circuit) == [0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5j]
    assert [item.name for item in circuit.to_ir().instructions if item.name != "h"] == [
        "cphase_quarter"
    ]


def test_the_new_names_stay_in_their_namespace_and_out_of_the_root() -> None:
    assert operators.register_operator is register_operator
    assert operators.OperatorRegistry is OperatorRegistry
    assert "register_operator" in operators.__all__
    assert "OperatorRegistry" in operators.__all__
    assert not hasattr(fq, "register_operator")
    assert not hasattr(fq, "OperatorRegistry")
    assert "register_operator" not in fq.__all__
    assert "OperatorRegistry" not in fq.__all__
