from types import MappingProxyType

import pytest
import torch

import flagquantum.operators as fqo
from flagquantum import Circuit
from flagquantum.core import (
    OPERATOR_ALIASES,
    OPERATOR_SCHEMAS,
    OperatorSchema,
    canonical_opcode,
)
from flagquantum.ops import (
    DEFAULT_LOWERING_REGISTRY,
    OperatorLoweringRegistry,
    UnsupportedLoweringError,
    matrices,
    register_gate,
    registered_gates,
    registry,
    validate_lowering,
)

pytestmark = pytest.mark.unit


def test_gate_info_exposes_user_facing_parameter_contract():
    info = fqo.gate_info("u")

    assert isinstance(info, fqo.GateInfo)
    assert info.name == "u3"
    assert info.n_wires == 1
    assert info.parameters == ("theta", "phi", "lbd")
    assert info.n_parameters == 3
    assert dict(info.parameter_shapes) == {"theta": (), "phi": (), "lbd": ()}
    with pytest.raises(ValueError, match="unknown FlagQuantum gate"):
        fqo.gate_info("missing")


def test_schema_is_immutable_and_aliases_are_canonical():
    assert isinstance(OPERATOR_SCHEMAS, MappingProxyType)
    assert isinstance(OPERATOR_ALIASES, MappingProxyType)
    assert canonical_opcode("CNOT") == "cx"
    assert canonical_opcode("fredkin") == "cswap"
    with pytest.raises(TypeError):
        OPERATOR_SCHEMAS["new"] = OPERATOR_SCHEMAS["x"]


def test_schema_drives_circuit_methods_and_parameter_order():
    circuit = Circuit(2).p(0, 0.2).u(1, 0.1, 0.2, 0.3).cnot(0, 1)
    instructions = circuit.to_ir().instructions
    assert [item.name for item in instructions] == ["phase", "u3", "cx"]
    assert tuple(instructions[1].params) == OPERATOR_SCHEMAS["u3"].parameters


def test_generated_gate_methods_accept_semantic_qubit_keywords():
    circuit = (
        Circuit(n_qubits=3)
        .h(qubit=0)
        .rx(qubit=1, theta=0.2)
        .cx(control=0, target=1)
        .rzz(qubit1=1, qubit2=2, theta=0.3)
        .ccx(control1=0, control2=1, target=2)
    )

    assert [instruction.wires for instruction in circuit.to_ir().instructions] == [
        (0,),
        (1,),
        (0, 1),
        (1, 2),
        (0, 1, 2),
    ]


def test_generated_gate_methods_allow_mixed_qubits_and_reject_ambiguity():
    circuit = Circuit(2).cx(0, target=1).rx(0.2, qubit=1)
    assert circuit.to_ir().instructions[-1].params["theta"] == 0.2

    with pytest.raises(TypeError, match="multiple names for qubit"):
        Circuit(1).h(qubit=0, target=0)
    with pytest.raises(TypeError, match="requires qubit arguments"):
        Circuit(2).cx(control=0)


def test_fixed_matrix_shapes_follow_schema_arity():
    for opcode, schema in OPERATOR_SCHEMAS.items():
        matrix = matrices.GATE_MAT_DICT.get(opcode)
        if schema.unitary and isinstance(matrix, torch.Tensor):
            assert matrix.shape == (2**schema.arity, 2**schema.arity)


def test_lowering_validation_rejects_unsupported_backend_operation():
    validate_lowering(Circuit(2).h(0).cx(0, 1).to_ir(), "qcis")
    unsupported = Circuit(2).cphase(0, 1, 0.3).to_ir()
    with pytest.raises(UnsupportedLoweringError, match="cphase"):
        validate_lowering(unsupported, "qcis")


def test_lowering_registry_is_copy_on_write():
    custom = OperatorSchema(
        opcode="custom",
        aliases=(),
        arity=1,
        parameters=(),
        dtype_policy=("complex64",),
        semantic_kind="unitary",
        adjoint="matrix_adjoint",
        decomposition=(),
        differentiable=False,
        wire_convention=("target",),
    )
    base = OperatorLoweringRegistry()
    extended = base.with_operator(custom)
    lowered = extended.with_capability(
        backend="jax", opcode="custom", strategy="native", implementation="example"
    )
    assert "custom" not in base.schemas
    assert extended.capability("jax", "custom") is None
    assert lowered.capability("jax", "custom").supported


def test_custom_registration_does_not_mutate_builtin_tables_or_globals():
    name = "issue074_custom_x"
    builtin_keys = tuple(matrices.GATE_MAT_DICT)
    record = register_gate(name, torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))
    assert record.arity == 1
    assert registered_gates()[name] == record
    assert tuple(matrices.GATE_MAT_DICT) == builtin_keys
    with pytest.raises(AttributeError):
        getattr(registry, name)


def test_default_manifest_covers_every_schema_and_backend():
    manifest = DEFAULT_LOWERING_REGISTRY.manifest()["backends"]
    for backend, capabilities in manifest.items():
        assert set(capabilities["supported"]) | set(capabilities["unsupported"]) == set(
            OPERATOR_SCHEMAS
        ), backend
