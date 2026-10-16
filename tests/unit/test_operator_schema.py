import math
from types import MappingProxyType

import pytest
import torch

import flagquantum.operators as fqo
from flagquantum import Circuit
from flagquantum.compiler.operator_lowering import (
    DEFAULT_LOWERING_REGISTRY,
    OperatorLoweringRegistry,
    UnsupportedLoweringError,
    validate_lowering,
)
from flagquantum.core import (
    OPERATOR_ALIASES,
    OPERATOR_SCHEMAS,
    OperatorSchema,
    canonical_opcode,
)
from flagquantum.core.operator_schema import parameter_shift_rule
from flagquantum.simulation import matrices

pytestmark = pytest.mark.unit


def test_every_parameterized_unitary_declares_its_frequencies():
    for opcode, schema in OPERATOR_SCHEMAS.items():
        if not schema.unitary or not schema.parameters:
            continue
        assert len(schema.parameter_frequencies) == len(schema.parameters), opcode
        assert schema.differentiable, opcode
        for frequencies in schema.parameter_frequencies:
            assert frequencies, opcode
            assert all(value > 0 for value in frequencies), opcode


def test_differentiable_follows_from_the_declared_frequencies():
    assert OPERATOR_SCHEMAS["rx"].differentiable
    assert OPERATOR_SCHEMAS["crx"].differentiable
    assert OPERATOR_SCHEMAS["crx"].parameter_frequencies == ((0.5, 1.0),)
    assert not OPERATOR_SCHEMAS["x"].differentiable
    assert not OPERATOR_SCHEMAS["depolarizing"].differentiable


def test_schema_rejects_a_frequency_set_that_cannot_be_used():
    declared = dict(
        opcode="odd",
        aliases=(),
        arity=1,
        parameters=("theta",),
        dtype_policy=("complex64",),
        semantic_kind="unitary",
        adjoint="matrix_adjoint",
        decomposition=(),
        qubit_convention=("target",),
    )
    with pytest.raises(ValueError, match="2 frequency set"):
        OperatorSchema(**declared, parameter_frequencies=((1.0,), (2.0,)))
    with pytest.raises(ValueError, match="only equidistant frequencies"):
        OperatorSchema(**declared, parameter_frequencies=((1.0, 2.0, 4.0),))
    with pytest.raises(ValueError, match="must be positive"):
        OperatorSchema(**declared, parameter_frequencies=((0.0,),))
    with pytest.raises(ValueError, match="must be unique"):
        OperatorSchema(**declared, parameter_frequencies=((1.0, 1.0),))
    with pytest.raises(ValueError, match="cannot declare parameter frequencies"):
        OperatorSchema(
            **{**declared, "semantic_kind": "channel", "adjoint": "not_applicable"},
            parameter_frequencies=((1.0,),),
        )


def test_shift_rule_is_a_two_term_rule_for_one_frequency():
    rule = parameter_shift_rule((1.0,))

    assert len(rule) == 2
    assert rule[0] == pytest.approx((0.5, math.pi / 2))
    assert rule[1] == pytest.approx((-0.5, -math.pi / 2))


def test_shift_rule_is_a_four_term_rule_for_the_controlled_rotations():
    rule = parameter_shift_rule((0.5, 1.0))
    shifts = sorted(abs(shift) for _, shift in rule)

    assert sorted(coefficient for coefficient, _ in rule) == pytest.approx(
        [
            -0.42677669529663687,
            -0.07322330470336312,
            0.07322330470336312,
            0.42677669529663687,
        ]
    )
    assert shifts == pytest.approx(
        [math.pi / 2, math.pi / 2, 3 * math.pi / 2, 3 * math.pi / 2]
    )
    assert sum(coefficient for coefficient, _ in rule) == pytest.approx(0.0)


def test_shift_rule_rejects_a_declaration_it_cannot_differentiate():
    with pytest.raises(ValueError, match="at least one frequency"):
        parameter_shift_rule(())


def test_schema_shift_rule_names_the_parameter_it_differentiates():
    assert OPERATOR_SCHEMAS["rx"].shift_rule("theta") == parameter_shift_rule((1.0,))
    assert OPERATOR_SCHEMAS["crx"].shift_rule("theta") == parameter_shift_rule(
        (0.5, 1.0)
    )
    with pytest.raises(ValueError, match="does not declare a differentiable"):
        OPERATOR_SCHEMAS["rx"].shift_rule("phi")
    with pytest.raises(ValueError, match="does not declare a differentiable"):
        OPERATOR_SCHEMAS["depolarizing"].shift_rule("theta")


def test_gate_info_exposes_user_facing_parameter_contract():
    info = fqo.gate_info("u")

    assert isinstance(info, fqo.GateInfo)
    assert info.name == "u3"
    assert info.n_qubits == 1
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


def test_schema_and_aliases_define_the_complete_circuit_method_set():
    for name in set(OPERATOR_SCHEMAS) | set(OPERATOR_ALIASES):
        assert callable(getattr(Circuit, name))


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
        qubit_convention=("target",),
    )
    base = OperatorLoweringRegistry()
    extended = base.with_operator(custom)
    lowered = extended.with_capability(
        backend="jax", opcode="custom", strategy="native", implementation="example"
    )
    assert "custom" not in base.schemas
    assert extended.capability("jax", "custom") is None
    assert lowered.capability("jax", "custom").supported


def test_default_manifest_covers_every_schema_and_backend():
    manifest = DEFAULT_LOWERING_REGISTRY.manifest()["backends"]
    for backend, capabilities in manifest.items():
        assert set(capabilities["supported"]) | set(capabilities["unsupported"]) == set(
            OPERATOR_SCHEMAS
        ), backend
