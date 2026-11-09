"""Pinned consistency decisions over native quantum and classical metadata."""

from enum import Enum

import numpy as np
import pytest
import sympy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
N = sympy.Symbol("n", integer=True, positive=True)


class Severity(Enum):
    LOOSE = 0
    ANY = 1
    STRICT = 2


@pytest.fixture(scope="module")
def register_cases():
    import qualtran as source

    from flagquantum.core.ir.program import register_types as native
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    cases = []
    factories = [
        (source.QAny, native.opaque_register_type),
        (source.QUInt, native.unsigned_register_type),
        (source.QInt, native.signed_register_type),
        (source.QIntOnesComp, native.ones_complement_register_type),
        (source.QMontgomeryUInt, native.montgomery_register_type),
        (source.CUInt, native.classical_unsigned_register_type),
        (source.CInt, native.classical_signed_register_type),
        (source.CIntOnesComp, native.classical_ones_complement_register_type),
        (source.CMontgomeryUInt, native.classical_montgomery_register_type),
    ]
    for original, replacement in factories:
        for width in (0, 1, 3, 4, N):
            if original in (source.QIntOnesComp, source.CIntOnesComp) and width == 1:
                continue
            cases.append((original(width), replacement(width)))
    cases.extend(
        [
            (source.QBit(), native.quantum_bit_type()),
            (source.CBit(), native.classical_bit_type()),
        ]
    )
    for width in (1, 3, 4, N):
        cases.extend(
            [
                (
                    source.BQUInt(width, 1),
                    native.bounded_unsigned_register_type(width, 1),
                ),
                (
                    source.BCUInt(width, 1),
                    native.classical_bounded_unsigned_register_type(width, 1),
                ),
                (
                    source.QMontgomeryUInt(width, 3),
                    native.montgomery_register_type(width, 3),
                ),
                (
                    source.CMontgomeryUInt(width, 3),
                    native.classical_montgomery_register_type(width, 3),
                ),
            ]
        )
        for fractional in (0, width):
            cases.extend(
                [
                    (
                        source.QFxp(width, fractional),
                        native.fixed_point_register_type(width, fractional),
                    ),
                    (
                        source.CFxp(width, fractional),
                        native.classical_fixed_point_register_type(width, fractional),
                    ),
                ]
            )
    for width in (3, 4, N):
        for signed in (False, True):
            cases.extend(
                [
                    (
                        source.QFxp(width, 1, signed),
                        native.fixed_point_register_type(width, 1, signed),
                    ),
                    (
                        source.CFxp(width, 1, signed),
                        native.classical_fixed_point_register_type(width, 1, signed),
                    ),
                ]
            )
    for characteristic, degree in ((2, 1), (2, 2), (2, 3), (3, 2), (2, N)):
        for classical in (False, True):
            for explicit_none in (False, True):
                constructor = source.CGF if classical else source.QGF
                reference = (
                    constructor(characteristic, degree, None)
                    if explicit_none
                    else constructor(characteristic, degree)
                )
                field = (
                    FiniteFieldEncoding(characteristic, degree, None)
                    if explicit_none
                    else FiniteFieldEncoding(characteristic, degree)
                )
                cases.append((reference, field.register_type(classical=classical)))
    for width in (np.int64(3), np.int64(-1)):
        cases.extend(
            [
                (source.QUInt(width), native.unsigned_register_type(width)),
                (source.QInt(width), native.signed_register_type(width)),
                (source.CUInt(width), native.classical_unsigned_register_type(width)),
                (
                    source.QFxp(width, width),
                    native.fixed_point_register_type(width, width),
                ),
            ]
        )
    for signed in (None, np.bool_(False), np.bool_(True)):
        cases.append(
            (source.QFxp(3, 0, signed), native.fixed_point_register_type(3, 0, signed))
        )
    return cases


def reference_check(left, right, severity):
    from qualtran._infra.data_types import (
        QDTypeCheckingSeverity,
        check_dtypes_consistent,
    )

    return check_dtypes_consistent(
        left[0], right[0], QDTypeCheckingSeverity[severity.name]
    )


def native_check(left, right, severity):
    from flagquantum.core.ir.program.register_compatibility import (
        RegisterTypeCheckingSeverity,
        check_register_types_consistent,
    )

    return check_register_types_consistent(
        left[1], right[1], RegisterTypeCheckingSeverity[severity.name]
    )


@pytest.fixture(params=["source", "native"])
def check(request):
    return reference_check if request.param == "source" else native_check


def outcome(call):
    try:
        result = call()
    except (TypeError, ValueError) as error:
        return type(error), str(error)
    return type(result), result


@pytest.mark.parametrize("severity", list(Severity))
def test_supported_type_cross_product(check, register_cases, severity):
    comparisons = 0
    for left in register_cases:
        for right in register_cases:
            assert outcome(
                lambda left=left, right=right: check(left, right, severity)
            ) == outcome(
                lambda left=left, right=right: reference_check(left, right, severity)
            ), (
                left[0],
                right[0],
                severity,
            )
            comparisons += 1
    assert comparisons == len(register_cases) ** 2


def test_enum_values_and_default_severity():
    from flagquantum.core.ir.program.register_compatibility import (
        RegisterTypeCheckingSeverity,
        check_register_types_consistent,
    )
    from flagquantum.core.ir.program.register_types import (
        signed_register_type,
        unsigned_register_type,
    )

    assert [(item.name, item.value) for item in RegisterTypeCheckingSeverity] == [
        (item.name, item.value) for item in Severity
    ]
    left, right = signed_register_type(3), unsigned_register_type(3)
    assert check_register_types_consistent(left, right)
    assert not check_register_types_consistent(
        left, right, RegisterTypeCheckingSeverity.ANY
    )
    assert not check_register_types_consistent(
        left, right, RegisterTypeCheckingSeverity.STRICT
    )


@pytest.mark.parametrize("severity", list(Severity))
def test_field_none_exception_precedes_severity(check, register_cases, severity):
    fields = [
        pair
        for pair in register_cases
        if pair[1].encoding == "finite_field" and pair[1].num_qubits == 3
    ]
    present = next(pair for pair in fields if "polynomial" in dict(pair[1].parameters))
    missing = next(
        pair for pair in fields if "polynomial" not in dict(pair[1].parameters)
    )
    with pytest.raises(TypeError, match="NoneType"):
        check(present, missing, severity)
    with pytest.raises(TypeError, match="NoneType"):
        check(missing, present, severity)


@pytest.mark.parametrize("severity", list(Severity))
def test_source_native_consistency_replacement_in_append_consumer(check, severity):
    from collections import Counter

    import torch
    from qualtran import QGF, QFxp
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.finite_field.field_multiplication import (
        bind_field_outputs,
        gf2_multiplication,
    )
    from flagquantum.core.ir.program.register_types import fixed_point_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
    from tests.unit.test_field_multiplication import (
        assert_gate_streams,
        reference_gates,
    )

    field = FiniteFieldEncoding(2, 3)
    left = (QGF(2, 3), field.register_type())
    expected_type = (QFxp(3, 0), fixed_point_register_type(3, 0))
    decision = check(left, expected_type, severity)
    assert decision == reference_check(left, expected_type, severity)
    if severity is not Severity.LOOSE:
        assert not decision
        return
    assert decision
    signature = Signature(
        tuple(Register(name, left[1]) for name in ("x", "y", "result"))
    )
    registers = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    plan = gf2_multiplication(
        *registers, (), modulus=int(field.irreducible_poly), algorithm="quadratic"
    )
    upstream = GF2Multiplication(left[0])
    gates, _ = reference_gates(
        upstream, {"x": registers[0], "y": registers[1]}, registers[2]
    )
    assert_gate_streams([(gate.name, gate.wires) for gate in plan.instructions], gates)
    emitted = Counter(gate.name for gate in plan.instructions)
    model = get_cost_value(upstream, QECGatesCost())
    assert (emitted["ccx"], emitted["cx"]) == (model.toffoli, model.clifford)
    generator = torch.Generator().manual_seed(27)
    initial = torch.randn(64, generator=generator, dtype=torch.complex128)
    initial /= initial.norm()
    state = torch.zeros(512, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for x in range(8):
        for y in range(8):
            index = (x * 8 + y) * 8
            state[index] = initial[x * 8 + y]
            product = int(field.gf_type(x) * field.gf_type(y))
            expected[index + product] = state[index]
    instructions = bind_field_outputs(plan, registers)
    result = (
        Circuit(int(signature.n_qubits()), inputs=state, dtype=torch.complex128)
        .append(FieldOperation(9, instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-12)


def test_neutral_consistency_has_no_galois_dependency(monkeypatch):
    import sys

    from flagquantum.core.ir.program.register_compatibility import (
        check_register_types_consistent,
    )
    from flagquantum.core.ir.program.register_types import fixed_point_register_type
    from flagquantum.core.ir.program.registers import RegisterType

    monkeypatch.setitem(sys.modules, "galois", None)
    field = RegisterType(
        "finite_field",
        3,
        parameters=(("characteristic", 2), ("degree", 3), ("order", 8)),
    )
    assert check_register_types_consistent(field, fixed_point_register_type(3, 0))
