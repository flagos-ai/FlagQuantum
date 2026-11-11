"""Default field codec wiring preserves the frozen source's numerical boundaries."""

from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_register_codec import (
    decode_register_bits,
    encode_register_bits,
)
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = [
    (1, "default"),
    (2, "default"),
    (3, "default"),
    (3, "none"),
    (3, "11"),
    (3, "13"),
]


def field_pair(width: int, mode: str) -> tuple[FiniteFieldEncoding, Any]:
    import galois
    from qualtran import QGF

    if mode == "default":
        return FiniteFieldEncoding(2, width), QGF(2, width)
    polynomial = galois.Poly.Int(int(mode)) if mode.isdecimal() else None
    return FiniteFieldEncoding(2, width, polynomial), QGF(2, width, polynomial)


@pytest.mark.parametrize("width,mode", CASES)
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_all_field_words_types_polynomial_identity_and_array_errors(
    width: int, mode: str, shape: tuple[int, ...]
) -> None:
    field, source = field_pair(width, mode)
    register = Register("x", field.register_type(), shape)
    for word in product((0, 1), repeat=width):
        bits = np.array(word, dtype=np.uint8)
        if shape:
            bits = np.broadcast_to(bits, shape + (width,))
        try:
            expected = (
                source.from_bits_array(bits)
                if shape
                else source.from_bits(list(bits.flat))
            )
        except ValueError as error:
            with pytest.raises(type(error)):
                decode_register_bits(register, bits)
            continue
        actual = decode_register_bits(register, bits)
        assert type(actual) is type(expected) is field.gf_type
        np.testing.assert_array_equal(actual, expected)
        assert isinstance(actual, np.ndarray)
        for values in (actual, np.asarray(actual)):
            try:
                encoded = source.to_bits_array(values)
            except ValueError as error:
                with pytest.raises(type(error)):
                    encode_register_bits(register, values)
            else:
                result = encode_register_bits(register, values)
                np.testing.assert_array_equal(result, encoded)
                assert result.dtype == encoded.dtype == np.uint8


@pytest.mark.parametrize("width,mode", CASES)
def test_original_default_field_tensor_identity_failure_is_preserved(
    width: int, mode: str
) -> None:
    from qualtran import Bloq
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    field, dtype = field_pair(width, mode)
    source_type: Any = type(
        "FieldIdentity",
        (Bloq,),
        {
            "signature": SourceSignature([SourceRegister("x", dtype)]),
            "on_classical_vals": lambda self, **v: v,
        },
    )
    source = source_type()
    signature = Signature([Register("x", field.register_type())])
    with pytest.raises(ValueError, match="should be a") as expected:
        bloq_to_dense_via_classical_action(source)
    with pytest.raises(ValueError, match="should be a") as actual:
        classical_action_tensor(
            signature, source.call_classically, operation_name=str(source)
        )
    assert str(actual.value) == str(expected.value)
    assert type(actual.value.__cause__) is type(expected.value.__cause__) is ValueError
    assert str(actual.value.__cause__) == str(expected.value.__cause__)


@pytest.mark.parametrize("width,mode", CASES)
def test_field_input_only_effect_uses_default_decoder_successfully(
    width: int, mode: str
) -> None:
    from qualtran import Bloq, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    field, dtype = field_pair(width, mode)
    source_type: Any = type(
        "FieldEffect",
        (Bloq,),
        {
            "signature": SourceSignature([SourceRegister("x", dtype, side=Side.LEFT)]),
            "on_classical_vals": lambda self, **v: {},
        },
    )
    source = source_type()
    signature = Signature(
        [Register("x", field.register_type(), side=RegisterSide.LEFT)]
    )
    actual = classical_action_tensor(
        signature, source.call_classically, operation_name=str(source)
    )
    np.testing.assert_array_equal(actual, bloq_to_dense_via_classical_action(source))
    np.testing.assert_array_equal(actual, np.ones(2**width))


@pytest.mark.parametrize("width,mode", CASES)
def test_typed_field_codecs_full_add_matrix_exact_gates_and_complex_append(
    width: int, mode: str
) -> None:
    from collections import Counter

    import cirq
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.cirq_interop import BloqAsCirqGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.test_binary_field_encoding import fixture

    field, dtype = field_pair(width, mode)
    register = Register("x", field.register_type())
    source = GF2Addition(dtype)
    size = 1 << (2 * width)
    matrix = torch.zeros((size, size), dtype=torch.complex128)
    for x, y in product(range(1 << width), repeat=2):

        def decode(value: int) -> Any:
            return decode_register_bits(
                register, np.array(list(f"{value:0{width}b}"), dtype=np.uint8)
            )

        left, right = decode(x), decode(y)
        output = left + right
        bits = encode_register_bits(register, output)
        encoded = int("".join(map(str, bits.flat)), 2)
        assert encoded == int(dtype.from_bits(list(bits.flat)))
        matrix[(x << width) | encoded, (x << width) | y] = 1
    torch.testing.assert_close(
        matrix, torch.as_tensor(source.tensor_contract()), rtol=0, atol=1e-12
    )
    record = fixture(field)
    bound = bind_register_program(
        record,
        (tuple(range(width)), tuple(range(width, 2 * width))),
        clean=(2 * width,),
    )
    wires = np.asarray(cirq.LineQubit.range(2 * width))
    circuit, _ = source.decompose_bloq().to_cirq_circuit_and_quregs(
        x=wires[:width], y=wires[width:]
    )
    expected = []
    for op in circuit.all_operations():
        assert op.gate in {cirq.CNOT, BloqAsCirqGate(CNOT())}
        expected.append(("cx", tuple(q.x for q in op.qubits)))
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == expected
    assert Counter(g.name for g in bound.circuit.instructions) == {"cx": width}
    assert get_cost_value(source, QECGatesCost()).clifford == width
    initial = torch.zeros(2 * size, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, size + 1, dtype=torch.float64),
        torch.arange(size, 0, -1, dtype=torch.float64),
    )
    initial[::2] = amplitudes / amplitudes.norm()
    target = torch.zeros_like(initial)
    target[::2] = matrix @ initial[::2]
    operation = FieldOperation(2 * width + 1, bound.circuit.instructions)
    actual = (
        Circuit(2 * width + 1, inputs=initial, dtype=torch.complex128)
        .append(operation)
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, target, rtol=0, atol=1e-12)
    assert torch.count_nonzero(actual[1::2]) == 0
    restored = (
        Circuit(2 * width + 1, inputs=actual, dtype=torch.complex128)
        .append(operation)
        .state()
        .flatten()
    )
    torch.testing.assert_close(restored, initial, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "defect", ["classical", "odd", "order", "width", "extra", "partial", "symbolic"]
)
def test_field_default_obeys_existing_core_admission(defect: str) -> None:
    import sympy

    field = FiniteFieldEncoding(2, 2)
    original = field.register_type()
    base = original.parameters[:3]
    dtype = {
        "classical": field.register_type(classical=True),
        "odd": FiniteFieldEncoding(3, 2).register_type(),
        "order": RegisterType(
            "finite_field",
            2,
            parameters=(("characteristic", 2), ("degree", 2), ("order", 5)),
        ),
        "width": RegisterType("finite_field", 3, parameters=original.parameters),
        "extra": RegisterType(
            "finite_field", 2, parameters=original.parameters + (("unknown", 1),)
        ),
        "partial": RegisterType(
            "finite_field", 2, parameters=base + (("polynomial", 7),)
        ),
        "symbolic": RegisterType("finite_field", sympy.Symbol("n"), parameters=base),
    }[defect]
    register = Register("x", dtype)
    with pytest.raises(ValueError):
        decode_register_bits(register, np.array([0, 1]))
    with pytest.raises(ValueError):
        encode_register_bits(register, np.array(1))


def test_integer_default_does_not_resolve_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.classical_register_codec as owner
    from flagquantum.core.ir.program.register_types import unsigned_register_type

    monkeypatch.setattr(
        owner,
        "resolve_binary_field_encoding",
        lambda dtype: pytest.fail("field resolution"),
    )
    signature = Signature([Register("x", unsigned_register_type(2))])
    np.testing.assert_array_equal(
        classical_action_tensor(
            signature, lambda **v: (v["x"],), operation_name="identity"
        ),
        np.eye(4),
    )
