"""Metadata-driven tensor codecs reuse each pinned source dtype's actual path."""

from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program import register_types as types
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_register_codec import (
    decode_register_bits,
    encode_register_bits,
)
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
    classical_action_tensor_node,
)
from tests.unit.test_component_opaque_values import identity_record

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def pair(kind: str) -> tuple[RegisterType, Any]:
    import qualtran as q

    constructors: dict[str, tuple[Any, Any]] = {
        "bit": (types.quantum_bit_type, q.QBit),
        "cbit": (types.classical_bit_type, q.CBit),
        "opaque": (lambda: types.opaque_register_type(2), lambda: q.QAny(2)),
        "uint": (lambda: types.unsigned_register_type(2), lambda: q.QUInt(2)),
        "cuint": (
            lambda: types.classical_unsigned_register_type(2),
            lambda: q.CUInt(2),
        ),
        "signed": (lambda: types.signed_register_type(2), lambda: q.QInt(2)),
        "csigned": (lambda: types.classical_signed_register_type(2), lambda: q.CInt(2)),
        "ones": (
            lambda: types.ones_complement_register_type(2),
            lambda: q.QIntOnesComp(2),
        ),
        "cones": (
            lambda: types.classical_ones_complement_register_type(2),
            lambda: q.CIntOnesComp(2),
        ),
        "bounded": (
            lambda: types.bounded_unsigned_register_type(2, 3),
            lambda: q.BQUInt(2, 3),
        ),
        "cbounded": (
            lambda: types.classical_bounded_unsigned_register_type(2, 3),
            lambda: q.BCUInt(2, 3),
        ),
        "fxp": (lambda: types.fixed_point_register_type(2, 1), lambda: q.QFxp(2, 1)),
        "cfxp": (
            lambda: types.classical_fixed_point_register_type(2, 1),
            lambda: q.CFxp(2, 1),
        ),
        "sfxp": (
            lambda: types.fixed_point_register_type(2, 1, True),
            lambda: q.QFxp(2, 1, True),
        ),
        "csfxp": (
            lambda: types.classical_fixed_point_register_type(2, 1, True),
            lambda: q.CFxp(2, 1, True),
        ),
        "mont": (
            lambda: types.montgomery_register_type(2, 3),
            lambda: q.QMontgomeryUInt(2, 3),
        ),
        "cmont": (
            lambda: types.classical_montgomery_register_type(2, 3),
            lambda: q.CMontgomeryUInt(2, 3),
        ),
    }
    native, source = constructors[kind]
    return native(), source()


KINDS = [
    "bit",
    "cbit",
    "opaque",
    "uint",
    "cuint",
    "signed",
    "csigned",
    "ones",
    "cones",
    "bounded",
    "cbounded",
    "fxp",
    "cfxp",
    "sfxp",
    "csfxp",
    "mont",
    "cmont",
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(), (2,), (1, 2)])
def test_all_small_bit_patterns_and_output_codecs(
    kind: str, shape: tuple[int, ...]
) -> None:
    dtype, source = pair(kind)
    register = Register("x", dtype, shape)
    width = int(register.bitsize)
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
        except (ValueError, TypeError) as error:
            with pytest.raises(type(error)):
                decode_register_bits(register, bits)
            continue
        actual = decode_register_bits(register, bits)
        np.testing.assert_equal(actual, expected)
        assert type(actual) is type(expected)
        if isinstance(actual, np.ndarray):
            assert isinstance(expected, np.ndarray)
            assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(
            encode_register_bits(register, np.asarray(actual)),
            source.to_bits_array(np.asarray(expected)),
        )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(0,), (2, 0)])
def test_empty_array_policy_is_delegated_to_each_original_codec(
    kind: str, shape: tuple[int, ...]
) -> None:
    dtype, source = pair(kind)
    register = Register("x", dtype, shape)
    bits = np.empty(shape + (int(register.bitsize),), dtype=np.uint8)
    cases: list[
        tuple[
            Callable[[np.ndarray[Any, Any]], object],
            Callable[[np.ndarray[Any, Any]], object],
            np.ndarray[Any, Any],
        ]
    ] = [
        (source.from_bits_array, lambda b: decode_register_bits(register, b), bits),
        (
            source.to_bits_array,
            lambda v: encode_register_bits(register, v),
            np.empty(shape, dtype=int),
        ),
    ]
    for reference, native, value in cases:
        try:
            expected = reference(value)
        except (ValueError, TypeError) as error:
            with pytest.raises(type(error)):
                native(value)
        else:
            np.testing.assert_equal(native(value), expected)


@pytest.mark.parametrize("kind", KINDS)
def test_default_dense_entry_replaces_handwritten_dispatch(kind: str) -> None:
    from qualtran import Bloq
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    dtype, source_dtype = pair(kind)
    # A full-capacity bound lets every binary input reach the value action.
    if kind in {"bounded", "cbounded"}:
        import qualtran as q

        dtype = (
            types.bounded_unsigned_register_type(2)
            if kind == "bounded"
            else types.classical_bounded_unsigned_register_type(2)
        )
        source_dtype = q.BQUInt(2) if kind == "bounded" else q.BCUInt(2)
    signature = Signature([Register("x", dtype)])
    source_type: Any = type(
        "IdentityCodec",
        (Bloq,),
        {
            "signature": SourceSignature([SourceRegister("x", source_dtype)]),
            "on_classical_vals": lambda self, **v: v,
        },
    )
    source = source_type()
    expected = bloq_to_dense_via_classical_action(source)
    actual = classical_action_tensor(
        signature, source.call_classically, operation_name=str(source)
    )
    explicit = classical_action_tensor(
        signature,
        source.call_classically,
        decode_register_bits,
        encode_register_bits,
        operation_name=str(source),
    )
    np.testing.assert_equal(actual, expected)
    np.testing.assert_equal(actual, explicit)
    # Ones' complement preserves source negative-zero collapse, not a repaired identity.
    assert actual.shape == (2 ** int(dtype.num_bits),) * 2


@pytest.mark.parametrize("kind", ["uint", "signed", "ones", "fxp", "sfxp", "mont"])
def test_node_uses_same_codec_and_preserves_complex_consumer(kind: str) -> None:
    dtype, _ = pair(kind)
    signature = Signature([Register("x", dtype)])
    like = torch.empty((), dtype=torch.complex128)
    node = classical_action_tensor_node(
        signature,
        {"x": ((2, 3),)},
        {"x": ((0, 1),)},
        lambda **v: (v["x"],),
        decode_register_bits,
        encode_register_bits,
        like,
        operation_name="identity",
    )
    matrix = classical_action_tensor(
        signature, lambda **v: (v["x"],), operation_name="identity"
    )
    np.testing.assert_equal(node.tensor.reshape(4, 4).numpy(), matrix)
    amplitudes = torch.tensor([1 + 2j, 3 - 4j, -2 + 1j, 4 + 3j], dtype=torch.complex128)
    torch.testing.assert_close(
        node.tensor.reshape(4, 4) @ amplitudes,
        torch.as_tensor(matrix, dtype=like.dtype) @ amplitudes,
    )
    assert node.labels == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "dtype",
    [
        RegisterType("foreign", 2),
        RegisterType("uint", -1),
        RegisterType("uint", 2, parameters=(("unknown", 1),)),
    ],
)
def test_unsupported_metadata_has_no_codec_fallback(dtype: RegisterType) -> None:
    register = Register("x", dtype)
    with pytest.raises(ValueError):
        decode_register_bits(register, np.array([0, 1]))
    with pytest.raises(ValueError):
        encode_register_bits(register, np.array(1))


def test_metadata_composition_does_not_expand_core_admission() -> None:
    from dataclasses import replace

    record = identity_record(2)
    register = Register("x", types.fixed_point_register_type(2, 1))
    changed = replace(
        record,
        functions=tuple(replace(f, registers=(register,)) for f in record.functions),
    )
    assert verify_program_record(changed)


@pytest.mark.parametrize("depth", [1, 3])
def test_default_tensor_codecs_match_add_gates_all_complex_branches(depth: int) -> None:
    from collections import Counter

    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.simulation.component_pure_evaluation import (
        simulate_pure_component_calls,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_classical_state import add_values
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_component_phased_evaluation import orders

    record, source = wrapped_add(depth)
    signature = Signature(record.functions[0].registers or ())
    output_names = tuple(reg.name for reg in signature.rights())

    def action(**values: object) -> tuple[object, ...]:
        result = simulate_pure_component_calls(
            record, orders(record), values, add_values
        )
        return tuple(result[name] for name in output_names)

    table = classical_action_tensor(signature, action, operation_name="stored_add")
    expected = bloq_to_dense_via_classical_action(source)
    np.testing.assert_array_equal(table, expected)
    initial = torch.zeros(256, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    amplitudes /= amplitudes.norm()
    initial[::4] = amplitudes
    target = torch.zeros_like(initial)
    target[::4] = torch.as_tensor(table, dtype=initial.dtype) @ amplitudes
    bound = bind_register_program(record, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    for branch in product((0, 1), repeat=4):
        actual = forced_branch(initial, bound.circuit.instructions, branch)
        torch.testing.assert_close(actual, target / 4, rtol=0, atol=1e-12)
        assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "cx": 40,
        "h": 12,
        "t": 8,
        "tdg": 8,
        "s": 4,
        "measure": 4,
        "cz": 4,
        "reset": 4,
    }
    cost = get_cost_value(source, QECGatesCost())
    assert cost.total_t_count() == 16 and cost.measurement == 4
