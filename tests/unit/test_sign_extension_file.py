"""Actual sign-extension factories compose through native physical capture."""

from collections import Counter
from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.bits.sign_extension import sign_extend_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.arithmetic.conversions import sign_extension

    for name, blob in (
        ("sign_extension.py", "16e8f2152796a823cbc54fafb1257b196fc1a515"),
        ("sign_extension_test.py", "e57180dda042e3f7c48c3ee809a013a65a6a773c"),
    ):
        raw = (Path(sign_extension.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_actual_factories_compose_capture_and_restore_full_complex_input() -> None:
    from qualtran import BloqBuilder, QInt, Signature
    from qualtran.bloqs.arithmetic.conversions.sign_extension import (
        _SIGN_EXTEND_DOC,
        _SIGN_TRUNCATE_DOC,
        _sign_extend,
        _sign_truncate,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    extend = _sign_extend.make()
    truncate = _sign_truncate.make()
    assert _SIGN_EXTEND_DOC.examples == (_sign_extend,)
    assert _SIGN_TRUNCATE_DOC.examples == (_sign_truncate,)
    assert extend.adjoint() == truncate and truncate.adjoint() == extend
    assert extend.extend_bitsize == truncate.truncate_bitsize == 8
    builder, inputs = BloqBuilder.from_signature(Signature.build_from_dtypes(x=QInt(8)))
    y = builder.add(extend, x=inputs["x"])
    x = builder.add(truncate, x=y)
    source = builder.finalize(x=x)
    with pytest.raises(ValueError, match="Wrong number of qubits"):
        import_register_emitter(source)
    # Use the original decomposition to expose the sided bookkeeping first.
    emitter = import_register_emitter(source.flatten_once())
    assert emitter.layout == (("x", 8),)
    assert emitter.requirements() == (8, 0) and emitter.binding_swaps == (0, 0)
    word = tuple(reversed(range(8)))
    work = tuple(range(9, 17))
    native = sign_extend_ir(word, work)
    initial = torch.zeros(1 << 17, dtype=torch.complex128)
    expected_extended = torch.zeros_like(initial)
    values = torch.tensor(
        [complex(i + 1, i % 7 - 3) for i in range(512)], dtype=torch.complex128
    )
    values /= values.norm()
    for raw in range(256):
        reversed_word = int(f"{raw:08b}"[::-1], 2)
        for spectator in (0, 1):
            label = (reversed_word << 9) | (spectator << 8)
            initial[label] = values[2 * raw + spectator]
            expected_extended[label | (255 if raw & 128 else 0)] = values[
                2 * raw + spectator
            ]
    for inverse in (False, True):
        gates = emitter({"x": word}, work, (), inverse)
        expected_half = tuple(reversed(native)) if inverse else native
        assert gates == (*expected_half, *expected_half)
        assert Counter(g.name for g in gates) == {"cx": 30}
        circuit = Circuit(17, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(17, gates[:15]))
        torch.testing.assert_close(
            circuit.state().flatten(), expected_extended, rtol=0, atol=2e-12
        )
        circuit.append(FieldOperation(17, gates[15:]))
        torch.testing.assert_close(
            circuit.state().flatten(), initial, rtol=0, atol=2e-12
        )
        assert torch.count_nonzero(circuit.state().reshape(-1, 256)[:, 1:]) == 0
        ir = Circuit(17).append(FieldOperation(17, gates)).to_ir()
        assert CircuitIR.from_json(ir.to_json()).instructions == gates
    # Each MultiTargetCNOT is one coarse Clifford in the original cost model.
    assert get_cost_value(source, QECGatesCost()).clifford == 2
    assert get_cost_value(source, QubitCount()) == 16


def test_original_truncate_join_dtype_loss_is_an_explicit_boundary() -> None:
    from qualtran import QAny, QInt
    from qualtran.bloqs.arithmetic.conversions.sign_extension import SignTruncate

    source = SignTruncate(QInt(16), QInt(8)).decompose_bloq()
    joins = [b.bloq for b in source.bloq_instances if type(b.bloq).__name__ == "Join"]
    assert len(joins) == 2
    assert all(join.dtype == QAny(8) for join in joins)
    assert source.signature.get_right("y").dtype == QInt(8)


@pytest.mark.parametrize("truncate", (False, True))
def test_default_native_score_preserves_source_sided_ports(truncate: bool) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.arithmetic.conversions.sign_extension import (
        _sign_extend,
        _sign_truncate,
    )
    from qualtran.drawing import get_musical_score_data

    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from tests.unit.test_drawer_component_score import assert_source_geometry
    from tests.unit.test_register_tensor import native_signature

    source = (_sign_truncate if truncate else _sign_extend).make()
    signature = native_signature(source.signature)
    inputs, outputs = tuple(signature.lefts()), tuple(signature.rights())
    step = RegisterStep(
        str(source),
        tuple(r.name for r in inputs),
        tuple(r.name for r in outputs),
        input_registers=inputs,
        output_registers=outputs,
    )
    score = layout_register_score(
        {r.name: int(r.bitsize) for r in signature},
        tuple(r.name for r in inputs),
        (step,),
    )
    builder, initial = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **initial))
    assert_source_geometry(
        register_score_data(score), get_musical_score_data(original), ordered_ports=True
    )
