"""Locked-source evidence for classical register ports before executable admission."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from flagquantum.core.ir.program.register_types import classical_bit_type

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_original_cbit_parity_preserves_owned_elements_and_big_endian_values() -> None:
    from qualtran import BloqBuilder, CBit, Register, Side, Signature
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity

    signature = Signature(
        [
            Register("x", CBit(), shape=(2,)),
            Register("parity", CBit(), side=Side.RIGHT),
        ]
    )
    builder, inputs = BloqBuilder.from_signature(signature)
    controls, parity = builder.add(Parity(2), x=inputs["x"])
    composite = builder.finalize(x=controls, parity=parity)
    assert signature.n_qubits() == 0
    assert classical_bit_type().num_qubits == 0
    assert classical_bit_type().num_cbits == 1
    for left, right in product((0, 1), repeat=2):
        data = np.array([left, right], dtype=np.uint8)
        returned, result = composite.call_classically(x=data)
        np.testing.assert_array_equal(returned, data)
        assert result == left ^ right


def test_original_cbit_output_is_not_cirq_exportable() -> None:
    import cirq
    from qualtran import BloqBuilder, CBit, Register, Side, Signature
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity

    signature = Signature(
        [
            Register("x", CBit(), shape=(2,)),
            Register("parity", CBit(), side=Side.RIGHT),
        ]
    )
    builder, inputs = BloqBuilder.from_signature(signature)
    controls, parity = builder.add(Parity(2), x=inputs["x"])
    composite = builder.finalize(x=controls, parity=parity)
    with pytest.raises(ValueError, match="Expected 0 qubits"):
        composite.to_cirq_circuit_and_quregs(
            x=np.array([cirq.LineQubit(0), cirq.LineQubit(1)]).reshape(2, 1)
        )


def test_original_measurement_creates_classical_right_port() -> None:
    from qualtran import BloqBuilder, CBit, QBit, Register, Side, Signature
    from qualtran.bloqs.basic_gates import MeasureZ
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    signature = Signature(
        [
            Register("q", QBit(), side=Side.LEFT),
            Register("c", CBit(), side=Side.RIGHT),
        ]
    )
    builder, inputs = BloqBuilder.from_signature(signature)
    measured = builder.add(MeasureZ(), q=inputs["q"])
    composite = builder.finalize(c=measured)
    assert [reg.name for reg in signature.lefts()] == ["q"]
    assert [reg.name for reg in signature.rights()] == ["c"]
    assert composite.call_classically(q=0) == (0,)
    assert composite.call_classically(q=1) == (1,)
    model = get_cost_value(composite, QECGatesCost())
    assert model.measurement == 1
    assert get_cost_value(composite, QubitCount()) == 1


def test_current_executable_register_boundary_rejects_classical_ports() -> None:
    from flagquantum.core.ir.program.register_verifier import executable_register_width

    with pytest.raises(ValueError, match="unsupported executable"):
        executable_register_width(classical_bit_type())
