"""Preserve raw Montgomery views when source serialization omits the modulus."""

from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def paired_views() -> Any:
    from qualtran import BloqBuilder, QMontgomeryUInt, Register, Signature
    from qualtran.bloqs.bookkeeping import Join, Split

    types = (QMontgomeryUInt(3), QMontgomeryUInt(3, 3))
    builder, inputs = BloqBuilder.from_signature(
        Signature(
            [
                Register(name, dtype)
                for name, dtype in zip(("a", "b"), types, strict=True)
            ]
        )
    )
    outputs = {}
    for name, dtype in zip(("a", "b"), types, strict=True):
        bits = builder.add(Split(dtype), reg=inputs[name])
        outputs[name] = builder.add(Join(dtype), reg=bits)
    return builder.finalize(**outputs)


@pytest.mark.parametrize("adjoint", (False, True))
def test_distinct_source_montgomery_views_preserve_full_raw_storage(
    adjoint: bool,
) -> None:
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.bookkeeping import Join, Split

    for value in range(8):
        outputs = []
        for dtype in (QMontgomeryUInt(3), QMontgomeryUInt(3, 3)):
            bits = Split(dtype).on_classical_vals(reg=value)["reg"]
            assert tuple(bits) == tuple((value >> shift) & 1 for shift in (2, 1, 0))
            assert Join(dtype).on_classical_vals(reg=bits)["reg"] == value
            outputs.append(tuple(bits))
        assert outputs[0] == outputs[1]
    graph = paired_views()
    program = import_register_graph(graph.adjoint() if adjoint else graph)
    assert program.functions[0].registers is not None
    assert (
        program.functions[0].registers[0].dtype
        == program.functions[0].registers[1].dtype
    )
    bound = bind_register_program(program, ((4, 0, 2), (5, 1, 3)))
    assert bound.outputs == ((4, 0, 2), (5, 1, 3))
    assert not bound.events and not bound.circuit.instructions
    circuit = Circuit(7)
    circuit.append(FieldOperation(7, bound.circuit.instructions))
    ir = circuit.to_ir()
    assert CircuitIR.from_json(ir.to_json()) == ir
    # All raw encodings and a phase-correlated spectator participate coherently.
    state = torch.tensor(
        [complex(k + 1, 3 - k) for k in range(128)], dtype=torch.complex128
    )
    state /= state.norm()
    torch.testing.assert_close(
        forced_branch(state, ir.instructions, ()), state, rtol=0, atol=0
    )


def test_non_view_collisions_still_refuse_ambiguous_live_semantics() -> None:
    from qualtran import BloqBuilder, QMontgomeryUInt, Signature
    from qualtran.bloqs.bookkeeping import Allocate, Free

    builder, _ = BloqBuilder.from_signature(Signature([]))
    for dtype in (QMontgomeryUInt(3), QMontgomeryUInt(3, 3)):
        word = builder.add(Allocate(dtype))
        builder.add(Free(dtype), reg=word)
    with pytest.raises(ValueError, match="Distinct live components"):
        import_register_graph(builder.finalize())
