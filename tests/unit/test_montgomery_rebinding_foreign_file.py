"""Preserve raw Montgomery views when source serialization omits the modulus."""

from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import montgomery_register_type
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def paired_views() -> Any:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QMontgomeryUInt
    from qualtran._infra.registers import Register, Signature
    from qualtran.bloqs.bookkeeping.join import Join
    from qualtran.bloqs.bookkeeping.split import Split

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
    from qualtran._infra.data_types import QMontgomeryUInt
    from qualtran.bloqs.bookkeeping.join import Join
    from qualtran.bloqs.bookkeeping.split import Split

    for value in range(8):
        outputs = []
        for dtype in (QMontgomeryUInt(3), QMontgomeryUInt(3, 3)):
            bits = Split(dtype).on_classical_vals(reg=value)["reg"]
            assert isinstance(bits, np.ndarray)
            assert tuple(bits) == tuple((value >> shift) & 1 for shift in (2, 1, 0))
            # The source Join annotation excludes Split's actual uint8 output.
            # Keep the unmodified array and validate the real callback result.
            original_join: Any = Join(dtype)
            restored = original_join.on_classical_vals(reg=bits)["reg"]
            assert isinstance(restored, (int, np.integer)) and restored == value
            outputs.append(tuple(bits))
        assert outputs[0] == outputs[1]
    graph = paired_views()
    program = import_register_graph(graph.adjoint() if adjoint else graph)
    assert program.functions[0].registers is not None
    assert tuple(r.dtype for r in program.functions[0].registers) == (
        montgomery_register_type(3),
        montgomery_register_type(3, 3),
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


def test_distinct_live_montgomery_lifetimes_preserve_nominal_types() -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QMontgomeryUInt
    from qualtran._infra.registers import Signature
    from qualtran.bloqs.bookkeeping.allocate import Allocate
    from qualtran.bloqs.bookkeeping.free import Free

    builder, _ = BloqBuilder.from_signature(Signature([]))
    for dtype in (QMontgomeryUInt(3), QMontgomeryUInt(3, 3)):
        word = builder.add(Allocate(dtype))
        builder.add(Free(dtype), reg=word)
    graph = builder.finalize()
    program = import_register_graph(graph)
    assert len(program.functions[0].blocks[0].operations) == 4
    bound = bind_register_program(program, (), clean=(0, 1, 2))
    assert bound.outputs == ()
    assert not bound.circuit.instructions
    assert [event.kind for event in bound.events] == ["allocate", "free"] * 2
    assert {r.dtype for f in program.functions[1:] for r in f.registers or ()} == {
        montgomery_register_type(3),
        montgomery_register_type(3, 3),
    }
