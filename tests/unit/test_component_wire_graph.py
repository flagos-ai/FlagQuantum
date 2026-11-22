"""Native wire replay preserves frozen SDK insertion, types and lifetimes."""

from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.comparison._component_gates import (
    comparison_bit_functions,
)
from flagquantum.compiler.registers.component_wire_graph import (
    WireCall,
    wire_component_program,
)
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source, _source_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("shaped", (False, True))
def test_lazy_words_earliest_layers_seed_cleanup_and_complex_tensor(
    shaped: bool,
) -> None:
    import cirq
    from qualtran.cirq_interop import cirq_optree_to_cbloq

    ports = Signature(
        (
            Register(
                "x", unsigned_register_type(2 if shaped else 4), (2,) if shaped else ()
            ),
            Register("target", quantum_bit_type()),
        )
    )
    x, cx = (ProgramRecord((function,)) for function in comparison_bit_functions())
    calls: list[WireCall] = [
        (x, {"q": (9,)}),
        (cx, {"ctrl": (0,), "target": (1,)}),
        (cx, {"ctrl": (9,), "target": (4,)}),
        (x, {"q": (9,)}),
        (cx, {"ctrl": (3,), "target": (2,)}),
    ]
    wires = {"x": (0, 1, 2, 3), "target": (4,)}
    native = wire_component_program(ports, wires, wires, calls)
    qs = cirq.LineQubit.range(10)
    source_ports = {
        r.name: np.array([qs[w] for w in wires[r.name]], dtype=object).reshape(
            (*r.shape, r.bitsize)
        )
        for r in ports
    }
    source: Any = cirq_optree_to_cbloq(
        (
            cirq.X(qs[9]),
            cirq.CNOT(qs[0], qs[1]),
            cirq.CNOT(qs[9], qs[4]),
            cirq.X(qs[9]),
            cirq.CNOT(qs[3], qs[2]),
        ),
        signature=_source(ports).signature(ports),
        in_quregs=source_ports,
        out_quregs=source_ports,
    )
    assert verify_program_record(native) == ()
    assert _native_trace(native) == _source_trace(_source(ports), source)
    groups = ((0, 1), (2, 3), (4,)) if shaped else ((0, 1, 2, 3), (4,))
    bound = bind_register_program(native, groups, clean=(5,))
    raw = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(raw, 11 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::2] = amplitudes
    actual = forced_branch(initial, bound.circuit.instructions, ())
    expected = (
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
            32, 32
        )
        @ amplitudes
    )
    torch.testing.assert_close(actual[::2], expected, rtol=0, atol=1e-14)
    assert torch.count_nonzero(actual[1::2]) == 0
    assert [event.kind for event in bound.events] == ["allocate", "free"]


def test_explicit_cast_has_one_physical_wire_and_no_primitive() -> None:
    source, target = quantum_bit_type(), opaque_register_type(1)
    function = cast_word_function("typed_cast", source, target)
    ports = Signature((Register("q", source),))
    native = wire_component_program(
        ports, {"q": (0,)}, {"q": (0,)}, ((ProgramRecord((function,)), {"reg": (0,)}),)
    )
    bound = bind_register_program(native, ((0,),))
    assert bound.circuit.instructions == () and bound.outputs == ((0,),)


@pytest.mark.parametrize("wires", ((-1,), (True,), (0, 0), (0, 1)))
def test_invalid_word_identities_or_width_are_rejected(wires: tuple[int, ...]) -> None:
    ports = Signature((Register("q", quantum_bit_type()),))
    with pytest.raises(ValueError):
        wire_component_program(ports, {"q": wires}, {"q": wires}, ())


def test_overlapping_input_ports_and_child_ports_are_rejected() -> None:
    bit = quantum_bit_type()
    ports = Signature((Register("a", bit), Register("b", bit)))
    with pytest.raises(ValueError, match="inputs overlap"):
        wire_component_program(
            ports, {"a": (0,), "b": (0,)}, {"a": (0,), "b": (0,)}, ()
        )
    _, cx = (ProgramRecord((function,)) for function in comparison_bit_functions())
    with pytest.raises(ValueError, match="call ports overlap"):
        wire_component_program(
            ports,
            {"a": (0,), "b": (1,)},
            {"a": (0,), "b": (1,)},
            ((cx, {"ctrl": (0,), "target": (0,)}),),
        )


@pytest.mark.parametrize("barriers", [(-1,), (1,), (True,), (0, 0)])
def test_invalid_explicit_group_positions_reject(barriers: tuple[int, ...]) -> None:
    ports = Signature((Register("q", quantum_bit_type()),))
    x = ProgramRecord((comparison_bit_functions()[0],))
    with pytest.raises(ValueError, match="barriers"):
        wire_component_program(
            ports, {"q": (0,)}, {"q": (0,)}, ((x, {"q": (0,)}),), barriers=barriers
        )
