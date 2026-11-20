"""Sided signed conversion composes shared And lifetimes through normal append."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.bits.signed_encoding import (
    ones_complement_to_twos_ir,
)
from flagquantum.compiler.registers.bits.signed_encoding_component import (
    signed_encoding_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (2, 3, 4, 5, 8, 10))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("expanded", (False, True))
def test_sided_types_aliases_shared_stream_and_resources(
    width: int, adjoint: bool, expanded: bool
) -> None:
    from qualtran._infra.data_types import QInt, QIntOnesComp
    from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
        SignedIntegerToTwosComplement,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    graph = signed_encoding_program(width, adjoint=adjoint)
    assert verify_program_record(graph) == ()
    ports = graph.functions[0].registers
    assert ports is not None
    before, after = (
        ("signed", "ones_complement") if adjoint else ("ones_complement", "signed")
    )
    assert [(r.name, r.dtype, r.side.name) for r in ports] == [
        ("x", RegisterType(before, width), "LEFT"),
        ("y", RegisterType(after, width), "RIGHT"),
    ]
    source = SignedIntegerToTwosComplement(width)
    assert source.signature.get_left("x").dtype == QIntOnesComp(width)
    assert source.signature.get_right("y").dtype == QInt(width)
    if expanded:
        graph, _ = expand_component_calls(graph)
    word = tuple(reversed(range(width)))
    work = tuple(range(width + 1, 2 * width - 1))
    bound = bind_register_program(graph, (word,), clean=work)
    assert bound.outputs == (word,)
    reference = ones_complement_to_twos_ir(word, work, adjoint=adjoint)
    # Typed lifetimes use fresh measurement SSA slots; the flat owner reuses one.
    slots = [
        g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    ]
    normalized = []
    for gate in bound.circuit.instructions:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            metadata["classical_bit"] = 0
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (0 if slot in slots else slot, value)
                for slot, value in metadata["conditions"]
            )
        normalized.append(
            type(gate)(gate.name, gate.wires, gate.params, metadata=metadata)
        )
    assert tuple(normalized) == reference
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 4 * (width - 2)
    assert counts["measure"] == counts["reset"] == counts["cz"] == width - 2
    assert counts["h"] == 3 * (width - 2)
    assert counts["s"] == width - 2
    assert counts["cx"] == 7 * (width - 2) + 1
    assert counts["x"] == 4 * (width - 2) * adjoint
    assert get_cost_value(source, QECGatesCost()).toffoli == width - 2


@pytest.mark.parametrize("width", (2, 3, 4))
@pytest.mark.parametrize("expanded", (False, True))
def test_all_unnormalized_histories_complex_spectator_and_zero_work(
    width: int, expanded: bool
) -> None:
    word = tuple(range(width))
    work = tuple(range(width + 1, 2 * width - 1))
    n = width + 1 + len(work)
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    coefficients = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coefficients, 7 - coefficients * 2)
    amplitudes /= amplitudes.norm()
    for raw, spectator in product(range(1 << width), (0, 1)):
        index = ((raw << 1) | spectator) << len(work)
        sign = 1 << (width - 1)
        converted = raw if raw < sign else sign | ((raw + 1) % sign)
        output = ((converted << 1) | spectator) << len(work)
        initial[index] = amplitudes[2 * raw + spectator]
        expected[output] = amplitudes[2 * raw + spectator]
    forward = signed_encoding_program(width)
    inverse = signed_encoding_program(width, adjoint=True)
    circuits = []
    for graph in (forward, inverse):
        if expanded:
            graph, _ = expand_component_calls(graph)
        circuits.append(
            bind_register_program(graph, (word,), clean=work).circuit.instructions
        )
    histories = tuple(product((0, 1), repeat=len(work)))
    for first, second in product(histories, repeat=2):
        state = forced_branch(initial, circuits[0], first)
        torch.testing.assert_close(
            state, expected / 2 ** (len(work) / 2), atol=3e-12, rtol=0
        )
        assert torch.count_nonzero(state.reshape(-1, 1 << len(work))[:, 1:]) == 0
        restored = forced_branch(state, circuits[1], second)
        torch.testing.assert_close(
            restored, initial / 2 ** len(work), atol=3e-12, rtol=0
        )


def test_nested_sided_calls_replace_flat_emitter_and_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.registers.bits import signed_encoding

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Typed composition must use its shared child owners")

    monkeypatch.setattr(signed_encoding, "ones_complement_to_twos_ir", refuse)
    dtype = RegisterType("ones_complement", 3)
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(x=dtype))
    converted = builder.call_program(signed_encoding_program(3), builder.inputs)
    restored = builder.call_program(
        signed_encoding_program(3, adjoint=True), {"x": converted["y"]}
    )
    graph = builder.finalize({"x": restored["y"]})
    bound = bind_register_program(graph, ((2, 0, 1),), clean=(4,))
    assert bound.outputs == ((2, 0, 1),)
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[::2] = torch.complex(
        torch.arange(1, 17).double(), torch.arange(16).double()
    )
    initial /= initial.norm()
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(5, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=37, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), initial.expand(8, -1), atol=3e-12, rtol=0
    )


@pytest.mark.parametrize(
    "width,adjoint", ((1, False), (0, False), (True, False), (3, 1))
)
def test_invalid_width_or_direction_is_rejected(width: int, adjoint: bool) -> None:
    with pytest.raises(ValueError):
        signed_encoding_program(width, adjoint=adjoint)
