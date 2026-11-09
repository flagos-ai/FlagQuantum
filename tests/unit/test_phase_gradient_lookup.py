"""Source graph and branch evidence for programmable phase-gradient lookup."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.lookup.source_qrom import source_qrom_workspace_size
from flagquantum.compiler.registers.phase.phase_gradient_lookup import (
    phase_gradient_lookup_ir,
    phase_gradient_lookup_requirements,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qrom import source_qrom_gates

pytestmark = pytest.mark.unit


def source_lookup_gates(
    source: Any,
    ports: SourcePorts,
    workspace: tuple[int, ...],
    bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    width = len(ports["phase_gradient"])
    angle, scratch = workspace[:width], workspace[width:]
    calls = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        calls.append(type(bloq))
        if isinstance(bloq, Allocate):
            assert (
                not bloq.dirty and bloq.signature.get_right("reg").total_bits() == width
            )
            return (), {"reg": angle}
        if isinstance(bloq, Free):
            assert inputs == {"reg": angle} and not bloq.dirty
            return (), {}
        if isinstance(bloq, QROM):
            words = tuple(int(value) for value in bloq.data[0])
            count = source_qrom_workspace_size(
                len(inputs.get("control", ())),
                (len(inputs.get("selection", ())),),
                (words,),
            )
            assert inputs["target0_"] == angle
            return (
                source_qrom_gates(bloq, inputs, scratch[:count], bits[:count]),
                inputs,
            )
        if isinstance(bloq, AddIntoPhaseGrad):
            assert (
                bloq.sign == 1 and bloq.right_shift == 0 and bloq.controlled_by is None
            )
            assert (
                inputs["x"] == angle and inputs["phase_grad"] == ports["phase_gradient"]
            )
            work = scratch[: width - 1]
            return (
                reference_arithmetic_gates(
                    Add(QUInt(width), QUInt(width)),
                    {"a": inputs["x"], "b": inputs["phase_grad"]},
                    work,
                    dict(zip(work, bits[: width - 1], strict=True)),
                ),
                inputs,
            )
        return None

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert calls == [type(source), Allocate, QROM, AddIntoPhaseGrad, QROM, Free]
    return gates


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "controls_count,words,width",
    [(0, (3,), 2), (1, (0, 2), 2), (2, (3, 3), 2), (2, (5, 255, 0, 93), 8)],
)
def test_phase_gradient_lookup_graph_and_resources(
    controls_count: int, words: tuple[int, ...], width: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        PRGAViaPhaseGradient,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    selection_width = (len(words) - 1).bit_length()
    controls = tuple(range(controls_count))
    selection = tuple(range(controls_count, controls_count + selection_width))
    start = controls_count + selection_width
    gradient = tuple(range(start, start + width))
    q, c = phase_gradient_lookup_requirements(
        controls_count, selection_width, width, words
    )
    work = tuple(range(start + width, start + width + q))
    bits = tuple(range(c))
    gates = phase_gradient_lookup_ir(controls, selection, gradient, words, work, bits)
    source = PRGAViaPhaseGradient(selection_width, width, words, controls_count)
    ports = {"phase_gradient": gradient}
    if controls:
        ports["control"] = controls
    if selection:
        ports["selection"] = selection
    expected = source_lookup_gates(source, ports, work, bits)
    assert_source(gates, expected)

    with pytest.raises(DecomposeNotImplementedError):
        source.add_into_phase_grad.decompose_bloq()
    counts = Counter(gate.name for gate in gates)
    qrom_model = get_cost_value(source.qrom_bloq, QECGatesCost())
    assert counts["t"] + counts["tdg"] == 4 * counts["measure"]
    assert counts["measure"] == 2 * qrom_model.measurement + width - 1
    assert counts["reset"] == counts["measure"]
    estimate = get_cost_value(source, QECGatesCost())
    assert estimate.toffoli == width - 2
    assert estimate.measurement == 2 * qrom_model.measurement

    if width == 2:
        # Coherent arbitrary addresses, controls, and gradient basis values.
        state = torch.zeros(1 << (start + width + q), dtype=torch.complex128)
        output = torch.zeros_like(state)
        for address, g in product(range(1 << start), range(1 << width)):
            index = (address << width | g) << q
            value = complex(index + 1, -(g + 1)) / state.numel()
            state[index] = value
            control = address >> selection_width
            word = (
                words[address % len(words)]
                if control == (1 << controls_count) - 1
                else 0
            )
            shifted = source.add_into_phase_grad.on_classical_vals(x=word, phase_grad=g)
            output[(address << width | int(shifted["phase_grad"])) << q] = value
        for history in product((0, 1), repeat=counts["measure"]):
            torch.testing.assert_close(
                forced_branch(state, gates, history),
                output * 2 ** (-counts["measure"] / 2),
                rtol=0,
                atol=2e-12,
            )
