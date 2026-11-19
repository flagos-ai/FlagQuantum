"""Owned clean lookup banks preserve source load and all small erase branches."""

from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.lookup.clean_lookup_component import (
    clean_lookup_program,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qroam_clean import source_clean_graph

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("log", [0, 1])
def test_source_forward_primitive_operands_and_per_event_feedback(log: int) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    table = QroamCleanTable((4,), (log,), ((1, 2, 3, 0),))
    graph = clean_lookup_program(table, 2)
    bound = bind_register_program(
        graph, ((0, 1),), clean=tuple(range(2, 14)), allow_workspace_outputs=True
    )
    outputs = {"selection": bound.outputs[0], "target0_": bound.outputs[1]}
    if log:
        outputs["junk_target0_"] = tuple(w for word in bound.outputs[2:] for w in word)
    bank = {w for word in bound.outputs[1:] for w in word}
    work = tuple(
        w
        for e in bound.events
        if e.kind == "allocate"
        for w in e.wires
        if w not in bank
    )
    source = QROAMClean.build_from_data(
        np.array(table.datasets[0]), target_bitsizes=(2,), log_block_sizes=(log,)
    )
    expected, _, _ = source_clean_graph(
        source, {"selection": (0, 1)}, outputs, work, tuple(range(len(work)))
    )
    slots = MeasurementSlots(bound.circuit.instructions)
    remapped = []
    active: dict[int, int] = {}
    for gate in expected:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            old = int(metadata["classical_bit"])
            active[old] = slots[gate.wires[0]]
            metadata["classical_bit"] = active[old]
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (active[bit], value) for bit, value in metadata["conditions"]
            )
        remapped.append(
            Instruction(gate.name, gate.wires, gate.params, metadata=metadata)
        )
    assert_source(bound.circuit.instructions, tuple(remapped))


@pytest.mark.parametrize("log", [0, 1])
def test_all_raw_lookup_erase_histories_complex_spectator_and_clean_work(
    log: int,
) -> None:
    table = QroamCleanTable((4,), (log,), ((1, 2, 3, 0),))
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(selection=unsigned_register_type(2))
    )
    loaded = builder.call_program(
        clean_lookup_program(table, 2), {"selection": builder.inputs["selection"]}
    )
    restored = builder.call_program(clean_lookup_program(table, 2, erase=True), loaded)
    graph = builder.finalize(restored)
    bound = bind_register_program(graph, ((0, 1),), clean=tuple(range(3, 15)))
    n = max(w for g in bound.circuit.instructions for w in g.wires) + 1
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    coefficients = torch.complex(
        torch.arange(1, 9, dtype=torch.float64),
        torch.arange(8, 0, -1, dtype=torch.float64),
    )
    coefficients /= coefficients.norm()
    initial[torch.arange(8) << (n - 3)] = coefficients
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert events <= 7
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            initial / 2 ** (events / 2),
            rtol=0,
            atol=3e-12,
        )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live and bound.outputs == ((0, 1),)
