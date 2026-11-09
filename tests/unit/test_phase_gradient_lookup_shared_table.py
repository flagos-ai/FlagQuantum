"""Phase kickback consumes shared QROM workspace and physical port binding."""

from itertools import product

import pytest

from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.phase.phase_gradient_lookup import (
    phase_gradient_lookup_ir,
    phase_gradient_lookup_requirements,
)
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "controls_count,selection_width", tuple(product((0, 1, 2), (0, 2)))
)
def test_replacement_qrom_workspace_and_binding_are_consumed(
    monkeypatch, controls_count, selection_width
):
    calls = []
    words = tuple(i % 4 for i in range(1 << selection_width))
    monkeypatch.setattr(QromTable, "workspace_size", property(lambda self: 5))

    def replacement(self, ports, workspace, classical_bits):
        calls.append((self, ports, workspace, classical_bits))
        return (Instruction("z", (ports[-1],)),)

    monkeypatch.setattr(QromTable, "bind_packed", replacement)
    width = 3
    quantum, classical = phase_gradient_lookup_requirements(
        controls_count, selection_width, width, words
    )
    assert (quantum, classical) == (8, 5)
    controls = tuple(range(controls_count))
    selection = tuple(range(controls_count, controls_count + selection_width))
    start = controls_count + selection_width
    gradient = tuple(range(start, start + width))
    work = tuple(range(start + width, start + width + quantum))
    slots = tuple(range(classical))
    gates = phase_gradient_lookup_ir(controls, selection, gradient, words, work, slots)
    assert len(calls) == 1
    table, ports, scratch, bits = calls[0]
    assert table == QromTable(
        (len(words),), (selection_width,), (width,), ((),), controls_count, (words,)
    )
    assert ports == (*controls, *selection, *work[:width])
    assert scratch == work[width:] and bits == slots
    sentinel = Instruction("z", (ports[-1],))
    assert gates[0] == gates[-1] == sentinel
    # The original addition still uses only its own scratch prefix.
    assert sum(g.name == "measure" for g in gates) == width - 1


@pytest.mark.parametrize(
    "controls,selection,width,words",
    [
        (-1, 1, 2, (0, 1)),
        (True, 1, 2, (0, 1)),
        (0, -1, 2, (0,)),
        (0, 1, 2, (0, 1, 2)),
        (0, 1, 0, (0, 1)),
    ],
)
def test_phase_lookup_input_validation_stays_at_existing_boundary(
    controls, selection, width, words
):
    with pytest.raises(ValueError):
        phase_gradient_lookup_requirements(controls, selection, width, words)
