"""Check ownership and physical placement of terminal register publications."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import (
    register_ports,
    verify_register_program,
)
from flagquantum.compiler._hybrid.model import HybridProgram
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.register_program_cases import Registers

pytestmark = pytest.mark.unit


def published_program(
    *,
    locations: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (1, 1)),
    initialize_source: bool = True,
    initialize_output: bool = False,
    keep_source_as_workspace: bool = False,
    after_publication: bool = False,
) -> HybridProgram:
    model = Registers()
    mixed, tail = model.declare("mixed", 2), model.declare("tail", 1)
    a, b = model.declare("a", 1, local=True), model.declare("b", 2, local=True)
    model.operations[-2] = replace(model.operations[-2], name="qreg.hidden")
    model.operations[-1] = replace(
        model.operations[-1],
        name="qreg.hidden",
        attributes={
            "name": "b",
            **({"workspace": True} if keep_source_as_workspace else {}),
        },
    )
    model.emit("qreg.allocate", a)
    model.emit("qreg.gate", a, gate="x", index=0)
    if initialize_source:
        model.emit("qreg.allocate", b)
        model.emit("qreg.gate", b, gate="h", index=0)
        model.emit("qreg.gate", b, gate="x", index=1)
    if initialize_output:
        model.emit("qreg.allocate", mixed)
    model.emit("qreg.publish", mixed, tail, a, b, destinations=2, locations=locations)
    if after_publication:
        model.emit("qreg.gate", tail, gate="z", index=0)
    return model.finish()


def test_publish_transfers_physical_ownership_and_uses_no_gates() -> None:
    program = published_program()
    verify_register_program(program, max_qubits=3)
    labels, layout, width = register_ports(program)
    assert labels == {"mixed": (0, 1), "tail": (2,)}
    assert width == 3
    assert list(layout.values()) == [(0, 1), (2,), (1,), (2, 0)]
    lowered = schedule_register_program(program, (), n_qubits=3)
    assert lowered.peak_workspace == 0
    assert lowered.circuit.n_wires == 3
    assert [(gate.name, gate.wires) for gate in lowered.circuit.instructions] == [
        ("x", (1,)),
        ("h", (0,)),
        ("x", (2,)),
    ]


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"locations": ((0, 0), (1, 0), (1, 0))}, "must be distinct"),
        ({"locations": ((0, 0), (1, 0))}, "cover the source and output widths"),
        ({"locations": ((0, 0), (1, 0), (1, 2))}, "must be in bounds"),
        ({"initialize_source": False}, "initialized hidden data"),
        ({"initialize_output": True}, "fresh outputs"),
        ({"keep_source_as_workspace": True}, "initialized hidden data"),
        ({"after_publication": True}, "terminal publication"),
    ],
)
def test_publish_rejects_invalid_layout_or_ownership(options, reason) -> None:
    with pytest.raises(HybridVerificationError, match=reason):
        verify_register_program(published_program(**options))
