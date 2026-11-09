"""Retained semantic scratch is accounted without asserting a clean release."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.model import Block, Region
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod

pytestmark = pytest.mark.unit

SOURCE = "qfunc prep(q:qbit) {inplace_prepare_state([0.93,0.07],0,q);} qfunc main(output q:qbit) {allocate(q);exact_amplitude_amplification(0.264575131106459,Z,prep,q);}"


def test_hidden_workspace_counts_and_drop_does_not_release_physical_storage() -> None:
    imported = parse_qmod(SOURCE, source_version=SUPPORTED_SOURCE_VERSION)
    model = imported.program
    block = model.body.blocks[0]
    hidden = next(op for op in block.operations if op.name == "qreg.hidden")
    assert hidden.attributes["workspace"] is True
    assert any(
        op.name == "qreg.drop" and hidden.results[0] in op.operands
        for op in block.operations
    )
    assert not any(
        op.name == "qreg.release" and hidden.results[0] in op.operands
        for op in block.operations
    )
    candidate = schedule_register_program(model, imported.functions, n_qubits=8)
    assert candidate.peak_workspace == 1


@pytest.mark.parametrize("marker", [False, 0, 1, "yes", None])
def test_owned_hidden_workspace_marker_is_exact_boolean_true(marker: object) -> None:
    imported = parse_qmod(SOURCE, source_version=SUPPORTED_SOURCE_VERSION)
    model = imported.program
    block = model.body.blocks[0]
    operations = tuple(
        (
            replace(op, attributes={**op.attributes, "workspace": marker})
            if op.name == "qreg.hidden"
            else op
        )
        for op in block.operations
    )
    altered = replace(model, body=Region((Block(block.arguments, operations),)))
    with pytest.raises(ValueError, match="workspace marker"):
        verify_register_program(altered, imported.functions)


def test_selection_counts_retained_workspace_once_in_physical_width() -> None:
    from flagquantum import synth

    program = synth.synthesize(
        SOURCE, source_format="qmod", max_qubits=2, max_workspace=1
    )
    assert program.circuit.n_wires == 2
    assert program.report["workspace_qubits"] == 1
