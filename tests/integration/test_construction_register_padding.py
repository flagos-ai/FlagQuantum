"""A fresh wider destination retains storage and zero high bits under coherence."""

from typing import Literal

import pytest

from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.selection import select_register_program
from tests.integration.test_construction_register_execution import _check as check_model
from tests.register_program_cases import Registers, square_plus_one

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_wider_fresh_output_preserves_low_weights_phases_and_zero_high_bits(
    objective: Literal["width", "depth"],
) -> None:
    p = Registers()
    x, result = p.declare("x", 2), p.declare("result", 6)
    p.emit("qreg.allocate", x)
    p.emit("qreg.gate", x, gate="h", index=0)
    p.emit("qreg.gate", x, gate="h", index=1)
    p.emit("qreg.gate", x, gate="s", index=1)
    graph = capture(square_plus_one, {"x": 2})
    p.emit("qreg.compute", result, x, callee=graph.name)
    selection = select_register_program(p.finish(), (graph,), objective=objective)
    assert selection.selected is not None
    assert len(selection.registers["result"]) == 6
    check_model(
        selection, {(x, x * x + 1): (1j if x >= 2 else 1) / 2 for x in range(4)}
    )
