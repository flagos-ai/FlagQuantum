"""Exercise the width model through verified native register programs."""

import pytest

from examples.synthesis_local_registers import local_composition
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_width_model import (
    estimate_register_program_width,
)
from flagquantum.compiler._hybrid.model import Operation
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.register_program_cases import Registers, product

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "overlapping,helpers,expected",
    [(False, False, 7), (True, False, 8), (False, True, 7)],
)
def test_storage_matches_lowering_band(
    overlapping: bool, helpers: bool, expected: int
) -> None:
    program, functions = local_composition(overlapping=overlapping, helpers=helpers)
    lowered = schedule_register_program(program, functions, n_qubits=24)
    assert estimate_register_program_width(program, functions) == expected
    assert (
        sum(map(len, lowered.registers.values()))
        + lowered.peak_workspace
        - max(segment.peak_workspace for segment in lowered.segments)
        == expected
    )


def test_later_outputs_stay_reserved_during_earlier_local_lifetime() -> None:
    registers = Registers()
    first = registers.declare("first", 1)
    registers.emit("qreg.allocate", first)
    temp = registers.declare("temp", 2, local=True)
    registers.emit("qreg.release", temp)
    later = registers.declare("later", 2)
    registers.emit("qreg.allocate", later)
    program = registers.finish()
    assert estimate_register_program_width(program) == 5
    lowered = schedule_register_program(program)
    assert sum(map(len, lowered.registers.values())) + lowered.peak_workspace == 5


def test_local_peak_replaces_distinct_operands_and_keeps_bystanders() -> None:
    function = capture(product, {"a": 1, "b": 1})
    registers = Registers()
    x = registers.declare("x", 1)
    result = registers.declare("result", 2)
    bystander = registers.declare("bystander", 2)
    registers.emit("qreg.allocate", x)
    registers.emit("qreg.compute", result, x, x, callee=function.name)
    registers.emit("qreg.allocate", bystander)
    program = registers.finish()

    def peak(op: Operation, operand_width: int) -> int:
        if op.name == "qreg.compute":
            assert operand_width == 3
            return 6
        return operand_width

    assert estimate_register_program_width(program, (function,)) == 5
    assert estimate_register_program_width(program, (function,), local_peak=peak) == 8
    with pytest.raises(ValueError, match="distinct operand registers"):
        estimate_register_program_width(program, (function,), local_peak=lambda *_: 0)


def test_uncompute_accounts_for_implicit_borrowed_inputs() -> None:
    function = capture(product, {"a": 1, "b": 1})
    registers = Registers()
    x = registers.declare("x", 1)
    result = registers.declare("result", 2, local=True)
    registers.emit("qreg.allocate", x)
    registers.emit("qreg.compute", result, x, x, callee=function.name)
    registers.emit("qreg.uncompute", result)
    registers.emit("qreg.release", result)

    def peak(op: Operation, operand_width: int) -> int:
        if op.name == "qreg.uncompute":
            assert operand_width == 3
            return 5
        return operand_width

    assert (
        estimate_register_program_width(
            registers.finish(), (function,), local_peak=peak
        )
        == 5
    )


def test_unreleased_local_is_rejected_before_estimation() -> None:
    registers = Registers()
    x = registers.declare("x", 1)
    registers.declare("temp", 1, local=True)
    registers.emit("qreg.allocate", x)
    with pytest.raises(HybridVerificationError, match="register.local"):
        estimate_register_program_width(registers.finish())
