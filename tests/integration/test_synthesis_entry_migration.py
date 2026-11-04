"""Check named results across scalar and owned construction entry points."""

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform, qfunc

pytestmark = pytest.mark.integration

QMOD = """qfunc main(output x: qnum<1>, output y: qnum<1>, output result: qnum) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    result = x + y;
}"""


def legacy(x: int, bias: int, /, y: int) -> int:
    return x + bias + y


@qfunc
def add(x: Const[QNum[1]], y: Const[QNum[1]], result: Output[QNum]) -> None:
    result |= x + y


@qfunc
def native(x: Output[QNum[1]], y: Output[QNum[1]], result: Output[QNum]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    add(x, y, result)


@pytest.mark.parametrize("topology", [None, "line"])
def test_migrate_scalar_value_to_native_or_qmod_with_named_readout(topology):
    programs = [
        synth.synthesize(
            legacy,
            inputs={"y": synth.UInt(1), "x": synth.UInt(1)},
            constants={"bias": 0},
            input_state=synth.InputState.uniform(),
            topology=topology,
        ),
        synth.synthesize(native, topology=topology),
        synth.synthesize(
            QMOD, source_format="qmod", source_name="addition.qmod", topology=topology
        ),
    ]
    expected = {(y, x + y, x): 0.25 for x in range(2) for y in range(2)}
    for program in programs:
        result = fq.run(program)
        assert result.plan.identity == program.execution.identity
        assert tuple(program.registers) == ("x", "y", "result")
        assert dict(
            program.probabilities(result, registers=("y", "result", "x"))
        ) == pytest.approx(expected, abs=1e-10)
        assert program.report["resources"] == program._checked.verify_result(result)
    assert programs[1].report["source_format"] == "python"
    assert programs[2].report["source_format"] == "qmod"
    assert programs[2].report["source_name"] == "addition.qmod"
