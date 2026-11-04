"""Check the complete fourth-section program through both frontends."""

import cmath

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QNum,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration

SOURCE = """qfunc cost(const x: qnum<2, False, 0>, const y: qnum<2, False, 0>) {
    phase(x * x + y, 0.8);
}
qfunc kick(const y: qnum<2, False, 0>) {
    phase(y, 0.3);
}
qfunc main(output x: qnum<2, False, 0>, output y: qnum<2, False, 0>) {
    allocate(x);
    allocate(y);
    hadamard_transform(x);
    hadamard_transform(y);
    cost(x, y);
    control (x == 1) { kick(y); }
}"""


@qfunc
def cost(x: Const[QNum[2, False, 0]], y: Const[QNum[2, False, 0]]) -> None:
    phase(x * x + y, 0.8)


@qfunc
def kick(y: Const[QNum[2, False, 0]]) -> None:
    phase(y, 0.3)


@qfunc
def main(x: Output[QNum[2, False, 0]], y: Output[QNum[2, False, 0]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    cost(x, y)
    control(x == 1, lambda: kick(y))


@pytest.mark.parametrize("source", [SOURCE, main])
def test_full_fourth_section_amplitudes_and_clean_workspace(source):
    options = {"source_format": "qmod"} if isinstance(source, str) else {}
    program = synth.synthesize(
        source, max_qubits=24, max_workspace=20, memory_limit_bytes=2**30, **options
    )
    expected = {
        (x, y): cmath.exp(1j * (0.8 * (x * x + y) + (0.3 * y if x == 1 else 0))) / 4
        for x in range(4)
        for y in range(4)
    }
    check_state(program, fq.run(program), expected)


@qfunc
def alias(x: Output[QNum[2]]) -> None:
    allocate(x)
    control(x == 1, lambda: kick(x))


@qfunc
def allocate_helper(y: Output[QNum[2]]) -> None:
    allocate(y)


@qfunc
def escape(x: Output[QNum[2]], y: Output[QNum[2]]) -> None:
    allocate(x)
    control(x == 1, lambda: allocate_helper(y))


@pytest.mark.parametrize(
    "source,diagnostic", [(alias, "disjoint"), (escape, "Const-only")]
)
def test_native_controlled_helper_rejects_alias_and_output_escape(source, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        synth.synthesize(source)


@pytest.mark.parametrize(
    "source,diagnostic",
    [
        (SOURCE.replace("kick(y); }", "kick(x); }"), "disjoint"),
        (
            SOURCE.replace(
                "const y: qnum<2, False, 0>) {\n    phase(y, 0.3);",
                "output y: qnum<2, False, 0>) {\n    allocate(y);",
            ),
            "Const-only",
        ),
    ],
)
def test_qmod_controlled_helper_rejects_alias_and_output_escape(source, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


@qfunc
def two_ports(y: Const[QNum[2]], other: Const[QNum[2]]) -> None:
    phase(y, 0.3)


@qfunc
def duplicate_actuals(x: Output[QNum[2]], y: Output[QNum[2]]) -> None:
    allocate(x)
    allocate(y)
    control(x == 1, lambda: two_ports(y, y))


def test_controlled_helper_rejects_two_formals_bound_to_same_register():
    with pytest.raises(ValueError, match="disjoint"):
        synth.synthesize(duplicate_actuals)
