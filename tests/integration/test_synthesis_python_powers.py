"""Python powers share checked execution with native and explicit products."""

from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.arithmetic.python_powers.main import cube
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.synthesis_workflow_helpers import check_state
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.integration


def product(x: int) -> int:
    return (x * x) * x


def correlated_power(x: int, ref: int) -> int:
    return x**8


def constant_power(x: int) -> int:
    return (x * 0 + 1) ** 8


def configured_power(x: int, exponent: int) -> int:
    return x**exponent


def increment(x: int) -> int:
    return x + 1


def composed_power(x: int) -> int:
    return increment(x) ** 3


def test_composed_base_is_called_once_and_executes_coherently() -> None:
    graph = capture(composed_power, {"x": 1}, functions=(increment,))
    assert sum(op.name == "qint.add" for op in graph.body.blocks[0].operations) == 1
    assert (
        sum(op.name == "qint.multiply" for op in graph.body.blocks[0].operations) == 2
    )
    p = synth.synthesize(
        composed_power,
        inputs={"x": synth.UInt(1)},
        functions=(increment,),
        input_state=synth.InputState.uniform(),
    )
    check_state(p, fq.run(p), {(0, 1): 2**-0.5, (1, 8): 2**-0.5})


def test_configured_exponent_remains_outside_literal_profile() -> None:
    with pytest.raises(HybridVerificationError, match="power"):
        synth.synthesize(
            configured_power, inputs={"x": synth.UInt(1)}, constants={"exponent": 3}
        )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_python_powers_preserve_complex_external_entanglement(
    objective: Literal["width", "depth"],
    mode: Literal["value", "phase"],
) -> None:
    preparation = fq.Circuit(2, dtype=torch.complex128).h(0).s(0).cx(0, 1)
    p = synth.synthesize(
        correlated_power,
        inputs={"x": synth.UInt(1), "ref": synth.UInt(1)},
        preparation=preparation.to_ir(),
        preparation_map={"x": (0,), "ref": (1,)},
        objective=objective,
        output_mode=mode,
    )
    expected = {
        (x, x, x) if mode == "value" else (x, x): 1j**x
        * 2**-0.5
        * ((-1) ** x if mode == "phase" else 1)
        for x in range(2)
    }
    check_state(p, fq.run(p), expected)


@pytest.mark.parametrize("mode", ["value", "phase"])
def test_quantum_dependent_constant_power_keeps_global_sign(
    mode: Literal["value", "phase"],
) -> None:
    p = synth.synthesize(
        constant_power,
        inputs={"x": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode=mode,
    )
    check_state(
        p,
        fq.run(p),
        {
            (x, 1) if mode == "value" else (x,): 2**-0.5
            * (-1 if mode == "phase" else 1)
            for x in range(2)
        },
    )


@pytest.mark.parametrize("power", range(1, 9))
@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_all_python_literal_powers_in_basis_states(
    power: int,
    x: int,
    objective: Literal["width", "depth"],
) -> None:
    function = _cell(
        f"def main(x: int) -> int:\n    return x**{power}\n",
        filename=f"<python-power-{power}>",
    )
    p = synth.synthesize(
        function,
        inputs={"x": synth.UInt(1)},
        input_state=synth.InputState.basis(x=x),
        objective=objective,
    )
    check_state(p, fq.run(p), {(x, x**power): 1})


@pytest.mark.parametrize("producer", ["python", "product", "native"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_coherent_cube_producers_share_named_consumer(
    producer: str,
    objective: Literal["width", "depth"],
) -> None:
    if producer == "native":
        p = synth.synthesize(
            "qfunc main(output x:qnum<2>,output result:qnum) {allocate(x);hadamard_transform(x);result=x**3;}",
            source_format="qmod",
            objective=objective,
        )
    else:
        p = synth.synthesize(
            cube if producer == "python" else product,
            inputs={"x": synth.UInt(2)},
            input_state=synth.InputState.uniform(),
            objective=objective,
        )
    check_state(p, fq.run(p), {(x, x**3): 0.5 for x in range(4)})


@pytest.mark.parametrize(
    "expression",
    [
        "x**0",
        "x**9",
        "x**True",
        "x**False",
        "x**-1",
        "x**1.5",
        "x**n",
        "x**(1+1)",
        "x**2**2",
    ],
)
def test_invalid_python_powers_are_rejected(expression: str) -> None:
    function = _cell(
        f"def main(x: int) -> int:\n    return {expression}\n",
        filename=f"<bad-power-{expression}>",
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(function, inputs={"x": synth.UInt(1)})


def test_base_is_captured_once_and_guarded_by_full_domain() -> None:
    function = _cell(
        "def main(x: int) -> int:\n    return (x+1)**3\n", filename="<power-base-once>"
    )
    graph = capture(function, {"x": 2})
    ops = graph.body.blocks[0].operations
    assert sum(op.name == "qint.add" for op in ops) == 1
    assert sum(op.name == "qint.multiply" for op in ops) == 2
    with pytest.raises(HybridVerificationError, match="256-bit"):
        capture(cube, {"x": 100})
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(cube, inputs={"x": synth.UInt(2)}, max_workspace=0)


@pytest.mark.parametrize("power", range(1, 9))
def test_classical_powers_fold_without_quantum_multiplication(power: int) -> None:
    function = _cell(
        f"def main(x: int) -> int:\n    return x+2**{power}\n",
        filename=f"<constant-python-power-{power}>",
    )
    graph = capture(function, {"x": 1})
    assert not any(op.name == "qint.multiply" for op in graph.body.blocks[0].operations)
    assert any(
        op.name == "arith.constant" and op.attributes["value"] == 2**power
        for op in graph.body.blocks[0].operations
    )
    returned = graph.body.blocks[0].operations[-1].operands[0]
    assert returned.type.parameters[0] == (2**power + 1).bit_length()
    if power <= 2:
        p = synth.synthesize(
            function,
            inputs={"x": synth.UInt(1)},
            input_state=synth.InputState.uniform(),
            memory_limit_bytes=64 * 1024**2,
        )
        check_state(p, fq.run(p), {(x, x + 2**power): 2**-0.5 for x in range(2)})


def test_wide_folded_constant_is_refused_before_state_allocation() -> None:
    function = _cell(
        "def main(x: int) -> int:\n    return x+2**8\n", filename="<wide-folded-power>"
    )
    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(function, inputs={"x": synth.UInt(1)}, memory_limit_bytes=1024)
