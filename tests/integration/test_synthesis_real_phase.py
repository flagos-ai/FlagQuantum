"""Native high-level radians functions compile without user gate design."""

import cmath
import math
from types import FunctionType
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_register_workflow import check_state
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.integration


def z_phase(q: int) -> float:
    return 1.9 * (q - 0.5)


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("state", ["zero", "one", "uniform"])
def test_high_level_phase_matches_analytic_native_rz_and_qmod(
    objective: Literal["width", "depth"], state: str
) -> None:
    input_state = (
        synth.InputState.uniform()
        if state == "uniform"
        else synth.InputState.basis(q=int(state == "one"))
    )
    p = synth.synthesize(
        z_phase,
        inputs={"q": synth.UInt(1)},
        input_state=input_state,
        output_mode="phase",
        objective=objective,
        max_qubits=1,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    native = fq.Circuit(1, dtype=torch.complex128)
    prefix = ""
    if state == "uniform":
        native.h(0)
        prefix = "H(q);"
        expected = {
            (q,): cmath.exp(1j * 1.9 * (q - 0.5)) / math.sqrt(2) for q in (0, 1)
        }
    else:
        bit = int(state == "one")
        if bit:
            native.x(0)
            prefix = "X(q);"
        expected = {(bit,): cmath.exp(1j * 1.9 * (bit - 0.5))}
    native.rz(0, theta=1.9)
    imported = synth.synthesize(
        f"qfunc main(output q:qbit) {{allocate(q);{prefix}RZ(1.9,q);}}",
        source_format="qmod",
        objective=objective,
    )
    result = fq.run(p)
    check_state(p, result, expected)
    check_state(imported, fq.run(imported), expected)
    torch.testing.assert_close(
        result.statevector(), fq.run(native).statevector(), atol=1e-12, rtol=0
    )
    assert p.report["output_mode"] == "phase"
    assert p.report["result_width"] == 0
    assert tuple(p.registers) == ("q",)
    assert p.report["workspace_qubits"] == 0
    assert p.execution.state_bytes == 32


@pytest.mark.parametrize(
    "expression",
    [
        "1.9*q",
        "0.3",
        "0.0",
        "0.7*(1-q)",
        "0.1+0.2*q",
        "q*q-0.5",
        "-(q-0.5)*1.9",
        "+q*0.25",
        "(1000000000000000+q*0.01)-1000000000000000",
    ],
)
def test_full_one_bit_function_including_constant_global_phase(expression: str) -> None:
    function = _cell(f"def main(q: int) -> float:\n    return {expression}\n")
    p = synth.synthesize(
        function,
        inputs={"q": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
    )
    # Host evaluation is an independent test oracle, never the compiler path.
    expected = {(q,): cmath.exp(1j * function(q)) / math.sqrt(2) for q in (0, 1)}
    check_state(p, fq.run(p), expected)


@pytest.mark.parametrize(
    "expression",
    [
        "True",
        "1j",
        "None",
        "1e309",
        "9007199254740993",
        "q+9007199254740992",
        "q/2",
        "q**2",
        "float(q)",
        "q if q else 0",
        "q==1",
        "theta*q",
        "[q][0]",
    ],
)
def test_unsupported_expression_is_refused_even_on_basis_zero(expression: str) -> None:
    function = _cell(f"def main(q: int) -> float:\n    return {expression}\n")
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            function,
            inputs={"q": synth.UInt(1)},
            input_state=synth.InputState.basis(q=0),
            output_mode="phase",
        )


@pytest.mark.parametrize("width", [2, 3])
def test_real_phase_does_not_enumerate_wider_domains(width: int) -> None:
    with pytest.raises(HybridVerificationError, match="phase.profile"):
        synth.synthesize(z_phase, inputs={"q": synth.UInt(width)}, output_mode="phase")


def test_custom_preparation_helpers_assignments_and_value_mode_remain_refused() -> None:
    with pytest.raises(ValueError, match="named input_state"):
        synth.synthesize(
            z_phase,
            inputs={"q": synth.UInt(1)},
            output_mode="phase",
            preparation=fq.Circuit(1).h(0).to_ir(),
        )
    function = _cell("def main(q: int) -> float:\n    x=q-0.5\n    return x\n")
    with pytest.raises(HybridVerificationError, match="phase.body"):
        synth.synthesize(function, inputs={"q": synth.UInt(1)}, output_mode="phase")
    with pytest.raises(HybridVerificationError, match="annotation"):
        synth.synthesize(z_phase, inputs={"q": synth.UInt(1)})
    with pytest.raises(HybridVerificationError, match="phase.profile"):
        synth.synthesize(
            z_phase,
            inputs={"q": synth.UInt(1)},
            output_mode="phase",
            functions=(_cell("def main(q: int) -> int:\n    return q\n"),),
        )


def test_existing_bool_and_integer_predicate_phase_still_means_pi() -> None:
    for annotation, expression in [("bool", "q==1"), ("int", "q")]:
        function = _cell(
            f"def main(q: int) -> {annotation}:\n    return {expression}\n"
        )
        p = synth.synthesize(
            function,
            inputs={"q": synth.UInt(1)},
            input_state=synth.InputState.uniform(),
            output_mode="phase",
        )
        check_state(p, fq.run(p), {(0,): 2**-0.5, (1,): -(2**-0.5)})


def test_rz_is_selected_from_intent_not_python_function_name() -> None:
    graph = capture(z_phase, {"q": 1}, real_phase=True)
    gate = next(op for op in graph.body.blocks[0].operations if op.name == "qreg.gate")
    assert dict(gate.attributes) == {"gate": "rz", "index": 0, "angle": 1.9}


def test_compiler_uses_retained_source_without_executing_host_body() -> None:
    function = _cell(
        "def main(q: int) -> float:\n    return 1.9*(q-0.5)\n",
        filename="<native-phase-no-execution>",
    )
    namespace: dict[str, object] = {}
    exec(
        compile(
            "def main(q):\n    raise RuntimeError('host body executed')\n",
            function.__code__.co_filename,
            "exec",
        ),
        namespace,
    )
    replacement = namespace["main"]
    assert isinstance(replacement, FunctionType)
    function.__code__ = replacement.__code__
    p = synth.synthesize(
        function,
        inputs={"q": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
    )
    check_state(
        p, fq.run(p), {(q,): cmath.exp(1.9j * (q - 0.5)) * 2**-0.5 for q in (0, 1)}
    )


def test_explicit_radians_annotation_and_full_profile_guards() -> None:
    cases = [
        (
            _cell(
                "def main(q: int):\n    return 1.9*(q-0.5)\n",
                filename="<native-phase-unannotated>",
            ),
            {"q": synth.UInt(1)},
            {},
        ),
        (
            _cell(
                "def main(q: int, r: int) -> float:\n    return q+r\n",
                filename="<native-phase-multiple-inputs>",
            ),
            {"q": synth.UInt(1), "r": synth.UInt(1)},
            {},
        ),
        (
            _cell(
                "def main(q: int, k: int) -> float:\n    return q*k\n",
                filename="<native-phase-constant-binding>",
            ),
            {"q": synth.UInt(1)},
            {"k": 1},
        ),
    ]
    for function, inputs, constants in cases:
        with pytest.raises(HybridVerificationError):
            synth.synthesize(
                function, inputs=inputs, constants=constants, output_mode="phase"
            )
    expression = "q"
    for _ in range(33):
        expression = f"({expression}+0.0)"
    with pytest.raises(HybridVerificationError, match="phase.depth"):
        synth.synthesize(
            _cell(f"def main(q: int) -> float:\n    return {expression}\n"),
            inputs={"q": synth.UInt(1)},
            output_mode="phase",
        )


def test_native_high_level_memory_and_actual_gate_budget() -> None:
    from flagquantum.errors import PlanningError

    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(
            z_phase,
            inputs={"q": synth.UInt(1)},
            output_mode="phase",
            memory_limit_bytes=16,
        )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            z_phase,
            inputs={"q": synth.UInt(1)},
            input_state=synth.InputState.uniform(),
            output_mode="phase",
            max_operations=1,
        )
    with pytest.raises(ValueError, match="logical CPU"):
        synth.synthesize(
            z_phase, inputs={"q": synth.UInt(1)}, output_mode="phase", topology="line"
        )
