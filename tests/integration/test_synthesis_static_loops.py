"""Static loops execute through existing value/phase schedules and named IO."""

import ast
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import (
    ArithmeticCandidate,
    Schedule,
    schedule_arithmetic,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_register_workflow import check_state
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.integration


def matched(x: int, y: int, steps: int) -> bool:
    flag = x == y
    for value in range(1, steps):
        flag = flag or (x == value)
    return flag


def repeated(x: int, y: int, steps: int) -> int:
    total = x
    for i in range(steps):
        term = y * i
        total = total + term
    return total


def expanded(x: int, y: int) -> int:
    first = x + y * 0
    second = first + y
    return second + y * 2


def raw(
    strategy: Schedule, function: Callable[..., object] = repeated
) -> ArithmeticCandidate:
    graph = (
        capture(function, {"x": 1, "y": 1}, constants={"steps": 3})
        if function is repeated
        else capture(function, {"x": 1, "y": 1})
    )
    bindings = {v.id: (i,) for i, v in enumerate(graph.body.blocks[0].arguments)}
    width = graph.body.blocks[0].operations[-1].operands[0].type.parameters[0]
    assert isinstance(width, int)
    output = tuple(range(2, 2 + width))
    start = 2 + width
    probe = schedule_arithmetic(
        (graph,),
        bindings,
        output,
        entry=graph.name,
        n_qubits=64,
        workspace=tuple(range(start, 64)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    n = start + probe.peak_workspace
    assert n <= 22
    return schedule_arithmetic(
        (graph,),
        bindings,
        output,
        entry=graph.name,
        n_qubits=n,
        workspace=tuple(range(start, n)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_real_basis_inputs_all_nonzero_xor_outputs_and_expanded_equivalence(
    strategy: Schedule,
) -> None:
    loop = raw(strategy)
    manual = raw(strategy, expanded)
    for lowered in (loop, manual):
        for x in range(2):
            for y in range(2):
                for target in range(8):
                    prefix = (x << 4) | (y << 3) | target
                    initial = prefix << lowered.peak_workspace
                    circuit = fq.Circuit(
                        lowered.circuit.n_wires, dtype=torch.complex128
                    )
                    for q in range(circuit.n_qubits):
                        if initial & (1 << (circuit.n_qubits - q - 1)):
                            circuit.x(q)
                    state = (
                        fq.run(
                            circuit.compose(lowered.circuit),
                            options=fq.ExecutionOptions(precision="complex128"),
                        )
                        .statevector()
                        .reshape(-1)
                    )
                    expected = torch.zeros_like(state)
                    expected[(prefix ^ (x + 3 * y)) << lowered.peak_workspace] = 1
                    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_loop_carry_and_xor_target_cohere_with_external_reference(
    strategy: Schedule,
) -> None:
    candidate = raw(strategy)
    n = candidate.circuit.n_wires
    assert n + 1 <= 23
    amplitudes = torch.randn(
        64, dtype=torch.complex128, generator=torch.Generator().manual_seed(978)
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for x in range(2):
        for y in range(2):
            for target in range(8):
                prefix = (x << 4) | (y << 3) | target
                for reference in range(2):
                    source = (prefix << (candidate.peak_workspace + 1)) | reference
                    destination = (
                        (prefix ^ (x + 3 * y)) << (candidate.peak_workspace + 1)
                    ) | reference
                    initial[source] = amplitudes[(prefix << 1) | reference]
                    expected[destination] = initial[source]
    embedded = fq.Circuit(n + 1, dtype=torch.complex128).compose(
        candidate.circuit, qubits=range(n)
    )
    state = (
        fq.Circuit.from_ir(embedded.to_ir(), inputs=initial, device="cpu")
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert (
        float(
            state.reshape(32, 1 << candidate.peak_workspace, 2)[:, 1:, :]
            .abs()
            .square()
            .sum()
        )
        <= 1e-12
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_public_numeric_loop_values_constants_and_coherent_preparation(
    objective: Literal["width", "depth"], steps: int
) -> None:
    program = synth.synthesize(
        synth.qfunc(repeated),
        inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
        constants={"steps": steps},
        input_state=synth.InputState.uniform(),
        objective=objective,
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    result = fq.run(program)
    factor = steps * (steps - 1) // 2
    expected = {(x, y, x + factor * y): 0.5 for x in range(2) for y in range(2)}
    check_state(program, result, expected)
    assert dict(
        program.probabilities(result, registers=("x", "y", "result"))
    ) == pytest.approx(dict.fromkeys(expected, 0.25), abs=1e-12)
    assert program.report["resources"] == program._checked.verify_result(result)
    if steps == 0:
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_public_all_16_predicate_inputs_match_loop_reference(
    objective: Literal["width", "depth"], mode: Literal["value", "phase"]
) -> None:
    for x in range(4):
        for y in range(4):
            program = synth.synthesize(
                matched,
                inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
                constants={"steps": 3},
                input_state=synth.InputState.basis(x=x, y=y),
                objective=objective,
                output_mode=mode,
                max_qubits=24,
                memory_limit_bytes=256 * 1024**2,
            )
            truth = int(x == y or x in (1, 2))
            key = (x, y, truth) if mode == "value" else (x, y)
            check_state(
                program, fq.run(program), {key: 1 if mode == "value" else (-1) ** truth}
            )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_public_loop_predicate_preserves_full_complex_state(
    objective: Literal["width", "depth"], mode: Literal["value", "phase"]
) -> None:
    preparation = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).h(2).h(3).s(0).t(2)
    program = synth.synthesize(
        matched,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        constants={"steps": 3},
        preparation=preparation.to_ir(),
        preparation_map={"x": (0, 1), "y": (2, 3)},
        output_mode=mode,
        objective=objective,
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    expected = {}
    for x in range(4):
        for y in range(4):
            phase = (1j if x & 2 else 1) * complex(
                math.cos((y >> 1) * math.pi / 4), math.sin((y >> 1) * math.pi / 4)
            )
            truth = int(x == y or x in (1, 2))
            key = (x, y, truth) if mode == "value" else (x, y)
            expected[key] = phase * (1 if mode == "value" else (-1) ** truth) / 4
    check_state(program, fq.run(program), expected)


@pytest.mark.parametrize("expression", ["range(0)", "range(3,1)"])
def test_empty_loop_does_not_add_quantum_work_and_validates_source(
    expression: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fn = _cell(
        f"def main(x):\n    total=x\n    for i in {expression}:\n        total=total+x\n    return total\n"
    )
    program = synth.synthesize(
        fn,
        inputs={"x": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        max_workspace=0,
    )
    check_state(program, fq.run(program), {(x, x): 1 / math.sqrt(2) for x in range(2)})
    assert program.report["workspace_qubits"] == 0
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned invalid empty body")
    )
    bad = _cell(
        "def main(x):\n    total=x\n    for i in range(0):\n        total=unavailable(x)\n    return total\n"
    )
    with pytest.raises(HybridVerificationError, match="construction.python.call"):
        synth.synthesize(bad, inputs={"x": synth.UInt(1)})


def test_dynamic_or_oversized_ranges_fail_before_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned invalid range")
    )
    for extent in ["x", "257", "2**100"]:
        fn = _cell(
            f"def main(x):\n    total=x\n    for i in range({extent}):\n        total=total+x\n    return total\n"
        )
        with pytest.raises(
            HybridVerificationError, match="loop.range|loop.budget|power"
        ):
            synth.synthesize(fn, inputs={"x": synth.UInt(1)})


def test_loop_constraints_and_memory_are_checked_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fq, "run", lambda *a, **k: pytest.fail("executed during synthesis")
    )
    for limits in [
        dict(max_workspace=0),
        dict(max_operations=1),
        dict(construction_budget=synth.ConstructionBudget(candidate_checks=1)),
    ]:
        with pytest.raises(synth.SynthesisError):
            synth.synthesize(
                repeated,
                inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
                constants={"steps": 3},
                **limits,
            )
    with pytest.raises(PlanningError, match="memory"):
        synth.synthesize(
            repeated,
            inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
            constants={"steps": 3},
            memory_limit_bytes=1,
        )


def test_complete_loop_example_uses_only_public_intent_and_named_readout() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_static_loop"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    rows = dict(
        line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line
    )
    assert ast.literal_eval(rows["weighted probabilities"]) == pytest.approx(
        {(x, y, x + 3 * y): 0.25 for x in range(2) for y in range(2)}, abs=1e-12
    )
    assert ast.literal_eval(rows["phase input probabilities"]) == pytest.approx(
        {(x, y): 1 / 16 for x in range(4) for y in range(4)}, abs=1e-12
    )
    source = (root / "examples/synthesis_static_loop.py").read_text()
    assert (
        "._" not in source
        and "fq.Circuit" not in source
        and "physical_of" not in source
    )


@pytest.mark.parametrize("extent", [0, 2, 256])
def test_classical_loop_scaffolding_preserves_zero_workspace_phase_mask(
    extent: int,
) -> None:
    function = _cell(
        f"def main(enable,flag):\n    marked=enable and flag\n    for i in range({extent}):\n        marked=marked\n    return marked\n"
    )
    program = synth.synthesize(
        function,
        inputs={"enable": synth.UInt(1), "flag": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        max_qubits=2,
        max_workspace=0,
        max_operations=3,
    )
    check_state(
        program,
        fq.run(program),
        {(c, x): (-1) ** (c * x) / 2 for c in range(2) for x in range(2)},
    )
    assert program.report["workspace_qubits"] == 0
    assert program.report["resources"].n_operations == 3
