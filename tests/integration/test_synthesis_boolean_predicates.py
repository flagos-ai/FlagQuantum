"""Composite predicates execute through the existing value and phase workflows."""

import ast
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.compiler._construction.frontend_python import Definition, capture
from flagquantum.compiler._construction.lowering import (
    ArithmeticCandidate,
    schedule_arithmetic,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_register_workflow import check_state
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.integration


def marked(x: int, y: int) -> bool:
    return (x < 2 and y == 1) or not (x == y)


def same(x: int, y: int) -> bool:
    return x == y


def composed(x: int, y: int, limit: int) -> bool:
    alias = same(x, y)
    return (x < limit and y == 1) or not alias


def raw(
    function: Definition,
    widths: Mapping[str, int],
    strategy: Literal["retain", "recompute"],
    mode: Literal["value", "phase"] = "value",
) -> ArithmeticCandidate:
    graph = capture(function, widths)
    bindings = {}
    position = 0
    for value in graph.body.blocks[0].arguments:
        width = value.type.parameters[0]
        bindings[value.id] = tuple(range(position, position + width))
        position += width
    output = (position,) if mode == "value" else ()
    start = position + len(output)
    kwargs = dict(
        entry=graph.name,
        workspace_state="fresh_zero",
        scheduling=strategy,
        output_mode=mode,
    )
    probe = schedule_arithmetic(
        (graph,),
        bindings,
        output,
        n_qubits=64,
        workspace=tuple(range(start, 64)),
        **kwargs,
    )
    n = start + probe.peak_workspace
    assert n <= 21
    return schedule_arithmetic(
        (graph,),
        bindings,
        output,
        n_qubits=n,
        workspace=tuple(range(start, n)),
        **kwargs,
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize(
    "expression",
    [
        "x and y",
        "x or y",
        "not x",
        "x and x",
        "x or x",
        "x and y and z",
        "x or y or z",
        "x or y and z",
        "(x or y) and not z",
        "x and 0",
        "x or 1",
    ],
)
def test_complete_truth_tables_and_nonzero_xor_targets_on_real_cpu(
    strategy: Literal["retain", "recompute"], expression: str
) -> None:
    function = _cell(f"def main(x,y,z):\n    return {expression}\n")
    lowered = raw(function, {"x": 1, "y": 1, "z": 1}, strategy)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                truth = int(function(x, y, z))
                for target in (0, 1):
                    prefix = (x << 3) | (y << 2) | (z << 1) | target
                    initial = prefix << lowered.peak_workspace
                    circuit = fq.Circuit(
                        lowered.circuit.n_wires, dtype=torch.complex128
                    )
                    for q in range(circuit.n_qubits):
                        if initial & (1 << (circuit.n_qubits - q - 1)):
                            circuit.x(q)
                    result = fq.run(
                        circuit.compose(lowered.circuit),
                        options=fq.ExecutionOptions(precision="complex128"),
                    )
                    state = result.statevector().reshape(-1)
                    expected = torch.zeros_like(state)
                    expected[(prefix ^ truth) << lowered.peak_workspace] = 1
                    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_complex_target_and_external_reference_entanglement_keep_absolute_phase(
    strategy: Literal["retain", "recompute"], mode: Literal["value", "phase"]
) -> None:
    lowered = raw(marked, {"x": 2, "y": 2}, strategy, mode)
    n = lowered.circuit.n_wires
    target_count = 2 if mode == "value" else 1
    amplitudes = torch.randn(
        16 * target_count * 2,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(731),
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    row = 0
    for x in range(4):
        for y in range(4):
            truth = int(marked(x, y))
            for target in range(target_count):
                prefix = (x << 2) | y
                if mode == "value":
                    prefix = (prefix << 1) | target
                for reference in range(2):
                    source = (prefix << (lowered.peak_workspace + 1)) | reference
                    destination = (
                        (prefix ^ (truth if mode == "value" else 0))
                        << (lowered.peak_workspace + 1)
                    ) | reference
                    initial[source] = amplitudes[row]
                    expected[destination] = amplitudes[row] * (
                        (-1) ** truth if mode == "phase" else 1
                    )
                    row += 1
    circuit = fq.Circuit(n + 1, dtype=torch.complex128).compose(
        lowered.circuit, qubits=range(n)
    )
    observed = (
        fq.Circuit.from_ir(circuit.to_ir(), inputs=initial, device="cpu")
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    dirty = observed.reshape(16 * target_count, 1 << lowered.peak_workspace, 2)[
        :, 1:, :
    ]
    assert float(dirty.abs().square().sum()) <= 1e-12


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_public_all_16_input_pairs_match_composite_reference(
    objective: Literal["width", "depth"], mode: Literal["value", "phase"]
) -> None:
    for x in range(4):
        for y in range(4):
            program = synth.synthesize(
                marked,
                inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
                input_state=synth.InputState.basis(x=x, y=y),
                objective=objective,
                output_mode=mode,
                max_qubits=24,
                memory_limit_bytes=64 * 1024**2,
            )
            values = (x, y, int(marked(x, y))) if mode == "value" else (x, y)
            amplitude = 1 if mode == "value" else (-1) ** int(marked(x, y))
            check_state(program, fq.run(program), {values: amplitude})
            assert program.report["output_mode"] == mode


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_public_coherent_helpers_constants_and_qfunc_preserve_actual_costs(
    objective: Literal["width", "depth"], mode: Literal["value", "phase"]
) -> None:
    preparation = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).h(2).h(3).s(0).t(2)
    program = synth.synthesize(
        synth.qfunc(composed),
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        constants={"limit": 2},
        functions=(same,),
        preparation=preparation.to_ir(),
        preparation_map={"x": (0, 1), "y": (2, 3)},
        objective=objective,
        output_mode=mode,
        max_qubits=24,
        memory_limit_bytes=64 * 1024**2,
    )
    expected = {}
    for x in range(4):
        for y in range(4):
            amplitude = (
                (1j if x & 2 else 1)
                * complex(
                    math.cos((y >> 1) * math.pi / 4), math.sin((y >> 1) * math.pi / 4)
                )
                / 4
            )
            truth = int(marked(x, y))
            values = (x, y, truth) if mode == "value" else (x, y)
            expected[values] = amplitude * ((-1) ** truth if mode == "phase" else 1)
    result = fq.run(program)
    check_state(program, result, expected)
    assert result.plan.identity == program.execution.identity
    assert program.report["resources"] == program._checked.verify_result(result)
    assert program.report["workspace_qubits"] == program.circuit.n_wires - (
        5 if mode == "value" else 4
    )
    assert dict(
        program.probabilities(result, registers=tuple(program.registers))
    ) == pytest.approx(dict.fromkeys(expected, 1 / 16), abs=1e-12)


@pytest.mark.parametrize("mode", ["value", "phase"])
def test_invalid_truthiness_and_hidden_calls_fail_before_plan(
    mode: Literal["value", "phase"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned unsupported source")
    )
    for expression in ["not x", "0 and x", "1 or x", "0 and absent(x)", "1 or True"]:
        function = _cell(f"def main(x):\n    return {expression}\n")
        with pytest.raises(HybridVerificationError, match="construction.python"):
            synth.synthesize(function, inputs={"x": synth.UInt(2)}, output_mode=mode)


@pytest.mark.parametrize("mode", ["value", "phase"])
def test_boolean_constraints_refuse_before_execution(
    mode: Literal["value", "phase"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned impossible candidate")
    )
    for kwargs in [
        dict(max_operations=1),
        dict(max_workspace=0),
        dict(construction_budget=synth.ConstructionBudget(candidate_checks=1)),
    ]:
        with pytest.raises(synth.SynthesisError):
            synth.synthesize(
                marked,
                inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
                output_mode=mode,
                **kwargs,
            )


@pytest.mark.parametrize("mode", ["value", "phase"])
def test_boolean_memory_limit_refuses_before_numerical_execution(
    mode: Literal["value", "phase"],
) -> None:
    with pytest.raises(PlanningError, match="memory"):
        synth.synthesize(
            marked,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            output_mode=mode,
            memory_limit_bytes=1,
        )


def test_complete_public_boolean_example_runs_without_private_imports_or_gate_design() -> (
    None
):
    source = (
        Path(__file__).resolve().parents[2] / "examples/synthesis_boolean_predicate.py"
    )
    completed = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_boolean_predicate"],
        cwd=source.parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    rows = dict(
        line.split(": ", 1) for line in completed.stdout.splitlines() if ": " in line
    )
    expected = {(x, y, int(marked(x, y))): 1 / 16 for x in range(4) for y in range(4)}
    assert ast.literal_eval(rows["value probabilities"]) == pytest.approx(
        expected, abs=1e-12
    )
    assert ast.literal_eval(rows["phase probabilities"]) == pytest.approx(
        {(x, y): 1 / 16 for x in range(4) for y in range(4)}, abs=1e-12
    )
    assert "Uniform probabilities alone" in completed.stdout
    text = source.read_text()
    assert "._" not in text and "fq.Circuit" not in text and "physical_of" not in text


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_composed_phase_interference_and_double_application(
    objective: Literal["width", "depth"],
) -> None:
    def inverse_bit(x: int) -> bool:
        return not (x and x)

    program = synth.synthesize(
        inverse_bit,
        inputs={"x": synth.UInt(1)},
        output_mode="phase",
        objective=objective,
    )
    n = program.circuit.n_wires
    observed = (
        fq.run(
            fq.Circuit(n, dtype=torch.complex128).h(0).compose(program.circuit).h(0),
            options=fq.ExecutionOptions(precision="complex128"),
        )
        .statevector()
        .reshape(-1)
    )
    expected = torch.zeros_like(observed)
    expected[1 << (n - 1)] = -1
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    twice = fq.Circuit(n, dtype=torch.complex128).h(0).s(0)
    twice = twice.compose(program.circuit).compose(program.circuit)
    observed = (
        fq.run(twice, options=fq.ExecutionOptions(precision="complex128"))
        .statevector()
        .reshape(-1)
    )
    expected.zero_()
    expected[0] = 1 / math.sqrt(2)
    expected[1 << (n - 1)] = 1j / math.sqrt(2)
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_existing_line_value_target_preserves_nonuniform_complex_preparation(
    objective: Literal["width", "depth"],
) -> None:
    angles = (0.2, 0.4, 0.8, 1.2)
    phases = (0.3, 0.5, 0.7, 0.9)
    preparation = fq.Circuit(4, dtype=torch.complex128)
    for q, (angle, phase) in enumerate(zip(angles, phases, strict=True)):
        preparation.ry(q, theta=angle).rz(q, theta=phase)
    program = synth.synthesize(
        marked,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        preparation=preparation.to_ir(),
        preparation_map={"x": (0, 1), "y": (2, 3)},
        objective=objective,
        topology="line",
        max_qubits=24,
        memory_limit_bytes=64 * 1024**2,
    )
    expected = {}
    for x in range(4):
        for y in range(4):
            prefix = (x << 2) | y
            amplitude = complex(1)
            for q, (angle, phase) in enumerate(zip(angles, phases, strict=True)):
                bit = (prefix >> (3 - q)) & 1
                magnitude = math.sin(angle / 2) if bit else math.cos(angle / 2)
                argument = phase / 2 if bit else -phase / 2
                amplitude *= magnitude * complex(math.cos(argument), math.sin(argument))
            expected[(x, y, int(marked(x, y)))] = amplitude
    result = fq.run(program)
    check_state(program, result, expected)
    assert program.report["resources"] == program._checked.verify_result(result)
    assert any(op.name == "swap" for op in program.circuit.instructions)
