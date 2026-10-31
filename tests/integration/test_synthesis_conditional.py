"""Pure conditional values execute coherently with exact XOR and phase cleanup."""

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
    Schedule,
    schedule_arithmetic,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_register_workflow import check_state
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.integration


def smaller(x: int, y: int) -> int:
    return x if x < y else y


def marked(p: int, x: int, y: int) -> bool:
    return (x < y) if p else (x == y)


def raw(
    function: Definition,
    widths: Mapping[str, int],
    strategy: Schedule,
    mode: Literal["value", "phase"] = "value",
) -> ArithmeticCandidate:
    graph = capture(function, widths)
    bindings = {}
    position = 0
    for value in graph.body.blocks[0].arguments:
        width = value.type.parameters[0]
        bindings[value.id] = tuple(range(position, position + width))
        position += width
    result_width = graph.body.blocks[0].operations[-1].operands[0].type.parameters[0]
    output = tuple(range(position, position + result_width)) if mode == "value" else ()
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
    assert n <= 20
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
    "expression,widths",
    [
        ("x if x<y else y", {"x": 2, "y": 2}),
        ("x if p else y", {"p": 1, "x": 2, "y": 1}),
        ("3 if p else 0", {"p": 1}),
        ("p if p else y", {"p": 1, "y": 1}),
        ("y if p else p", {"p": 1, "y": 1}),
        ("p if p else p", {"p": 1}),
        ("x if y else 0", {"x": 1, "y": 1}),
        ("x if p else (y if x<y else 0)", {"p": 1, "x": 1, "y": 1}),
        ("x if 1 else x*x", {"x": 1}),
    ],
)
def test_complete_basis_tables_and_every_xor_target(
    strategy: Schedule, expression: str, widths: Mapping[str, int]
) -> None:
    function = _cell(f"def main({','.join(widths)}):\n    return {expression}\n")
    candidate = raw(function, widths, strategy)
    graph = capture(function, widths)
    output_width = graph.body.blocks[0].operations[-1].operands[0].type.parameters[0]
    input_width = sum(widths.values())
    n = candidate.circuit.n_wires
    for source in range(1 << input_width):
        values = []
        offset = input_width
        for width in widths.values():
            offset -= width
            values.append((source >> offset) & ((1 << width) - 1))
        truth = function(*values)
        for target in range(1 << output_width):
            prefix = (source << output_width) | target
            initial = torch.zeros(1 << n, dtype=torch.complex128)
            initial[prefix << candidate.peak_workspace] = 1
            observed = (
                fq.Circuit.from_ir(candidate.circuit, inputs=initial, device="cpu")
                .state()
                .reshape(-1)
            )
            expected = torch.zeros_like(observed)
            expected[(prefix ^ truth) << candidate.peak_workspace] = 1
            torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_entangled_complex_reference_and_absolute_phase(
    strategy: Schedule, mode: Literal["value", "phase"]
) -> None:
    candidate = raw(marked, {"p": 1, "x": 1, "y": 1}, strategy, mode)
    n = candidate.circuit.n_wires
    targets = 2 if mode == "value" else 1
    amplitudes = torch.randn(
        8 * targets * 2,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(980),
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    row = 0
    for p in range(2):
        for x in range(2):
            for y in range(2):
                truth = int(x < y if p else x == y)
                for target in range(targets):
                    prefix = (p << 2) | (x << 1) | y
                    if mode == "value":
                        prefix = (prefix << 1) | target
                    for reference in range(2):
                        source = (prefix << (candidate.peak_workspace + 1)) | reference
                        destination = (
                            (prefix ^ (truth if mode == "value" else 0))
                            << (candidate.peak_workspace + 1)
                        ) | reference
                        initial[source] = amplitudes[row]
                        expected[destination] = amplitudes[row] * (
                            (-1) ** truth if mode == "phase" else 1
                        )
                        row += 1
    composed = fq.Circuit(n + 1, dtype=torch.complex128).compose(
        candidate.circuit, qubits=range(n)
    )
    observed = (
        fq.Circuit.from_ir(composed.to_ir(), inputs=initial, device="cpu")
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    assert (
        float(
            observed.reshape(8 * targets, 1 << candidate.peak_workspace, 2)[:, 1:, :]
            .abs()
            .square()
            .sum()
        )
        <= 1e-12
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("topology", [None, "line"])
def test_public_minimum_has_exact_width_and_named_uniform_results(
    objective: Literal["width", "depth"], topology: Literal["line"] | None
) -> None:
    program = synth.synthesize(
        smaller,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        objective=objective,
        topology=topology,
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    result = fq.run(program)
    expected = {(x, y, min(x, y)): 0.25 for x in range(4) for y in range(4)}
    check_state(program, result, expected)
    assert program.report["register_widths"]["result"] == 2
    assert dict(
        program.probabilities(result, registers=("x", "y", "result"))
    ) == pytest.approx(dict.fromkeys(expected, 1 / 16), abs=1e-12)
    assert result.plan.identity == program.execution.identity


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_public_one_bit_selection_with_full_complex_preparation(
    objective: Literal["width", "depth"], mode: Literal["value", "phase"]
) -> None:
    preparation = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2).s(0).t(1)
    program = synth.synthesize(
        marked,
        inputs={"p": synth.UInt(1), "x": synth.UInt(1), "y": synth.UInt(1)},
        preparation=preparation.to_ir(),
        preparation_map={"p": (0,), "x": (1,), "y": (2,)},
        output_mode=mode,
        objective=objective,
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    expected = {}
    for p in range(2):
        for x in range(2):
            for y in range(2):
                truth = int(x < y if p else x == y)
                key = (p, x, y, truth) if mode == "value" else (p, x, y)
                expected[key] = (
                    (1j if p else 1)
                    * complex(math.cos(x * math.pi / 4), math.sin(x * math.pi / 4))
                    / math.sqrt(8)
                    * ((-1) ** truth if mode == "phase" else 1)
                )
    check_state(program, fq.run(program), expected)
    if mode == "value":
        assert program.report["register_widths"]["result"] == 1


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_helpers_qfunc_constants_and_looped_carried_selection(
    objective: Literal["width", "depth"],
) -> None:
    def choose(p: int, x: int, y: int) -> int:
        return x if p else y

    @synth.qfunc
    def main(p: int, x: int, y: int, n: int) -> int:
        total = x
        for i in range(n):
            total = choose(p, total, y)
        return total

    program = synth.synthesize(
        main,
        inputs={"p": synth.UInt(1), "x": synth.UInt(1), "y": synth.UInt(1)},
        constants={"n": 2},
        functions=(choose,),
        input_state=synth.InputState.uniform(),
        objective=objective,
        max_qubits=20,
    )
    check_state(
        program,
        fq.run(program),
        {
            (p, x, y, x if p else y): 1 / math.sqrt(8)
            for p in range(2)
            for x in range(2)
            for y in range(2)
        },
    )


def test_invalid_predicates_and_unselected_errors_never_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned invalid conditional")
    )
    for expr in [
        "x if x else y",
        "x if 1 else unavailable(y)",
        "unavailable(x) if 0 else y",
        "x if p else -1",
    ]:
        with pytest.raises(HybridVerificationError):
            synth.synthesize(
                _cell(f"def main(p,x,y):\n    return {expr}\n"),
                inputs={"p": synth.UInt(1), "x": synth.UInt(2), "y": synth.UInt(1)},
            )


def test_resource_and_memory_limits_still_fail_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fq, "run", lambda *a, **k: pytest.fail("executed during synthesis")
    )
    for limits in [
        dict(max_workspace=0),
        dict(max_operations=1),
        dict(max_depth=1),
        dict(construction_budget=synth.ConstructionBudget(candidate_checks=1)),
    ]:
        with pytest.raises(synth.SynthesisError):
            synth.synthesize(
                smaller, inputs={"x": synth.UInt(2), "y": synth.UInt(2)}, **limits
            )
    with pytest.raises(PlanningError, match="memory"):
        synth.synthesize(
            smaller,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            memory_limit_bytes=1,
        )


def test_public_conditional_example_runs_without_manual_gates_or_decoding() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_conditional"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    rows = dict(
        line.split(": ", 1) for line in completed.stdout.splitlines() if ": " in line
    )
    assert ast.literal_eval(rows["minimum probabilities"]) == pytest.approx(
        {(x, y, min(x, y)): 1 / 16 for x in range(4) for y in range(4)}, abs=1e-12
    )
    assert ast.literal_eval(rows["phase probabilities"]) == pytest.approx(
        {(p, x, y): 1 / 8 for p in range(2) for x in range(2) for y in range(2)},
        abs=1e-12,
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_numeric_minimum_preserves_complex_target_and_external_entanglement(
    strategy: Schedule,
) -> None:
    candidate = raw(smaller, {"x": 2, "y": 2}, strategy)
    n = candidate.circuit.n_wires
    amplitudes = torch.randn(
        128, dtype=torch.complex128, generator=torch.Generator().manual_seed(981)
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for prefix in range(64):
        x, y = prefix >> 4, (prefix >> 2) & 3
        for reference in range(2):
            source = (prefix << (candidate.peak_workspace + 1)) | reference
            destination = (
                (prefix ^ min(x, y)) << (candidate.peak_workspace + 1)
            ) | reference
            initial[source] = amplitudes[2 * prefix + reference]
            expected[destination] = amplitudes[2 * prefix + reference]
    composed = fq.Circuit(n + 1, dtype=torch.complex128).compose(
        candidate.circuit, qubits=range(n)
    )
    observed = (
        fq.Circuit.from_ir(composed.to_ir(), inputs=initial, device="cpu")
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "expression", ["0 if p else 3", "0 if p else 0", "1 if p else 1"]
)
def test_literal_branches_zero_extend_and_restore_workspace(expression: str) -> None:
    function = _cell(f"def main(p):\n    return {expression}\n")
    program = synth.synthesize(
        function, inputs={"p": synth.UInt(1)}, input_state=synth.InputState.uniform()
    )
    check_state(
        program, fq.run(program), {(p, function(p)): 1 / math.sqrt(2) for p in range(2)}
    )


def test_public_basis_example_uses_explicit_classical_config_and_named_reading() -> (
    None
):
    program = synth.synthesize(
        smaller,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        input_state=synth.InputState.basis(x=3, y=1),
    )
    result = fq.run(program)
    check_state(program, result, {(3, 1, 1): 1})
    assert dict(program.probabilities(result)) == {(1,): 1.0}
