"""Complete integer references for comparison normalization and coherent chains."""

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
RELATIONS = ("==", "!=", "<", "<=", ">", ">=")
CASES = (
    [(f"x {op} y", {"x": 2, "y": 1}) for op in RELATIONS]
    + [
        (f"{left} {op} {right}", {"x": 2})
        for op in RELATIONS
        for left, right in [("x", "2"), ("2", "x")]
    ]
    + [(f"x {op} x", {"x": 2}) for op in RELATIONS]
    + [
        (expr, {"x": 2, "y": 1})
        for expr in [
            "0<=x<=2",
            "3>=x>y",
            "x<y>x",
            "x<=y>=x",
            "x!=y==0",
            "x==y==x",
            "x<=y<=x",
            "0 and (x<x<2)",
        ]
    ]
)


def interval(x: int, low: int, high: int, excluded: int) -> bool:
    return low <= x <= high and x != excluded


def mixed(x: int, y: int) -> bool:
    return 3 >= x > y != 0


def raw(
    function: Definition,
    widths: Mapping[str, int],
    strategy: Schedule,
    mode: Literal["value", "phase"] = "value",
) -> ArithmeticCandidate:
    graph = capture(function, widths)
    position = 0
    bindings = {}
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
@pytest.mark.parametrize("expression,widths", CASES)
def test_every_relation_and_chain_over_all_inputs_and_xor_destinations(
    strategy: Schedule, expression: str, widths: Mapping[str, int]
) -> None:
    function = _cell(f"def main({','.join(widths)}):\n    return {expression}\n")
    candidate = raw(function, widths, strategy)
    n = candidate.circuit.n_wires
    for source in range(1 << sum(widths.values())):
        values = []
        offset = sum(widths.values())
        for width in widths.values():
            offset -= width
            values.append((source >> offset) & ((1 << width) - 1))
        truth = int(function(*values))
        for target in range(2):
            prefix = (source << 1) | target
            initial = torch.zeros(1 << n, dtype=torch.complex128)
            initial[prefix << candidate.peak_workspace] = 1
            observed = (
                fq.Circuit.from_ir(
                    candidate.circuit,
                    inputs=initial,
                    dtype=torch.complex128,
                    device="cpu",
                )
                .state()
                .reshape(-1)
            )
            expected = torch.zeros_like(initial)
            expected[(prefix ^ truth) << candidate.peak_workspace] = 1
            torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_mixed_chain_preserves_arbitrary_complex_targets_and_external_reference(
    strategy: Schedule, mode: Literal["value", "phase"]
) -> None:
    candidate = raw(mixed, {"x": 2, "y": 1}, strategy, mode)
    n = candidate.circuit.n_wires
    targets = 2 if mode == "value" else 1
    amplitudes = torch.randn(
        8 * targets * 2,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(982),
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    row = 0
    for x in range(4):
        for y in range(2):
            truth = int(3 >= x > y != 0)
            for target in range(targets):
                prefix = (x << 1) | y
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
    circuit = fq.Circuit(n + 1, dtype=torch.complex128).compose(
        candidate.circuit, qubits=range(n)
    )
    observed = (
        fq.Circuit.from_ir(
            circuit.to_ir(), inputs=initial, dtype=torch.complex128, device="cpu"
        )
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
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_public_all_eight_basis_values_match_interval_and_exclusion(
    objective: Literal["width", "depth"], mode: Literal["value", "phase"]
) -> None:
    for x in range(8):
        program = synth.synthesize(
            interval,
            inputs={"x": synth.UInt(3)},
            constants={"low": 1, "high": 6, "excluded": 3},
            input_state=synth.InputState.basis(x=x),
            objective=objective,
            output_mode=mode,
            max_qubits=24,
            memory_limit_bytes=256 * 1024**2,
        )
        truth = int(1 <= x <= 6 and x != 3)
        key = (x, truth) if mode == "value" else (x,)
        check_state(
            program, fq.run(program), {key: (-1) ** truth if mode == "phase" else 1}
        )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_public_full_complex_interval_preparation_and_named_reading(
    objective: Literal["width", "depth"], mode: Literal["value", "phase"]
) -> None:
    preparation = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2).s(0).t(1)
    program = synth.synthesize(
        interval,
        inputs={"x": synth.UInt(3)},
        constants={"low": 1, "high": 6, "excluded": 3},
        preparation=preparation.to_ir(),
        preparation_map={"x": (0, 1, 2)},
        objective=objective,
        output_mode=mode,
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    expected = {}
    for x in range(8):
        truth = int(1 <= x <= 6 and x != 3)
        amplitude = (
            (1j if x & 4 else 1)
            * complex(
                math.cos(((x >> 1) & 1) * math.pi / 4),
                math.sin(((x >> 1) & 1) * math.pi / 4),
            )
            / math.sqrt(8)
        )
        key = (x, truth) if mode == "value" else (x,)
        expected[key] = amplitude * ((-1) ** truth if mode == "phase" else 1)
    result = fq.run(program)
    check_state(program, result, expected)
    names = ("x", "result") if mode == "value" else ("x",)
    assert dict(program.probabilities(result, registers=names)) == pytest.approx(
        dict.fromkeys(expected, 1 / 8), abs=1e-12
    )
    assert result.plan.identity == program.execution.identity


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_public_line_target_keeps_zero_extended_comparison_semantics(
    objective: Literal["width", "depth"],
) -> None:
    function = _cell("def main(x,y):\n    return x >= y\n")
    program = synth.synthesize(
        function,
        inputs={"x": synth.UInt(2), "y": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        objective=objective,
        topology="line",
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    check_state(
        program,
        fq.run(program),
        {(x, y, int(x >= y)): 1 / math.sqrt(8) for x in range(4) for y in range(2)},
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_helpers_qfunc_loops_and_conditional_values_share_chain_results(
    objective: Literal["width", "depth"],
) -> None:
    def accepted(x: int) -> bool:
        return 0 <= x <= 2

    @synth.qfunc
    def main(x: int, n: int) -> bool:
        flag = accepted(x)
        for i in range(n):
            flag = (flag and x != i) if x >= 1 else (x == 0)
        return flag

    program = synth.synthesize(
        main,
        inputs={"x": synth.UInt(2)},
        constants={"n": 2},
        functions=(accepted,),
        input_state=synth.InputState.uniform(),
        objective=objective,
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    check_state(
        program, fq.run(program), {(x, int(x in (0, 2))): 0.5 for x in range(4)}
    )


def test_false_prefix_unknown_calls_and_identity_never_reach_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned unsupported source")
    )
    for expression in ["0>x<unavailable(x)", "x is x", "0>x not in x", "0<x<True"]:
        with pytest.raises(HybridVerificationError):
            synth.synthesize(
                _cell(f"def main(x):\n    return {expression}\n"),
                inputs={"x": synth.UInt(2)},
            )


def test_budgets_and_memory_remain_checked_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fq, "run", lambda *a, **k: pytest.fail("executed during synthesis")
    )
    for constraints in [
        dict(max_qubits=4),
        dict(max_workspace=0),
        dict(max_operations=1),
        dict(construction_budget=synth.ConstructionBudget(candidate_checks=1)),
    ]:
        with pytest.raises(synth.SynthesisError):
            synth.synthesize(
                mixed, inputs={"x": synth.UInt(2), "y": synth.UInt(1)}, **constraints
            )
    with pytest.raises(PlanningError, match="memory"):
        synth.synthesize(
            mixed, inputs={"x": synth.UInt(2), "y": synth.UInt(1)}, memory_limit_bytes=1
        )


def test_public_interval_cli_has_no_gate_or_decoding_requirement() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_interval"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    rows = dict(
        line.split(": ", 1) for line in completed.stdout.splitlines() if ": " in line
    )
    assert ast.literal_eval(rows["value probabilities"]) == pytest.approx(
        {(x, int(1 <= x <= 6 and x != 3)): 1 / 8 for x in range(8)}, abs=1e-12
    )
    assert ast.literal_eval(rows["phase probabilities"]) == pytest.approx(
        {(x,): 1 / 8 for x in range(8)}, abs=1e-12
    )
