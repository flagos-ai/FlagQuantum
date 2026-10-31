"""Real CPU predicates through both schedules and the existing public workflow."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from flagquantum.compiler._construction.selection import ConstructionBudget
from flagquantum.ecosystem import InteropConversionError
from flagquantum.runtime.execution_plan_contract import plan_execution_program
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


def polynomial_less(x: int, y: int) -> bool:
    return x * x + y < 5


def polynomial_equal(x: int, y: int) -> bool:
    return x * x + y == 5


def direct_equal(x: int, y: int) -> bool:
    return x == y


def direct_less(x: int, y: int) -> bool:
    return x < y


def candidate(function, strategy):
    graph = capture(function, {"x": 2, "y": 2})
    bindings = {
        v.id: ports
        for v, ports in zip(
            graph.body.blocks[0].arguments, ((0, 1), (2, 3)), strict=True
        )
    }
    options = dict(entry=graph.name, workspace_state="fresh_zero", scheduling=strategy)
    probe = schedule_arithmetic(
        (graph,), bindings, (4,), n_qubits=64, workspace=tuple(range(5, 64)), **options
    )
    n = 5 + probe.peak_workspace
    assert (
        n <= 22
    )  # Bound real state buffers to at most 64 MiB, not a tiny-width rewrite.
    return schedule_arithmetic(
        (graph,), bindings, (4,), n_qubits=n, workspace=tuple(range(5, n)), **options
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("function", [polynomial_less, direct_equal])
def test_all_16_basis_inputs_and_both_xor_targets_on_real_cpu(strategy, function):
    lowered = candidate(function, strategy)
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    for x in range(4):
        for y in range(4):
            truth = int(x * x + y < 5) if function is polynomial_less else int(x == y)
            for target in (0, 1):
                prefix = ((x << 2 | y) << 1) | target
                expected = (
                    ((x << 2 | y) << 1) | (target ^ truth)
                ) << lowered.peak_workspace
                initial = prefix << lowered.peak_workspace
                circuit = fq.Circuit(lowered.circuit.n_wires, dtype=torch.complex128)
                for q in range(circuit.n_qubits):
                    if initial & (1 << (circuit.n_qubits - q - 1)):
                        circuit.x(q)
                state = (
                    fq.run(circuit.compose(lowered.circuit), options=options)
                    .statevector()
                    .reshape(-1)
                )
                reference = torch.zeros_like(state)
                reference[expected] = 1
                torch.testing.assert_close(state, reference, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("function", [polynomial_less, polynomial_equal])
def test_complex_amplitudes_and_target_superposition_return_workspace_zero(
    strategy, function
):
    lowered = candidate(function, strategy)
    amplitudes = torch.randn(
        32, dtype=torch.complex128, generator=torch.Generator().manual_seed(208)
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << lowered.circuit.n_wires, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for x in range(4):
        for y in range(4):
            truth = (
                int(x * x + y < 5)
                if function is polynomial_less
                else int(x * x + y == 5)
            )
            for target in (0, 1):
                prefix = (x << 3) | (y << 1) | target
                initial[prefix << lowered.peak_workspace] = amplitudes[prefix]
                expected[(prefix ^ truth) << lowered.peak_workspace] = amplitudes[
                    prefix
                ]
    state = (
        fq.Circuit.from_ir(
            lowered.circuit, inputs=initial, dtype=torch.complex128, device="cpu"
        )
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert (
        float(
            state.reshape(32, 1 << lowered.peak_workspace)[:, 1:].abs().square().sum()
        )
        <= 1e-12
    )
    assert lowered.peak_workspace == max(e.live_workspace for e in lowered.events)


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("topology", [None, "line"])
def test_public_predicate_has_same_named_workflow_and_actual_target_costs(
    objective, topology
):
    program = synth.synthesize(
        direct_less,
        inputs={"x": synth.UInt(1), "y": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        objective=objective,
        topology=topology,
    )
    result = fq.run(program)
    expected = {
        (x, y, int(x < y)): 1 / math.sqrt(8) for x in range(2) for y in range(4)
    }
    check_state(program, result, expected)
    assert dict(
        program.probabilities(result, registers=("x", "y", "result"))
    ) == pytest.approx(dict.fromkeys(expected, 1 / 8), abs=1e-12)
    assert dict(program.report["register_widths"]) == {"x": 1, "y": 2, "result": 1}
    assert program.report["resources"] == estimate_resources(
        plan_execution_program(program.execution)
    )
    assert program.report["resources"] == program._checked.verify_result(result)
    if topology:
        assert any(op.name == "swap" for op in program.circuit.instructions)


@pytest.mark.parametrize(
    "expression,truth",
    [
        ("x == x", lambda x: 1),
        ("x < x", lambda x: 0),
        ("3 < x", lambda x: int(x > 3)),
        ("x < 0", lambda x: 0),
        ("x == 7", lambda x: int(x == 7)),
        ("x == 8", lambda x: 0),
        ("x < 8", lambda x: 1),
    ],
)
def test_aliases_constants_and_boundaries_keep_full_numeric_semantics(
    expression, truth
):
    from tests.team.compiler.test_construction_python import _cell

    function = _cell(f"def main(x):\n    return {expression}\n")
    program = synth.synthesize(
        function, inputs={"x": synth.UInt(3)}, input_state=synth.InputState.uniform()
    )
    check_state(
        program, fq.run(program), {(x, truth(x)): 1 / math.sqrt(8) for x in range(8)}
    )


def test_constraints_and_construction_budget_fail_before_run():
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            direct_equal,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            max_operations=1,
        )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            direct_equal,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            construction_budget=ConstructionBudget(candidate_checks=1),
        )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            polynomial_less,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            max_workspace=0,
        )


def test_qmod_grammar_does_not_expand_with_python_predicates():
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            "qfunc main(output x:qnum<2>, output result:qnum) { allocate(x); result=x<2; }",
            source_format="qmod",
        )


def test_complete_public_predicate_example_runs_without_gate_design():
    import ast
    import subprocess
    import sys
    from pathlib import Path

    completed = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_predicate"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    lines = dict(line.split(": ", 1) for line in completed.stdout.splitlines())
    assert ast.literal_eval(lines["basis"]) == {(1, 2, 1): 1.0}
    expected = {(x, y, int(x * x + y < 5)): 1 / 16 for x in range(4) for y in range(4)}
    assert ast.literal_eval(lines["joint"]) == pytest.approx(expected, abs=1e-12)
    assert "n_qubits=19," in lines["resources"]
    assert lines["workspace"] == "14"


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_one_bit_helper_result_reused_in_arithmetic_is_cleaned(objective):
    def helper(x: int, y: int) -> bool:
        return x == y

    def main(x: int, y: int) -> int:
        flag = helper(x, y)
        return flag + x

    program = synth.synthesize(
        main,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        functions=(helper,),
        input_state=synth.InputState.uniform(),
        objective=objective,
    )
    expected = {(x, y, x + int(x == y)): 0.25 for x in range(4) for y in range(4)}
    check_state(program, fq.run(program), expected)
    assert program.report["register_widths"]["result"] == 3
