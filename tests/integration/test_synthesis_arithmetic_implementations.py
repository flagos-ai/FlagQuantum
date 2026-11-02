"""Actual AO-02 resources, complete-domain XOR, relative phases and clean work."""

from collections.abc import Callable
from itertools import product
from typing import Literal

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from examples.synthesis_arithmetic_implementations import QMOD, main
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import (
    ArithmeticCandidate,
    Schedule,
    _logical_cost,
    adjoint_value_candidate,
    schedule_arithmetic,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


def square(x: int) -> int:
    return x * x


def cube_constant(x: int) -> int:
    return 3 * x


def fifth_constant(x: int) -> int:
    return 5 * x


def add(x: int, y: int) -> int:
    return x + y


def multiply_add(x: int, y: int, z: int) -> int:
    return x * y + z


def energy(x: int, y: int) -> int:
    return x * x + y


def candidate(
    function: Callable[..., int],
    widths: dict[str, int],
    strategy: Schedule = "retain",
    *,
    capacity: int = 128,
    max_operations: int = 100_000,
) -> tuple[ArithmeticCandidate, tuple[tuple[int, ...], ...]]:
    graph = capture(function, widths)
    block = graph.body.blocks[0]
    position = 0
    registers = []
    for value in block.arguments:
        width = value.type.parameters[0]
        registers.append(tuple(range(position, position + width)))
        position += width
    output_width = block.operations[-1].operands[0].type.parameters[0]
    registers.append(tuple(range(position, position + output_width)))
    position += output_width
    result = schedule_arithmetic(
        (graph,),
        {arg.id: registers[i] for i, arg in enumerate(block.arguments)},
        registers[-1],
        entry=graph.name,
        n_qubits=capacity,
        workspace=tuple(range(position, capacity)),
        workspace_state="fresh_zero",
        scheduling=strategy,
        max_operations=max_operations,
    )
    return result, tuple(registers)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_native_and_qmod_complete_uniform_state_and_actual_costs(
    objective: Literal["width", "depth"],
) -> None:
    expected = {(x, y, x * x + y): 0.25 for x, y in product(range(4), repeat=2)}
    for definition in (main, QMOD):
        program = (
            synth.synthesize(definition, source_format="qmod", objective=objective)
            if isinstance(definition, str)
            else synth.synthesize(definition, objective=objective)
        )
        assert len(program.registers["result"]) == 4
        check_state(program, fq.run(program), expected)
        resource = program.report["resources"]
        assert (
            resource.n_qubits,
            program.report["workspace_qubits"],
            resource.n_operations,
            resource.depth,
        ) == (16, 8, 81, 59)
        assert program.selection.selected is not None
        assert program.selection.selected.arithmetic is not None
        segment = program.selection.selected.arithmetic.segments[0]
        assert {rule for _, rule in segment.implementation_rules} == {
            "square_symmetry",
            "terminal_xor_add",
        }
        assert segment.conservative is not None
        assert _logical_cost(segment) < _logical_cost(segment.conservative)


@pytest.mark.parametrize("x,y", list(product(range(4), repeat=2)))
def test_all_sixteen_basis_inputs_execute_with_zero_workspace(x: int, y: int) -> None:
    source = QMOD.replace(
        "allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);",
        f"x={x};y={y};",
    )
    program = synth.synthesize(source, source_format="qmod")
    check_state(program, fq.run(program), {(x, y, x * x + y): 1})


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize(
    "function,widths,rule",
    [
        (square, {"x": 2}, "square_symmetry"),
        (square, {"x": 3}, "square_symmetry"),
        (cube_constant, {"x": 4}, "constant_shift_add"),
        (cube_constant, {"x": 5}, "constant_shift_add"),
        (fifth_constant, {"x": 4}, "constant_shift_add"),
        (add, {"x": 2, "y": 2}, "terminal_xor_add"),
        (add, {"x": 1, "y": 3}, "terminal_xor_add"),
        (multiply_add, {"x": 1, "y": 1, "z": 2}, "terminal_xor_add"),
    ],
)
def test_each_structure_has_nonexample_strict_actual_gain_and_every_xor_target(
    function: Callable[..., int], widths: dict[str, int], rule: str, strategy: Schedule
) -> None:
    result, registers = candidate(function, widths, strategy)
    assert result.conservative is not None
    assert any(name == rule for _, name in result.implementation_rules)
    before, after = _logical_cost(result.conservative), _logical_cost(result)
    assert all(a <= b for a, b in zip(after, before, strict=True)) and after != before
    output_width = len(registers[-1])
    for values in product(*(range(1 << width) for width in widths.values())):
        for target in range(1 << output_width):
            initial = encode((*values, target), registers)
            expected = encode((*values, target ^ function(*values)), registers)
            assert classical_apply(result.circuit, initial) == expected


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_nonzero_targets_complex_interference_external_control_and_adjoint(
    strategy: Schedule, controlled: bool, inverse: bool
) -> None:
    probe, registers = candidate(energy, {"x": 2, "y": 2}, strategy)
    n = 8 + probe.peak_workspace
    result, _ = candidate(energy, {"x": 2, "y": 2}, strategy, capacity=n)
    if inverse:
        result = adjoint_value_candidate(result)
    block = fq.Circuit.from_ir(result.circuit, dtype=torch.complex128)
    circuit = (
        block.control(1, ctrl_qubits=(n,))
        if controlled
        else fq.Circuit(n + 1, dtype=torch.complex128).compose(result.circuit)
    )
    rows = list(product(range(4), range(4), range(16), range(2)))
    amplitudes = torch.randn(
        len(rows),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(20261102),
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for i, (x, y, target, control) in enumerate(rows):
        offset = result.peak_workspace + 1
        initial[(((x << 2 | y) << 4) | target) << offset | control] = amplitudes[i]
        answer = x * x + y if not controlled or control else 0
        expected[(((x << 2 | y) << 4) | (target ^ answer)) << offset | control] = (
            amplitudes[i]
        )
    observed = (
        fq.Circuit.from_ir(circuit.to_ir(), inputs=initial, dtype=torch.complex128)
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    assert (
        float(
            observed.reshape(256, 1 << result.peak_workspace, 2)[:, 1:, :]
            .abs()
            .square()
            .sum()
        )
        <= 1e-12
    )


def test_direct_output_can_fit_when_the_conservative_candidate_is_refused() -> None:
    result, registers = candidate(add, {"x": 2, "y": 2}, capacity=9, max_operations=14)
    assert result.conservative is None and result.conservative_errors
    assert result.peak_workspace == 2 and len(result.circuit.instructions) == 14
    for x, y, target in product(range(4), range(4), range(8)):
        assert classical_apply(
            result.circuit, encode((x, y, target), registers)
        ) == encode((x, y, target ^ (x + y)), registers)


def test_output_alias_and_unknown_workspace_are_rejected_before_optimized_emission() -> (
    None
):
    graph = capture(add, {"x": 2, "y": 2})
    args = graph.body.blocks[0].arguments
    bindings = {args[0].id: (0, 1), args[1].id: (2, 3)}
    with pytest.raises(HybridVerificationError, match="binding.alias"):
        schedule_arithmetic(
            (graph,),
            bindings,
            (0, 4, 5),
            entry=graph.name,
            n_qubits=16,
            workspace=tuple(range(6, 16)),
            workspace_state="fresh_zero",
        )
    with pytest.raises(HybridVerificationError, match="workspace.precondition"):
        schedule_arithmetic(
            (graph,),
            bindings,
            (4, 5, 6),
            entry=graph.name,
            n_qubits=16,
            workspace=tuple(range(7, 16)),
        )


def test_constant_tradeoff_retains_the_actual_conservative_candidate() -> None:
    result, _ = candidate(cube_constant, {"x": 2})
    assert result.conservative is not None and result.implementation_cost is not None
    assert result.implementation == "conservative"
    assert _logical_cost(result) == _logical_cost(result.conservative)
    assert any(
        a > b
        for a, b in zip(
            result.implementation_cost, _logical_cost(result.conservative), strict=True
        )
    )


def test_optional_implementation_probe_has_a_recorded_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler._construction import lowering

    monkeypatch.setattr(lowering, "_NORMALIZATION_PROBE_MAX_OPERATIONS", 10)
    result, _ = candidate(cube_constant, {"x": 4})
    assert result.implementation == "conservative"
    assert result.implementation_cost is None
    assert result.implementation_errors[0].code == "construction.implementation.budget"
