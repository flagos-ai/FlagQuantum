"""Private phase construction through existing targets, Program and real CPU."""

import cmath
import math
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from flagquantum.compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    select_arithmetic,
)
from flagquantum.compiler._construction.target import (
    PHASE_BASIS,
    Target,
    compile_target,
)
from flagquantum.compiler.coupling import CouplingMap
from flagquantum.core.ir import Instruction
from flagquantum.errors import PlanningError
from flagquantum.services.synthesis import SynthesisError, _complete
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


def marked(x: int, y: int) -> bool:
    return x * x + y < 5


def equal(x: int, y: int) -> bool:
    return x == y


def true(x: int) -> bool:
    return x * 0 + 1


def false(x: int) -> bool:
    return x * 0


def bit(x: int) -> bool:
    return x


def is_one(x: int) -> bool:
    return x == 1


def pair_marked(x: int, y: int) -> bool:
    return x < y


def target(width, routed, *, optimize=True):
    return Target(
        basis=PHASE_BASIS if routed else None,
        couplings=(CouplingMap.line(width),) if routed else (),
        optimize=False if routed else optimize,
        restore_layout=not routed,
    )


def raw(function, widths, strategy):
    graph = capture(function, widths)
    bindings = {}
    position = 0
    for v in graph.body.blocks[0].arguments:
        width = v.type.parameters[0]
        bindings[v.id] = tuple(range(position, position + width))
        position += width
    options = dict(
        entry=graph.name,
        workspace_state="fresh_zero",
        scheduling=strategy,
        output_mode="phase",
    )
    probe = schedule_arithmetic(
        (graph,),
        bindings,
        (),
        n_qubits=64,
        workspace=tuple(range(position, 64)),
        **options,
    )
    n = position + probe.peak_workspace
    assert n <= 21
    return schedule_arithmetic(
        (graph,),
        bindings,
        (),
        n_qubits=n,
        workspace=tuple(range(position, n)),
        **options,
    )


def program(function=marked, *, objective="width", routed=False):
    graph = capture(function, {"x": 2, "y": 2})
    preparation = fq.Circuit(4, dtype=torch.complex128)
    for q in range(4):
        preparation.h(q)
    preparation.s(0).t(2)
    # The private target supplies explicit supported widths, never a resized device.
    profile = Target(
        basis=PHASE_BASIS if routed else None,
        couplings=tuple(CouplingMap.line(n) for n in range(4, 33)) if routed else (),
        optimize=not routed,
        restore_layout=not routed,
    )
    selection = select_arithmetic(
        (graph,),
        entry=graph.name,
        input_names=("x", "y"),
        preparation=preparation.to_ir(),
        output_mode="phase",
        objective=objective,
        target=profile,
        budget=ConstructionBudget(qubits=32),
    )
    return _complete(
        selection,
        {
            "function": graph.name,
            "source_format": "python",
            "output_mode": "phase",
            "scope": "actual_cpu_preparation_predicate_phase_cleanup",
            "source_locations": tuple(
                op.location for op in graph.body.blocks[0].operations if op.location
            ),
        },
        memory_limit_bytes=64 * 1024**2,
        memory_limit_origin="explicit",
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("function", [marked, equal])
def test_all_16_basis_amplitudes_have_exact_marked_sign_and_clean_workspace(
    strategy, function
):
    candidate = raw(function, {"x": 2, "y": 2}, strategy)
    n = candidate.circuit.n_wires
    for x in range(4):
        for y in range(4):
            index = ((x << 2) | y) << candidate.peak_workspace
            circuit = fq.Circuit(n, dtype=torch.complex128)
            for q in range(n):
                if index & (1 << (n - q - 1)):
                    circuit.x(q)
            state = (
                fq.run(
                    circuit.compose(candidate.circuit),
                    options=fq.ExecutionOptions(
                        precision="complex128",
                        device="cpu",
                        mode="statevector",
                        backend="torch",
                    ),
                )
                .statevector()
                .reshape(-1)
            )
            expected = torch.zeros_like(state)
            expected[index] = (-1) ** int(
                x * x + y < 5 if function is marked else x == y
            )
            torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "function,routed", [(marked, False), (equal, False), (marked, True)]
)
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_checked_program_preserves_full_relative_phase_input_names_and_actual_costs(
    function, objective, routed
):
    p = program(function, objective=objective, routed=routed)
    result = fq.run(p)
    expected = {
        (x, y): (-1) ** int(x * x + y < 5 if function is marked else x == y)
        * cmath.exp(1j * ((x >> 1) * math.pi / 2 + (y >> 1) * math.pi / 4))
        / 4
        for x in range(4)
        for y in range(4)
    }
    check_state(p, result, expected)
    assert dict(p.probabilities(result, registers=("x", "y"))) == pytest.approx(
        dict.fromkeys(expected, 1 / 16), abs=1e-12
    )
    assert dict(p.report["register_widths"]) == {"x": 2, "y": 2}
    assert p.report["resources"] == p._checked.verify_result(result)
    assert p.report["workspace_qubits"] == p.report["resources"].n_qubits - 4
    with pytest.raises(ValueError, match="available names.*x.*y"):
        p.probabilities(result)


@pytest.mark.parametrize("function", [true, false, bit, is_one, pair_marked])
@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("routed", [False, True])
def test_external_control_reference_preserves_even_constant_global_phase_after_target(
    function, strategy, routed
):
    widths = {"x": 1, "y": 1} if function is pair_marked else {"x": 1}
    n_inputs = len(widths)
    candidate = raw(function, widths, strategy)
    n = candidate.circuit.n_wires
    # Control only the phase marker; the same U/U-adjoint cancels on control=0.
    gates = candidate.circuit.instructions
    middle = len(gates) // 2
    controlled = replace(
        candidate.circuit,
        n_wires=n + 1,
        shape=(),
        instructions=(
            *gates[:middle],
            Instruction("cz", (n, gates[middle].wires[0])),
            *gates[middle + 1 :],
        ),
    )
    product = compile_target(controlled, target(n + 1, routed), 100_000)
    amplitudes = torch.randn(
        1 << (n_inputs + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(2081),
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for value in range(1 << n_inputs):
        args = tuple((value >> (n_inputs - bit - 1)) & 1 for bit in range(n_inputs))
        for c in range(2):
            initial[(value << (n + 1 - n_inputs)) | c] = amplitudes[2 * value + c]
            physical = c << (n - product.layout.physical_of(n))
            for logical, actual in enumerate(args):
                physical |= actual << (n - product.layout.physical_of(logical))
            expected[physical] = amplitudes[2 * value + c] * (
                (-1) ** int(function(*args)) if c else 1
            )
    state = (
        fq.Circuit.from_ir(
            product.executable, inputs=initial, dtype=torch.complex128, device="cpu"
        )
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_interference_and_double_application_show_phase_without_a_fake_flag(strategy):
    candidate = raw(is_one, {"x": 1}, strategy)
    n = candidate.circuit.n_wires
    circuit = fq.Circuit(n, dtype=torch.complex128).h(0).compose(candidate.circuit).h(0)
    state = (
        fq.run(circuit, options=fq.ExecutionOptions(precision="complex128"))
        .statevector()
        .reshape(-1)
    )
    expected = torch.zeros_like(state)
    expected[1 << (n - 1)] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    identity = (
        fq.Circuit(n, dtype=torch.complex128)
        .h(0)
        .s(0)
        .compose(candidate.circuit)
        .compose(candidate.circuit)
    )
    state = (
        fq.run(identity, options=fq.ExecutionOptions(precision="complex128"))
        .statevector()
        .reshape(-1)
    )
    expected.zero_()
    expected[0] = 1 / math.sqrt(2)
    expected[1 << (n - 1)] = 1j / math.sqrt(2)
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_constraints_budget_and_memory_refuse_without_an_authorized_partial_program():
    graph = capture(marked, {"x": 2, "y": 2})
    selection = select_arithmetic(
        (graph,),
        entry=graph.name,
        output_mode="phase",
        constraints=Constraints(CircuitLimits(max_operations=1)),
    )
    assert selection.selected is None and selection.recoveries
    with pytest.raises(SynthesisError):
        _complete(selection, {})
    incomplete = select_arithmetic(
        (graph,),
        entry=graph.name,
        output_mode="phase",
        budget=ConstructionBudget(candidate_checks=1),
    )
    with pytest.raises(SynthesisError):
        _complete(incomplete, {})
    valid = select_arithmetic((graph,), entry=graph.name, output_mode="phase")
    with pytest.raises(PlanningError):
        _complete(valid, {}, memory_limit_bytes=1)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_unverified_line_equality_remains_refused_by_the_existing_swap_witness_guard(
    objective,
):
    with pytest.raises(
        SynthesisError, match="target.layout.*lost routed SWAP evidence"
    ):
        program(equal, objective=objective, routed=True)


def test_private_contributor_example_reports_real_phase_evidence():
    import subprocess
    import sys
    from pathlib import Path

    completed = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_phase_oracle"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert "public output_mode remains proposed" in completed.stdout
    assert "Uniform probabilities alone do not verify the phase" in completed.stdout
    values = dict(
        line.split(": ", 1) for line in completed.stdout.splitlines() if ": " in line
    )
    assert float(values["phase/workspace residual"]) <= 1e-12
    assert "n_qubits=18," in values["resources"]
