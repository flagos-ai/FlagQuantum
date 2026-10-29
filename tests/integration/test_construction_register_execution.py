"""Real CPU register effects and arithmetic composition, not native Qmod runs."""

import math
from collections.abc import Mapping
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.selection import (
    ConstructionBudget,
    Selection,
    select_register_program,
)
from flagquantum.compiler._construction.target import Target
from flagquantum.compiler.coupling import CouplingMap
from tests.register_program_cases import (
    Registers,
    double,
    pressure,
    pressure_program,
    product,
    tutorial_program,
)

pytestmark = pytest.mark.integration
BASIS = ("x", "h", "s", "sdg", "t", "tdg", "ry", "rz", "cx", "u3", "swap")


def _check(selection: Selection, expected: Mapping[tuple[int, ...], complex]) -> None:
    row = selection.selected
    assert row is not None and row.target is not None and row.arithmetic is not None
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    assert row.target.executable.n_wires <= 22
    observed = fq.run(row.target.submitted, options=options).statevector().reshape(-1)
    reference = torch.zeros_like(observed)
    n = row.target.executable.n_wires
    for values, amplitude in expected.items():
        logical = 0
        for value, ports in zip(values, selection.registers.values(), strict=True):
            for bit, port in enumerate(reversed(ports)):
                logical |= ((value >> bit) & 1) << (n - 1 - port)
        physical = sum(
            ((logical >> (n - 1 - port)) & 1)
            << (n - 1 - row.target.layout.physical_of(port))
            for port in range(n)
        )
        reference[physical] = torch.tensor(amplitude, dtype=reference.dtype)
    torch.testing.assert_close(observed, reference, atol=1e-12, rtol=0)
    # Every amplitude in the independent reference has a zero workspace tail.
    outputs = sum(len(ports) for ports in selection.registers.values())
    indices = torch.arange(observed.numel())
    dirty = torch.zeros_like(indices, dtype=torch.bool)
    for port in range(outputs, n):
        physical = row.target.layout.physical_of(port)
        dirty |= ((indices >> (n - 1 - physical)) & 1).bool()
    assert float(observed[dirty].abs().square().sum()) <= 1e-12


def test_fresh_qbit_x_uses_existing_runtime_without_expression_inputs() -> None:
    p = Registers()
    q = p.declare("q", 1)
    p.emit("qreg.allocate", q)
    p.emit("qreg.gate", q, gate="x", index=0)
    _check(select_register_program(p.finish()), {(1,): 1})


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_fresh_tutorial_model_is_coherent_and_cleans_workspace(
    objective: Literal["width", "depth"],
) -> None:
    program, functions = tutorial_program(2)
    _check(
        select_register_program(program, functions, objective=objective),
        {(x, x * x + 1): 0.5 for x in range(4)},
    )


def test_quantum_constant_outputs_remain_live_beside_the_product() -> None:
    p = Registers()
    a, b, result = p.declare("a", 2), p.declare("b", 2), p.declare("res", 4)
    graph = capture(product, {"a": 2, "b": 2})
    p.emit("qreg.initialize", a, value=2)
    p.emit("qreg.initialize", b, value=3)
    p.emit("qreg.compute", result, a, b, callee=graph.name)
    _check(select_register_program(p.finish(), (graph,)), {(2, 3, 6): 1})


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_repeated_actual_register_reads_preserve_alias_identity(
    objective: Literal["width", "depth"],
) -> None:
    p = Registers()
    x, result = p.declare("x", 2), p.declare("result", 4)
    graph = capture(product, {"a": 2, "b": 2})
    p.emit("qreg.allocate", x)
    for index in (0, 1):
        p.emit("qreg.gate", x, gate="h", index=index)
    p.emit("qreg.gate", x, gate="s", index=0)
    p.emit("qreg.compute", result, x, x, callee=graph.name)
    _check(
        select_register_program(p.finish(), (graph,), objective=objective),
        {(x, x * x): (1j if x % 2 else 1) / 2 for x in range(4)},
    )


def test_source_numeric_bit_indices_are_mapped_per_register() -> None:
    p = Registers()
    x = p.declare("x", 3)
    p.emit("qreg.allocate", x)
    p.emit("qreg.gate", x, gate="x", index=0)
    p.emit("qreg.gate", x, gate="h", index=1)
    p.emit("qreg.gate", x, gate="s", index=1)
    _check(
        select_register_program(p.finish()),
        {(1,): 1 / math.sqrt(2), (3,): 1j / math.sqrt(2)},
    )


def test_input_mutation_after_compute_does_not_corrupt_inverse_dependencies() -> None:
    p = Registers()
    x, first, second = p.declare("x", 1), p.declare("first", 2), p.declare("second", 2)
    graph = capture(double, {"x": 1})
    p.emit("qreg.allocate", x)
    p.emit("qreg.gate", x, gate="h", index=0)
    p.emit("qreg.compute", first, x, callee=graph.name)
    p.emit("qreg.gate", x, gate="x", index=0)
    p.emit("qreg.compute", second, x, callee=graph.name)
    selection = select_register_program(p.finish(), (graph,))
    assert selection.selected is not None and selection.selected.arithmetic is not None
    assert len(selection.selected.arithmetic.segments) == 2
    assert selection.selected.arithmetic.peak_workspace == max(
        segment.peak_workspace for segment in selection.selected.arithmetic.segments
    )
    _check(selection, {(1, 0, 2): 1 / math.sqrt(2), (0, 2, 0): 1 / math.sqrt(2)})


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_whole_program_target_pipeline_preserves_phases_and_named_weights(
    routed: bool,
    objective: Literal["width", "depth"],
) -> None:
    program, functions = pressure_program()
    target = Target(
        basis=BASIS if routed else None,
        couplings=tuple(CouplingMap.line(n) for n in range(4, 21)) if routed else (),
        optimize=not routed,
        restore_layout=not routed,
    )
    selection = select_register_program(
        program,
        functions,
        target=target,
        objective=objective,
        budget=ConstructionBudget(qubits=20),
    )
    if routed:
        assert selection.selected is not None and selection.selected.target is not None
        assert any(
            op.name == "swap"
            for op in selection.selected.target.executable.instructions
        )
        assert selection.selected.target.layout.logical_to_physical != tuple(
            range(selection.selected.target.executable.n_wires)
        )
    _check(
        selection,
        {
            (x, y, pressure(x, y)): (1j if x else 1) / 2
            for x in range(2)
            for y in range(2)
        },
    )
