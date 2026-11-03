"""Shifted copies preserve exact multiplication and coherent inverse cleanup."""

from dataclasses import replace
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_construction_addition_execution import _encode
from tests.integration.test_synthesis_register_workflow import check_state
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.integration


def scaled(x: int) -> int:
    return x * 256


def shared(x: int) -> int:
    raised = x * 2
    doubled = raised + raised
    return doubled + raised


@pytest.mark.parametrize("order", [False, True])
@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_every_input_and_arbitrary_xor_output_on_permuted_bindings(
    order: bool, strategy: Literal["retain", "recompute"]
) -> None:
    expression = "4*x" if order else "x*4"
    graph = capture(
        _cell(f"def main(x: int) -> int:\n    return {expression}\n"), {"x": 2}
    )
    arg = graph.body.blocks[0].arguments[0]
    row = schedule_arithmetic(
        (graph,),
        {arg.id: (8, 1)},
        (6, 0, 9, 3),
        entry=graph.name,
        n_qubits=10,
        workspace=(2, 7, 4, 5),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    assert row.peak_workspace == 4
    assert all(not e.scratch for e in row.events)
    assert all(g.name == "cx" for g in row.circuit.instructions)
    for x in range(4):
        for z in range(16):
            initial = _encode(x, (8, 1), 10) | _encode(z, (6, 0, 9, 3), 10)
            expected = _encode(x, (8, 1), 10) | _encode(z ^ (4 * x), (6, 0, 9, 3), 10)
            circuit = fq.Circuit(10, dtype=torch.complex128)
            for wire in range(10):
                if initial & (1 << (9 - wire)):
                    circuit.x(wire)
            actual = fq.run(circuit.compose(row.circuit)).statevector().reshape(-1)
            target = torch.zeros_like(actual)
            target[expected] = 1
            torch.testing.assert_close(actual, target, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_shared_rematerialized_dependencies_preserve_complex_reference_state(
    strategy: Literal["retain", "recompute"],
) -> None:
    graph = capture(shared, {"x": 1})
    arg = graph.body.blocks[0].arguments[0]
    row = schedule_arithmetic(
        (graph,),
        {arg.id: (0,)},
        (1, 2, 3),
        entry=graph.name,
        n_qubits=48,
        workspace=tuple(range(5, 48)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    n = max(max(g.wires) for g in row.circuit.instructions) + 1
    assert n <= 20
    amplitudes = torch.randn(
        32, generator=torch.Generator().manual_seed(421), dtype=torch.complex128
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    i = 0
    for x in range(2):
        for z in range(8):
            for ref in range(2):
                base = _encode(x, (0,), n) | _encode(ref, (4,), n)
                initial[base | _encode(z, (1, 2, 3), n)] = amplitudes[i]
                expected[base | _encode(z ^ (6 * x), (1, 2, 3), n)] = amplitudes[i]
                i += 1
    actual = (
        fq.Circuit.from_ir(
            replace(row.circuit, n_wires=n), dtype=torch.complex128, inputs=initial
        )
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert (row.recomputations > 0) == (strategy == "recompute")


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("producer", ["python", "native"])
@pytest.mark.usefixtures("legacy_register_selection")
def test_wide_constant_executes_under_existing_public_memory_limit(
    objective: Literal["width", "depth"], producer: str
) -> None:
    if producer == "python":
        p = synth.synthesize(
            scaled,
            inputs={"x": synth.UInt(1)},
            input_state=synth.InputState.uniform(),
            objective=objective,
            memory_limit_bytes=64 * 1024**2,
        )
    else:
        p = synth.synthesize(
            "qfunc main(output x:qbit,output result:qnum) {allocate(x);H(x);result=x*256;}",
            source_format="qmod",
            objective=objective,
            memory_limit_bytes=64 * 1024**2,
        )
    check_state(p, fq.run(p), {(x, 256 * x): 2**-0.5 for x in range(2)})
    assert p.report["resources"].n_qubits == 19
    assert p.execution.state_bytes == 8 * 1024**2
    assert p.report["workspace_qubits"] == 9


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_nested_cleanup_restores_reference_and_exact_sign(
    objective: Literal["width", "depth"],
) -> None:
    p = synth.synthesize(
        """qfunc main(output x:qnum<2>,output ref:qnum<2>) {
          allocate(x);hadamard_transform(x);phase(x<2,pi);ref=x;t:qnum;
          within {t=x*4;} apply {phase(t<8,pi);}
        }""",
        source_format="qmod",
        objective=objective,
    )
    check_state(p, fq.run(p), {(x, x): 0.5 for x in range(4)})


@pytest.mark.parametrize(
    "constant,shifted",
    [
        (0, False),
        (1, False),
        (2, True),
        (3, False),
        (4, True),
        (6, False),
        (8, True),
        (12, False),
        (256, True),
    ],
)
def test_constant_proof_controls_actual_scratch_reservation(
    constant: int, shifted: bool
) -> None:
    graph = capture(
        _cell(f"def main(x: int) -> int:\n    return x*{constant}\n"), {"x": 2}
    )
    arg = graph.body.blocks[0].arguments[0]
    width = max(1, (3 * constant).bit_length())
    row = schedule_arithmetic(
        (graph,),
        {arg.id: (0, 1)},
        tuple(range(2, 2 + width)),
        entry=graph.name,
        n_qubits=64,
        workspace=tuple(range(2 + width, 64)),
        workspace_state="fresh_zero",
    )
    original = row.baseline or row
    assert len(original.events) == 2
    assert all(bool(e.scratch) == (not shifted) for e in original.events)
    assert original.peak_workspace == width * (1 if shifted else 2)
    if constant == 1:
        assert row.events == () and row.peak_workspace == 0
        assert [gate.name for gate in row.circuit.instructions] == ["cx", "cx"]
    else:
        assert len(row.events) == 2
        assert all(bool(e.scratch) == (not shifted) for e in row.events)
        assert row.peak_workspace == width * (1 if shifted else 2)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_derived_zero_keeps_generic_kernel_without_truncation(
    strategy: Literal["retain", "recompute"],
) -> None:
    graph = capture(_cell("def main(x: int) -> int:\n    return (x*0)*4\n"), {"x": 1})
    arg = graph.body.blocks[0].arguments[0]
    row = schedule_arithmetic(
        (graph,),
        {arg.id: (0,)},
        (1,),
        entry=graph.name,
        n_qubits=12,
        workspace=tuple(range(2, 12)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    assert row.events and all(e.scratch for e in row.events)
    circuit = fq.Circuit(12, dtype=torch.complex128)
    circuit.h(0)
    actual = fq.run(circuit.compose(row.circuit)).statevector().reshape(-1)
    expected = torch.zeros_like(actual)
    expected[0] = expected[1 << 11] = 2**-0.5
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("expression", ["x*x", "x*3"])
def test_quantum_and_non_power_operands_do_not_specialize_from_basis_zero(
    expression: str,
) -> None:
    p = synth.synthesize(
        _cell(f"def main(x: int) -> int:\n    return {expression}\n"),
        inputs={"x": synth.UInt(2)},
        input_state=synth.InputState.basis(x=0),
        memory_limit_bytes=64 * 1024**2,
    )
    assert any(g.name == "ccx" for g in p.circuit.instructions)
    check_state(p, fq.run(p), {(0, 0): 1})


def test_memory_and_width_guards_remain_enforced() -> None:
    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(
            scaled, inputs={"x": synth.UInt(1)}, memory_limit_bytes=1024**2
        )
    with pytest.raises(HybridVerificationError, match="256-bit safety cap"):
        synth.synthesize(scaled, inputs={"x": synth.UInt(256)})
