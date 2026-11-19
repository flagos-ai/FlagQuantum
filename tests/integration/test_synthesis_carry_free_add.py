"""Range-proven addition preserves arbitrary XOR targets and coherent cleanup."""

from dataclasses import replace
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from tests.integration.test_construction_addition_execution import _encode
from tests.synthesis_workflow_helpers import check_state
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.integration


def wide(x: int) -> int:
    return x + 256


def shared(x: int) -> int:
    raised = x + 2
    doubled = raised + raised
    return doubled + raised


@pytest.mark.parametrize("order", [False, True])
@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_every_input_and_nonzero_xor_target_on_permuted_bindings(
    order: bool,
    strategy: Literal["retain", "recompute"],
) -> None:
    expression = "4+x" if order else "x+4"
    function = _cell(
        f"def main(x: int) -> int:\n    return {expression}\n",
        filename=f"<carry-free-{order}>",
    )
    graph = capture(function, {"x": 2})
    arg = graph.body.blocks[0].arguments[0]
    row = schedule_arithmetic(
        (graph,),
        {arg.id: (5, 1)},
        (7, 0, 3),
        entry=graph.name,
        n_qubits=8,
        workspace=(2, 4, 6),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    assert row.peak_workspace == 0
    assert row.conservative is not None and row.conservative.peak_workspace == 3
    assert all(g.name in {"x", "cx"} for g in row.circuit.instructions)
    for x in range(4):
        for z in range(8):
            initial = _encode(x, (5, 1), 8) | _encode(z, (7, 0, 3), 8)
            expected = _encode(x, (5, 1), 8) | _encode(z ^ (x + 4), (7, 0, 3), 8)
            circuit = fq.Circuit(8, dtype=torch.complex128)
            for wire in range(8):
                if initial & (1 << (7 - wire)):
                    circuit.x(wire)
            circuit.compose(row.circuit)
            actual = fq.run(circuit).statevector().reshape(-1)
            target = torch.zeros_like(actual)
            target[expected] = 1
            torch.testing.assert_close(actual, target, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_shared_dependencies_and_rematerialization_keep_complex_xor_state(
    strategy: Literal["retain", "recompute"],
) -> None:
    graph = capture(shared, {"x": 1})
    arg = graph.body.blocks[0].arguments[0]
    row = schedule_arithmetic(
        (graph,),
        {arg.id: (0,)},
        (1, 2, 3, 4),
        entry=graph.name,
        n_qubits=48,
        workspace=tuple(range(6, 48)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    n = max(max(g.wires) for g in row.circuit.instructions) + 1
    assert n <= 20
    ir = replace(row.circuit, n_wires=n)
    generator = torch.Generator().manual_seed(418)
    amplitudes = torch.randn(64, generator=generator, dtype=torch.complex128)
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    i = 0
    for x in range(2):
        for z in range(16):
            for ref in range(2):
                base = _encode(x, (0,), n) | _encode(ref, (5,), n)
                initial[base | _encode(z, (1, 2, 3, 4), n)] = amplitudes[i]
                expected[base | _encode(z ^ (3 * (x + 2)), (1, 2, 3, 4), n)] = (
                    amplitudes[i]
                )
                i += 1
    actual = (
        fq.Circuit.from_ir(ir, dtype=torch.complex128, inputs=initial)
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert (row.recomputations > 0) == (strategy == "recompute")


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("producer", ["python", "native"])
def test_previously_eight_gibibyte_example_executes_under_sixty_four_mib(
    objective: Literal["width", "depth"],
    producer: str,
) -> None:
    if producer == "python":
        p = synth.synthesize(
            wide,
            inputs={"x": synth.UInt(1)},
            input_state=synth.InputState.uniform(),
            objective=objective,
            memory_limit_bytes=64 * 1024**2,
        )
    else:
        p = synth.synthesize(
            "qfunc main(output x:qbit,output result:qnum) {allocate(x);H(x);result=x+256;}",
            source_format="qmod",
            objective=objective,
            memory_limit_bytes=64 * 1024**2,
        )
    check_state(p, fq.run(p), {(x, x + 256): 2**-0.5 for x in range(2)})
    assert p.report["resources"].n_qubits == 10
    assert p.execution.state_bytes == 16 * 1024
    assert p.report["workspace_qubits"] == 0


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_nested_cleanup_keeps_reference_and_exact_relative_sign(
    objective: Literal["width", "depth"],
) -> None:
    source = """qfunc main(output x:qnum<2>,output ref:qnum<2>) {
      allocate(x);hadamard_transform(x);phase(x<2,pi);ref=x;t:qnum;
      within {t=x+4;} apply {phase(t<6,pi);}
    }"""
    p = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(p, fq.run(p), {(x, x): 0.5 for x in range(4)})


@pytest.mark.parametrize(
    "constant,carry_free",
    [(0, False), (1, False), (3, False), (4, True), (6, False), (8, True)],
)
def test_proof_uses_full_domain_not_basis_initialization(
    constant: int, carry_free: bool
) -> None:
    function = _cell(
        f"def main(x: int) -> int:\n    return x+{constant}\n",
        filename=f"<carry-proof-{constant}>",
    )
    p = synth.synthesize(
        function,
        inputs={"x": synth.UInt(2)},
        input_state=synth.InputState.basis(x=0),
        memory_limit_bytes=64 * 1024**2,
    )
    gates = p.circuit.instructions
    assert any(g.name == "ccx" for g in gates) == (not carry_free and constant != 0)
    if constant == 0:
        assert len(p.registers["result"]) == 2
        selected = p.selection.selected
        assert selected is not None and selected.arithmetic is not None
        assert selected.arithmetic.baseline is not None
        assert any(
            g.name == "ccx" for g in selected.arithmetic.baseline.circuit.instructions
        )
        assert selected.arithmetic.peak_workspace == 0
