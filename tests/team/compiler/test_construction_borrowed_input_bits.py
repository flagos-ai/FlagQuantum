"""Borrow immutable input bits without allocating copies of numeric casts."""

from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum.compiler._construction.codegen.arithmetic import (
    ArithmeticCandidate,
    Schedule,
    _emit_arithmetic,
)
from flagquantum.compiler._construction.frontend.program import capture_program
from flagquantum.core.ir import CircuitIR
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.unit


def candidate(
    width: int,
    signed_width: int,
    scheduling: Schedule,
    *,
    optimized: bool,
    expression: str = "x+y",
) -> ArithmeticCandidate:
    entry = program_definition(
        "from flagquantum.synth import *\n"
        "@qfunc\n"
        f"def main(x: Output[QNum[{width}]], "
        f"y: Output[QNum[{signed_width},True,1]], result: Output[QNum]):\n"
        f"    allocate(x)\n    allocate(y)\n    result |= {expression}\n"
    )
    root, functions = capture_program(entry)
    graph = functions[0]
    output_width = graph.body.blocks[0].operations[-1].operands[0].type.parameters[0]
    base = width + signed_width
    registers = {"x": tuple(range(width)), "y": tuple(range(width, base))}
    declarations = {
        op.results[0].id: op.attributes["name"]
        for op in root.body.blocks[0].operations
        if op.name == "qreg.declare"
    }
    call = next(
        op for op in root.body.blocks[0].operations if op.name == "qreg.compute"
    )
    bindings = {
        argument.id: registers[declarations[actual.id]]
        for argument, actual in zip(
            graph.body.blocks[0].arguments, call.operands[2:], strict=True
        )
    }
    return _emit_arithmetic(
        functions,
        bindings,
        tuple(range(base, base + output_width)),
        entry=graph.name,
        n_qubits=64,
        workspace=tuple(range(base + output_width, 64)),
        workspace_state="fresh_zero",
        scheduling=scheduling,
        implementation="arithmetic" if optimized else "conservative",
    )


@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
def test_original_workshop_all_inputs_and_nonzero_xor_targets(
    scheduling: Schedule,
) -> None:
    original = candidate(3, 5, scheduling, optimized=False)
    selected = candidate(3, 5, scheduling, optimized=True)
    assert selected.peak_workspace <= original.peak_workspace
    assert len(selected.circuit.instructions) < len(original.circuit.instructions)
    assert selected.peak_workspace == 11
    assert len(selected.circuit.instructions) == 49
    registers = (tuple(range(3)), tuple(range(3, 8)), tuple(range(8, 14)))
    for x, y, target in product(range(8), range(32), range(64)):
        signed = y - 32 if y & 16 else y
        expected = encode((x, y, target ^ ((2 * x + signed) % 64)), registers)
        basis = encode((x, y, target), registers)
        assert classical_apply(selected.circuit, basis) == expected
        assert classical_apply(original.circuit, basis) == expected
        # The inverse restores operands, arbitrary target and every work bit.
        assert classical_apply(selected.circuit, expected) == basis


@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
def test_complex_superposition_preserves_absolute_phase_and_cleans_workspace(
    scheduling: Schedule,
) -> None:
    compiled = candidate(1, 2, scheduling, optimized=True).circuit
    size = max(wire for gate in compiled.instructions for wire in gate.wires) + 1
    registers = ((0,), (1, 2), (3, 4, 5))
    amplitudes = torch.randn(
        64, dtype=torch.complex128, generator=torch.Generator().manual_seed(71)
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << size, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for amplitude, (x, y, target) in zip(
        amplitudes, product(range(2), range(4), range(8)), strict=True
    ):
        answer = target ^ ((2 * x + (y - 4 if y & 2 else y)) % 8)
        source = encode((x, y, target), registers)
        destination = encode((x, y, answer), registers)
        initial[int(f"{source:0{size}b}"[::-1], 2)] = amplitude
        expected[int(f"{destination:0{size}b}"[::-1], 2)] = amplitude
    observed = (
        fq.Circuit.from_ir(
            CircuitIR(size, compiled.instructions),
            inputs=initial,
            dtype=torch.complex128,
        )
        .state()
        .flatten()
    )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
def test_signed_product_keeps_injective_multiplier_operands(
    scheduling: Schedule,
) -> None:
    compiled = candidate(2, 3, scheduling, optimized=True, expression="x*y")
    registers = ((0, 1), (2, 3, 4), (5, 6, 7, 8, 9))
    for x, y, target in product(range(4), range(8), range(32)):
        signed = y - 8 if y & 4 else y
        expected = encode((x, y, target ^ ((x * signed) % 32)), registers)
        assert (
            classical_apply(compiled.circuit, encode((x, y, target), registers))
            == expected
        )
