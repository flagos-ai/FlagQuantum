"""Whole-program coherence and mutation boundaries for arithmetic normalization."""

from dataclasses import replace
from itertools import product
from typing import Literal

import pytest

import flagquantum as fq
from examples.synthesis.tutorials.arithmetic.normalization.main import QMOD, main
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import (
    _logical_cost,
    schedule_arithmetic,
)
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.usefixtures("legacy_register_selection")
def test_unified_program_and_qmod_share_and_keep_all_complex_amplitudes(
    objective: Literal["width", "depth"],
) -> None:
    expected = {(x, y, (x + y) ** 2): 0.5 for x in range(2) for y in range(2)}
    for definition in (main, QMOD):
        program = (
            synth.synthesize(definition, source_format="qmod", objective=objective)
            if isinstance(definition, str)
            else synth.synthesize(definition, objective=objective)
        )
        assert len(program.registers["result"]) == 3
        check_state(program, fq.run(program), expected)
        selected = program.selection.selected
        assert selected is not None and selected.arithmetic is not None
        segment = selected.arithmetic.segments[0]
        assert segment.normalization is not None
        assert len(segment.normalization.squares) == 1
        assert (
            sum(
                op.name == "qint.add"
                for op in segment.normalization.program.body.blocks[0].operations
            )
            == 1
        )
        if segment.baseline is not None:
            assert all(
                a <= b
                for a, b in zip(
                    _logical_cost(segment), _logical_cost(segment.baseline), strict=True
                )
            )


@pytest.mark.parametrize("x,y", list(product(range(2), repeat=2)))
def test_all_basis_preparations_keep_complete_domain_and_clean_workspace(
    x: int, y: int
) -> None:
    source = QMOD.replace(
        "allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);",
        f"x={x};y={y};",
    )
    program = synth.synthesize(source, source_format="qmod")
    assert len(program.registers["result"]) == 3
    check_state(program, fq.run(program), {(x, y, (x + y) ** 2): 1})


def test_repeated_value_is_not_shared_across_a_quantum_write() -> None:
    source = """qfunc copy(const a:qbit,output b:qnum) { b=a+0; }
qfunc main(output x:qbit,output first:qnum,output second:qnum) {
 allocate(x);H(x);copy(x,first);X(x);copy(x,second);
}"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(program, fq.run(program), {(0, 1, 0): 2**-0.5, (1, 0, 1): 2**-0.5})
    selected = program.selection.selected
    assert selected is not None and selected.arithmetic is not None
    assert len(selected.arithmetic.segments) == 2
    assert (
        selected.arithmetic.segments[0].normalization
        is not selected.arithmetic.segments[1].normalization
    )


def repeated(x: int, y: int) -> int:
    return (x + y) * (y + x)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_exact_xor_for_every_input_and_target_on_the_normalized_graph(strategy) -> None:
    from benchmarks.synthesis.verification import classical_apply, encode

    graph = capture(repeated, {"x": 2, "y": 2})
    block = graph.body.blocks[0]
    width = block.operations[-1].operands[0].type.parameters[0]
    wires = ((0, 1), (2, 3), tuple(range(4, 4 + width)))
    candidate = schedule_arithmetic(
        (graph,),
        {arg.id: wires[i] for i, arg in enumerate(block.arguments)},
        wires[-1],
        entry=graph.name,
        n_qubits=64,
        workspace=tuple(range(4 + width, 64)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    for x, y, target in product(range(4), range(4), range(1 << width)):
        assert classical_apply(
            candidate.circuit, encode((x, y, target), wires)
        ) == encode((x, y, target ^ ((x + y) ** 2)), wires)


def test_rewrite_keeps_input_relative_phase_and_output_interference() -> None:
    from benchmarks.synthesis.arithmetic_baseline import synthesize_case
    from benchmarks.synthesis.expressions import EXPRESSIONS, Case
    from benchmarks.synthesis.verification import coherent_check

    case = Case(EXPRESSIONS[6], (1, 1))
    program = synthesize_case(case, phased=True)
    coherent_check(case, program, phased=True)
    source = QMOD.replace(
        "energy(x,y,result);", "energy(x,y,result);hadamard_transform(result);"
    )
    program = synth.synthesize(source, source_format="qmod")
    expected = {
        (x, y, z): (-1) ** ((z & ((x + y) ** 2)).bit_count()) / (2 * 2**1.5)
        for x, y, z in product(range(2), range(2), range(8))
    }
    check_state(program, fq.run(program), expected)


def test_zero_rewrite_does_not_hide_an_uninitialized_quantum_read() -> None:
    from flagquantum.ecosystem.contracts import InteropConversionError

    source = """qfunc zero(const a:qbit,output b:qnum) {b=a*0;}
qfunc main(output x:qbit,output result:qnum) {zero(x,result);}"""
    with pytest.raises(InteropConversionError, match="uninitialized"):
        synth.synthesize(source, source_format="qmod")


def test_independent_circuit_reference_matches_all_targets_and_complete_state() -> None:
    import torch

    from benchmarks.synthesis.verification import classical_apply, encode
    from examples.synthesis.tutorials.arithmetic.normalization.main import (
        circuit_reference,
    )

    reference = circuit_reference().to_ir()
    permutation = replace(
        reference,
        instructions=tuple(g for g in reference.instructions if g.name != "h"),
    )
    wires = ((0,), (1,), (2, 3, 4))
    for x, y, target in product(range(2), range(2), range(8)):
        assert classical_apply(permutation, encode((x, y, target), wires)) == encode(
            (x, y, target ^ ((x + y) ** 2)), wires
        )
    state = fq.run(reference).statevector().reshape(2, 2, 8)
    expected = torch.zeros_like(state)
    for x, y in product(range(2), repeat=2):
        expected[x, y, (x + y) ** 2] = 0.5
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_wide_declared_return_preserves_all_256_nonzero_xor_targets() -> None:
    from benchmarks.synthesis.verification import classical_apply, encode
    from flagquantum.compiler._construction.profile import uint_type
    from flagquantum.compiler._hybrid.model import (
        INDEX,
        Block,
        HybridProgram,
        Operation,
        Region,
        Value,
        ValueId,
    )

    x = Value(ValueId("wide", 0), uint_type(1))
    zero = Value(ValueId("wide", 1), INDEX)
    result = Value(ValueId("wide", 2), uint_type(8))
    graph = HybridProgram(
        "wide",
        Region(
            (
                Block(
                    (x,),
                    (
                        Operation(
                            "arith.constant", results=(zero,), attributes={"value": 0}
                        ),
                        Operation("qint.add", (x, zero), (result,)),
                        Operation("qint.return", (result,)),
                    ),
                ),
            )
        ),
    )
    outputs = tuple(range(1, 9))
    candidate = schedule_arithmetic(
        (graph,),
        {x.id: (0,)},
        outputs,
        entry="wide",
        n_qubits=16,
        workspace=tuple(range(9, 16)),
        workspace_state="fresh_zero",
    )
    assert candidate.lifetimes.program.body.blocks[0].operations[-1].operands[
        0
    ].type == uint_type(8)
    for value, target in product(range(2), range(256)):
        assert classical_apply(
            candidate.circuit, encode((value, target), ((0,), outputs))
        ) == encode((value, target ^ value), ((0,), outputs))
