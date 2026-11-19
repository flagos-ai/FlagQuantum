"""Routing witnesses survive exact SWAP decomposition without accepting tampering."""

from dataclasses import replace

import pytest
import torch

from flagquantum.compiler._construction.scheduling.target import Target, compile_target
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.layout import _routing_swap_phases, remove_layout_restore
from flagquantum.compiler.pipeline import compile
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit


def marked(wires: tuple[int, int], phase: str = "forward") -> Instruction:
    return Instruction(
        "swap", wires, metadata={"routing_phase": phase, "source_instruction_index": 1}
    )


def test_disjoint_interleaved_cx_triples_are_one_witness_each() -> None:
    a = convert_basis(
        CircuitIR(4, (marked((0, 1)),)), gates=("cx",)
    ).program.instructions
    b = convert_basis(
        CircuitIR(4, (marked((3, 2)),)), gates=("cx",)
    ).program.instructions
    program = CircuitIR(4, tuple(g for pair in zip(a, b, strict=True) for g in pair))
    assert _routing_swap_phases(program) == [
        (0, "forward", (0, 1)),
        (1, "forward", (3, 2)),
    ]
    assert compile(program).instructions == program.instructions


@pytest.mark.parametrize(
    "kind", ("missing", "direction", "source", "phase", "overlap", "interruption")
)
def test_incomplete_reordered_and_spoofed_decompositions_refuse(kind: str) -> None:
    gates = list(
        convert_basis(
            CircuitIR(3, (marked((0, 1)),)), gates=("cx",)
        ).program.instructions
    )
    if kind == "missing":
        gates.pop()
    elif kind == "direction":
        gates[1] = replace(gates[1], wires=(0, 1))
    elif kind in {"source", "phase"}:
        key = "source_instruction_index" if kind == "source" else "routing_phase"
        gates[1] = replace(gates[1], metadata=dict(gates[1].metadata) | {key: 9})
    elif kind == "overlap":
        gates.insert(1, marked((1, 2)))
    else:
        gates.insert(1, Instruction("h", (0,)))
    with pytest.raises(ValueError, match="SWAP evidence"):
        _routing_swap_phases(CircuitIR(3, tuple(gates)))


def test_actual_routed_cx_u_program_preserves_absolute_unitary_and_cleanup() -> None:
    source = CircuitIR(
        3,
        (Instruction("h", (0,)), Instruction("t", (2,)), Instruction("cx", (0, 2))),
        dtype="complex128",
    )
    product = compile_target(
        source,
        Target(basis=("cx", "u3"), couplings=(CouplingMap(3, ((0, 1), (1, 2))),)),
        10000,
    )
    assert _routing_swap_phases(product.executable)
    assert {g.name for g in product.executable.instructions} <= {"cx", "u3"}
    torch.testing.assert_close(
        get_unitary(product.executable), get_unitary(source), atol=1e-12, rtol=0
    )
    # Removal is performed before ASAP scheduling, when restore is a suffix.
    reduced = remove_layout_restore(product.submitted)
    assert all(
        phase != "final_restore" for _, phase, _ in _routing_swap_phases(reduced)
    )


def test_decomposed_restore_counts_swaps_instead_of_cx_leaves() -> None:
    source = CircuitIR(3, (Instruction("cx", (0, 2)),))
    routed = compile(
        source,
        coupling_map=CouplingMap(3, ((0, 1), (1, 2))),
        routing_strategy="persistent_layout",
        optimize=False,
    )
    converted = convert_basis(routed, gates=("cx",)).program
    restore_count = sum(
        phase == "final_restore" for _, phase, _ in _routing_swap_phases(converted)
    )
    assert restore_count > 0
    reduced = remove_layout_restore(converted)
    assert (
        reduced.metadata["routing"]["removed_layout_restore_swap_count"]
        == restore_count
    )
    assert len(converted.instructions) - len(reduced.instructions) == 3 * restore_count
