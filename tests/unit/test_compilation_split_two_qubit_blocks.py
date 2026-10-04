"""Unit contract for the local two-qubit split W9-23 adds.

`split_two_qubit_blocks` is the inverse of the two-qubit run fold: one two-wire
instruction carrying a matrix that is a Kronecker product becomes the two
single-qubit instructions that product names. It is the counterpart of Qiskit's
``Split2QUnitaries``, with one difference this file exists to pin: Qiskit's pass
sets ``new_dag.global_phase`` on the branch it acts on, and this one must not need
to, because its two factors are read off the product's own blocks rather than
normalized.

That is the whole exactness claim, and it is asserted on raw statevectors rather
than on an overlap magnitude. A phase-blind fold of the same input -- the one
`two_qubit_synthesis` produces, and the one `legalize_native_gates` took before
this pass existed -- has an overlap magnitude of 1 and a raw difference of 0.75,
so an overlap assertion would accept it. Nothing here is divided out.
"""

import math
import random

import pytest
import torch

from flagquantum.compiler.one_qubit_optimization import collapse_one_qubit_runs
from flagquantum.compiler.pipeline import optimize
from flagquantum.compiler.two_qubit_optimization import (
    _MATCH_EPS,
    _product_factors,
    split_two_qubit_blocks,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import get_operator_schema
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: The declared single-qubit opcodes the population's factors are drawn from, so
#: every case is a product the schema table can already name rather than a random
#: matrix. `i` is in the list on purpose: it makes a factor-free case a plain
#: identity rather than a special case in the pass.
FACTORS: tuple[str, ...] = (
    "i",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "sx",
    "sxdg",
    "t",
    "tdg",
)


def _one_qubit_matrix(opcode: str, angle: float = 0.6) -> list[list[complex]]:
    schema = get_operator_schema(opcode)
    assert schema is not None
    params = dict.fromkeys(schema.parameters, angle)
    values = gate_matrix(
        Instruction(opcode, (0,), params=params),
        bsz=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    ).reshape(2, 2)
    return [[complex(values[row][column]) for column in range(2)] for row in range(2)]


def _product(
    left: list[list[complex]], right: list[list[complex]]
) -> list[list[complex]]:
    """``left (x) right`` with `left` on wire 0, which is the executor's convention."""

    return [
        [
            left[row // 2][column // 2] * right[row % 2][column % 2]
            for column in range(4)
        ]
        for row in range(4)
    ]


def _state(program: CircuitIR) -> torch.Tensor:
    from flagquantum.simulation.statevector.local import run_local_statevector

    return run_local_statevector(
        program,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    ).reshape(-1)


def _random_case(index: int) -> tuple[str, str, list[list[complex]]]:
    rng = random.Random(20261024 + index)
    left_opcode = rng.choice(FACTORS)
    right_opcode = rng.choice(FACTORS)
    return (
        left_opcode,
        right_opcode,
        _product(_one_qubit_matrix(left_opcode), _one_qubit_matrix(right_opcode)),
    )


def _haar_unitary(seed: int) -> list[list[complex]]:
    """A seeded SU(4) from a complex Gram-Schmidt, in pure Python."""

    rng = random.Random(seed)
    rows = [
        [complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(4)]
        for _ in range(4)
    ]
    columns: list[list[list[complex]]] = []
    for index in range(4):
        vector = [rows[row][index] for row in range(4)]
        for previous in columns:
            overlap = sum(previous[row].conjugate() * vector[row] for row in range(4))
            vector = [vector[row] - overlap * previous[row] for row in range(4)]
        norm = math.sqrt(sum(abs(item) ** 2 for item in vector))
        columns.append([item / norm for item in vector])
    return [[columns[column][row] for column in range(4)] for row in range(4)]


def test_the_split_reproduces_the_source_statevector_exactly() -> None:
    """The pass's one claim, on raw amplitudes, over a seeded population.

    Each source is one two-wire instruction carrying a product, placed after a
    Hadamard so that the comparison is on an entangled state rather than on a
    basis vector. The split has to leave the amplitudes unchanged -- not unchanged
    up to a phase -- and the largest raw difference over the whole population is
    asserted to be float noise.
    """

    worst = 0.0
    for index in range(160):
        _, _, matrix = _random_case(index)
        instructions = (
            Instruction("h", (0,)),
            Instruction("blob", (0, 1), matrix=matrix),
        )
        source = CircuitIR(2, instructions, dtype="complex128")
        split = split_two_qubit_blocks(source)
        assert [item.wires for item in split.instructions] == [(0,), (0,), (1,)]
        difference = float(torch.max(torch.abs(_state(source) - _state(split))).item())
        worst = max(worst, difference)
    assert worst < 1e-14, worst

    # Falsifiability control. The tolerance above is only worth something if a
    # phase-blind route of the same input actually misses it, and it does: the KAK
    # route this pass replaces has an overlap magnitude of 1 and a raw difference
    # of order 1, which is why nothing in this file compares overlaps.
    from flagquantum.compiler.two_qubit_synthesis import synthesize_two_qubit

    _, _, matrix = _random_case(0)
    source = CircuitIR(
        2,
        (Instruction("h", (0,)), Instruction("blob", (0, 1), matrix=matrix)),
        dtype="complex128",
    )
    leaves = synthesize_two_qubit(
        matrix, wires=(0, 1), entangler="cx", z_rotation="rz", pulse_opcode="sx"
    )
    assert leaves is not None
    blind = CircuitIR(2, (Instruction("h", (0,)), *leaves), dtype="complex128")
    raw = float(torch.max(torch.abs(_state(source) - _state(blind))).item())
    assert raw > 0.1, raw
    assert abs(float(torch.abs(torch.vdot(_state(source), _state(blind))))) > 0.999


def test_the_emitted_factors_multiply_back_to_the_source_matrix() -> None:
    """Entry for entry, with nothing divided out.

    The statevector above is the runtime's reading; this is the arithmetic the
    pass actually performs, so a convention slip that the runtime happens to
    normalize away would still fail here.
    """

    worst = 0.0
    for index in range(160):
        _, _, matrix = _random_case(index)
        source = CircuitIR(
            2, (Instruction("blob", (0, 1), matrix=matrix),), dtype="complex128"
        )
        split = split_two_qubit_blocks(source)
        left = split.instructions[0].matrix
        right = split.instructions[1].matrix
        assert left is not None and right is not None
        rebuilt = _product(
            [[complex(left[row][column]) for column in range(2)] for row in range(2)],
            [[complex(right[row][column]) for column in range(2)] for row in range(2)],
        )
        worst = max(
            worst,
            max(
                abs(matrix[row][column] - rebuilt[row][column])
                for row in range(4)
                for column in range(4)
            ),
        )
    assert worst < 1e-14, worst


def test_a_non_product_is_refused_rather_than_rewritten() -> None:
    """The fail-closed direction, on the population that must not be accepted.

    A Haar-random SU(4) is entangling and has no factorisation, and every declared
    arity-2 opcode that is not a product of two declared single-qubit gates is in
    the same class. The pass returns the input object itself for both, which is
    how a caller can tell nothing happened.
    """

    for index in range(200):
        matrix = _haar_unitary(20261100 + index)
        assert _product_factors(matrix) is None, index
        source = CircuitIR(
            2, (Instruction("blob", (0, 1), matrix=matrix),), dtype="complex128"
        )
        assert split_two_qubit_blocks(source) is source

    # `swap` and a controlled rotation are exactly the shapes Qiskit's own pass
    # declines as well, and they are declined here by arithmetic rather than by a
    # specialization name.
    for opcode, params in (
        ("swap", {}),
        ("cx", {}),
        ("cz", {}),
        ("rzz", {"theta": 0.4}),
        ("cry", {"theta": 0.4}),
    ):
        values = gate_matrix(
            Instruction(opcode, (0, 1), params=params),
            bsz=1,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        ).reshape(4, 4)
        entries = [[complex(values[row][col]) for col in range(4)] for row in range(4)]
        assert _product_factors(entries) is None, opcode


def test_a_declared_opcode_is_never_a_candidate() -> None:
    """The schema's name wins over any matrix the instruction happens to carry.

    Two one-wire instructions carrying a declared two-qubit name would contradict
    the arity the schema declares, so the pass refuses before it reads the matrix.
    A caller that wants the matrix split writes an undeclared name, which is what
    an adapter produces.
    """

    for opcode in sorted(
        name
        for name in ("cx", "cz", "swap", "rzz", "crx")
        if get_operator_schema(name) is not None
    ):
        source = CircuitIR(2, (Instruction(opcode, (0, 1), params={"theta": 0.4}),))
        assert split_two_qubit_blocks(source) is source, opcode
        values = gate_matrix(
            Instruction(opcode, (0, 1), params={"theta": 0.4}),
            bsz=1,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        ).reshape(4, 4)
        carrying = CircuitIR(
            2,
            (
                Instruction(
                    opcode,
                    (0, 1),
                    params={"theta": 0.4},
                    matrix=values.tolist(),
                ),
            ),
            dtype="complex128",
        )
        assert split_two_qubit_blocks(carrying) is carrying, opcode


def test_a_one_wire_instruction_is_never_a_candidate() -> None:
    source = CircuitIR(
        1, (Instruction("blob", (0,), matrix=[[0, 1], [1, 0]]),), dtype="complex128"
    )
    assert split_two_qubit_blocks(source) is source


def test_a_product_of_identities_is_the_identity_and_still_splits() -> None:
    """The degenerate end, where a naive block detector divides by zero.

    ``i (x) i`` is a product, so it must be accepted and produce two identity
    factors; the zero matrix is not a product and must be refused by the same
    arithmetic that accepts the first case rather than by a special case.
    """

    identity = _one_qubit_matrix("i")
    product = _product(identity, identity)
    factors = _product_factors(product)
    assert factors is not None
    assert max(abs(item) for row in factors[0] for item in row) > 0.5
    source = CircuitIR(
        2, (Instruction("blob", (0, 1), matrix=product),), dtype="complex128"
    )
    split = split_two_qubit_blocks(source)
    assert [item.wires for item in split.instructions] == [(0,), (1,)]
    assert float(torch.max(torch.abs(_state(source) - _state(split))).item()) < 1e-14

    zero = [[0j for _ in range(4)] for _ in range(4)]
    assert _product_factors(zero) is None
    assert _MATCH_EPS > 0.0


def test_the_pipeline_can_read_the_wire_after_the_split_and_not_before() -> None:
    """What the split buys, measured as reach rather than as a shorter program.

    A two-wire matrix instruction spans two wires and is not readable by the
    single-qubit fold, which works one wire at a time, so it blocks every
    simplification around it. The same operator written as its two factors does
    not. Both programs below are three instructions; the difference is that the
    second has had its two ``rz`` gates on wire 0 merged with the factor between
    them, which is exactly the reach the split adds.

    The factor is chosen to be a ``z`` rotation so the merged run is a declared
    opcode the runtime can also execute, which is what makes the statevector
    assertion at the end an independent check of the merge.
    """

    product = _product(_one_qubit_matrix("rz"), _one_qubit_matrix("rz"))
    source = CircuitIR(
        2,
        (
            Instruction("rz", (0,), params={"theta": 0.3}),
            Instruction("blob", (0, 1), matrix=product),
            Instruction("rz", (0,), params={"theta": 0.4}),
        ),
        dtype="complex128",
    )
    # The fold cannot see through the two-wire matrix, so the shipped pipeline
    # leaves all three instructions in place.
    assert [item.name for item in optimize(source).instructions] == [
        "rz",
        "blob",
        "rz",
    ]
    split = split_two_qubit_blocks(source)
    assert [item.wires for item in split.instructions] == [(0,), (0,), (1,), (0,)]
    folded = optimize(split)
    # Wire 0's three-member run is now two declared opcodes and wire 1 keeps its
    # factor, which is the whole of what the split bought: before it, wire 0's run
    # was not readable at all. Three in, three out -- the point is the reach, not
    # the length.
    assert len(folded.instructions) == 3
    assert [item.name for item in folded.instructions] == ["phase", "rz", "blob"]
    assert [item.wires for item in folded.instructions] == [(0,), (0,), (1,)]
    assert [item.matrix is not None for item in folded.instructions] == [
        False,
        False,
        True,
    ]
    # And the merged program still computes the same operator as the source.
    assert float(torch.max(torch.abs(_state(source) - _state(folded))).item()) < 1e-13

    # The reach is the fold's, not the split's: the wire-0 factor of this product
    # is a `z` rotation, so a one-wire run around it collapses where the two-wire
    # instruction alone did not.
    run = CircuitIR(
        1,
        (
            Instruction("blob", (0,), matrix=_one_qubit_matrix("rz")),
            Instruction("blob", (0,), matrix=_one_qubit_matrix("rz")),
        ),
        dtype="complex128",
    )
    assert len(collapse_one_qubit_runs(run).instructions) == 1


def test_the_split_is_idempotent_and_leaves_a_native_program_alone() -> None:
    for index in range(40):
        _, _, matrix = _random_case(index)
        source = CircuitIR(
            2, (Instruction("blob", (0, 1), matrix=matrix),), dtype="complex128"
        )
        once = split_two_qubit_blocks(source)
        assert split_two_qubit_blocks(once) is once

    native = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("rz", (1,), params={"theta": 0.3}),
        ),
        dtype="complex128",
    )
    assert split_two_qubit_blocks(native) is native
