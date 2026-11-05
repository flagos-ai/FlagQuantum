"""Original count-tree gates, coherent logical output and every erasure branch."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.hamming_weight import (
    hamming_weight_ir,
    hamming_weight_requirements,
)
from flagquantum.core.ir import Instruction
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    j, o = hamming_weight_requirements(n)
    return (
        tuple(range(n)),
        tuple(range(n, n + j)),
        tuple(range(n + j, n + j + o)),
        tuple(range(40, 40 + j)),
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 16])
def test_source_compute_and_measured_adjoint_stream_and_cost(n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QUInt, Signature
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    x, junk, out, bits = layout(n)
    source = HammingWeightCompute(n)
    compute = hamming_weight_ir(x, junk, out)
    erase = hamming_weight_ir(x, junk, out, adjoint=True, classical_bits=bits)
    # Direct source generator keeps explicit logical RIGHT outputs and their bit order.
    expected_forward = reference_arithmetic_gates(
        source, {"x": x, "junk": junk, "out": out}, (), {}, cirq_style=True
    )
    assert_source(compute, expected_forward)
    bb, soqs = BloqBuilder.from_signature(Signature.build_from_dtypes(x=QUInt(n)))
    value, garbage, count = bb.add(source, **soqs)
    value = bb.add(source.adjoint(), x=value, junk=garbage, out=count)
    pair = bb.finalize(x=value)
    if n == 1:
        # Pinned Bloq adjoint conversion cannot bind its empty QAny junk.
        with pytest.raises(KeyError):
            source.adjoint().decompose_bloq()
        assert_source(erase, expected_forward)
        return
    expected_pair = reference_arithmetic_gates(
        pair,
        {"x": x},
        (*junk, *out),
        dict(zip(junk, bits, strict=True)),
        cirq_style=True,
    )
    assert_source((*compute, *erase), expected_pair)
    model_forward = get_cost_value(source, QECGatesCost())
    model_pair = get_cost_value(pair, QECGatesCost())
    counts = Counter(g.name for g in (*compute, *erase))
    assert (
        counts["t"] + counts["tdg"]
        == model_forward.total_t_count()
        == model_pair.total_t_count()
        == 4 * len(junk)
    )
    assert counts["measure"] == model_pair.measurement == len(junk)
    assert model_forward.measurement == 0
    assert counts["cx"] == 16 * len(junk) + 2 * n.bit_count()


def logical_output(n: int, value: int) -> int:
    """Independent full-adder truth oracle, including overwritten input/junk."""
    x, junk, out, _ = layout(n)
    wires = [(value >> (n - 1 - i)) & 1 for i in range(n)] + [0] * (
        len(junk) + len(out)
    )
    layer, available = list(x[::-1]), list(junk)  # Source pop order is ascending junk.

    def compress(a: int, b: int, c: int) -> int:
        carry = available.pop(0)
        aa, bb, cc = wires[a], wires[b], wires[c]
        wires[c] = aa ^ bb ^ cc
        wires[carry] = (aa & bb) | (aa & cc) | (bb & cc)
        return carry

    for target in reversed(out):
        higher = [
            compress(layer[i], layer[i + 1], layer[i + 2])
            for i in range(0, len(layer) - 2, 2)
        ]
        if len(layer) % 2:
            wires[target] ^= wires[layer[-1]]
        else:
            higher.append(compress(layer[-2], layer[-1], target))
        layer = higher
    packed = 0
    for bit in wires:
        packed = (packed << 1) | bit
    assert packed & ((1 << len(out)) - 1) == value.bit_count()
    return packed


def forced_branch(
    state: torch.Tensor, gates: tuple[Instruction, ...], outcomes: tuple[int, ...]
) -> torch.Tensor:
    """Postselect actual primitive IR, retaining unnormalized branch amplitudes."""
    width = state.numel().bit_length() - 1
    data = torch.arange(state.numel(), dtype=torch.int64)
    classical: dict[int, int] = {}
    pending: list[Instruction] = []
    measurements = iter(outcomes)

    def flush() -> None:
        nonlocal state
        if pending:
            state = (
                Circuit(width, inputs=state, dtype=torch.complex128)
                .append(FieldOperation(width, tuple(pending)))
                .state()
                .flatten()
            )
            pending.clear()

    for gate in gates:
        if gate.name == "measure":
            flush()
            outcome = next(measurements)
            classical[int(gate.metadata["classical_bit"])] = outcome
            mask = (data >> (width - 1 - gate.wires[0])) & 1
            state = state * (mask == outcome)
        elif gate.name == "reset":
            flush()
            mask = 1 << (width - 1 - gate.wires[0])
            # Reset follows a projective measurement; sum the disjoint sectors.
            state = torch.where((data & mask) == 0, state + state[data ^ mask], 0)
        elif all(
            classical[int(bit)] == value
            for bit, value in gate.metadata.get("conditions", ())
        ):
            pending.append(Instruction(gate.name, gate.wires, gate.params))
    flush()
    return state


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_full_complex_count_and_all_measured_cleanup_branches(n: int) -> None:
    x, junk, out, bits = layout(n)
    work = len(junk) + len(out)
    coeff = torch.arange(1 << n, dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << (n + work), dtype=torch.complex128)
    state[:: 1 << work] = amps
    expected = torch.zeros_like(state)
    expected[torch.tensor([logical_output(n, value) for value in range(1 << n)])] = amps
    compute = hamming_weight_ir(x, junk, out)
    forward = (
        Circuit(n + work, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(n + work, compute))
        .state()
        .flatten()
    )
    torch.testing.assert_close(forward, expected, rtol=0, atol=1e-12)
    erase = hamming_weight_ir(x, junk, out, adjoint=True, classical_bits=bits)
    # Exhaustive, unnormalized branches prove phases, cleanup and branch probabilities.
    for outcomes in product((0, 1), repeat=len(junk)):
        restored = forced_branch(forward, erase, outcomes)
        torch.testing.assert_close(
            restored, state / 2 ** (len(junk) / 2), rtol=0, atol=1e-12
        )


def test_hamming_validation() -> None:
    for width in (0, -1, True):
        with pytest.raises(ValueError, match="positive"):
            hamming_weight_requirements(width)
    x, j, o, b = layout(3)
    with pytest.raises(ValueError, match="boolean"):
        hamming_weight_ir(x, j, o, adjoint=1)
    with pytest.raises(ValueError, match="junk"):
        hamming_weight_ir(x, (), o)
    with pytest.raises(ValueError, match="distinct"):
        hamming_weight_ir(x, j, (0, 8))
    for bits in ((), (-1,), (True,)):
        with pytest.raises(ValueError, match="classical"):
            hamming_weight_ir(x, j, o, adjoint=True, classical_bits=bits)


def test_user_append_example() -> None:
    from examples.algorithms.register_hamming_weight import main

    main()
