"""Raw source-chain parity and phase-preserving clean-output inversion."""

from collections import Counter

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.field_inverse import (
    gf2_inverse_decomposition,
    gf2_inverse_ir,
    gf2_inverse_workspace_size,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "width,modulus", [(1, 3), (2, 7), (3, 11), (4, 19), (5, 37), (6, 67), (8, 283)]
)
def test_inverse_upstream_call_chain_and_resource_closure(
    width: int, modulus: int
) -> None:
    pytest.importorskip("qualtran")
    from galois import Poly
    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = tuple(range(width))
    workspace = tuple(range(width, 9 * width))
    upstream = GF2Inverse(QGF(2, width, Poly.Int(modulus)))
    raw = gf2_inverse_decomposition(source, workspace, modulus=modulus)
    expected, outputs = reference_gates(
        upstream, {"x": source}, workspace, repair_adjoint=True
    )
    assert_gate_streams([(g.name, g.wires) for g in raw.instructions], expected)
    assert raw.outputs[:2] == (outputs["x"], outputs["result"])
    assert tuple(w for reg in raw.outputs[2:] for w in reg) == outputs.get("junk", ())
    target = tuple(range(9 * width, 10 * width))
    clean = gf2_inverse_ir(source, target, workspace, modulus=modulus)
    actual_counts = Counter(g.name for g in clean)
    raw_counts = Counter(name for name, _ in expected)
    assert actual_counts == raw_counts + raw_counts + Counter(cx=width)
    model = get_cost_value(upstream, QECGatesCost())
    assert (raw_counts["ccx"], raw_counts["cx"]) == (model.toffoli, model.clifford)


@pytest.mark.parametrize(
    "width,modulus", [(1, 3), (2, 7), (3, 11), (4, 19), (5, 37), (6, 67), (8, 283)]
)
def test_inverse_all_field_values_and_every_workspace_bit(
    width: int, modulus: int
) -> None:
    galois = pytest.importorskip("galois")
    field = (
        galois.GF(2**width, irreducible_poly=galois.Poly.Int(modulus))
        if width > 1
        else galois.GF(2)
    )
    needed = gf2_inverse_workspace_size(width, modulus=modulus)
    source = tuple(range(width))
    target = tuple(range(width, 2 * width))
    work = tuple(range(2 * width, 2 * width + needed))
    gates = gf2_inverse_ir(source, target, work, modulus=modulus)
    with pytest.raises(ValueError, match="Insufficient"):
        gf2_inverse_ir(source, target, work[:-1], modulus=modulus)
    for value in range(1 << width):
        bits = [int(b) for b in f"{value:0{width}b}"] + [0] * (9 * width)
        for gate in gates:
            bits[gate.wires[-1]] ^= all(bits[w] for w in gate.wires[:-1])
        assert bits[:width] == [int(b) for b in f"{value:0{width}b}"]
        inverse = int(field(value) ** -1) if value else 0
        assert int("".join(map(str, bits[width : 2 * width])), 2) == inverse
        assert not any(bits[2 * width :])


class Inverse:
    def __init__(self, width: int, modulus: int, work_count: int) -> None:
        self.width, self.modulus, self.work_count = width, modulus, work_count

    def append_to(self, circuit: Circuit) -> Circuit:
        n = self.width
        gates = gf2_inverse_ir(
            tuple(range(n)),
            tuple(range(n, 2 * n)),
            tuple(range(2 * n, 2 * n + self.work_count)),
            modulus=self.modulus,
        )
        return circuit.compose(CircuitIR(2 * n + self.work_count, gates))


@pytest.mark.parametrize(
    "width,modulus,work_count", [(1, 3, 1), (2, 7, 2), (3, 11, 6), (4, 19, 8)]
)
@pytest.mark.integration
def test_inverse_append_complex_amplitudes_and_cleanup(
    width: int, modulus: int, work_count: int
) -> None:
    galois = pytest.importorskip("galois")
    field = (
        galois.GF(2**width, irreducible_poly=galois.Poly.Int(modulus))
        if width > 1
        else galois.GF(2)
    )
    count = 2 * width + work_count
    state = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    # An arbitrary target superposition also checks the XOR-target contract.
    for x in range(1 << width):
        inverse = int(field(x) ** -1) if x else 0
        for y in range(1 << width):
            index = ((x << width) | y) << work_count
            state[index] = complex(x + 1, 2 * y - 1)
            expected[((x << width) | (y ^ inverse)) << work_count] = state[index]
    expected /= state.norm()
    state /= state.norm()
    result = Circuit(count, dtype=torch.complex128, inputs=state).append(
        Inverse(width, modulus, work_count)
    )
    torch.testing.assert_close(result.state().flatten(), expected, rtol=0, atol=1e-12)


def test_inverse_rejects_aliases_and_insufficient_workspace() -> None:
    with pytest.raises(ValueError, match="distinct"):
        gf2_inverse_ir((0, 1), (1, 2), (3, 4), modulus=7)
    with pytest.raises(ValueError, match="equal"):
        gf2_inverse_ir((0, 1), (2,), (3, 4), modulus=7)
    with pytest.raises(ValueError, match="Insufficient"):
        gf2_inverse_ir((0, 1, 2), (3, 4, 5), (6, 7), modulus=11)
