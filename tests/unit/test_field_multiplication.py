"""Pinned gate networks, wire maps, phases, and clean scratch for field products."""

from __future__ import annotations

from collections import Counter
from importlib.metadata import version
from typing import Any, Literal

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.field_multiplication import (
    bind_field_outputs,
    gf2_multiplication,
)
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit


def reference_gates(
    bloq: Any,
    inputs: dict[str, tuple[int, ...]],
    allocations: tuple[int, ...],
    *,
    repair_adjoint: bool = False,
) -> tuple[list[tuple[str, tuple[int, ...]]], dict[str, tuple[int, ...]]]:
    """Keep upstream's allocation order and compare each wire's ordered gate stream."""
    pytest.importorskip("qualtran")
    if version("qualtran") != "0.7.0":
        pytest.skip("Gate comparison pins Qualtran v0.7.0")
    import cirq
    import numpy as np
    from qualtran.bloqs.basic_gates import CNOT, Toffoli
    from qualtran.cirq_interop import BloqAsCirqGate

    class Manager(cirq.QubitManager):
        def __init__(self) -> None:
            self.available = list(reversed(allocations))

        def qalloc(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            assert dim == 2
            return [cirq.LineQubit(self.available.pop()) for _ in range(n)]

        def qborrow(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            return self.qalloc(n, dim)

        def qfree(self, qubits: list[cirq.Qid]) -> None:
            self.available.extend(reversed([q.x for q in qubits]))

    names = {
        cirq.CNOT: "cx",
        cirq.CCX: "ccx",
        BloqAsCirqGate(CNOT()): "cx",
        BloqAsCirqGate(Toffoli()): "ccx",
    }
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit

    original = SynthesizeLRCircuit.build_composite_bloq

    def corrected(self: Any, bb: Any, *, q: Any) -> Any:
        if self.is_adjoint:
            forward = SynthesizeLRCircuit(self.matrix).decompose_bloq()
            return bb.add_d(forward.adjoint(), q=q)
        return original(self, bb, q=q)

    with pytest.MonkeyPatch.context() as patch:
        if repair_adjoint:
            patch.setattr(SynthesizeLRCircuit, "build_composite_bloq", corrected)
        circuit, outputs = (
            bloq.decompose_bloq()
            .flatten()
            .to_cirq_circuit_and_quregs(
                qubit_manager=Manager(),
                **{
                    name: np.array([cirq.LineQubit(w) for w in wires])
                    for name, wires in inputs.items()
                },
            )
        )
    gates = [
        (names[op.gate], tuple(q.x for q in op.qubits))
        for op in circuit.all_operations()
    ]
    return gates, {
        name: tuple(q.x for q in wires.flat) for name, wires in outputs.items()
    }


def assert_gate_streams(
    actual: list[tuple[str, tuple[int, ...]]],
    expected: list[tuple[str, tuple[int, ...]]],
) -> None:
    # Qualtran's DAG exporter can reorder disjoint gates. Each wire must see the
    # identical ordered operations and operand order; no cancellation is allowed.
    assert Counter(actual) == Counter(expected)
    for wire in {w for _, wires in expected for w in wires}:
        assert [g for g in actual if wire in g[1]] == [
            g for g in expected if wire in g[1]
        ]


@pytest.mark.parametrize(
    "width,modulus", [(2, 7), (3, 11), (4, 19), (5, 37), (6, 67), (8, 283)]
)
@pytest.mark.parametrize("algorithm", ["quadratic", "karatsuba"])
def test_multiplication_upstream_gates_and_outputs(
    width: int, modulus: int, algorithm: Literal["quadratic", "karatsuba"]
) -> None:
    pytest.importorskip("qualtran")
    from galois import Poly
    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2Multiplication,
        GF2MulViaKaratsuba,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    x, y, target = [tuple(range(i * width, (i + 1) * width)) for i in range(3)]
    scratch = tuple(range(3 * width, 4 * width))
    actual = gf2_multiplication(
        x, y, target, scratch, modulus=modulus, algorithm=algorithm
    )
    cls = GF2Multiplication if algorithm == "quadratic" else GF2MulViaKaratsuba
    upstream = cls(QGF(2, width, Poly.Int(modulus)))
    expected, outputs = reference_gates(
        upstream,
        {"x": x, "y": y},
        target + scratch,
        repair_adjoint=algorithm == "karatsuba",
    )
    assert_gate_streams([(g.name, g.wires) for g in actual.instructions], expected)
    assert actual.outputs == (outputs["x"], outputs["y"], outputs["result"])
    counts = Counter(g.name for g in actual.instructions)
    model = get_cost_value(upstream, QECGatesCost())
    assert (counts["ccx"], counts["cx"]) == (model.toffoli, model.clifford)
    if algorithm == "quadratic":
        assert Counter(g.name for g in actual.instructions)["ccx"] == width**2
    elif width & (width - 1) == 0:
        assert Counter(g.name for g in actual.instructions)["ccx"] == 3 ** (
            width.bit_length() - 1
        )


class Product:
    def __init__(
        self, width: int, modulus: int, algorithm: Literal["quadratic", "karatsuba"]
    ) -> None:
        self.width, self.modulus, self.algorithm = width, modulus, algorithm

    def append_to(self, circuit: Circuit) -> Circuit:
        n = self.width
        x, y, target = [tuple(range(i * n, (i + 1) * n)) for i in range(3)]
        plan = gf2_multiplication(
            x,
            y,
            target,
            tuple(range(3 * n, 4 * n)),
            modulus=self.modulus,
            algorithm=self.algorithm,
        )
        return circuit.compose(
            CircuitIR(4 * n, bind_field_outputs(plan, (x, y, target)))
        )


@pytest.mark.parametrize("algorithm", ["quadratic", "karatsuba"])
@pytest.mark.integration
def test_product_complex_amplitudes_and_clean_workspace(
    algorithm: Literal["quadratic", "karatsuba"],
) -> None:
    galois = pytest.importorskip("galois")
    n = 3
    field = galois.GF(2**n, irreducible_poly=galois.Poly.Int(11))
    state = torch.zeros(1 << (4 * n), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for x in range(8):
        for y in range(8):
            index = (x << (3 * n)) | (y << (2 * n))
            state[index] = complex(x + 1, y - 3)
            expected[index | (int(field(x) * field(y)) << n)] = state[index]
    expected /= state.norm()
    state /= state.norm()
    result = Circuit(4 * n, dtype=torch.complex128, inputs=state).append(
        Product(n, 11, algorithm)
    )
    torch.testing.assert_close(result.state().flatten(), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("modulus", [0, 5, 9, 15, True])
def test_invalid_field_modulus(modulus: int) -> None:
    with pytest.raises(ValueError, match="Modulus"):
        gf2_multiplication((0, 1), (2, 3), (4, 5), (), modulus=modulus)


def test_multiplication_invalid_registers_and_workspace() -> None:
    with pytest.raises(ValueError, match="distinct"):
        gf2_multiplication((0, 1), (0, 3), (4, 5), (), modulus=7)
    with pytest.raises(ValueError, match="equal"):
        gf2_multiplication((0, 1), (2,), (4, 5), (), modulus=7)
    with pytest.raises(ValueError, match="Insufficient"):
        gf2_multiplication(
            tuple(range(5)), tuple(range(5, 10)), tuple(range(10, 15)), (), modulus=37
        )


@pytest.mark.parametrize("width,modulus", [(5, 37), (8, 283)])
def test_recursive_scratch_and_sparse_complex_amplitudes(
    width: int, modulus: int
) -> None:
    """Execute every operand pair without allocating a dense 32-qubit vector."""
    np = pytest.importorskip("numpy")
    galois = pytest.importorskip("galois")
    n = width
    x, y, target = [tuple(range(i * n, (i + 1) * n)) for i in range(3)]
    plan = gf2_multiplication(x, y, target, tuple(range(3 * n, 4 * n)), modulus=modulus)
    if n == 5:
        assert plan.scratch
    gates = bind_field_outputs(plan, (x, y, target))
    size = 1 << n
    a, b = np.repeat(np.arange(size), size), np.tile(np.arange(size), size)
    initial = (a << (3 * n)) | (b << (2 * n))
    indices = initial.copy()
    amplitudes = ((a + 1) + 1j * (2 * b - 3)).astype(np.complex128)
    amplitudes /= np.linalg.norm(amplitudes)
    # These exact permutation gates have no phase multiplier. Keep amplitudes
    # attached to basis labels while propagating the sparse complex state.
    for gate in gates:
        bits = [4 * n - 1 - w for w in gate.wires]
        if gate.name == "swap":
            flip = ((indices >> bits[0]) ^ (indices >> bits[1])) & 1
            indices ^= (flip << bits[0]) | (flip << bits[1])
        else:
            assert gate.name in {"cx", "ccx"}
            condition = np.ones(len(indices), dtype=indices.dtype)
            for bit in bits[:-1]:
                condition &= (indices >> bit) & 1
            indices ^= condition << bits[-1]
    field = galois.GF(size, irreducible_poly=galois.Poly.Int(modulus))
    products = np.asarray(field(a) * field(b), dtype=np.int64)
    expected = initial | (products << n)
    assert not np.any(indices & (size - 1))
    assert dict(zip(indices.tolist(), amplitudes.tolist(), strict=True)) == dict(
        zip(expected.tolist(), amplitudes.tolist(), strict=True)
    )


def test_pinned_upstream_adjoint_counterexample() -> None:
    """Unmodified v0.7.0 emits its forward linear network for the adjoint too."""
    pytest.importorskip("qualtran")
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulViaKaratsuba

    inputs = {"x": (0, 1), "y": (2, 3)}
    literal, _ = reference_gates(GF2MulViaKaratsuba(2), inputs, (4, 5))
    corrected, _ = reference_gates(
        GF2MulViaKaratsuba(2), inputs, (4, 5), repair_adjoint=True
    )
    assert literal != corrected

    # x=y=1 in GF(4): upstream's literal forward decomposition returns 0,
    # while the field product and corrected decomposition return 1.
    def evaluate(gates: list[tuple[str, tuple[int, ...]]]) -> int:
        bits = [0, 1, 0, 1, 0, 0]
        for _, wires in gates:
            bits[wires[-1]] ^= all(bits[w] for w in wires[:-1])
        return 2 * bits[5] + bits[4]

    assert evaluate(literal) == 0
    assert evaluate(corrected) == 1
