"""Check polynomial-basis register addition against both pinned references."""

from dataclasses import dataclass
from importlib.metadata import version

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field import gf2_add_constant_ir, gf2_add_ir
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [1, 3, 8])
def test_qualtran_binary_field_addition_gate_order(width: int) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    if version("qualtran") != "0.7.0":
        pytest.skip("Gate comparison pins Qualtran v0.7.0")
    import cirq
    from qualtran import QGF
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import GF2AddK
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.cirq_interop import BloqAsCirqGate

    actual = gf2_add_ir(tuple(range(width)), tuple(range(width, 2 * width)))
    wires = np.asarray(cirq.LineQubit.range(2 * width))
    upstream, _ = (
        GF2Addition(QGF(2, width))
        .decompose_bloq()
        .to_cirq_circuit_and_quregs(x=wires[:width], y=wires[width:])
    )
    expected = []
    for op in upstream.all_operations():
        assert op.gate in {cirq.CNOT, BloqAsCirqGate(CNOT())}
        expected.append(("cx", tuple(wire.x for wire in op.qubits)))
    assert [(op.name, op.wires) for op in actual] == expected

    value = ((1 << width) - 1) & 0b10100101
    upstream, _ = (
        GF2AddK(width, value)
        .decompose_bloq()
        .to_cirq_circuit_and_quregs(x=wires[:width])
    )
    actual = gf2_add_constant_ir(tuple(range(width)), value)
    assert [(op.name, op.wires) for op in actual] == [
        ("x", tuple(wire.x for wire in op.qubits)) for op in upstream.all_operations()
    ]


@pytest.mark.parametrize("controlled", [False, True])
def test_native_binary_field_register_addition_gate_order(controlled: bool) -> None:
    km = pytest.importorskip("kickmix")
    if getattr(km, "__version__", None) not in {"0.1", "0.1.dev0"}:
        pytest.skip("Native comparison requires the pinned reference build")
    builder = km.CircuitBuilder()
    control = (
        builder.create_quantum_register(1, name="control")[0] if controlled else True
    )
    builder.gf2_iadd(
        offset=builder.create_quantum_register(8, name="source"),
        target=builder.create_quantum_register(8, name="target"),
        control=control,
    )
    expected = []
    for line in str(builder.finish_circuit()).splitlines():
        name, *operands = line.split()
        if name in {"CX", "CCX"}:
            expected.append((name.lower(), tuple(int(wire[1:]) for wire in operands)))
    start = int(controlled)
    actual = gf2_add_ir(
        tuple(range(start, start + 8)),
        tuple(range(start + 8, start + 16)),
        control=0 if controlled else None,
    )
    assert [(op.name, op.wires) for op in actual] == expected


@dataclass(frozen=True)
class _ControlledFieldAdd:
    def append_to(self, circuit: Circuit) -> Circuit:
        return circuit.compose(CircuitIR(5, gf2_add_ir((1, 2), (3, 4), control=0)))


def test_binary_field_addition_preserves_complex_amplitudes() -> None:
    basis = torch.arange(32)
    state = torch.tensor([1, 1j, -1, -1j], dtype=torch.complex128)[basis % 4] / 32**0.5
    expected = torch.empty_like(state)
    for prefix in range(32):
        control, source, target = prefix >> 4, (prefix >> 2) & 3, prefix & 3
        output = (prefix & ~3) | (target ^ (source if control else 0))
        expected[output] = state[prefix]
    actual = Circuit(5, dtype=torch.complex128, inputs=state).append(
        _ControlledFieldAdd()
    )
    torch.testing.assert_close(actual.state().flatten(), expected, rtol=0, atol=1e-12)


def test_binary_field_operands_and_constant_validation() -> None:
    assert gf2_add_ir((), ()) == gf2_add_constant_ir((), 0) == ()
    with pytest.raises(ValueError, match="equal register widths"):
        gf2_add_ir((0,), (1, 2))
    with pytest.raises(ValueError, match="disjoint"):
        gf2_add_ir((0,), (1,), control=0)
    with pytest.raises(ValueError, match="fit"):
        gf2_add_constant_ir((0, 1), 4)
