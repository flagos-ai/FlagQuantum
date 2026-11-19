"""Verify the frozen default adjoint's display and physical gate protocols."""

from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.register_score_data import SymbolClass
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

try:
    import cirq
    import numpy as np

    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
except ModuleNotFoundError:
    pytest.skip("Cirq reference dependency is absent", allow_module_level=True)

pytest.importorskip("qualtran")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_phase(*, diagram: bool = True) -> Any:
    from qualtran import Bloq, BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard, TGate

    signature = SourceSignature.build(q=1)
    builder, inputs = BloqBuilder.from_signature(signature)
    q = builder.add(Hadamard(), q=inputs["q"])
    q = builder.add(TGate(), q=q)
    composite = builder.finalize(q=q)

    def decompose(self: object) -> Any:
        return composite

    def info(self: object, args: object) -> Any:
        return cirq.CircuitDiagramInfo(("Phase",)) if diagram else NotImplemented

    source_type = type(
        "Phase",
        (Bloq,),
        {
            "signature": signature,
            "decompose_bloq": decompose,
            "_circuit_diagram_info_": info,
        },
    )
    return source_type()


def phase_gate() -> RegisterCirqGate:
    return RegisterCirqGate(
        CircuitIR(1, (Instruction("h", (0,)), Instruction("t", (0,)))),
        Signature.build(q=1),
        wire_symbol=lambda register, index: ("TextBox", {"text": "Phase"}),
    )


@pytest.mark.parametrize("inverse_count", [0, 1, 2, 3])
def test_adjoint_display_gate_stream_amplitudes_and_resources(
    inverse_count: int,
) -> None:
    from qualtran import Adjoint
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original, gate = source_phase(), phase_gate()
    reference = original
    for _ in range(inverse_count):
        reference = reference.adjoint()
        gate = gate**-1
    assert cirq.circuit_diagram_info(gate) == cirq.circuit_diagram_info(reference)
    assert_source(
        gate._body.instructions,
        reference_arithmetic_gates(
            reference.decompose_bloq(), {"q": (0,)}, (), {}, cirq_style=True
        ),
    )
    expected = torch.as_tensor(np.asarray(reference.tensor_contract()))
    actual = torch.as_tensor(cirq.unitary(gate))
    torch.testing.assert_close(actual, expected, atol=3e-16, rtol=0)
    initial = torch.tensor([1 + 2j, -3 + 4j, 5 - 1j, 2 + 3j], dtype=torch.complex128)
    initial /= initial.norm()
    state = (
        Circuit(2, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(2, gate._body.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(
        state,
        torch.kron(expected.contiguous(), torch.eye(2, dtype=torch.complex128))
        @ initial,
        atol=3e-16,
        rtol=0,
    )
    model = get_cost_value(reference, QECGatesCost())
    assert (model.t, model.clifford, model.measurement) == (1, 1, 0)
    assert Counter(op.name for op in gate._body.instructions) == (
        {"h": 1, "tdg": 1} if inverse_count % 2 else {"h": 1, "t": 1}
    )
    assert len(gate._body.instructions) == 2
    assert isinstance(reference, Adjoint) == bool(inverse_count % 2)


@pytest.mark.parametrize("side", [RegisterSide.LEFT, RegisterSide.RIGHT])
def test_directional_label_supplier_receives_its_original_register(
    side: RegisterSide,
) -> None:
    register = Register("q", quantum_bit_type(), shape=(2,), side=side)
    calls: list[tuple[Register, tuple[int, ...]]] = []

    def symbol(
        actual: Register, index: tuple[int, ...]
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        assert actual == register
        calls.append((actual, index))
        return "TextBox", {"text": f"q{index[0]}"}

    gate = RegisterCirqGate(
        CircuitIR(2, ()), Signature((register,)), wire_symbol=symbol
    )
    inverse = gate**-1
    assert inverse._signature == Signature((register.adjoint(),))
    info = cirq.circuit_diagram_info(inverse)
    assert info.wire_symbols == ("q0", "q1") and info.exponent == -1
    assert calls == [(register, (0,)), (register, (1,))] * 2
    assert inverse**-1 is gate
    assert hash(inverse**-1) == hash(gate)


def test_display_identity_is_stable_and_includes_the_power() -> None:
    gate = phase_gate()
    inverse = gate**-1
    independently_lowered = RegisterCirqGate(
        inverse._body,
        Signature.build(q=1),
        wire_symbol=lambda register, index: ("TextBox", {"text": "Phase"}),
    )
    assert inverse != independently_lowered
    assert inverse == phase_gate() ** -1
    assert hash(inverse) == hash(phase_gate() ** -1)
    assert inverse**-1 is gate
    assert len({gate, inverse, phase_gate(), phase_gate() ** -1}) == 2
    before = cirq.circuit_diagram_info(gate)
    info = cirq.circuit_diagram_info(inverse)
    info.exponent = 17
    assert cirq.circuit_diagram_info(inverse).exponent == -1
    assert cirq.circuit_diagram_info(gate) == before


@pytest.mark.parametrize("power", [-3, -2, 2, 3])
def test_repeated_power_retains_labels_and_exact_body(power: int) -> None:
    gate = phase_gate()
    powered = gate**power
    assert cirq.circuit_diagram_info(powered).wire_symbols == ("Phase",)
    assert cirq.circuit_diagram_info(powered).exponent == power
    np.testing.assert_allclose(
        cirq.unitary(powered),
        np.linalg.matrix_power(cirq.unitary(gate), power),
        atol=8e-16,
        rtol=0,
    )
    assert len(powered._body.instructions) == 2 * abs(power)
    assert cirq.circuit_diagram_info(powered**-1).exponent == -power


def test_missing_source_diagram_protocol_and_native_default_labels() -> None:
    from qualtran import Adjoint

    assert (
        cirq.circuit_diagram_info(Adjoint(source_phase(diagram=False)), default=None)
        is None
    )
    # Native register gates have a concrete owned-register label even when no
    # optional source diagram protocol exists.
    gate = RegisterCirqGate(
        CircuitIR(1, (Instruction("t", (0,)),)), Signature.build(q=1)
    )
    assert cirq.circuit_diagram_info(gate**-1).wire_symbols == ("q",)
    assert cirq.circuit_diagram_info(gate**-1).exponent == -1


@pytest.mark.parametrize("as_set", [False, True])
@pytest.mark.parametrize("merge", [False, True])
def test_declared_symbolic_callees_use_shared_generalization(
    as_set: bool, merge: bool
) -> None:
    from qualtran import Adjoint, Bloq
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.for_testing.atom import TestAtom
    from qualtran.resource_counting import SympySymbolAllocator
    from sympy import Symbol

    from flagquantum.compiler._component_trace.call_graph import component_callee_counts

    signature = SourceSignature.build(q=1)
    if merge:

        def specialized_adjoint(self: object) -> Any:
            return Hadamard()

        first = type(
            "First", (Bloq,), {"signature": signature, "adjoint": specialized_adjoint}
        )()
        second = type(
            "Second", (Bloq,), {"signature": signature, "adjoint": specialized_adjoint}
        )()
    else:
        first, second = TestAtom("first"), TestAtom("second")
    n = Symbol("n")
    entries = ((first, n), (second, 3))
    raw = set(entries) if as_set else dict(entries)

    def source_calls(self: object, ssa: object) -> Any:
        return raw

    root = type(
        "Declared", (Bloq,), {"signature": signature, "build_call_graph": source_calls}
    )()
    expected = {
        str(k): count
        for k, count in Adjoint(root).build_call_graph(SympySymbolAllocator()).items()
    }
    names = {str(key): str(key.adjoint()) for key, _ in entries}
    native_entries = [
        (str(key), count)
        for key, count in (raw.items() if isinstance(raw, dict) else raw)
    ]
    native_raw = set(native_entries) if as_set else dict(native_entries)
    actual = component_callee_counts(
        "root",
        lambda name, allocator: native_raw,
        generalizer=names.__getitem__,
        ignore_unavailable=False,
    )
    assert actual == expected
    if merge:
        assert actual == {"H": n + 3}
