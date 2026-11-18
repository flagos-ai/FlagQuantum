"""Complete textbook QPE composition, source powers and measured PREPARE."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha1
from itertools import product
from math import pi, sqrt
from pathlib import Path

import cirq
import numpy as np
import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.black_box_prepare import PrepareEmitter
from flagquantum.algorithms.primitives.luis_perina_state import LuisPerinaState
from flagquantum.algorithms.primitives.phase_estimation_window import (
    RectangularWindowState,
)
from flagquantum.algorithms.primitives.textbook_phase_estimation import (
    PoweredUnitaryEmitter,
    TextbookPhaseEstimation,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.primitives.circuit_binding import (
    bind_primitive_circuit,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def z_power(theta: float) -> PoweredUnitaryEmitter:
    def emit(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        repetitions: int,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert not work and not bits
        return (
            Instruction(
                "cphase",
                (control, ports["q"][0]),
                {"theta": 2 * pi * theta * (-1 if adjoint else 1)},
            ),
        ) * repetitions

    return emit


def source_qft(width: int) -> tuple[Instruction, ...]:
    from qualtran.bloqs.basic_gates import CZPowGate
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook
    from qualtran.cirq_interop import BloqAsCirqGate

    qubits: list[cirq.Qid] = list(cirq.LineQubit.range(width))
    operation = QFTTextBook(width).adjoint().on(*qubits)
    leaves = cirq.decompose(
        operation,
        keep=lambda op: isinstance(op.gate, (cirq.HPowGate, cirq.SwapPowGate))
        or (
            isinstance(op.gate, BloqAsCirqGate) and isinstance(op.gate.bloq, CZPowGate)
        ),
    )
    gates = []
    for op in leaves:
        wires = tuple(qubits.index(q) for q in op.qubits)
        if isinstance(op.gate, cirq.HPowGate):
            assert op.gate.exponent % 2 == 1
            gates.append(Instruction("h", wires))
        elif isinstance(op.gate, BloqAsCirqGate):
            gates.append(
                Instruction(
                    "cphase", wires, {"theta": float(op.gate.bloq.exponent) * pi}
                )
            )
        else:
            assert isinstance(op.gate, cirq.SwapPowGate) and op.gate.exponent % 2 == 1
            gates.append(Instruction("swap", wires))
    return tuple(gates)


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_source_powers_full_stream_complex_amplitudes_and_inverse(width: int) -> None:
    from qualtran.bloqs.basic_gates import ZPowGate
    from qualtran.bloqs.phase_estimation import RectangularWindowState as Window
    from qualtran.bloqs.phase_estimation import TextbookQPE

    theta = 0.234
    window = RectangularWindowState(width)
    native = TextbookPhaseEstimation(
        Signature.build(q=1), window, z_power(theta), window.bind()
    )
    original = TextbookQPE(ZPowGate(exponent=2 * theta), Window(width))
    phase, target = tuple(range(width)), width
    operation = native.make_on(phase=phase, target=(target,))
    gates = operation.instructions
    q = cirq.LineQubit.range(width + 1)
    stages = tuple(
        cirq.flatten_op_tree(
            original.decompose_from_registers(
                cirq.DecompositionContext(cirq.SimpleQubitManager()),
                qpe_reg=np.array(q[:-1]),
                q=np.array(q[-1:]),
            )
        )
    )
    assert len(stages) == width + 2
    expected = [Instruction("h", (wire,)) for wire in phase]
    for index, op in enumerate(stages[1:-1]):
        assert op.qubits == (q[width - 1 - index], q[-1])
        np.testing.assert_allclose(
            cirq.unitary(op),
            np.diag([1, 1, 1, np.exp(2j * pi * theta * 2**index)]),
            atol=1e-13,
            rtol=0,
        )
        expected.extend(
            (
                Instruction(
                    "cphase", (width - 1 - index, target), {"theta": 2 * pi * theta}
                ),
            )
            * 2**index
        )
    expected.extend(source_qft(width))
    assert_source(gates, tuple(expected))
    for actual_gate, source_gate in zip(gates, expected, strict=True):
        assert actual_gate.params == pytest.approx(source_gate.params, abs=1e-14)
    spectator = np.array([sqrt(0.3), 1j * sqrt(0.7)])
    eigen_superposition = np.array([sqrt(0.4), 1j * sqrt(0.6)])
    zero = np.zeros(2**width, dtype=complex)
    zero[0] = 1
    initial = np.kron(np.kron(zero, eigen_superposition), spectator)
    circuit = Circuit(
        width + 2, inputs=torch.tensor(initial), dtype=torch.complex128
    ).append(operation)
    expected_state = original.tensor_contract() @ eigen_superposition
    np.testing.assert_allclose(
        circuit.state().numpy().flatten(),
        np.kron(expected_state, spectator),
        atol=1e-13,
        rtol=0,
    )
    circuit.append(native.make_on(phase=phase, target=(target,), adjoint=True))
    np.testing.assert_allclose(
        circuit.state().numpy().flatten(), initial, atol=1e-13, rtol=0
    )
    assert Counter(g.name for g in gates) == {
        "h": 2 * width,
        "cphase": 2**width - 1 + width * (width - 1) // 2,
        **({"swap": width // 2} if width > 1 else {}),
    }
    model = native.resource_model(
        ComponentCost("window", width, GateCost(clifford=width)),
        ComponentCost(
            "CZPow", 3, GateCost(and_bloq=1, clifford=1, rotation=1, measurement=1)
        ),
    )
    assert_source_model(model, original)
    assert (
        CircuitIR.from_json(CircuitIR(width + 1, gates).to_json()).instructions == gates
    )


@pytest.mark.parametrize("width", [1, 2])
def test_measured_preparation_all_histories_complex_phase_and_clean_work(
    width: int,
) -> None:
    from qualtran.bloqs.basic_gates import ZPowGate
    from qualtran.bloqs.phase_estimation import TextbookQPE
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState

    window = LuisPerinaState(width)
    native = TextbookPhaseEstimation(
        Signature.build(q=1), window, z_power(0.234), window.bind()
    )
    original = TextbookQPE(ZPowGate(exponent=0.468), LPResourceState(width))
    work_count, slots = window.requirements()
    phase = tuple(range(width))
    workspace = tuple(range(width + 2, width + 2 + work_count))
    forward = native.make_on(
        phase=phase,
        target=(width,),
        preparation_workspace=workspace,
        preparation_classical_bits=tuple(range(slots)),
    )
    reverse = native.make_on(
        phase=phase,
        target=(width,),
        preparation_workspace=workspace,
        preparation_classical_bits=tuple(range(slots)),
        adjoint=True,
    )
    circuit = DynamicCircuit(width + 2 + work_count, dtype=torch.complex128).append(
        forward
    )
    gates = circuit.to_ir().instructions
    logical = np.array([sqrt(0.4), 1j * sqrt(0.6)])
    spectator = np.array([sqrt(0.3), 1j * sqrt(0.7)])
    zero, clean = np.zeros(2**width), np.zeros(2**work_count)
    zero[0] = clean[0] = 1
    initial = torch.tensor(np.kron(np.kron(np.kron(zero, logical), spectator), clean))
    expected = torch.tensor(
        np.kron(np.kron(original.tensor_contract() @ logical, spectator), clean)
    )
    assert sum(g.name == "measure" for g in gates) == slots
    for history in product((0, 1), repeat=slots):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / sqrt(2**slots),
            atol=1e-12,
            rtol=0,
        )
    for history in product((0, 1), repeat=2 * slots):
        torch.testing.assert_close(
            forced_branch(initial, (*gates, *reverse.instructions), history),
            initial / 2**slots,
            atol=2e-12,
            rtol=0,
        )


def test_all_source_examples_signatures_symbolic_graphs_and_source_dictionary() -> None:
    from qualtran.bloqs.phase_estimation.text_book_qpe import (
        _textbook_qpe_from_precision_and_delta,
        _textbook_qpe_from_standard_deviation_eps,
        _textbook_qpe_small,
        _textbook_qpe_using_m_bits,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    for factory in (
        _textbook_qpe_small,
        _textbook_qpe_using_m_bits,
        _textbook_qpe_from_precision_and_delta,
        _textbook_qpe_from_standard_deviation_eps,
    ):
        original = factory()
        native = TextbookPhaseEstimation(
            Signature.build(q=1),
            RectangularWindowState(original.m_bits),
            z_power(0.234),
        )
        assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
            (r.name, r.bitsize, r.side.name) for r in original.signature
        ]
        assert native.m_bits == original.m_bits
        source_calls = original.build_call_graph(SympySymbolAllocator())
        graph = native.source_graph(
            root="root",
            preparation=original.ctrl_state_prep,
            controlled_unitary=original.unitary.controlled(),
            inverse_qft=original.qft_inv,
        )
        assert graph.leaf_counts == source_calls
        if not isinstance(original.m_bits, int):
            with pytest.raises(ValueError, match="positive concrete"):
                native.make_on(phase=(0,), target=(1,))
    # The original dictionary overwrites colliding child keys.
    native = TextbookPhaseEstimation(
        Signature.build(q=1), RectangularWindowState(3), z_power(0.234)
    )
    assert native.source_graph(
        root="root", preparation="same", controlled_unitary="same", inverse_qft="same"
    ).leaf_counts == {"same": 1}
    n = sympy.Symbol("n")
    with pytest.raises(ValueError, match="Concrete"):
        replace(native, ctrl_state_prep=RectangularWindowState(n)).resource_model(
            ComponentCost("p", 1), ComponentCost("u", 2)
        )


def test_qft_replacement_named_targets_and_binding_failures() -> None:
    from flagquantum.algorithms.primitives.qft import qft

    window = RectangularWindowState(3)
    native = TextbookPhaseEstimation(
        Signature.build(q=1), window, z_power(0.234), window.bind()
    )
    phase, target = (3, 1, 4), (0,)
    captured = source_qft(3)

    def replacement(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert not work and not bits
        local = CircuitIR(3, captured) if not adjoint else qft(3).to_ir()
        return bind_primitive_circuit(local, ports["q"])

    emitter: PrepareEmitter = replacement
    replaced = replace(native, qft_inverse=emitter)
    assert_source(
        replaced.make_on(phase=phase, target=target).instructions,
        native.make_on(phase=phase, target=target).instructions,
    )
    with pytest.raises(ValueError, match="signature"):
        replace(native, qft_inv_signature=Signature.build(other=2))
    with pytest.raises(ValueError, match="PREPARE"):
        replace(native, preparation=None).make_on(phase=phase, target=target)
    with pytest.raises(ValueError):
        native.make_on(phase=phase, target=target, preparation_workspace=(3,))

    def escaped(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        repetitions: int,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return (Instruction("x", (2,)),)

    with pytest.raises(ValueError, match="partition"):
        replace(native, unitary=escaped).make_on(phase=phase, target=target)


def test_frozen_complete_sources() -> None:
    from qualtran.bloqs.phase_estimation import text_book_qpe

    root = Path(text_book_qpe.__file__).parent
    for name, blob in (
        ("text_book_qpe.py", "9349ca32908fbba054cab61f095852f880960b0f"),
        ("text_book_qpe_test.py", "39a0d095da29abccf3f9c6d7bf8d48d54a8a8f59"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_complete_named_multiport_unitary_power_schedule() -> None:
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.phase_estimation import RectangularWindowState as Window
    from qualtran.bloqs.phase_estimation import TextbookQPE

    calls: list[tuple[int, dict[str, tuple[int, ...]], int, bool]] = []

    def controlled_cnot(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        repetitions: int,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert not work and not bits
        calls.append((control, dict(ports), repetitions, adjoint))
        return (
            Instruction("ccx", (control, ports["ctrl"][0], ports["target"][0])),
        ) * repetitions

    window = RectangularWindowState(2)
    native = TextbookPhaseEstimation(
        Signature.build(ctrl=1, target=1), window, controlled_cnot, window.bind()
    )
    original = TextbookQPE(CNOT(), Window(2))
    rng = np.random.default_rng(15)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    initial = np.kron([1, 0, 0, 0], target)
    circuit = Circuit(4, inputs=torch.tensor(initial), dtype=torch.complex128)
    circuit.append(native.make_on(phase=(0, 1), target=(2, 3)))
    assert calls == [
        (1, {"ctrl": (2,), "target": (3,)}, 1, False),
        (0, {"ctrl": (2,), "target": (3,)}, 2, False),
    ]
    np.testing.assert_allclose(
        circuit.state().numpy().flatten(),
        original.tensor_contract() @ target,
        atol=1e-13,
        rtol=0,
    )
    circuit.append(native.make_on(phase=(0, 1), target=(2, 3), adjoint=True))
    np.testing.assert_allclose(
        circuit.state().numpy().flatten(), initial, atol=1e-13, rtol=0
    )
    assert [item[2:] for item in calls[-2:]] == [(2, True), (1, True)]
