"""Map the frozen phase tutorial fixtures to the existing append boundary."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from math import pi

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.phase_block_encoding import PhaseBlockEncoding
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _hadamard(
    ports: Mapping[str, tuple[int, ...]],
    work: tuple[int, ...],
    bits: tuple[int, ...],
    inverse: bool,
) -> tuple[Instruction, ...]:
    return (Instruction("h", ports["q"]),)


@pytest.mark.parametrize("phi", [0.25, 1.0])
def test_original_phase_hadamard_fixture_codec_and_absolute_phase(phi: float) -> None:
    from qualtran.bloqs.basic_gates import GlobalPhase, Hadamard
    from qualtran.bloqs.block_encoding.phase import Phase, _phase_block_encoding
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = (
        _phase_block_encoding() if phi == 0.25 else Phase(Unitary(Hadamard()), phi, 0)
    )
    child = UnitaryBlockEncoding((("q", 1),), _hadamard)
    block = PhaseBlockEncoding(child, phi, 0)
    assert (block.alpha, block.epsilon) == (source.alpha, source.epsilon)
    assert (block.system_bitsize, block.ancilla_bitsize, block.resource_bitsize) == (
        1,
        0,
        0,
    )
    assert [(reg.name, int(reg.total_bits())) for reg in source.signature] == [
        ("system", 1)
    ]
    original_stages = tuple(
        instance.bloq for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    )
    assert set(original_stages) == {
        GlobalPhase(exponent=phi, eps=0),
        Unitary(Hadamard()),
    }
    assert len(original_stages) == 2 and not GlobalPhase(exponent=phi, eps=0).signature

    circuit = Circuit(1, dtype=torch.complex128)
    for inverse in (False, True):
        emitted = Circuit(1, dtype=torch.complex128)
        operation = block.make_on(
            child.make_on(system=(0,), adjoint=inverse), anchor=0, adjoint=inverse
        )
        emitted.append(operation)
        ir = emitted.to_ir()
        assert (
            CircuitIR.from_json(ir.to_json()).validate().instructions == ir.instructions
        )
        assert Counter(gate.name for gate in ir.instructions) == {
            "x": 2,
            "phase": 2,
            "h": 1,
        }
        body = ir.instructions[:1] if inverse else ir.instructions[-1:]
        assert body == (Instruction("h", (0,)),)
        assert [
            gate.params["theta"] for gate in ir.instructions if gate.name == "phase"
        ] == [(-phi if inverse else phi) * pi] * 2
        locked = source.adjoint() if inverse else source
        torch.testing.assert_close(
            get_unitary(emitted),
            torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128),
            atol=2e-12,
            rtol=0,
        )
        model = estimate_component_resources(
            block.resource_model(ComponentCost("Hadamard", 1, GateCost(clifford=1)))
        )
        assert {
            key: val for key, val in asdict(model.gates).items() if val
        } == get_cost_value(locked, QECGatesCost()).asdict()
        assert model.peak_logical_qubits == get_cost_value(locked, QubitCount()) == 1
        circuit.append(operation)
    torch.testing.assert_close(
        get_unitary(circuit), torch.eye(2, dtype=torch.complex128), atol=2e-12, rtol=0
    )


def test_original_phase_forwards_custom_signal_preparation(
    request: pytest.FixtureRequest,
) -> None:
    from qualtran.bloqs.block_encoding.phase import Phase

    request.config.addinivalue_line("markers", "notebook: upstream notebook execution")
    from qualtran.bloqs.block_encoding.product_test import (
        TestBlockEncoding,
        TestPrepareOracle,
    )

    source = Phase(TestBlockEncoding(), phi=1, eps=0)

    def prepare(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        return (Instruction("h", ports["z"]),)

    signal = BlackBoxPrepare((("z", 1),), (("a", 5),), prepare)

    class Child(UnitaryBlockEncoding):
        @property
        def signal_state(self) -> BlackBoxPrepare:
            return signal

    child = Child((("q", 1),), _hadamard, ancilla_bitsize=1, resource_bitsize=1)
    block = PhaseBlockEncoding(child, phi=1, eps=0)
    assert isinstance(source.signal_state.prepare, TestPrepareOracle)
    assert (block.ancilla_bitsize, block.resource_bitsize) == (
        source.ancilla_bitsize,
        source.resource_bitsize,
    )
    assert block.signal_state is signal
    assert (signal.selection_width, signal.junk_width) == tuple(
        int(reg.total_bits()) for reg in source.signal_state.signature
    )
    assert block.signal_state.emit((0,), (1, 2, 3, 4, 5), (), ()) == (
        Instruction("h", (0,)),
    )
