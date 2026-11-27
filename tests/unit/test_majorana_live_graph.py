"""Complete Majorana live graphs use the accepted unary-iteration owner."""

import builtins
from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import assert_source_wire_streams
from tests.unit.test_majorana_file import source as source
from tests.unit.test_majorana_source_binding import _capture, _expected_matrix

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("controls", (0, 1, 2))
@pytest.mark.parametrize(
    "gate_name", ("T", "Rx", "Ry", "sqrtH", "matrix", "random_matrix")
)
def test_complete_live_graph_all_complex_measurement_branches(
    source: Any, controls: int, gate_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cirq
    from qualtran._infra.data_types import BQUInt, QAny
    from qualtran._infra.registers import Register
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    gate = {
        "T": cirq.T,
        "Rx": cirq.rx(0.3),
        "Ry": cirq.ry(0.3),
        "sqrtH": cirq.H**0.5,
        "matrix": cirq.MatrixGate(cirq.unitary(cirq.rx(0.3))),
        "random_matrix": cirq.MatrixGate(
            cirq.testing.random_unitary(2, random_state=17)
        ),
    }[gate_name]
    original = source(
        Register("selection", BQUInt(1, 2)),
        control_regs=() if controls == 0 else (Register("enable", QAny(controls)),),
        target_gate=gate,
    )
    inputs = (() if controls == 0 else (tuple(range(controls)),)) + (
        (controls,),
        (controls + 1, controls + 2),
    )
    logical = controls + 3
    matrix = _expected_matrix(gate, 2, controls)
    values = torch.randn(
        1 << logical,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(734),
    )
    values /= values.norm()
    real_import = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split(".")[0] not in {"qualtran", "cirq", "kickmix"}
        return real_import(name, *args, **kwargs)

    for adjoint in (False, True):
        directed = original.adjoint() if adjoint else original
        program = import_register_graph(directed)
        bound = bind_register_program(
            program, inputs, clean=tuple(range(logical, logical + 5))
        )
        assert bound.outputs == inputs
        graph = directed.decompose_bloq().flatten()
        captured, swaps = _snapshot(
            graph, tuple((reg.name, int(reg.total_bits())) for reg in graph.signature)
        )
        assert swaps == 0
        assert Counter(g.name for g in captured.instructions) == Counter(
            g.name for g in bound.circuit.instructions
        )
        if gate_name not in {"matrix", "random_matrix"}:
            assert_source_wire_streams(
                bound.circuit.instructions, captured.instructions
            )
        assert sum(event.kind == "allocate" for event in bound.events) == sum(
            event.kind == "free" for event in bound.events
        )
        counts = Counter(g.name for g in bound.circuit.instructions)
        events = counts["measure"]
        assert events == controls + {"T": 2, "random_matrix": 4}.get(gate_name, 0)
        assert counts["t"] + counts["tdg"] == 4 * events + 2 * (gate_name == "T")
        model = get_cost_value(directed, QECGatesCost())
        assert model.measurement == events
        width = bound.circuit.n_wires
        work = width - logical
        state = torch.zeros(1 << width, dtype=torch.complex128)
        state[:: 1 << work] = values
        expected = torch.zeros_like(state)
        expected[:: 1 << work] = (matrix.mH if adjoint else matrix) @ values
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", native_import)
            for stream in (bound.circuit.instructions, captured.instructions):
                ir = Circuit(width).append(FieldOperation(width, stream)).to_ir()
                ir = CircuitIR.from_json(ir.to_json())
                for history in product((0, 1), repeat=events):
                    torch.testing.assert_close(
                        forced_branch(state, ir.instructions, history),
                        expected / 2 ** (events / 2),
                        atol=4e-12,
                        rtol=0,
                    )


def test_opaque_unitary_uses_owned_capture_without_source_json(source: Any) -> None:
    import cirq
    import numpy as np
    from numpy.typing import NDArray
    from qualtran._infra.data_types import BQUInt
    from qualtran._infra.registers import Register

    class OpaqueUnitary(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _unitary_(self) -> NDArray[np.complex128]:
            matrix: NDArray[np.complex128] = np.array(
                [[1, 1j], [1j, 1]], dtype=np.complex128
            )
            return matrix / 2**0.5

        def __pow__(self, exponent: object) -> Any:
            if exponent == -1:
                return cirq.MatrixGate(self._unitary_().conj().T)
            return NotImplemented

    gate = OpaqueUnitary()
    original = source(Register("selection", BQUInt(1, 2)), target_gate=gate)
    live_programs = tuple(
        import_register_graph(original.adjoint() if adjoint else original)
        for adjoint in (False, True)
    )
    native, emitter = _capture(original)
    work, bits = emitter.requirements()
    assert (work, bits) == (2, 1)
    matrix = _expected_matrix(gate, 2, 1)
    values = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128)
    values += 1j * values.flip(0)
    values /= values.norm()
    for adjoint in (False, True):
        snapshot = (
            Circuit(4 + work)
            .append(
                native.make_on(
                    selection=(0, 1),
                    target=(2, 3),
                    workspace=(4, 5),
                    classical_bits=(0,),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert snapshot.instructions == emitter.snapshots[int(adjoint)].instructions
        snapshot = CircuitIR.from_json(snapshot.to_json())
        state = torch.zeros(1 << (4 + work), dtype=torch.complex128)
        state[:: 1 << work] = values
        expected = torch.zeros_like(state)
        expected[:: 1 << work] = (matrix.mH if adjoint else matrix) @ values
        bound = bind_register_program(
            live_programs[int(adjoint)], ((0,), (1,), (2, 3)), clean=(4, 5)
        )
        for outcome in (0, 1):
            torch.testing.assert_close(
                forced_branch(state, snapshot.instructions, (outcome,)),
                expected / 2**0.5,
                atol=3e-12,
                rtol=0,
            )
            torch.testing.assert_close(
                forced_branch(state, bound.circuit.instructions, (outcome,)),
                expected / 2**0.5,
                atol=3e-12,
                rtol=0,
            )


@pytest.mark.parametrize("gate_name", ("T", "Rx", "random_matrix"))
def test_three_control_word_live_capture_bypasses_source_serialization_failure(
    source: Any, gate_name: str
) -> None:
    import cirq
    from qualtran._infra.data_types import BQUInt, QAny
    from qualtran._infra.registers import Register
    from qualtran.serialization.bloq import bloqs_to_proto

    gate = {
        "T": cirq.T,
        "Rx": cirq.rx(0.3),
        "random_matrix": cirq.MatrixGate(
            cirq.testing.random_unitary(2, random_state=17)
        ),
    }[gate_name]
    original = source(
        Register("selection", BQUInt(1, 2)),
        control_regs=(Register("enable", QAny(3)),),
        target_gate=gate,
    )
    with pytest.raises(ValueError, match="Cannot serialize.*numpy.uint8"):
        bloqs_to_proto(
            original.decompose_bloq().flatten(), max_depth=1, pred=lambda _: False
        )
    live_programs = tuple(
        import_register_graph(original.adjoint() if adjoint else original)
        for adjoint in (False, True)
    )
    # The actual adjoint owns a different graph and serializes successfully.
    assert bloqs_to_proto(
        original.adjoint().decompose_bloq().flatten(), max_depth=1, pred=lambda _: False
    )
    assert import_register_graph(original.adjoint()).functions
    native, emitter = _capture(original)
    work, bits = emitter.requirements()
    matrix = _expected_matrix(gate, 2, 3)
    values = torch.randn(
        64, dtype=torch.complex128, generator=torch.Generator().manual_seed(821)
    )
    values /= values.norm()
    for adjoint in (False, True):
        ir = (
            Circuit(6 + work)
            .append(
                native.make_on(
                    selection=(0, 1, 2, 3),
                    target=(4, 5),
                    workspace=tuple(range(6, 6 + work)),
                    classical_bits=tuple(range(bits)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert ir.instructions == emitter.snapshots[int(adjoint)].instructions
        ir = CircuitIR.from_json(ir.to_json())
        state = torch.zeros(1 << (6 + work), dtype=torch.complex128)
        state[:: 1 << work] = values
        expected = torch.zeros_like(state)
        expected[:: 1 << work] = (matrix.mH if adjoint else matrix) @ values
        bound = bind_register_program(
            live_programs[int(adjoint)],
            ((0, 1, 2), (3,), (4, 5)),
            clean=tuple(range(6, 6 + work)),
        )
        for stream in (ir.instructions, bound.circuit.instructions):
            stream_events = sum(g.name == "measure" for g in stream)
            for history in product((0, 1), repeat=stream_events):
                torch.testing.assert_close(
                    forced_branch(state, stream, history),
                    expected / 2 ** (stream_events / 2),
                    atol=1e-11,
                    rtol=0,
                )
