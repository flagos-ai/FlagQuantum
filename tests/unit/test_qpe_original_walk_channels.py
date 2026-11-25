# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Original scope: Qualtran v0.7.0 phase_estimation/*qpe_test.py.

"""Original five-bit walk examples through actual native child channels."""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.algorithms.primitives.luis_perina_state import LuisPerinaState
from flagquantum.algorithms.primitives.phase_estimation_window import (
    RectangularWindowState,
)
from flagquantum.algorithms.primitives.qft import qft
from flagquantum.algorithms.primitives.qubitization_phase_estimation import (
    QubitizationPhaseEstimation,
)
from flagquantum.algorithms.primitives.textbook_phase_estimation import (
    TextbookPhaseEstimation,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sources_and_native_reference() -> None:
    pytest.importorskip("cirq")
    from importlib.metadata import version

    import kickmix

    from tests.unit.test_qubitization_phase_estimation import (
        test_complete_frozen_source_and_original_tests,
    )
    from tests.unit.test_textbook_phase_estimation import test_frozen_complete_sources

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    test_frozen_complete_sources()
    test_complete_frozen_source_and_original_tests()


@dataclass(frozen=True)
class Child:
    source: Any
    emitter: CircuitRegisterEmitter
    forward: torch.Tensor
    reverse: torch.Tensor
    events: int


def capture_child(source: Any) -> Child:
    emitter = import_register_emitter(source)
    work, slots = emitter.requirements()
    ports: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for register in source.signature:
        width = int(register.total_bits())
        ports[register.name] = tuple(range(cursor, cursor + width))
        cursor += width
    matrices = []
    counts = []
    for adjoint in (False, True):
        gates = emitter(
            ports, tuple(range(cursor, cursor + work)), tuple(range(slots)), adjoint
        )
        matrices.append(logical_operator(CircuitIR(cursor + work, gates), cursor))
        counts.append(Counter(g.name for g in gates)["measure"])
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(matrices[0], expected, atol=8e-12, rtol=0)
    torch.testing.assert_close(matrices[1], expected.mH, atol=8e-12, rtol=0)
    assert counts[0] == counts[1]
    return Child(source, emitter, matrices[0], matrices[1], counts[0])


@pytest.fixture(scope="module", params=(2, 3, 4))
def walk_case(request: pytest.FixtureRequest) -> tuple[Any, Any, dict[str, Child]]:
    pytest.importorskip("cirq")
    from qualtran import CtrlSpec
    from qualtran.bloqs.for_testing.qubitization_walk_test import (
        get_uniform_pauli_qubitized_walk,
    )

    ham, walk = get_uniform_pauli_qubitized_walk(request.param)
    children = {
        name: capture_child(source)
        for name, source in (
            ("walk", walk),
            ("controlled", walk.controlled()),
            ("reflection", walk.reflect.controlled(CtrlSpec(cvs=0))),
        )
    }
    return ham, walk, children


def apply_operator(
    state: torch.Tensor, matrix: torch.Tensor, wires: tuple[int, ...]
) -> torch.Tensor:
    """Apply a certified child on named MSB axes; the final axis batches inputs."""
    rest = tuple(i for i in range(state.ndim) if i not in wires)
    permutation = (*wires, *rest)
    reordered = state.permute(permutation)
    result = (matrix @ reordered.reshape(1 << len(wires), -1)).reshape(reordered.shape)
    inverse = tuple(permutation.index(i) for i in range(state.ndim))
    return result.permute(inverse)


@pytest.mark.parametrize("measured", (False, True))
@pytest.mark.parametrize("qubitization", (False, True))
def test_all_original_walk_eigenstates_exact_schedule_channels_and_resources(
    walk_case: tuple[Any, Any, dict[str, Child]],
    measured: bool,
    qubitization: bool,
    record_property: Callable[[str, object], None],
) -> None:
    pytest.importorskip("cirq")
    import cirq

    pytest.importorskip("numpy")
    import numpy as np
    from qualtran.bloqs.basic_gates import Power
    from qualtran.bloqs.phase_estimation import QubitizationQPE, TextbookQPE
    from qualtran.bloqs.phase_estimation import RectangularWindowState as SourceWindow
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState

    ham, walk, children = walk_case
    n = int(walk.signature.get_left("target").bitsize)
    select_width = (n - 1).bit_length()
    phase = tuple(range(5))
    target = tuple(range(5, 5 + select_width + n))
    ports = {"selection": target[:select_width], "target": target[select_width:]}
    window = LuisPerinaState(5) if measured else RectangularWindowState(5)
    prep_work, prep_slots = (
        window.requirements() if isinstance(window, LuisPerinaState) else (0, 0)
    )
    walk_work = max(c.emitter.requirements()[0] for c in children.values())
    walk_slots = max(c.emitter.requirements()[1] for c in children.values())
    work = tuple(range(5 + len(target), 5 + len(target) + max(prep_work, walk_work)))
    bits = tuple(range(max(prep_slots, walk_slots)))

    def emit(
        name: str, named: Mapping[str, tuple[int, ...]], adjoint: bool
    ) -> tuple[Instruction, ...]:
        emitter = children[name].emitter
        w, s = emitter.requirements()
        return emitter(named, work[:w], bits[:s], adjoint)

    def body(
        named: Mapping[str, tuple[int, ...]],
        scratch: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert scratch == work[:walk_work] and slots == bits[:walk_slots]
        return emit("walk", named, adjoint)

    def controlled(
        control: int,
        named: Mapping[str, tuple[int, ...]],
        scratch: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert scratch == work[:walk_work] and slots == bits[:walk_slots]
        return emit("controlled", {"ctrl": (control,), **named}, adjoint)

    def reflection(
        control: int,
        named: Mapping[str, tuple[int, ...]],
        scratch: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert scratch == work[:walk_work] and slots == bits[:walk_slots]
        return emit("reflection", {"control": (control,), **named}, adjoint)

    def powered(
        control: int,
        named: Mapping[str, tuple[int, ...]],
        scratch: tuple[int, ...],
        slots: tuple[int, ...],
        repetitions: int,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return controlled(control, named, scratch, slots, adjoint) * repetitions

    selection = Register("selection", bounded_unsigned_register_type(select_width, n))
    signature = Signature((selection, Register("target", opaque_register_type(n))))
    native = (
        QubitizationPhaseEstimation(
            signature,
            Signature((selection,)),
            window,
            body,
            controlled,
            reflection,
            window.bind(),
        )
        if qubitization
        else TextbookPhaseEstimation(signature, window, powered, window.bind())
    )
    if isinstance(native, QubitizationPhaseEstimation):
        bound = native.make_on(
            phase=phase,
            target=target,
            preparation_workspace=work[:prep_work],
            preparation_classical_bits=bits[:prep_slots],
            walk_workspace=work[:walk_work],
            walk_classical_bits=bits[:walk_slots],
        )
        reverse = native.make_on(
            phase=phase,
            target=target,
            preparation_workspace=work[:prep_work],
            preparation_classical_bits=bits[:prep_slots],
            walk_workspace=work[:walk_work],
            walk_classical_bits=bits[:walk_slots],
            adjoint=True,
        )
    else:
        bound = native.make_on(
            phase=phase,
            target=target,
            preparation_workspace=work[:prep_work],
            preparation_classical_bits=bits[:prep_slots],
            unitary_workspace=work[:walk_work],
            unitary_classical_bits=bits[:walk_slots],
        )
        reverse = native.make_on(
            phase=phase,
            target=target,
            preparation_workspace=work[:prep_work],
            preparation_classical_bits=bits[:prep_slots],
            unitary_workspace=work[:walk_work],
            unitary_classical_bits=bits[:walk_slots],
            adjoint=True,
        )
    n_wires = 5 + len(target) + len(work)
    gates = tuple(DynamicCircuit(n_wires).append(bound).to_ir().instructions)
    assert (
        tuple(DynamicCircuit(n_wires).append(reverse).to_ir().instructions)
        == reverse.instructions
    )
    prep = window.bind().emit(
        phase, (), work[:prep_work], bits[:prep_slots], adjoint=False
    )
    prep_events = certify_erasure_histories(
        prep, n_wires=n_wires, zero_wires=(*phase, *work), check_clean_output=False
    )
    initial = torch.zeros(1 << (5 + prep_work), dtype=torch.complex128)
    initial[0] = 1
    local_prep = window.bind().emit(
        phase, (), tuple(range(5, 5 + prep_work)), bits[:prep_slots], adjoint=False
    )
    prepared = forced_branch(initial, local_prep, (0,) * prep_events) * sqrt(
        2**prep_events
    )
    vector = prepared[:: 1 << prep_work]
    clean = torch.zeros_like(prepared)
    clean[:: 1 << prep_work] = vector
    torch.testing.assert_close(prepared, clean, atol=8e-12, rtol=0)
    source_window = LPResourceState(5) if measured else SourceWindow(5)
    expected_vector = torch.as_tensor(
        source_window.tensor_contract().reshape(-1), dtype=torch.complex128
    )
    torch.testing.assert_close(vector, expected_vector, atol=8e-12, rtol=0)

    source = (QubitizationQPE if qubitization else TextbookQPE)(walk, source_window)
    qubits = cirq.LineQubit.range(5 + len(target))
    stages = tuple(
        cirq.flatten_op_tree(
            source.decompose_from_registers(
                cirq.DecompositionContext(cirq.SimpleQubitManager()),
                qpe_reg=np.array(qubits[:5]),
                selection=np.array(qubits[5 : 5 + select_width]),
                target=np.array(qubits[5 + select_width :]),
            )
        )
    )
    assert stages[0] == source_window.on(*qubits[:5])
    assert stages[-1] == source.qft_inv.on(*qubits[:5])
    schedule: list[tuple[str, tuple[int, ...], int]] = []
    emitted = list(prep)
    for i, control in enumerate(reversed(phase)):
        if not qubitization or i == 0:
            expected_op = (
                cirq.pow(
                    walk.on_registers(
                        selection=np.array(qubits[5 : 5 + select_width]),
                        target=np.array(qubits[5 + select_width :]),
                    ).controlled_by(qubits[control]),
                    2**i,
                )
                if not qubitization
                else walk.controlled().on_registers(
                    ctrl=qubits[control],
                    selection=np.array(qubits[5 : 5 + select_width]),
                    target=np.array(qubits[5 + select_width :]),
                )
            )
            assert stages[1 + i if not qubitization else 1] == expected_op
            schedule.append(("controlled", (control, *target), 2**i))
            emitted.extend(
                emit("controlled", {"ctrl": (control,), **ports}, False) * 2**i
            )
        else:
            start = 2 + 3 * (i - 1)
            before, power, after = stages[start : start + 3]
            assert before == after
            assert before.qubits == tuple(
                qubits[w] for w in (control, *ports["selection"])
            )
            assert isinstance(power, cirq.Operation) and isinstance(power.gate, Power)
            assert power.gate.bloq == walk and power.gate.power == 2**i
            schedule.extend(
                (
                    ("reflection", (control, *ports["selection"]), 1),
                    ("walk", target, 2**i),
                    ("reflection", (control, *ports["selection"]), 1),
                )
            )
            emitted.extend(
                emit(
                    "reflection",
                    {"control": (control,), "selection": ports["selection"]},
                    False,
                )
            )
            emitted.extend(emit("walk", ports, False) * 2**i)
            emitted.extend(
                emit(
                    "reflection",
                    {"control": (control,), "selection": ports["selection"]},
                    False,
                )
            )
    qft_ir = qft(5, inverse=True).to_ir()
    emitted.extend(qft_ir.instructions)
    assert gates == tuple(emitted)
    native_qft = logical_operator(qft_ir, 5)
    source_qft = torch.as_tensor(
        source.qft_inv.tensor_contract(), dtype=torch.complex128
    )
    torch.testing.assert_close(native_qft, source_qft, atol=8e-12, rtol=0)
    values, eigenvectors = np.linalg.eigh(ham.matrix())
    selection_state = torch.zeros(1 << select_width, dtype=torch.complex128)
    selection_state[:n] = 1 / sqrt(n)
    inputs = torch.kron(
        selection_state[:, None], torch.as_tensor(eigenvectors, dtype=torch.complex128)
    )
    actual = (vector[:, None, None] * inputs[None, :, :]).reshape(
        *(2,) * (5 + len(target)), 1 << n
    )
    expected = (expected_vector[:, None, None] * inputs[None, :, :]).reshape_as(actual)
    prepared_inputs = actual.clone()
    events = prep_events
    for name, wires, repetitions in schedule:
        child = children[name]
        reference = torch.as_tensor(
            child.source.tensor_contract(), dtype=torch.complex128
        )
        for _ in range(repetitions):
            actual = apply_operator(actual, child.forward, wires)
            expected = apply_operator(expected, reference, wires)
            events += child.events
    actual = apply_operator(actual, native_qft, phase)
    expected = apply_operator(expected, source_qft, phase)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=0)
    probabilities = actual.reshape(32, -1, 1 << n).abs().square().sum(dim=1)
    torch.testing.assert_close(
        probabilities.sum(dim=0),
        torch.ones(1 << n, dtype=torch.float64),
        atol=4e-11,
        rtol=0,
    )
    modes = probabilities.argmax(dim=0).to(torch.float64) / 32
    eigenvalues = torch.as_tensor(values / n)
    if qubitization:
        cosine = (torch.cos(torch.pi * modes).abs() - eigenvalues.abs()).abs()
        sine = (torch.sin(torch.pi * modes).abs() - eigenvalues.abs()).abs()
        assert bool((torch.minimum(cosine, sine) < 0.05).all())
    else:
        assert bool(
            ((torch.cos(2 * torch.pi * modes) - eigenvalues).abs() < 0.05).all()
        )
    recovered = apply_operator(actual, native_qft.mH, phase)
    for name, wires, repetitions in reversed(schedule):
        for _ in range(repetitions):
            recovered = apply_operator(recovered, children[name].reverse, wires)
    torch.testing.assert_close(recovered, prepared_inputs, atol=5e-11, rtol=0)
    reverse_prep = window.bind().emit(
        phase, (), tuple(range(5, 5 + prep_work)), bits[:prep_slots], adjoint=True
    )
    assert (
        certify_erasure_histories(
            reverse_prep,
            n_wires=5 + prep_work,
            zero_wires=tuple(range(5, 5 + prep_work)),
            check_clean_output=False,
        )
        == prep_events
    )
    independent_reverse = list(qft(5).to_ir().instructions)
    for name, wires, repetitions in reversed(schedule):
        named = dict(ports)
        if name == "controlled":
            named["ctrl"] = (wires[0],)
        elif name == "reflection":
            named = {"control": (wires[0],), "selection": ports["selection"]}
        independent_reverse.extend(emit(name, named, True) * repetitions)
    independent_reverse.extend(
        window.bind().emit(phase, (), work[:prep_work], bits[:prep_slots], adjoint=True)
    )
    assert reverse.instructions == tuple(independent_reverse)
    unprepared = forced_branch(prepared, reverse_prep, (0,) * prep_events) * sqrt(
        2**prep_events
    )
    torch.testing.assert_close(unprepared, initial, atol=8e-12, rtol=0)
    assert Counter(g.name for g in gates)["measure"] == events
    if n == 3 and measured and qubitization:
        # One independent full Runtime execution retains a coherent spectator.
        coefficients = torch.complex(
            torch.arange(1, 17, dtype=torch.float64).reshape(8, 2),
            torch.arange(16, dtype=torch.float64).reshape(8, 2).remainder(7) - 3,
        )
        coefficients /= coefficients.norm()
        input_columns = torch.zeros((32, inputs.shape[0], 2), dtype=torch.complex128)
        input_columns[0] = inputs @ coefficients
        runtime_initial = torch.zeros(1 << (n_wires + 1), dtype=torch.complex128)
        addresses = (
            torch.arange(1 << (5 + len(target)))[:, None] << (len(work) + 1)
        ) | torch.arange(2)
        runtime_initial[addresses] = input_columns.reshape(-1, 2)
        circuit = DynamicCircuit(
            n_wires + 1, inputs=runtime_initial, dtype=torch.complex128
        )
        circuit.append(bound)
        runtime = run_dynamic(
            circuit,
            shots=1,
            seed=20261125,
            strategy="trajectory",
        )
        runtime_expected = torch.zeros_like(runtime_initial)
        runtime_expected[addresses] = expected.reshape(-1, 8) @ coefficients
        torch.testing.assert_close(
            runtime.final_states[0].flatten(), runtime_expected, atol=4e-11, rtol=0
        )
        record_property("runtime_spectator_physical_wires", n_wires + 1)
        record_property(
            "runtime_max_complex_error",
            float((runtime.final_states[0].flatten() - runtime_expected).abs().max()),
        )
    record_property("target_width", n)
    record_property("family", "qubitization" if qubitization else "textbook")
    record_property("window", "LP" if measured else "rectangular")
    record_property("eigenstates", 1 << n)
    record_property("max_complex_error", float((actual - expected).abs().max()))
    record_property("all_raw_histories_exponent", events)
    record_property("joint_forward_inverse_histories_exponent", 2 * events)
    record_property(
        "emitted_instruction_inventory", dict(Counter(g.name for g in gates))
    )
    record_property(
        "opaque_matrix_leaf_widths",
        dict(Counter(len(g.wires) for g in gates if g.name == "any")),
    )
    record_property("physical_wires", n_wires)
    record_property(
        "model_root_children",
        {
            str(child): str(weight)
            for child, weight in source.build_call_graph(None).items()
        },
    )
