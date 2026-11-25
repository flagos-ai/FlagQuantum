"""Complete Hubbard call graphs retain the original atomic arithmetic boundary."""

from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.hubbard_prepare_component import (
    hubbard_prepare_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_terminals(record: ProgramRecord) -> set[str]:
    """Stop native arithmetic descendants beneath original atomic ModAddK."""
    return atomic_leaves(record) | {
        f.name
        for f in record.functions
        if tuple(r.name for r in f.registers or ()) == ("ctrl", "x")
    }


@pytest.mark.parametrize("dimension", [2, 3, 4, 5, 8, 9])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_hubbard_calls_views_and_exact_inverse(
    dimension: int, adjoint: bool
) -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard as Source,
    )

    source = Source(dimension, dimension, 1.0, 4.0)
    record = hubbard_prepare_program(
        PrepareHubbard(dimension, 1.0, 4.0), adjoint=adjoint
    )
    assert verify_program_record(record) == ()
    pairs = compare_layers(
        record, source.adjoint() if adjoint else source, source_terminals(record)
    )
    for name, child in pairs.items():
        rotation = child.subbloq if type(child).__name__ == "Controlled" else child
        if type(rotation).__name__ != "Ry":
            continue
        count = len(next(f for f in record.functions if f.name == name).registers or ())
        emitted = bind_register_program(
            record, tuple((i,) for i in range(count)), entry=name
        ).circuit.instructions
        assert len(emitted) == 1
        assert emitted[0].name == ("cry" if count == 2 else "ry")
        assert emitted[0].params["theta"] == pytest.approx(
            float(rotation.angle), abs=5e-15, rel=0
        )


def original_hubbard_state(
    dimension: int, initial: np.ndarray[Any, Any], *, t: float = 1.0, u: float = 4.0
) -> np.ndarray[Any, Any]:
    """Run the untouched generator, using its exact atomic word permutation."""
    import cirq
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard as Source,
    )
    from qualtran.bloqs.mod_arithmetic import ModAddK

    logical = 4 * (dimension - 1).bit_length() + 6
    wires = cirq.LineQubit.range(logical)
    context = cirq.DecompositionContext(cirq.GreedyQubitManager(prefix="reference"))
    operations = cirq.decompose_once(
        Source(dimension, dimension, t, u).on(*wires), context=context
    )
    replaced = []
    for operation in operations:
        gate = operation.gate
        if isinstance(gate, ModAddK):
            words = 1 << gate.bitsize
            matrix = np.zeros((4 * words, 4 * words))
            for ctrl, value in product(range(4), range(words)):
                output = gate.on_classical_vals(ctrl=ctrl, x=value)
                matrix[ctrl * words + output["x"], ctrl * words + value] = 1
            operation = cirq.MatrixGate(matrix).on(*operation.qubits)
        replaced.append(operation)
    circuit = cirq.Circuit(cirq.decompose(replaced, context=context))
    scratch = sorted(circuit.all_qubits() - set(wires))
    state = np.zeros(1 << (logical + len(scratch)), dtype=np.complex128)
    state[:: 1 << len(scratch)] = initial
    result = (
        cirq.Simulator(dtype=np.complex128, seed=11)
        .simulate(circuit, qubit_order=(*wires, *scratch), initial_state=state)
        .final_state_vector
    )
    rows = result.reshape(1 << logical, -1)
    np.testing.assert_allclose(rows[:, 1:], 0, atol=2e-11, rtol=0)
    return rows[:, 0]


def test_hubbard_all_small_raw_histories_complex_logical_junk_and_inverse() -> None:
    from tests.unit.measurement_certificate import certify_erasure_histories

    prepare = PrepareHubbard(2, 1.0, 4.0)
    logical = 10
    work, _ = prepare.requirements()
    inputs = []
    cursor = 0
    for register in prepare.signature:
        size = int(register.total_qubits())
        inputs.append(tuple(range(cursor, cursor + size)))
        cursor += size
    clean = tuple(range(logical, logical + work))
    rng = np.random.default_rng(2084)
    initial = rng.normal(size=1 << logical) + 1j * rng.normal(size=1 << logical)
    initial /= np.linalg.norm(initial)
    expected = original_hubbard_state(2, initial)
    for adjoint in (False, True):
        bound = bind_register_program(
            hubbard_prepare_program(prepare, adjoint=adjoint),
            tuple(inputs),
            clean=clean,
        )
        events = certify_erasure_histories(
            bound.circuit.instructions,
            n_wires=bound.circuit.n_wires,
            zero_wires=clean,
            check_clean_output=False,
        )
        state = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
        desired = torch.zeros_like(state)
        stride = 1 << work
        state[::stride] = torch.as_tensor(expected if adjoint else initial)
        desired[::stride] = torch.as_tensor(initial if adjoint else expected)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(state, bound.circuit.instructions, history),
                desired / 2 ** (events / 2),
                atol=3e-11,
                rtol=0,
            )


def test_original_singleton_stored_refusal() -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard as Source,
    )

    with pytest.raises(ValueError):
        Source(1, 1, 1.0, 4.0).decompose_bloq()
    with pytest.raises(ValueError, match="singleton"):
        hubbard_prepare_program(PrepareHubbard(1, 1.0, 4.0))
