"""Check measured Hubbard GQSP against the locked source and its free boundary."""

from collections import Counter, defaultdict
from math import sqrt
from typing import Any

import cirq
import pytest

from flagquantum.algorithms.primitives.hubbard_hamiltonian_gqsp import (
    HubbardHamiltonianGQSP,
)
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.circuit import Circuit
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _modadd_leaf(op: cirq.Operation) -> Any:
    from qualtran.bloqs.mod_arithmetic.mod_addition import ModAddK

    gate = op.gate
    gate = getattr(gate, "subbloq", gate)
    return gate if isinstance(gate, ModAddK) else None


def _source_sparse_branch(
    operations: list[cirq.Operation],
    visible: tuple[cirq.Qid, ...],
    initial: dict[int, complex],
    outcome: int,
) -> dict[int, complex]:
    """Run the source Cirq stream with only its opaque one-bit ModAddK adapted."""
    hidden = sorted({q for op in operations for q in op.qubits} - set(visible))
    wires = (*visible, *hidden)
    positions = {q: 1 << (len(wires) - i - 1) for i, q in enumerate(wires)}
    state = {key << len(hidden): value for key, value in initial.items()}
    measured: dict[str, int] = {}
    for op in operations:
        if isinstance(op, cirq.ClassicallyControlledOperation):
            if not all(
                measured[str(key)]
                for condition in op.classical_controls
                for key in condition.keys
            ):
                continue
            op = op.without_classical_controls()
        masks = tuple(positions[q] for q in op.qubits)
        if cirq.is_measurement(op):
            (mask,) = masks
            measured[cirq.measurement_key_name(op)] = outcome
            projected = {k: v for k, v in state.items() if bool(k & mask) == outcome}
            assert sum(abs(v) ** 2 for v in projected.values()) == pytest.approx(
                sum(abs(v) ** 2 for v in state.values()) / 2, abs=1e-12
            )
            state = {k: v * sqrt(2) for k, v in projected.items()}
            continue
        if isinstance(op.gate, cirq.ResetChannel):
            (mask,) = masks
            assert len({bool(k & mask) for k in state}) == 1
            state = {k & ~mask: v for k, v in state.items()}
            continue
        modadd = _modadd_leaf(op)
        if modadd is not None:
            assert (modadd.bitsize, modadd.mod, modadd.add_val, modadd.cvs) == (
                1,
                2,
                1,
                (0, 0),
            )
            a, b, target = masks
            state = {k if k & (a | b) else k ^ target: v for k, v in state.items()}
            continue
        unitary = cirq.unitary(op)
        chunks = [
            sum(
                mask
                for i, mask in enumerate(masks)
                if col & (1 << (len(masks) - i - 1))
            )
            for col in range(len(unitary))
        ]
        transitions = {
            chunk: [
                (dest, unitary[row, col])
                for row, dest in enumerate(chunks)
                if abs(unitary[row, col]) > 1e-14
            ]
            for col, chunk in enumerate(chunks)
        }
        mask = sum(masks)
        values: dict[int, complex] = defaultdict(complex)
        for key, amp in state.items():
            for destination, coefficient in transitions[key & mask]:
                values[(key & ~mask) | destination] += amp * coefficient
        state = {k: v for k, v in values.items() if abs(v) > 1e-12}
    assert all(k % (1 << len(hidden)) == 0 for k in state)
    return {k >> len(hidden): v for k, v in state.items()}


def test_hubbard_measured_append_and_outer_temp_leakage() -> None:
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        get_walk_operator_for_hubbard_model,
    )
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
    )
    from qualtran.cirq_interop import BloqAsCirqGate

    prep = PrepareHubbard(2, 1, 1)
    operation = HubbardHamiltonianGQSP(prep, 0.002, 0.01)
    assert operation.requirements() == (7, 7)
    program = operation.make_on(
        signal=0,
        target=tuple(range(1, 17)),
        prepare_temp=(17, 18),
        reflection_temp=(19, 20),
        workspace=tuple(range(21, 28)),
        classical_bits=tuple(range(7)),
    )
    with pytest.raises(ValueError, match="requires a DynamicCircuit"):
        Circuit(28).append(program)
    gates = DynamicCircuit(28).append(program).to_ir().instructions
    counts = Counter(g.name for g in gates)
    assert (len(gates), counts["measure"], counts["reset"]) == (2379, 68, 68)
    assert sum(counts[g] for g in ("t", "tdg")) == 566
    state = {0: 1 / sqrt(3), 129: 1j / sqrt(3), 36: -1 / sqrt(3)}
    initial = {k << 11: v for k, v in state.items()}
    histories = [
        (pattern * counts["measure"])[: counts["measure"]]
        for pattern in ((0,), (1,), (0, 1), (1, 0))
    ]
    branches = [
        _branch(gates, 28, history, initial, normalize_measurements=True)
        for history in histories
    ]
    for branch in branches[1:]:
        assert branch == pytest.approx(branches[0], abs=2e-12)
    assert all(k % (1 << 9) == 0 for k in branches[0])
    actual = {k >> 9: v for k, v in branches[0].items()}

    source = HamiltonianSimulationByGQSP(
        get_walk_operator_for_hubbard_model(2, 2, 1, 1), t=0.002, precision=0.01
    )
    assert source.degree == 1
    named = get_named_qubits(source.signature)
    context = cirq.DecompositionContext(cirq.SimpleQubitManager(prefix="hubbard"))
    top = BloqAsCirqGate(source).on(
        *(q for r in source.signature for q in named[r.name].flat)
    )
    stages = list(cirq.flatten_op_tree(cirq.decompose_once(top, context=context)))
    assert len(stages) == 3 and type(stages[1].gate).__name__ == "GeneralizedQSP"
    ordered = ("signal", *(name for name, _ in prep.selection_layout), "target")
    logical = tuple(q for name in ordered for q in named[name].flat)
    outer_temp = tuple(q for q in stages[0].qubits if q not in logical)
    assert len(outer_temp) == 2 and stages[-1].qubits[-2:] == outer_temp
    ops = cirq.decompose(
        stages,
        context=context,
        keep=lambda op: len(op.qubits) <= 3
        and (
            cirq.is_measurement(op)
            or cirq.has_unitary(op)
            or isinstance(op, cirq.ClassicallyControlledOperation)
            or isinstance(op.gate, cirq.ResetChannel)
            or _modadd_leaf(op) is not None
        ),
    )
    expected = _source_sparse_branch(
        ops, (*logical, *outer_temp), {k << 2: v for k, v in state.items()}, 0
    )
    assert actual == pytest.approx(expected, abs=5e-11)
    assert sum(abs(v) ** 2 for k, v in expected.items() if k % 4) == pytest.approx(
        0.016501741620346923, abs=2e-12
    )
    assert sum(
        abs(v) ** 2 for k, v in expected.items() if k < 1024 and k % 4
    ) == pytest.approx(3.027031372650665e-05, abs=2e-12)
