"""Pinned uniform preparation, exact phases and measured workspace cleanup."""

from collections import Counter
from itertools import product
from math import acos, sin, sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.basis_translation import translate
from flagquantum.compiler.registers.uniform_preparation import (
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = pytest.mark.unit


def layout(count, values):
    width, work, slots = uniform_preparation_requirements(count, len(values))
    controls = tuple(range(len(values)))
    target = tuple(range(len(values), len(values) + width))
    workspace = tuple(range(len(values) + width, len(values) + width + work))
    return target, workspace, tuple(range(slots)), controls


def emitted(count, values, adjoint=False):
    target, work, slots, controls = layout(count, values)
    return uniform_preparation_ir(
        count,
        target,
        work,
        slots,
        controls=controls,
        control_values=values,
        adjoint=adjoint,
    )


def embedded(logical, work):
    state = torch.zeros((len(logical), 1 << work), dtype=torch.complex128)
    state[:, 0] = logical
    return state.flatten()


@pytest.mark.parametrize(
    "count,values",
    [(2, ()), (4, (0, 1)), (3, ()), (5, (1,)), (6, (1, 0)), (12, (0, 1, 0))],
)
def test_uniform_support_relative_phase_inverse_and_counts(count, values):
    target, work, _, controls = layout(count, values)
    full = 1 << len(target)
    active = sum(value << i for i, value in enumerate(values[::-1]))
    weights = torch.arange(1, (1 << len(values)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(weights, 2 - weights)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (len(values) + len(target)), dtype=torch.complex128)
    initial[::full] = amplitudes
    expected = initial.clone()
    odd = count // (count & -count)
    high = (odd - 1).bit_length()
    if odd > 1:
        theta = acos(1 - (1 << (high - 1)) / odd)
        probability = odd / (1 << high)
        phase = sqrt(probability) * complex(1 / (2 * probability), sin(theta))
    else:
        phase = 1
    expected[active * full : active * full + count] = (
        amplitudes[active] * phase / sqrt(count)
    )
    gates = emitted(count, values)
    counts = Counter(gate.name for gate in gates)
    ands = len(target) * max(0, len(values) - 1)
    if odd > 1:
        ands += 3 * high + len(values) - 1
    assert counts["measure"] == counts["reset"] == ands
    assert counts["t"] + counts["tdg"] == 4 * ands
    assert counts["rz"] == (2 if odd > 1 else 0)
    assert counts["ch"] == (len(target) if controls else 0)
    circuit = DynamicCircuit(
        len(controls) + len(target) + len(work),
        inputs=embedded(initial, len(work)),
        dtype=torch.complex128,
    ).append(FieldOperation(len(controls) + len(target) + len(work), gates))
    result = run_dynamic(circuit, shots=8, seed=33, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1),
        embedded(expected, len(work)).expand(8, -1),
        atol=3e-12,
        rtol=0,
    )
    inverse = emitted(count, values, adjoint=True)
    restored = run_dynamic(
        circuit.append(
            FieldOperation(len(controls) + len(target) + len(work), inverse)
        ),
        shots=8,
        seed=47,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(8, -1),
        embedded(initial, len(work)).expand(8, -1),
        atol=3e-12,
        rtol=0,
    )


def test_empty_target_and_invalid_inputs():
    assert emitted(1, ()) == emitted(1, (1, 0), True) == ()
    with pytest.raises(ValueError, match="positive integer"):
        uniform_preparation_requirements(0)
    with pytest.raises(ValueError, match="nonnegative"):
        uniform_preparation_requirements(3, -1)
    target, work, bits, controls = layout(3, (0,))
    with pytest.raises(ValueError, match="adjoint"):
        uniform_preparation_ir(3, target, work, bits, adjoint=1)
    with pytest.raises(ValueError, match="zero/one"):
        uniform_preparation_ir(3, target, work, bits, controls=controls)
    with pytest.raises(ValueError, match="clean work"):
        uniform_preparation_ir(3, target, (), bits)
    with pytest.raises(ValueError, match="classical slots"):
        uniform_preparation_ir(3, target, work, (-1, -1))
    with pytest.raises(ValueError, match="distinct nonnegative"):
        uniform_preparation_ir(
            3, (work[0], target[1]), work, bits, controls=controls, control_values=(0,)
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "count,values", [(3, ()), (3, (0,)), (4, (1, 0)), (6, (1, 0)), (5, (1, 0, 1))]
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_full_source_complex_map_and_erasure_histories(count, values, adjoint):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    target, workspace, _, controls = layout(count, values)
    width = len(target) + len(controls)
    wires = tuple(cirq.LineQubit.range(width))
    src = PrepareUniformSuperposition(count, values).on(*wires)
    if adjoint:
        src = cirq.inverse(src)
    manager = cirq.GreedyQubitManager("work", maximize_reuse=True)
    ops = cirq.decompose(
        src,
        context=cirq.DecompositionContext(manager),
        on_stuck_raise=None,
        keep=lambda op: op.gate == cirq.H
        or isinstance(op.gate, (cirq.ControlledGate, cirq.Rz, cirq.ResetChannel))
        or cirq.is_measurement(op)
        or isinstance(op, cirq.ClassicallyControlledOperation),
    )
    reference = cirq.Circuit(ops)
    ancilla = tuple(sorted(reference.all_qubits() - set(wires)))
    spectator = cirq.NamedQubit("spectator")
    logical = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    initial = torch.complex(logical, 3 - 2 * logical)
    initial /= initial.norm()
    reference_state = embedded(initial, len(ancilla))
    ref_simulator = cirq.Simulator(dtype=np.complex128, seed=3)
    output = ref_simulator.simulate(
        reference,
        qubit_order=(*wires, spectator, *ancilla),
        initial_state=reference_state.numpy(),
    ).final_state_vector.reshape(len(logical), -1)
    np.testing.assert_allclose(output[:, 1:], 0, rtol=0, atol=3e-12)
    expected = embedded(torch.from_numpy(output[:, 0].copy()), len(workspace))

    gates = emitted(count, values, adjoint)
    n_qubits = width + 1 + len(workspace)
    # Keep native ancillas contiguous by placing the spectator between the
    # logical inputs and workspace, using an explicit rebinding.
    gates = tuple(
        Instruction(
            g.name,
            tuple(q + (q >= width) for q in g.wires),
            g.params,
            metadata=g.metadata,
        )
        for g in gates
    )
    circuit = DynamicCircuit(
        n_qubits, inputs=embedded(initial, len(workspace)), dtype=torch.complex128
    )
    circuit.append(FieldOperation(n_qubits, gates))
    ir = circuit.to_ir()
    events = sum(g.name == "measure" for g in gates)
    if events <= 6:
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(
                    embedded(initial, len(workspace)), tuple(ir.instructions), history
                ),
                expected / sqrt(2**events),
                atol=3e-12,
                rtol=0,
            )
    else:
        torch.testing.assert_close(
            forced_branch(
                embedded(initial, len(workspace)), tuple(ir.instructions), (0,) * events
            ),
            expected / sqrt(2**events),
            atol=3e-12,
            rtol=0,
        )
        actual = run_dynamic(
            circuit,
            shots=12,
            seed=39,
            strategy="trajectory",
        )
        torch.testing.assert_close(
            actual.final_states.reshape(12, -1),
            expected.expand(12, -1),
            atol=3e-12,
            rtol=0,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "count,values",
    [
        (2, ()),
        (4, (1, 0)),
        (3, ()),
        (6, (0,)),
        (6, (1, 0)),
        (10, (1, 0)),
        (5, (1, 0, 1)),
    ],
)
def test_pinned_outer_generator_and_inner_source_streams(count, values):
    pytest.importorskip("qualtran")
    import cirq
    from qualtran._infra.gate_with_registers import split_qubits
    from qualtran.bloqs.arithmetic import LessThanConstant
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    target, workspace, slots, controls = layout(count, values)
    source = PrepareUniformSuperposition(count, values)
    manager = cirq.DecompositionContext(
        cirq.GreedyQubitManager("stage", maximize_reuse=True)
    )
    quregs = split_qubits(
        source.signature, cirq.LineQubit.range(len(controls) + len(target))
    )
    ops = list(
        cirq.flatten_op_tree(source.decompose_from_registers(context=manager, **quregs))
    )

    def source_predicate(bloq, ctrl, prefix):
        qubits = tuple(cirq.LineQubit(q) for q in (*ctrl, *prefix))
        source_leaves = (
            tuple(
                cirq.flatten_op_tree(
                    bloq.decompose_from_registers(
                        context=manager, **split_qubits(bloq.signature, qubits)
                    )
                )
            )
            if len(prefix) > 1
            else (bloq.on(*qubits),)
        )
        compute, erase = [], []
        for leaf in source_leaves:
            left, right, flag = (q.x for q in leaf.qubits)
            binding = {"ctrl": (left, right), "target": (flag,)}
            compute.extend(
                reference_arithmetic_gates(leaf.gate, binding, (), {}, cirq_style=True)
            )
            erase.append(
                reference_arithmetic_gates(
                    leaf.gate.adjoint(),
                    binding,
                    (),
                    {flag: slots[prefix.index(flag)]},
                    cirq_style=True,
                )
            )
        return tuple(compute), tuple(g for block in reversed(erase) for g in block)

    initial = []
    for op in ops[: len(target)]:
        *ctrl, wire = (q.x for q in op.qubits)
        if op.gate == cirq.H:
            initial.append((Instruction("h", (wire,)),))
            continue
        assert op.gate.sub_gate == cirq.H
        pattern = tuple(next(iter(v)) for v in op.gate.control_values)
        assert tuple(ctrl) == controls and pattern == values
        if len(ctrl) == 1:
            flip = (Instruction("x", tuple(ctrl)),) if pattern == (0,) else ()
            initial.append((*flip, Instruction("ch", (ctrl[0], wire)), *flip))
        else:
            work = workspace[: len(ctrl) - 1]
            predicate = And(*pattern) if len(ctrl) == 2 else MultiAnd(pattern)
            compute, erase = source_predicate(predicate, tuple(ctrl), work)
            initial.append((*compute, Instruction("ch", (work[-1], wire)), *erase))

    _, odd, high = source.k_l_logL()
    assert tuple(q.x for q in quregs["target"]) == target
    for adjoint in (False, True):
        h_stage = tuple(
            gate
            for part in (reversed(initial) if adjoint else initial)
            for gate in part
        )
        if high == 0:
            expected = h_stage
        else:
            rest = ops[len(target) :]
            lt, rz, undo = rest[:3]
            assert lt.gate == undo.gate == LessThanConstant(high, odd)
            middle = tuple(
                Instruction("h", (op.qubits[0].x,)) for op in rest[3 : 3 + high]
            )
            reflect, rotate, _ = rest[3 + high : 6 + high]
            final = tuple(
                Instruction("h", (op.qubits[0].x,)) for op in rest[6 + high :]
            )
            assert middle == final
            assert all(
                op.gate == cirq.H for op in (*rest[3 : 3 + high], *rest[6 + high :])
            )
            less_work = workspace[1 : high + 2]
            compare = reference_arithmetic_gates(
                lt.gate,
                {"x": target[:high], "target": workspace[:1]},
                less_work,
                dict(zip(less_work[1:], slots[:high], strict=True)),
                cirq_style=True,
            )
            prefix = workspace[: high + len(values) - 1]
            assert tuple(q.x for q in reflect.qubits[: high + len(values)]) == (
                *target[:high],
                *controls,
            )
            compute, erase = source_predicate(
                getattr(reflect.gate, "bloq", reflect.gate),
                (*target[:high], *controls),
                prefix,
            )
            sign = -1 if adjoint else 1
            threshold = (
                *compare,
                Instruction("rz", workspace[:1], {"theta": sign * rz.gate._rads}),
                *compare,
            )
            zero = (
                *compute,
                Instruction("rz", prefix[-1:], {"theta": sign * rotate.gate._rads}),
                *erase,
            )
            expected = (
                (*final[::-1], *zero, *middle[::-1], *threshold, *h_stage)
                if adjoint
                else (*h_stage, *threshold, *middle, *zero, *final)
            )

        def fixed_basis(stream):
            for gate in stream:
                if gate.name == "ch":
                    leaves = translate(
                        gate, can_run=lambda candidate: candidate.name != "ch"
                    )
                    assert leaves is not None
                    yield from leaves
                else:
                    yield gate

        actual = tuple(fixed_basis(emitted(count, values, adjoint)))
        expected = tuple(fixed_basis(expected))
        assert [(g.name, g.wires, g.metadata) for g in actual] == [
            (g.name, g.wires, g.metadata) for g in expected
        ]
        assert Counter(g.name for g in actual) == Counter(g.name for g in expected)
        assert [g.params["theta"] for g in actual if g.name == "rz"] == pytest.approx(
            [g.params["theta"] for g in expected if g.name == "rz"], abs=2e-15, rel=0
        )


@pytest.mark.ftqc_reference
def test_empty_source_binding_and_concrete_resource_model():
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
    from qualtran.cirq_interop.t_complexity_protocol import t_complexity

    empty = PrepareUniformSuperposition(1)
    with pytest.raises(KeyError, match="target"):
        cirq.decompose_once(empty.on())
    context = cirq.DecompositionContext(cirq.SimpleQubitManager())
    assert not tuple(
        cirq.flatten_op_tree(
            empty.decompose_from_registers(
                context=context, target=np.empty(0, dtype=object)
            )
        )
    )
    for count, values, model_t, rotations, native_t in (
        (3, (), 20, 2, 20),
        (3, (0,), 24, 6, 28),
        (6, (1, 0), 40, 8, 46),
        (5, (1, 0, 1), 68, 8, 74),
    ):
        model = t_complexity(PrepareUniformSuperposition(count, values))
        assert (model.t, model.rotations) == (model_t, rotations)
        counts = Counter(g.name for g in emitted(count, values))
        assert counts["t"] + counts["tdg"] + 2 * counts["ch"] == native_t


def test_controlled_h_exact_target_translation_and_resources():
    basis = translate(
        Instruction("ch", (0, 1)),
        can_run=lambda g: g.name in {"h", "s", "sdg", "t", "tdg", "cx"},
    )
    assert basis is not None
    assert Counter(g.name for g in basis) == {
        "s": 1,
        "h": 2,
        "t": 1,
        "cx": 1,
        "tdg": 1,
        "sdg": 1,
    }
    from flagquantum.circuit import Circuit

    for i in range(4):
        state = torch.eye(4, dtype=torch.complex128)[i]
        a = Circuit(2, inputs=state, dtype=torch.complex128).append(
            FieldOperation(2, basis)
        )
        b = Circuit(2, inputs=state, dtype=torch.complex128).append(
            FieldOperation(2, (Instruction("ch", (0, 1)),))
        )
        torch.testing.assert_close(a.state(), b.state(), atol=2e-12, rtol=0)
