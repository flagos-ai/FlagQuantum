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
from tests.unit.test_register_subtraction import assert_source

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
    ands = max(0, len(values) - 1)
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
@pytest.mark.parametrize("count,values", [(3, ()), (6, (0,)), (10, (1, 0))])
def test_pinned_outer_generator_and_inner_source_streams(count, values):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran._infra.gate_with_registers import split_qubits
    from qualtran.bloqs.arithmetic import LessThanConstant
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
    _, odd, high = source.k_l_logL()
    assert [op.qubits for op in ops[: len(target)]] == [
        tuple(cirq.LineQubit(q) for q in (*controls, wire)) for wire in target
    ]
    assert all(
        op.gate.sub_gate == cirq.H if controls else op.gate == cirq.H
        for op in ops[: len(target)]
    )
    lt, rz, undo = ops[len(target) : len(target) + 3]
    assert lt.gate == undo.gate == LessThanConstant(high, odd)
    reflect, rotate, _ = ops[len(target) + 3 + high : len(target) + 6 + high]
    angle = rz.gate._rads
    assert angle == rotate.gate._rads
    gates = emitted(count, values)
    rz_locations = [i for i, g in enumerate(gates) if g.name == "rz"]
    assert [gates[i].params["theta"] for i in rz_locations] == pytest.approx(
        [angle, angle]
    )
    less_work = workspace[1 : high + 2]
    compare = reference_arithmetic_gates(
        lt.gate,
        {"x": target[:high], "target": workspace[:1]},
        less_work,
        dict(zip(less_work[1:], slots[:high], strict=True)),
        cirq_style=True,
    )
    a, b = rz_locations
    assert_source(gates[a - len(compare) : a], compare)
    assert_source(gates[a + 1 : a + 1 + len(compare)], compare)

    prefix = workspace[: high + len(values) - 1]
    bindings = {"ctrl": (*target[:high], *controls), "target": prefix[-1:]}
    if len(prefix) > 1:
        bindings["junk"] = prefix[:-1]
    bloq = getattr(reflect.gate, "bloq", reflect.gate)
    expected = reference_arithmetic_gates(bloq, bindings, (), {}, cirq_style=True)
    assert_source(gates[b - len(expected) : b], expected)
    source_prefix = (
        list(
            cirq.flatten_op_tree(
                bloq.decompose_from_registers(
                    context=manager,
                    **split_qubits(
                        bloq.signature,
                        tuple(cirq.LineQubit(q) for q in (*bindings["ctrl"], *prefix)),
                    ),
                )
            )
        )
        if len(prefix) > 1
        else [bloq.on(*(cirq.LineQubit(q) for q in (*bindings["ctrl"], *prefix)))]
    )
    erased = []
    for leaf in reversed(source_prefix):
        left, right, flag = (q.x for q in leaf.qubits)
        erased.extend(
            reference_arithmetic_gates(
                leaf.gate.adjoint(),
                {"ctrl": (left, right), "target": (flag,)},
                (),
                {flag: slots[prefix.index(flag)]},
                cirq_style=True,
            )
        )
    assert_source(gates[b + 1 : b + 1 + len(erased)], tuple(erased))
    np.testing.assert_array_equal(np.asarray([q.x for q in quregs["target"]]), target)


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
        (6, (1, 0), 40, 8, 38),
        (5, (1, 0, 1), 68, 8, 58),
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
