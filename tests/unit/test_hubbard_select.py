"""Verify Hubbard SELECT's source resources and exact complex Pauli action."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("side,control_value", tuple(product((2, 3, 5), (None, 0, 1))))
def test_hubbard_select_locked_signature_and_cost(side, control_value):
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = Source(side, side, control_val=1 if control_value == 1 else None)
    select = SelectHubbard(
        side,
        controlled=control_value is not None,
        control_value=1 if control_value is None else control_value,
    )
    model = select.source_resource_model()
    if control_value == 0:
        original = original.controlled(CtrlSpec(cvs=0))
        stages = tuple(original.decompose_bloq().iter_bloqsoqs())
        assert [type(stage.bloq).__name__ for stage, _, _ in stages] == [
            "XGate",
            "SelectHubbard",
            "XGate",
        ]
        assert model.name == type(original).__name__
        assert (
            model.calls[0].component
            == SelectHubbard(side, controlled=True).source_resource_model()
        )
        assert select.layout[1:] == tuple(
            (reg.name, reg.total_bits()) for reg in tuple(original.signature)[1:]
        )
    else:
        assert select.layout == tuple(
            (reg.name, reg.total_bits()) for reg in original.signature
        )
    result = estimate_component_resources(model)
    assert {name: value for name, value in asdict(result.gates).items() if value} == (
        get_cost_value(original, QECGatesCost()).asdict()
    )
    assert result.peak_logical_qubits == get_cost_value(original, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("side", [2, 3, 5, 11])
@pytest.mark.parametrize("control_value", [None, 0, 1])
def test_hubbard_select_gate_dependencies_match_source(side, control_value):
    import cirq
    import numpy as np
    from qualtran import QAny, Register
    from qualtran._infra.gate_with_registers import split_qubits
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardMajorannaOperator,
        HubbardSpinUpZ,
    )
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )
    from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    controlled = control_value is not None
    select = SelectHubbard(
        side, controlled, control_value=1 if control_value is None else control_value
    )
    logical = sum(size for _, size in select.layout)
    work, slots = select.requirements()
    workspace = tuple(range(logical, logical + work))
    ports = partition_register(tuple(range(logical)), select.layout)
    source = Source(side, side, control_val=1 if controlled else None)

    class SourceGenerator:
        signature = source.signature

        def decompose_from_registers(self, *, context, **quregs):
            # Use each frozen generator in the original SELECT call order.
            # Flattening the composite DAG first reorders independent gates.
            if control_value == 0:
                yield cirq.X(*quregs["control"])
            for op in cirq.flatten_op_tree(
                source.decompose_from_registers(context=context, **quregs)
            ):
                child = getattr(op.gate, "bloq", op.gate)
                if isinstance(child, HubbardMajorannaOperator):
                    majorana = SelectedMajoranaFermion(
                        child.selection_registers,
                        child.control_registers,
                        child._target_cirq_gate,
                    )
                    yield from majorana.decompose_from_registers(
                        context, **split_qubits(child.signature, op.qubits)
                    )
                elif isinstance(child, HubbardSpinUpZ):
                    bound = split_qubits(child.signature, op.qubits)
                    controls = np.asarray([*bound["V"], *bound.get("control", ())])
                    unary = ApplyGateToLthQubit(
                        (child.signature.get_left("y"), child.signature.get_left("x")),
                        nth_gate=lambda *_: cirq.Z,
                        control_regs=Register("control", QAny(len(controls))),
                    )
                    yield from unary.decompose_from_registers(
                        context=context,
                        control=controls,
                        y=bound["y"],
                        x=bound["x"],
                        target=bound["target"][side * side :],
                    )
                else:
                    yield op
            if control_value == 0:
                yield cirq.X(*quregs["control"])

    original = reference_arithmetic_gates(
        SourceGenerator(),
        ports,
        workspace,
        dict(zip(workspace, workspace, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
        preserve_components=True,
        scalar_phase_anchor=ports["U"][0],
    )
    circuit = DynamicCircuit(logical + work).append(
        select.make_on(ports, workspace=workspace)
    )
    actual = circuit.to_ir().instructions

    def feedback_by_measured_wire(stream):
        # The native emitter reuses logical slots across the three children.
        active = {}
        result = []
        for gate in stream:
            metadata = dict(gate.metadata)
            if gate.name == "measure":
                wire = gate.wires[0]
                active[metadata["classical_bit"]] = wire
                metadata["classical_bit"] = wire
            if "conditions" in metadata:
                metadata["conditions"] = tuple(
                    (active[bit], value) for bit, value in metadata["conditions"]
                )
            result.append(Instruction(gate.name, gate.wires, gate.params, metadata))
        return tuple(result)

    assert slots == work - 1
    assert sum(gate.name == "cy" for gate in actual) == 2 * side**2
    assert_source(
        feedback_by_measured_wire(actual), feedback_by_measured_wire(original)
    )
    if control_value == 1:
        exported = reference_arithmetic_gates(
            source,
            ports,
            workspace,
            dict(zip(workspace, workspace, strict=True)),
            cirq_style=True,
            keep_pauli_leaves=True,
            keep_exchange_leaves=True,
            preserve_components=True,
        )
        assert Counter(gate.name for gate in exported) == Counter(
            gate.name for gate in actual
        )
        if side == 2:
            # The smallest graph also matches when exported as a whole DAG.
            assert_source(
                feedback_by_measured_wire(actual), feedback_by_measured_wire(exported)
            )
        else:
            assert [gate.name for gate in exported] != [gate.name for gate in actual]


@pytest.mark.parametrize(
    "side,control_value", tuple(product((1, 2, 3, 5, 11), (None, 0, 1)))
)
def test_hubbard_select_complex_amplitudes_branches_and_work(side, control_value):
    controlled = control_value is not None
    select = SelectHubbard(
        side, controlled, control_value=1 if control_value is None else control_value
    )
    logical = sum(size for _, size in select.layout)
    work, _ = select.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), select.layout)
    operation = select.make_on(ports, workspace=tuple(range(logical, width)))
    ir = DynamicCircuit(width).append(operation).to_ir()
    if control_value == 0:
        positive = SelectHubbard(side, controlled=True).make_on(
            ports, workspace=tuple(range(logical, width))
        )
        flip = Instruction("x", ports["control"])
        assert ir.instructions == (
            flip,
            *DynamicCircuit(width).append(positive).to_ir().instructions,
            flip,
        )
    counts = count_emitted_instructions(ir)
    events = counts.unconditional.get("measure", 0)
    inverse = select._emit(
        ports,
        tuple(range(logical, width)),
        tuple(range(select.requirements()[1])),
        adjoint=True,
    )
    assert sum(gate.name == "measure" for gate in inverse) == events
    assert (counts.t_count, events, work) == {
        (1, False): (14, 0, 1),
        (1, True): (26, 3, 2),
        (2, False): (102, 15, 3),
        (2, True): (114, 18, 4),
        (3, False): (230, 40, 5),
        (3, True): (242, 43, 6),
        (5, False): (578, 120, 7),
        (5, True): (590, 123, 8),
        (11, False): (2526, 600, 9),
        (11, True): (2538, 603, 10),
    }[side, controlled]

    def pack(fields):
        result = 0
        for name, size in select.layout:
            result = (result << size) | fields[name]
        return result << work

    initial, expected = {}, {}
    n = 2 * side**2
    coordinates = range(side) if side <= 3 else (0, side // 2, side - 1)
    rows = tuple(
        product(range(1 + int(controlled)), (0, 1), (0, 1), coordinates, coordinates)
    )
    fields_per_row = (
        (
            (row, alpha, beta, data)
            for row in rows
            for alpha, beta, data in product((0, 1), (0, 1), range(4))
        )
        if side == 1
        else (
            (row, i % 2, (i // 2) % 2, (12345 * (i + 1)) % (1 << n))
            for i, row in enumerate(rows)
        )
    )
    norm = sqrt(
        sum(i * i for i in range(1, 16 * len(rows) + 1)) if side == 1 else len(rows)
    )
    for index, ((ctrl, u, v, x, y), alpha, beta, data) in enumerate(fields_per_row):
        qx, qy = (x + 1) % side, (y + 1) % side
        fields = dict(
            control=ctrl,
            U=u,
            V=v,
            p_x=x,
            p_y=y,
            q_x=qx,
            q_y=qy,
            alpha=alpha,
            beta=beta,
            target=data,
        )
        amp = 1j**index * (index + 1 if side == 1 else 1) / norm
        initial[pack(fields)] = amp
        if not controlled or ctrl == control_value:
            p = 2 * (x * side + y) + alpha
            q = p if v else 2 * (qx * side + qy) + beta
            p_bit, q_bit = 1 << (n - p - 1), 1 << (n - q - 1)
            amp *= (1j if not data & p_bit else -1j) * (-1) ** (
                (data >> (n - p)).bit_count()
            )
            data ^= p_bit
            amp *= (-1) ** ((data >> (n - q)).bit_count())
            data ^= q_bit
            amp *= -1j * (-1) ** u
            if v:
                spin_up = side**2 + qy * side + qx
                if data & (1 << (n - spin_up - 1)):
                    amp *= -1
        fields["target"] = data
        expected[pack(fields)] = amp

    histories = (
        product((0, 1), repeat=events)
        if side == 1
        else ((pattern * events)[:events] for pattern in ((0,), (1,), (0, 1), (1, 0)))
    )
    for history in histories:
        result = _branch(
            ir.instructions, width, history, initial, normalize_measurements=True
        )
        assert result == pytest.approx(expected, abs=5e-12)
        assert _branch(
            ir.instructions,
            width,
            tuple(1 - bit for bit in history),
            result,
            normalize_measurements=True,
        ) == pytest.approx(initial, abs=5e-12)
        assert _branch(
            inverse,
            width,
            tuple(1 - bit for bit in history),
            result,
            normalize_measurements=True,
        ) == pytest.approx(initial, abs=5e-12)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("control_value", [None, 0, 1])
def test_side_one_select_matches_normalized_locked_schedule(control_value):
    import cirq
    import numpy as np
    from qualtran import BloqError, BQUInt, CtrlSpec, QBit, Register
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardMajorannaOperator,
        HubbardSpinUpZ,
    )
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    controlled = control_value is not None
    source = Source(1, 1, control_val=1 if controlled else None)
    native = SelectHubbard(
        1, controlled, control_value=1 if control_value is None else control_value
    )
    assert native.layout == tuple((r.name, r.bitsize) for r in source.signature)
    with pytest.raises(BloqError, match=r"LeftDangle.q_y is not an available Soquet"):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(
        RuntimeError, match=r"LeftDangle.q_y is not an available Soquet"
    ):
        get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="undefined for side one"):
        native.source_resource_model()
    with pytest.raises(AssertionError):
        HubbardMajorannaOperator(1, 1).decompose_bloq()
    with pytest.raises(
        NotImplementedError, match="Selection register must not be empty"
    ):
        HubbardSpinUpZ(1, 1).decompose_bloq()
    if control_value == 0:
        negative = Source(1, 1).controlled(CtrlSpec(cvs=0))
        assert [
            type(instance.bloq).__name__
            for instance, *_ in negative.decompose_bloq().iter_bloqsoqs()
        ] == ["XGate", "SelectHubbard", "XGate"]

    class NormalizedSource:
        signature = source.signature

        def decompose_from_registers(self, *, context, **ports):
            if control_value == 0:
                yield cirq.X(*ports["control"])
            dropped = 0
            for op in cirq.flatten_op_tree(
                source.decompose_from_registers(context=context, **ports)
            ):
                original_gate = getattr(op.gate, "bloq", op.gate)
                if isinstance(original_gate, CSwap):
                    assert original_gate.bitsize == 0 and len(op.qubits) == 1
                    dropped += 1
                elif isinstance(original_gate, HubbardMajorannaOperator):
                    *controls, spin, first, second = op.qubits
                    # The actual source child delegates here with two extra
                    # singleton coordinates. Only the spin address has a bit.
                    majorana = SelectedMajoranaFermion(
                        (Register("spin", BQUInt(1, 2)),),
                        (Register("control", QBit()),) if controls else (),
                        original_gate._target_cirq_gate,
                    )
                    yield from majorana.decompose_from_registers(
                        context,
                        spin=np.asarray([spin]),
                        target=np.asarray([first, second]),
                        control=np.asarray(controls),
                    )
                elif isinstance(original_gate, HubbardSpinUpZ):
                    *controls, v, _, spin_up = op.qubits
                    if controls:
                        (pred,) = context.qubit_manager.qalloc(1)
                        yield And().on(v, controls[0], pred)
                        yield cirq.CZ(pred, spin_up)
                        yield And().adjoint().on(v, controls[0], pred)
                        context.qubit_manager.qfree([pred])
                    else:
                        yield cirq.CZ(v, spin_up)
                else:
                    yield op
            assert dropped == 4
            if control_value == 0:
                yield cirq.X(*ports["control"])

    logical = sum(width for _, width in native.layout)
    ports = partition_register(tuple(range(logical)), native.layout)
    work_count, slots = native.requirements()
    work = tuple(range(logical, logical + work_count))
    actual = DynamicCircuit(logical + work_count).append(
        native.make_on(ports, workspace=work)
    )
    expected = reference_arithmetic_gates(
        NormalizedSource(),
        ports,
        work,
        dict.fromkeys(work, 0) if slots else {},
        cirq_style=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
        preserve_components=True,
        scalar_phase_anchor=ports["U"][0],
    )
    assert_source(actual.to_ir().instructions, expected)
