"""Standalone spin-sector phase binding, original graphs and clean feedback."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import sqrt
from typing import Any

import pytest

from flagquantum import Circuit
from flagquantum.algorithms.primitives.hubbard_spin_up import HubbardSpinUpZ
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("side,controlled", tuple(product((2, 3, 5), (False, True))))
def test_actual_spin_up_graph_and_declared_model(side: int, controlled: bool) -> None:
    from qualtran.bloqs.bookkeeping import Join2, Split2
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardSpinUpZ as Source,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Source(side, side, control_val=1 if controlled else None)
    native = HubbardSpinUpZ(side, controlled)
    assert native.layout == tuple(
        (reg.name, reg.total_bits()) for reg in source.signature
    )
    logical = sum(size for _, size in native.layout)
    work, slots = native.requirements()
    ports = partition_register(tuple(range(logical)), native.layout)
    workspace = tuple(range(logical, logical + work))
    children: list[Any] = []

    class SourceCalls:
        def join(self, bits: Any) -> str:
            assert list(bits) == ["V", "control"]
            return "packed"

        def split(self, value: Any) -> tuple[str, str]:
            assert value == "packed"
            return "V", "control"

        def add(self, child: Any, **ports: Any) -> Any:
            if isinstance(child, Split2):
                assert ports == {"x": "target"}
                return "down", "up"
            assert isinstance(child, Join2)
            return "target"

        def add_from(self, child: Any, **ports: Any) -> tuple[Any, ...]:
            children.append(child)
            return tuple(ports[reg.name] for reg in child.signature)

    source.build_composite_bloq(
        SourceCalls(), V="V", x="x", y="y", target="target", control="control"
    )
    assert len(children) == 1
    child = children[0]
    child_ports = {
        "control": (*ports["V"], *ports.get("control", ())),
        "y": ports["y"],
        "x": ports["x"],
        "target": ports["target"][side**2 :],
    }
    expected = reference_arithmetic_gates(
        child,
        child_ports,
        workspace,
        dict(zip(workspace, range(slots), strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
        preserve_components=True,
    )
    # The full original graph may recycle temporary wire names in a different order.
    exported = reference_arithmetic_gates(
        source,
        ports,
        workspace,
        dict(zip(workspace, range(slots), strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
        preserve_components=True,
    )
    assert Counter(gate.name for gate in expected) == Counter(
        gate.name for gate in exported
    )
    actual = (
        DynamicCircuit(logical + work)
        .append(native.make_on(ports, workspace=workspace))
        .to_ir()
        .instructions
    )
    assert_source(actual, expected)
    model = estimate_component_resources(native.source_resource_model())
    assert {
        key: value for key, value in asdict(model.gates).items() if value
    } == get_cost_value(source, QECGatesCost()).asdict()
    assert model.peak_logical_qubits == get_cost_value(source, QubitCount())
    counts = Counter(gate.name for gate in actual)
    assert counts["measure"] == model.gates.measurement
    assert counts["t"] + counts["tdg"] == 4 * model.gates.and_bloq
    assert (
        sum(gate.name == "cz" and gate.wires[-1] in ports["target"] for gate in actual)
        == side**2
    )


@pytest.mark.parametrize("side,control_value", tuple(product((1, 2, 3), (None, 0, 1))))
def test_spin_up_append_full_complex_branches_and_involution(
    side: int, control_value: int | None
) -> None:
    native = HubbardSpinUpZ(
        side,
        control_value is not None,
        control_value=1 if control_value is None else control_value,
    )
    logical = sum(size for _, size in native.layout)
    work, slots = native.requirements()
    width = logical + work + 1
    ports = partition_register(tuple(reversed(range(logical))), native.layout)
    workspace = tuple(range(logical, logical + work))
    bound = native.make_on(ports, workspace=workspace)
    circuit = DynamicCircuit(width).append(bound)
    gates = CircuitIR.from_json(circuit.to_ir().to_json()).instructions
    if control_value == 0:
        positive = HubbardSpinUpZ(side, True)._emit(
            ports, workspace, tuple(range(slots))
        )
        flip = Instruction("x", ports["control"])
        assert gates == (flip, *positive, flip)
        if side > 1:
            positive_model = estimate_component_resources(
                HubbardSpinUpZ(side, True).source_resource_model()
            )
            assert (
                estimate_component_resources(
                    native.source_resource_model()
                ).gates.clifford
                == positive_model.gates.clifford + 2
            )
    initial: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    rows = tuple(
        product(
            range(1 + int(native.controlled)), (0, 1), range(side), range(side), (0, 1)
        )
    )
    modes = 2 * side**2
    for i, (control, v, x, y, spectator) in enumerate(rows):
        data = (i * 12345 + 3) % (1 << modes)
        fields = dict(control=control, V=v, x=x, y=y, target=data)
        index = (
            sum(
                ((fields[name] >> (len(wires) - 1 - bit)) & 1) << (width - 1 - wire)
                for name, wires in ports.items()
                for bit, wire in enumerate(wires)
            )
            | spectator
        )
        amplitude = 1j**i / sqrt(len(rows))
        initial[index] = amplitude
        selected = side**2 + y * side + x
        enabled = v and (control_value is None or control == control_value)
        expected[index] = amplitude * (
            -1 if enabled and ((data >> (modes - selected - 1)) & 1) else 1
        )
    events = sum(gate.name == "measure" for gate in gates)
    histories = (
        tuple(product((0, 1), repeat=events))
        if side <= 2
        else tuple(tuple((i + offset) % 2 for i in range(events)) for offset in (0, 1))
    )
    for history in histories:
        result = _branch(gates, width, history, initial, normalize_measurements=True)
        assert result == pytest.approx(expected, abs=3e-12)
        assert all(
            not any(index & (1 << (width - 1 - wire)) for wire in workspace)
            for index in result
        )
    twice = circuit.append(bound).to_ir().instructions
    assert _branch(
        twice, width, (0,) * (events * 2), initial, normalize_measurements=True
    ) == pytest.approx(initial, abs=3e-12)
    if events:
        measured = [
            gate.metadata["classical_bit"] for gate in twice if gate.name == "measure"
        ]
        assert min(measured[events:]) > max(measured[:events])
    with pytest.raises(ValueError, match="DynamicCircuit"):
        Circuit(width).append(bound)
    with pytest.raises(ValueError, match="matching clean work"):
        native.make_on(ports, workspace=(*workspace, width))


@pytest.mark.ftqc_reference
def test_original_standalone_control_zero_and_singleton_boundaries() -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardMajorannaOperator,
    )
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardSpinUpZ as Source,
    )

    # The spin-up source stores the control value but always packs a positive bit.
    assert (
        Source(2, 2, control_val=0).decompose_bloq()
        == Source(2, 2, control_val=1).decompose_bloq()
    )
    with pytest.raises(KeyError, match="control"):
        HubbardMajorannaOperator(2, 2, control_val=0).decompose_bloq()
    with pytest.raises(
        NotImplementedError, match="Selection register must not be empty"
    ):
        Source(1, 1).decompose_bloq()
    with pytest.raises(AssertionError):
        HubbardMajorannaOperator(1, 1).decompose_bloq()
