"""Standalone Hubbard Majorana source calls and coherent involutions."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import sqrt
from typing import Any, Literal

import pytest

from flagquantum import Circuit
from flagquantum.algorithms.primitives.hubbard_majorana import HubbardMajorana
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
@pytest.mark.parametrize("pauli", ("X", "Y"))
def test_actual_majorana_child_generator_and_declared_model(
    side: int, controlled: bool, pauli: Literal["X", "Y"]
) -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardMajorannaOperator,
    )
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = HubbardMajorannaOperator(
        side, side, pauli, control_val=1 if controlled else None
    )
    children: list[Any] = []

    class SourceCalls:
        def add_from(self, child: Any, **ports: Any) -> tuple[Any, ...]:
            children.append(child)
            return tuple(ports.values())

    source.build_composite_bloq(
        SourceCalls(), x="x", y="y", spin="spin", target="target", control="control"
    )
    assert len(children) == 1
    child = children[0]
    assert isinstance(child, SelectedMajoranaFermion)
    native = HubbardMajorana(side, pauli, controlled)
    assert native.layout == tuple(
        (reg.name, reg.total_bits()) for reg in source.signature
    )
    logical = sum(size for _, size in native.layout)
    work, slots = native.requirements()
    ports = partition_register(tuple(range(logical)), native.layout)
    workspace = tuple(range(logical, logical + work))
    expected = reference_arithmetic_gates(
        child,
        ports,
        workspace,
        dict(zip(workspace[1:], range(slots), strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
        preserve_components=True,
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
    assert (
        sum(
            gate.name == "c" + pauli.lower() and gate.wires[-1] in ports["target"]
            for gate in actual
        )
        == 2 * side**2
    )
    assert counts["measure"] == model.gates.measurement
    assert counts["t"] + counts["tdg"] == 4 * model.gates.and_bloq


@pytest.mark.parametrize(
    "side,control_value,pauli", tuple(product((1, 2, 3), (None, 0, 1), ("X", "Y")))
)
def test_majorana_append_complex_phase_clean_work_and_involution(
    side: int, control_value: int | None, pauli: Literal["X", "Y"]
) -> None:
    native = HubbardMajorana(
        side,
        pauli,
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
        positive = HubbardMajorana(side, pauli, True)._emit(
            ports, workspace, tuple(range(slots))
        )
        flip = Instruction("x", ports["control"])
        assert gates == (flip, *positive, flip)
        if side > 1:
            positive_model = estimate_component_resources(
                HubbardMajorana(side, pauli, True).source_resource_model()
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
            range(1 + int(native.controlled)), range(side), range(side), (0, 1), (0, 1)
        )
    )
    modes = 2 * side**2
    for i, (control, x, y, spin, spectator) in enumerate(rows):
        data = (i * 12345 + 3) % (1 << modes)
        fields = dict(control=control, x=x, y=y, spin=spin, target=data)
        index = (
            sum(
                ((fields[name] >> (len(wires) - 1 - bit)) & 1) << (width - 1 - wire)
                for name, wires in ports.items()
                for bit, wire in enumerate(wires)
            )
            | spectator
        )
        amplitude = (1j**i) / sqrt(len(rows))
        initial[index] = amplitude
        destination, phase = index, 1 + 0j
        if control_value is None or control == control_value:
            selected = (x * side + y) * 2 + spin
            phase *= (-1) ** ((data >> (modes - selected)).bit_count())
            if pauli == "Y":
                phase *= -1j if (data >> (modes - 1 - selected)) & 1 else 1j
            destination ^= 1 << (width - 1 - ports["target"][selected])
        expected[destination] = amplitude * phase
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
        twice, width, (0,) * (2 * events), initial, normalize_measurements=True
    ) == pytest.approx(initial, abs=3e-12)
    if events:
        measured = [
            gate.metadata["classical_bit"] for gate in twice if gate.name == "measure"
        ]
        assert min(measured[events:]) > max(measured[:events])
    with pytest.raises(ValueError, match="DynamicCircuit"):
        Circuit(width).append(bound)
    with pytest.raises(ValueError, match="matching clean work"):
        native.make_on(ports, workspace=())


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dimension": 0},
        {"dimension": True},
        {"dimension": 2, "controlled": 1},
        {"dimension": 2, "control_value": 0},
        {"dimension": 2, "control_value": True},
    ),
)
def test_majorana_rejects_invalid_binding_configuration(
    kwargs: dict[str, Any],
) -> None:
    # Runtime validation covers values arriving from untyped callers.
    with pytest.raises(ValueError):
        HubbardMajorana(**kwargs)
