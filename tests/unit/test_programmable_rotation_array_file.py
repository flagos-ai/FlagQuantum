"""Audit the exact source rotation array and its original parameter grid."""

from cmath import exp
from collections import Counter
from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.programmable_rotation_array_catalog import (
    ProgrammableRotationArrayExample,
)
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.lookup.source_qrom import source_qrom_workspace_size
from flagquantum.compiler.registers.phase.programmable_rotation_array import (
    programmable_rotation_array_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_source_qrom import source_qrom_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.rotations import programmable_rotation_gate_array

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(programmable_rotation_gate_array.__file__).parent
    for name, digest in (
        ("__init__.py", "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"),
        (
            "programmable_rotation_gate_array.ipynb",
            "a08ae58ab5eb1ddb3ba6e5cd21a95defade9f864",
        ),
        (
            "programmable_rotation_gate_array.py",
            "7c6b33d09150c49574a9e5a4fb218a504254a1b4",
        ),
        (
            "programmable_rotation_gate_array_test.py",
            "f3486cab5190ecb4b482f8ff0af11b68c0149794",
        ),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return programmable_rotation_gate_array


def source_bound_ops(source: Any, **wires: tuple[int, ...]) -> tuple[Any, ...]:
    import cirq

    return tuple(
        source.decompose_from_registers(
            context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
            **{
                name: np.array([cirq.LineQubit(q) for q in bound])
                for name, bound in wires.items()
            },
        )
    )


def qrom_gates(
    gate: Any, selection: tuple[int, ...], load: tuple[int, ...], work: tuple[int, ...]
) -> tuple[Instruction, ...]:
    data = tuple(map(int, gate.data[0]))
    required = source_qrom_workspace_size(0, (len(selection),), (data,))
    return source_qrom_gates(
        gate,
        {"selection": selection, "target0_": load},
        work[:required],
        tuple(range(required)),
    )


@pytest.mark.parametrize("measured", [False, True])
def test_four_entry_tutorial(measured: bool, source_module: Any) -> None:
    from cirq import CZPowGate as CirqCZPowGate
    from qualtran.bloqs.basic_gates.rotation import CZPowGate
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    pinned = source_module._programmable_rotation_gate_array.make()
    op = ProgrammableRotationArrayExample(measured)
    assert [(r.name, r.dtype.num_qubits, r.side.name) for r in op.signature] == [
        (r.name, r.bitsize, r.side.name) for r in pinned.signature
    ]
    selection, load, target, work = (0, 1), (2, 3), (4,), (5,)
    children = source_bound_ops(
        pinned,
        selection=selection,
        kappa_load_target=load,
        rotations_target=target,
    )
    reference: list[Instruction] = []
    calls: Counter[str] = Counter()
    declared = (
        instance.bloq
        for instance, _, _ in pinned.decompose_bloq().iter_bloqsoqs()
        if not isinstance(instance.bloq, (Split, Join))
    )
    for child, bloq in zip(children, declared, strict=True):
        gate = child.gate
        if isinstance(gate, QROM):
            assert gate == bloq
            calls[f"QROM {tuple(map(int, gate.data[0]))}"] += 1
            reference.extend(qrom_gates(gate, selection, load, work))
        else:
            assert isinstance(gate, CirqCZPowGate) and isinstance(bloq, CZPowGate)
            assert gate.exponent == bloq.exponent and gate.global_shift == 0
            calls[f"CZPowGate({float(gate.exponent)})"] += 1
            pair = tuple(q.x for q in child.qubits)
            if measured:
                body, _ = source_gradient_gates(bloq, {"q": pair}, {"q": pair}, work=5)
                reference.extend(body)
            else:
                reference.append(
                    Instruction("cphase", pair, {"theta": pi * float(gate.exponent)})
                )
    assert calls == op.call_graph().leaf_counts
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    gates = circuit.to_ir().instructions
    assert_source_dependencies(gates, tuple(reference))
    source_cost = get_cost_value(pinned, QECGatesCost())
    model = estimate_component_resources(op.resource_model())
    assert model.gates == GateCost(
        **{f.name: getattr(source_cost, f.name) for f in fields(GateCost)}
    )
    assert (
        model.peak_logical_qubits
        == get_cost_value(pinned, QubitCount())
        == op.qubit_count
    )
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert events == (9 if measured else 6)
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["cx"] == (70 if measured else 52)
    assert counts["phase"] == 3 * measured and counts["cphase"] == 3 * (not measured)
    initial = torch.zeros(64, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for address, bit in product(range(4), (0, 1)):
        i = address << 4 | bit << 1
        amplitude = (1 + (address + 0.25) * 1j) * (0.5 if bit else 1)
        exponent = sum(2.0 ** (-1 - j) for j in range(3) if (address + 1) >> j & 1)
        initial[i] = torch.tensor(amplitude, dtype=torch.complex128)
        expected[i] = torch.tensor(
            amplitude * exp(1j * pi * exponent * bit), dtype=torch.complex128
        )
    norm = initial.norm()
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial / norm, gates, history),
            expected / norm / sqrt(2**events),
            rtol=0,
            atol=2e-12,
        )


@pytest.mark.parametrize(
    "angles",
    [
        ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)),
        ((3, 4, 5), (10, 11, 12)),
    ],
)
@pytest.mark.parametrize("kappa", range(1, 12))
@pytest.mark.parametrize("kind", ["custom", "phase", "identity"])
def test_original_constructor_grid(
    angles: tuple[tuple[int, ...], ...],
    kappa: int,
    kind: str,
    source_module: Any,
) -> None:
    import cirq
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.rotations import programmable_rotation_gate_array_test as cases

    constructors = {
        "custom": cases.construct_custom_prga,
        "phase": cases.construct_prga_with_phase,
        "identity": cases.construct_prga_with_identity,
    }
    source = constructors[kind](*angles, kappa=kappa, rotation_gate=cirq.X)
    extent = len(angles[0])
    select = tuple(range((extent - 1).bit_length()))
    load = tuple(range(len(select), len(select) + kappa))
    target = (len(select) + kappa,)
    extras = (target[0] + 1,) if kind == "custom" else ()
    ports = {"selection": select, "kappa_load_target": load, "rotations_target": target}
    if extras:
        ports["unrelated_target"] = extras
    children = source_bound_ops(source, **ports)
    tables = [
        tuple(map(int, op.gate.data[0])) for op in children if isinstance(op.gate, QROM)
    ]
    size = max(
        source_qrom_workspace_size(0, (len(select),), (data,)) for data in tables
    )
    offset = target[0] + 1 + len(extras)
    work = tuple(range(offset, offset + size))
    width = offset + size

    def matrix_gate(gate: Any, wires: tuple[int, ...]) -> Instruction:
        return Instruction("any", wires, matrix=cirq.unitary(gate).tolist())

    interleavers = (
        (cirq.CNOT, cirq.CZ)
        if extras
        else ((cirq.Z, cirq.Z) if kind == "phase" else (cirq.I, cirq.I))
    )
    gates = programmable_rotation_array_ir(
        select,
        load,
        target,
        angles,
        work,
        tuple(range(size)),
        rotation=lambda control, wires, power: (
            matrix_gate(
                cirq.ControlledGate(cirq.X ** (2.0 ** (-1 - power))), (control, *wires)
            ),
        ),
        interleaved=tuple(
            (matrix_gate(g, (*target, *extras)),)
            for g in interleavers[: len(angles) - 1]
        ),
        interleaved_target=extras,
    )

    def qrom_addresses(address: int, word: int) -> tuple[int, ...]:
        return tuple(
            int(bit)
            for bit in f"{address:0{len(select)}b}{word:0{kappa}b}"
            + "0" * (width - len(select) - kappa)
        )

    source_gates: list[Instruction] = []
    current = (0,) * extent
    for child in children:
        gate = child.gate
        if isinstance(gate, QROM):
            segment = qrom_gates(gate, select, load, work)
            expected = tuple(
                old ^ int(delta)
                for old, delta in zip(current, gate.data[0], strict=True)
            )
            assert certified_and_channel_map(
                segment,
                tuple(qrom_addresses(i, old) for i, old in enumerate(current)),
            ) == (
                tuple(qrom_addresses(i, new) for i, new in enumerate(expected)),
                sum(g.name == "measure" for g in segment),
            )
            current = expected
            source_gates.extend(segment)
        else:
            physical = tuple(q.x for q in child.qubits)
            source_gates.append(matrix_gate(gate, physical))
    assert current == (0,) * extent
    assert_source_dependencies(gates, tuple(source_gates))

    # Compare the whole coherent state to independent per-address multiplexer matrices.
    target_width = 1 + len(extras)
    dimension = 1 << target_width
    norm = sqrt(sum(abs(1 + 0.3j * j) ** 2 for j in range(dimension)) * extent)
    initial: dict[int, complex] = {}
    expected_state: dict[int, complex] = {}
    for address in range(extent):
        angles_state = np.array(
            [exp(0.13j * address) * (1 + 0.3j * j) / norm for j in range(dimension)]
        )
        indices = tuple(
            (address << (width - len(select))) | (j << size) for j in range(dimension)
        )
        initial.update(zip(indices, map(complex, angles_state), strict=True))
        for i, row in enumerate(angles):
            theta = sum(
                2.0 ** (-1 - k)
                for k in range(row[address].bit_length())
                if row[address] >> k & 1
            )
            matrix = cirq.unitary(cirq.X**theta)
            if extras:
                matrix = np.kron(matrix, np.eye(2))
            angles_state = matrix @ angles_state
            if i < len(angles) - 1:
                angles_state = cirq.unitary(interleavers[i]) @ angles_state
        expected_state.update(zip(indices, map(complex, angles_state), strict=True))
    for seed in (4, 23):
        actual = execute_sparse_register(
            CircuitIR(width, gates), dict(initial), seed=seed
        )
        assert all(
            abs(actual.get(i, 0) - expected_state.get(i, 0)) < 2e-12
            for i in actual.keys() | expected_state.keys()
        )


@pytest.mark.parametrize(
    "work, slot, message",
    [
        ((3,), 0, "drawn from the QROM workspace"),
        ((4,), 1, "undeclared workspace slot"),
    ],
)
def test_rotation_cannot_escape_its_shared_work(
    work: tuple[int, ...],
    slot: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        programmable_rotation_array_ir(
            (0, 1),
            (2,),
            (3,),
            ((0, 1, 0, 1),),
            (4,),
            (0,),
            rotation=lambda *_: (
                Instruction(
                    "measure",
                    (4,),
                    metadata={"is_dynamic": True, "classical_bit": slot},
                ),
            ),
            interleaved=(),
            rotation_workspace=work,
        )
