"""Verify the pinned three-clean-wire comparison and its branch cleanup."""

import json
from collections import Counter
from hashlib import sha256
from random import Random

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.comparison.low_space_comparison import (
    low_space_less_than_ir,
)
from flagquantum.compiler.registers.comparison.mixed_comparison import (
    mixed_less_than_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "width,phase_target", [(2, False), (3, True), (4, False), (4, True)]
)
def test_comparison_restores_every_complex_measurement_history(
    width: int, phase_target: bool
) -> None:
    # Controls and target (a spectator in phase mode) can both be entangled.
    control, target = width, width + 1
    workspace = tuple(range(width + 2, width + 5))
    witness = width + 5
    total = witness + 1
    basis = torch.arange(1 << (width + 2))
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::16] = amplitudes
    ir = low_space_less_than_ir(
        tuple(range(width)),
        (ClassicalFieldCoefficient(0), True),
        None if phase_target else target,
        workspace,
        control=control,
        or_equal=True,
        inverted=True,
        measurement_bits=(1, 2, 3),
    )
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    recorded = []
    slots: list[int] = []
    for gate in (*measure_x_clear_ir(witness, 0), *ir):
        recorded.append(gate)
        if gate.name == "measure":
            slot = 8 + len(slots)
            slots.append(slot)
            recorded.append(
                ClassicalXor((gate.metadata["classical_bit"],), slot).instruction()
            )
    circuit.append(FieldOperation(total, tuple(recorded)))
    result = run_dynamic(circuit, shots=128, seed=41, strategy="trajectory")
    histories = result.classical_bits[:, slots]
    assert len(set(map(tuple, histories.tolist()))) == 1 << len(slots)
    for coefficient in (0, 1):
        enabled = ((basis >> 2) > 2 * coefficient + 1) & (((basis >> 1) & 1) == 1)
        expected = torch.zeros_like(state)
        if phase_target:
            expected[::16] = amplitudes * (1 - 2 * enabled.long())
        else:
            expected[::16] = amplitudes[basis ^ enabled.long()]
        actual = result.final_states.reshape(128, -1)[histories[:, 0] == coefficient]
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=0, atol=1e-12
        )


def test_dispatch_matches_native_eight_bit_resource_counts() -> None:
    left = tuple(range(8))
    right = tuple(ClassicalFieldCoefficient(bit) for bit in reversed(range(8)))
    clean = (9, 10, 11)
    slots = (9, 10, 11)
    equal = ClassicalFieldCoefficient(8)
    direct = low_space_less_than_ir(
        left, right, 8, clean, measurement_bits=slots, or_equal=equal
    )
    dispatched = mixed_less_than_ir(
        left, right, 8, clean, low_space_bits=slots, or_equal=equal
    )
    assert direct == dispatched
    initialize = tuple(ClassicalXor((), bit).instruction() for bit in range(9))
    program = lower_reversible_phase(CircuitIR(12, (*initialize, *direct)))
    counts = Counter(op.opcode for op in program.operations[len(initialize) :])
    # Locked comparator: 13 Toffoli, 5 HMR, 3 clean wires. Its scope
    # wrappers and resets are represented separately.
    assert counts["ccx"] == 13
    assert counts["hmr"] == 5
    assert program.n_bits == 12


def test_deep_prefix_keeps_conditional_ccz_and_checks_slot_count() -> None:
    left = tuple(range(20))
    right = (False,) * 20
    with pytest.raises(ValueError, match="6 result slots"):
        low_space_less_than_ir(
            left, right, 20, (21, 22, 23), measurement_bits=tuple(range(5))
        )
    ir = low_space_less_than_ir(
        left, right, 20, (21, 22, 23), measurement_bits=tuple(range(6))
    )
    program = lower_reversible_phase(CircuitIR(24, ir))
    assert any(
        op.opcode == "ccz" and op.condition_bit is not None for op in program.operations
    )
    assert program.n_bits == 6


@pytest.mark.ftqc_reference
def test_deep_classical_compare_runs_with_native_phase_and_clean_workspace() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"} and hasattr(km, "Simulator")
    width, shots = 20, 64
    rng = Random(20261107)
    left = [rng.getrandbits(width) for _ in range(shots)]
    right = [value ^ rng.randrange(4) for value in left]
    enabled = [rng.randrange(2) for _ in range(shots)]
    source = tuple(range(width))
    rhs = tuple(ClassicalFieldCoefficient(i) for i in range(width))
    control, workspace = width, (width + 1, width + 2, width + 3)
    ir = low_space_less_than_ir(
        source,
        rhs,
        None,
        workspace,
        or_equal=True,
        control=control,
        measurement_bits=tuple(range(width, width + 6)),
    )
    initialize = tuple(ClassicalXor((), bit).instruction() for bit in range(width))
    program = lower_reversible_phase(CircuitIR(width + 4, (*initialize, *ir)))
    text = encode_kmx(program).splitlines()[width + 1 :]
    sim = km.Simulator(batch_size=shots, seed=41)

    def pack(column: list[int]) -> int:
        return sum(value << shot for shot, value in enumerate(column))

    for wire in source:
        shift = width - 1 - wire
        sim.write_across_shots(
            km.q(wire), pack([(value >> shift) & 1 for value in left])
        )
        sim.write_across_shots(
            km.b(wire), pack([(value >> shift) & 1 for value in right])
        )
    sim.write_across_shots(km.q(control), pack(enabled))
    sim.do(km.Circuit("\n".join(text)))
    assert [sim.read_shot_phase(i) for i in range(shots)] == [
        int(bool(enable) and lhs <= rhs_value)
        for lhs, rhs_value, enable in zip(left, right, enabled, strict=True)
    ]
    for wire in source:
        shift = width - 1 - wire
        assert sim.read_across_shots(km.q(wire), out=int) == pack(
            [(value >> shift) & 1 for value in left]
        )
    assert sim.read_across_shots(km.q(control), out=int) == pack(enabled)
    assert all(sim.read_across_shots(km.q(wire), out=int) == 0 for wire in workspace)


@pytest.mark.parametrize(
    "width,operations,expected_digest",
    [
        (5, 54, "29cb8479258bc9d3df2774188b541c60a24f737fc1040033e434e10d43bb7277"),
        (8, 95, "a81f9d69ee639f371555d43b5692f8151b28d2a4dad195edf8a69c7ca50669d9"),
        (13, 164, "eade4676069e92149c45753d06e7bbac1ca53ae87974ad567e9d3b8a08a0ed05"),
        (19, 224, "f1d541295341809ee02dc16e9869b62709fda87f70816c89b7fd54e957006dc2"),
        (24, 312, "5d7d9b62308eb19c29b42df6a665ab87c66b5e07960143e622b34efc90ce9e2d"),
    ],
)
def test_strided_prefix_round_trip_matches_locked_source_schedule(
    width: int, operations: int, expected_digest: str
) -> None:
    # C++ gen_cmp_low_space.cc blob 7de1b7a8bd5ba36226385d7c319a5260635ebd8e.
    # The locked builder was invoked with little-endian inputs, three clean
    # wires, one external equality bit and no additional quantum control.
    source_order = tuple(reversed(range(width)))
    initial = tuple(ClassicalXor((), bit).instruction() for bit in range(width + 1))
    emitted = low_space_less_than_ir(
        source_order,
        tuple(ClassicalFieldCoefficient(bit) for bit in source_order),
        width,
        tuple(range(width + 1, width + 4)),
        or_equal=ClassicalFieldCoefficient(width),
        measurement_bits=tuple(range(width + 1, width + 7)),
    )
    program = lower_reversible_phase(CircuitIR(width + 4, (*initial, *emitted)))
    lowered = encode_kmx(program).splitlines()[len(initial) + 1 :]
    assert not any(line.startswith("R ") for line in lowered)
    stream = normalized_stream("\n".join(lowered))
    assert len(stream) == operations
    digest = sha256(json.dumps(stream, separators=(",", ":")).encode()).hexdigest()
    assert digest == expected_digest
