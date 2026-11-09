"""Check locked controlled-add streams, mixed carry semantics and branch phases."""

import hashlib
import json
from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.arithmetic.carry_controlled_addition import (
    carry_controlled_addition_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit
MODES = (
    (0, 0, False),
    (1, 0, False),
    (2, 0, False),
    (3, 0, False),
    (3, 1, False),
    (0, 0, True),
)


def _instructions(
    n: int,
    m: int,
    kind: int,
    bit_value: int,
    subtract: bool,
    extra: int = 0,
    *,
    source_order: bool = False,
) -> tuple[Instruction, ...]:
    inputs = n + m + 2
    carry: int | bool | ClassicalFieldCoefficient = bool(kind)
    if kind == 0:
        carry = False
    elif kind == 2:
        carry = inputs - 1
    elif kind == 3:
        carry = ClassicalFieldCoefficient(0)
    return (
        ClassicalXor((), 0, bool(bit_value)).instruction(),
        *carry_controlled_addition_ir(
            tuple(reversed(range(n))) if source_order else tuple(range(n)),
            (
                tuple(reversed(range(n, n + m)))
                if source_order
                else tuple(range(n, n + m))
            ),
            control=n + m,
            carry_in=carry,
            clean=tuple(range(inputs, inputs + (n if n >= 2 else 0) + extra)),
            measurement_bit=1,
            subtract=subtract,
        ),
    )


def test_all_locked_source_stream_digests() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_controlled_addition.json")
        .read_text()
    )
    assert fixture["source_blob"] == "ee8062ae063ce241383bdb633ead74c4dafc5db1"
    assert len(fixture["cases"]) == 180
    for case in fixture["cases"]:
        n, tail, extra = case["width"], case["offset_tail"], case["extra_clean"]
        m = n + tail
        instructions = _instructions(
            n, m, case["carry_kind"], 0, case["subtract"], extra, source_order=True
        )
        artifact = lower_reversible_phase(
            CircuitIR(n + m + 2 + (n if n >= 2 else 0) + extra, instructions)
        )
        lines = encode_kmx(artifact).splitlines()[
            2:
        ]  # header and caller's b0 initialization
        stream = normalized_stream("\n".join(lines))
        assert (
            hashlib.sha256(json.dumps(stream).encode()).hexdigest() == case["sha256"]
        ), case
        assert (
            dict(Counter(primitive.split()[0] for primitive, _ in stream))
            == case["opcodes"]
        )


def _expected_indices(
    n: int, m: int, kind: int, bit_value: int, subtract: bool
) -> torch.Tensor:
    words = torch.arange(1 << (n + m + 2))
    lhs, rhs = words >> (m + 2), (words >> 2) & ((1 << m) - 1)
    enable = (words >> 1) & 1
    carry = words & 1 if kind == 2 else bit_value if kind == 3 else int(kind == 1)
    updated = (lhs + (-rhs if subtract else rhs + carry) * enable) % (1 << n)
    return (updated << (m + 2)) | (words & ((1 << (m + 2)) - 1))


@pytest.mark.parametrize("n,m", [(1, 1), (2, 2), (2, 4), (3, 3)])
@pytest.mark.parametrize("kind,bit_value,subtract", MODES)
def test_every_complex_branch_restores_inputs_and_carries(
    n: int, m: int, kind: int, bit_value: int, subtract: bool
) -> None:
    inputs = n + m + 2
    work = n if n >= 2 else 0
    total = inputs + work + 1  # preserve one extra unused clean wire
    generator = torch.Generator().manual_seed(409)
    amplitudes = torch.complex(
        torch.randn(1 << inputs, generator=generator, dtype=torch.float64),
        torch.randn(1 << inputs, generator=generator, dtype=torch.float64),
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[:: 1 << (work + 1)] = amplitudes
    expected = torch.zeros_like(initial)
    expected[_expected_indices(n, m, kind, bit_value, subtract) << (work + 1)] = (
        amplitudes
    )
    gates = _instructions(n, m, kind, bit_value, subtract, extra=1)
    events = max(0, n - 1)
    artifact = lower_reversible_phase(CircuitIR(total, gates))
    counts = Counter(op.opcode for op in artifact.operations)
    assert counts["hmr"] == events
    if n >= 2:
        assert counts["ccx"] == 2 * n - 1
        assert counts["cz"] == n - 1
    for outcomes in product((0, 1), repeat=events):
        branch = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            branch, expected / (2 ** (events / 2)), atol=2e-14, rtol=2e-14
        )
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(total, gates)
    )
    result = run_dynamic(circuit, shots=16, seed=51, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=2e-14,
        rtol=2e-14,
    )
    assert torch.all(result.classical_bits[:, 0] == bool(bit_value))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", range(5))
@pytest.mark.parametrize("kind,bit_value,subtract", MODES)
def test_native_full_basis_and_absolute_phase(
    n: int, kind: int, bit_value: int, subtract: bool
) -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    m = n + 2 if n == 2 else n
    inputs = n + m + 2
    work = n if n >= 2 else 0
    artifact = lower_reversible_phase(
        CircuitIR(inputs + work, _instructions(n, m, kind, bit_value, subtract))
    )
    shots = 1 << inputs
    sim = km.Simulator(batch_size=shots, seed=47)
    for wire in range(inputs):
        sim.write_across_shots(
            km.q(wire),
            sum(((shot >> (inputs - 1 - wire)) & 1) << shot for shot in range(shots)),
        )
    sim.do(km.Circuit(encode_kmx(artifact)))
    columns = [
        sim.read_across_shots(km.q(wire), out=int) for wire in range(inputs + work)
    ]
    indices = _expected_indices(n, m, kind, bit_value, subtract).tolist()
    for shot, updated in enumerate(indices):
        expected = [
            ((updated >> (inputs - 1 - wire)) & 1) for wire in range(inputs)
        ] + [0] * work
        assert [(column >> shot) & 1 for column in columns] == expected
        assert Fraction(sim.read_shot_phase(shot)) % 2 == 0
    assert sim.read_across_shots(km.b(0), out=int) == (
        (1 << shots) - 1 if bit_value else 0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control": True},
        {"control": -1},
        {"control": 0},
        {"control": 4, "clean": ()},
        {"control": 4, "clean": (5, 6), "measurement_bit": None},
        {"control": 4, "clean": (5, 6), "measurement_bit": True},
        {
            "control": 4,
            "clean": (5, 6),
            "measurement_bit": 0,
            "carry_in": ClassicalFieldCoefficient(0),
        },
        {
            "control": 4,
            "clean": (5, 6),
            "measurement_bit": 0,
            "carry_in": True,
            "subtract": True,
        },
    ],
)
def test_invalid_bindings_refuse(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        carry_controlled_addition_ir((0, 1), (2, 3), **kwargs)


def test_short_offset_refuses_like_source() -> None:
    with pytest.raises(ValueError, match="at least as wide"):
        carry_controlled_addition_ir((0, 1), (2,), control=3)


def test_append_example() -> None:
    from examples.algorithms.controlled_carry_addition import main

    main()


@pytest.mark.parametrize("carry", [2.0, None])
def test_nonintegral_carry_is_not_silently_dropped(carry: object) -> None:
    with pytest.raises(ValueError, match="Carry must"):
        carry_controlled_addition_ir((0,), (1,), control=2, carry_in=carry)


def test_integer_subclass_is_not_silently_dropped() -> None:
    class WireSubclass(int):
        pass

    with pytest.raises(ValueError, match="Carry must"):
        carry_controlled_addition_ir((0,), (1,), control=2, carry_in=WireSubclass(3))
