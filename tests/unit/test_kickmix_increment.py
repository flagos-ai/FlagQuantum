"""Verify source increment order, complete measured branches and clean carries."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.arithmetic.increment import kickmix_increment_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def test_emitted_streams_match_locked_source_fixtures() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/kickmix_increment.json").read_text()
    )
    assert fixture["source_blob"] == "890b484fb6043d45d8e60a91dd241ab8092f7b21"
    assert len(fixture["cases"]) == 68
    for case in fixture["cases"]:
        n, controlled, extra = case["width"], case["controlled"], case["extra_clean"]
        needed = max(0, n - (2 if controlled else 3))
        gates = kickmix_increment_ir(
            tuple(reversed(range(n))),
            control=n if controlled else True,
            clean=tuple(range(n + 1, n + 1 + needed + extra)),
            measurement_bit=0,
        )
        total = n + 1 + needed + extra
        lines = encode_kmx(
            lower_reversible_phase(CircuitIR(total, gates))
        ).splitlines()[1:]
        assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == case["sha256"]
        assert dict(Counter(line.split()[0] for line in lines)) == case["opcodes"]


@pytest.mark.parametrize("n,controlled", tuple(product(range(1, 6), (False, True))))
def test_all_complex_branches_restore_clean_work(n: int, controlled: bool) -> None:
    needed = max(0, n - (2 if controlled else 3))
    total = n + 1 + needed + 1  # one unused clean wire must stay unchanged
    generator = torch.Generator().manual_seed(127 + n)
    amplitudes = torch.complex(
        torch.randn(1 << (n + 1), generator=generator, dtype=torch.float64),
        torch.randn(1 << (n + 1), generator=generator, dtype=torch.float64),
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[:: 1 << (needed + 1)] = amplitudes
    gates = kickmix_increment_ir(
        tuple(range(n)),
        control=n if controlled else True,
        clean=tuple(range(n + 1, total)),
        measurement_bit=0,
    )
    expected = torch.zeros_like(initial)
    for word in range(1 << (n + 1)):
        value, enable = word >> 1, word & 1
        updated = (value + (enable if controlled else 1)) % (1 << n)
        expected[((updated << 1) | enable) << (needed + 1)] = amplitudes[word]
    program = lower_reversible_phase(CircuitIR(total, gates))
    counts = Counter(op.opcode for op in program.operations)
    assert counts["hmr"] == needed
    effective = n if controlled else n - 1
    assert counts["ccx"] == max(0, effective - 1)
    assert counts["cz"] == needed
    for outcomes in product((0, 1), repeat=needed):
        branch = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            branch, expected / (2 ** (needed / 2)), atol=2e-14, rtol=2e-14
        )
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(total, gates)
    )
    result = run_dynamic(circuit, shots=16, seed=83, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=2e-14,
        rtol=2e-14,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clean": ()},
        {"clean": (4, 5), "measurement_bit": None},
        {"clean": (4, 5), "measurement_bit": -1},
        {"clean": (4, 5), "measurement_bit": True},
        {"clean": (1, 5), "measurement_bit": 0},
        {"clean": (4, 5), "measurement_bit": 0, "control": False},
    ],
)
def test_invalid_increment_bindings_refuse_before_emission(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        kickmix_increment_ir((0, 1, 2, 3), **kwargs)


def test_empty_increment_has_no_work_or_measurement() -> None:
    assert kickmix_increment_ir(()) == ()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,controlled", tuple(product(range(7), (False, True))))
def test_native_all_basis_words_keep_absolute_phase_and_clean_work(
    n: int,
    controlled: bool,
) -> None:
    from fractions import Fraction

    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    needed = max(0, n - (2 if controlled else 3))
    total = n + 1 + needed
    gates = kickmix_increment_ir(
        tuple(range(n)),
        control=n if controlled else True,
        clean=tuple(range(n + 1, total)),
        measurement_bit=0,
    )
    artifact = lower_reversible_phase(CircuitIR(total, gates))
    shots = 1 << (n + 1)
    sim = km.Simulator(batch_size=shots, seed=29)
    for wire in range(n + 1):
        sim.write_across_shots(
            km.q(wire), sum(((shot >> (n - wire)) & 1) << shot for shot in range(shots))
        )
    sim.do(km.Circuit(encode_kmx(artifact)))
    columns = [sim.read_across_shots(km.q(wire), out=int) for wire in range(total)]
    for shot in range(shots):
        value, enable = shot >> 1, shot & 1
        updated = (value + (enable if controlled else 1)) % (1 << n)
        expected = (
            [((updated >> (n - 1 - wire)) & 1) for wire in range(n)]
            + [enable]
            + [0] * needed
        )
        assert [(column >> shot) & 1 for column in columns] == expected
        assert Fraction(sim.read_shot_phase(shot)) % 2 == 0


def test_append_example() -> None:
    from examples.algorithms.measured_register_increment import main

    main()
