"""Replay locked sampled counts and matched phase trajectories."""

import json
from pathlib import Path
from random import Random

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128
from flagquantum.compiler.formats.kmb import encode_kmb
from flagquantum.compiler.formats.kmx import decode_kmx
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.formats.kmx_sampled_report import format_sampled_counts
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.kmx_sampled_counts import (
    main,
    parse_register_initialization,
    sampled_operation_counts,
)
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.physical_basis_shots import sample_basis_batch

_FIXTURES = Path(__file__).parent / "fixtures"
_CASES = json.loads((_FIXTURES / "kickmix_sampled_counts.json").read_text())


@pytest.mark.parametrize("name", [name for name in _CASES["reports"] if name != "hmr"])
def test_locked_sampled_report_exact(name: str) -> None:
    case = _CASES["reports"][name]
    circuit = read_kmx_circuit(case["kmx"])
    counts = sampled_operation_counts(
        circuit, case["shots"], init=case.get("init", ""), rng=Random(23)
    )
    assert (
        format_sampled_counts(circuit, counts, case["shots"], groups=True)
        == case["grouped"]
    )


def test_hmr_complementary_counts_keep_the_exact_total() -> None:
    case = _CASES["reports"]["hmr"]
    circuit = read_kmx_circuit(case["kmx"])
    counts = sampled_operation_counts(circuit, 193, rng=Random(17))
    assert 0 < counts["X"] < 193
    assert counts["X"] + counts["Z"] == 193
    assert counts["HMR"] == counts["BIT_INVERT"] == 193
    assert sum(counts.values()) == 579


def test_source_matched_histories_preserve_complex_amplitudes_and_work() -> None:
    reference = _CASES["basis"]
    kmx: str = reference["kmx"]
    rows: list[list[str]] = reference["rows"]
    masks = [
        sum(int(row[bit]) << shot for shot, row in enumerate(rows)) for bit in (0, 1)
    ]
    outcomes = iter([*masks, 0])
    circuit = read_kmx_circuit(kmx)
    batch = sample_basis_batch(
        circuit, shots=64, rng=Random(0), outcome_mask=lambda: next(outcomes)
    )
    assert tuple(batch.bits[bit] for bit in (0, 1)) == tuple(masks)
    assert (
        batch.counts["X"]
        == 64 + masks[0].bit_count() + (masks[0] & masks[1]).bit_count()
    )
    assert batch.counts["HMR"] == 128
    assert batch.counts["NEG"] == masks[1].bit_count()
    expected = {}
    for shot, (b0, b1, basis, angle) in enumerate(rows):
        assert "".join(str((batch.qubits[q] >> shot) & 1) for q in range(3)) == basis
        exact = PhaseAngle128.from_rounded_decimal_half_turns(angle)
        assert batch.phase_units[shot] == exact.units
        state = torch.zeros(8, dtype=torch.complex128)
        state[int(basis, 2)] = torch.exp(
            torch.tensor(
                1j * torch.pi * exact.signed_half_turns_float(), dtype=torch.complex128
            )
        )
        expected[int(b0), int(b1)] = state
    assert len(expected) == 4 and batch.qubits[2] == 0

    header = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=3 bits=2"'
    program = decode_kmx(f"{header}\n{kmx}")
    dynamic = DynamicCircuit(3, dtype=torch.complex128)
    dynamic.append(FieldOperation(3, bind_reversible_phase_execution(program)))
    result = run_dynamic(dynamic, shots=64, seed=92, strategy="batched")
    actual = result.final_states.reshape(64, -1)
    for i, bits in enumerate(result.classical_bits.tolist()):
        torch.testing.assert_close(
            actual[i], expected[bits[0], bits[1]], atol=1e-12, rtol=0
        )
    assert not torch.count_nonzero(actual[:, 1::2])


def test_original_all_operation_file_and_binary_cli(tmp_path: Path) -> None:
    text = json.loads((_FIXTURES / "kickmix_circuit_ownership.json").read_text())[
        "canonical"
    ]
    source, binary, out = (tmp_path / name for name in ("in.kmx", "in.kmb", "stats"))
    source.write_text(text)
    binary.write_bytes(encode_kmb(text))
    for path in (source, binary):
        assert main(["--shots", "64", "--in", str(path), "--out", str(out)]) == 0
        lines = out.read_text().splitlines()
        assert len(lines) == 20
        assert [value.strip() for value in lines[-1].split(",")] == [
            "PUSH_CONDITION",
            "128",
            "2.0",
            "100.0%",
        ]


def test_register_random_initialization_preserves_bounds_and_overwrites() -> None:
    circuit = read_kmx_circuit(
        "APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER b0 r0\nAPPEND_TO_REGISTER q0 r1"
    )
    init = parse_register_initialization(circuit, "r0<3,r1=1,")
    batch = sample_basis_batch(circuit, shots=64, rng=Random(31), initialization=init)
    assert batch.qubits[0] == (1 << 64) - 1
    assert 0 < batch.bits[0].bit_count() < 64
    init = parse_register_initialization(circuit, "r0=*")
    sample = sample_basis_batch(circuit, shots=64, rng=Random(31), initialization=init)
    assert 0 < sample.qubits[0].bit_count() < 64
    assert 0 < sample.bits[0].bit_count() < 64


@pytest.mark.parametrize("init", ["r99=0", "r0=", "r0<0x0", "r0=-1", "r0<3"])
def test_bad_initialization_is_rejected(init: str) -> None:
    circuit = read_kmx_circuit("APPEND_TO_REGISTER q0 r0")
    with pytest.raises(ValueError):
        sampled_operation_counts(circuit, 64, init=init, rng=Random(0))
