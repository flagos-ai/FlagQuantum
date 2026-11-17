"""Certify direct specialized comparison streams and masked-prefix diagrams."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from flagquantum.compiler.formats.kmb import KmxRegister
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.comparison.physical_comparison import (
    physical_low_space_less_than,
    physical_quantum_less_than,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.phase.masked_prefix_phase import (
    physical_masked_prefix_phase,
)
from flagquantum.drawer.reversible_phase_diagram import render_circuit_text
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

_Case = tuple[str, int, int, int, int, int, int, int, str, str, int, str]


class _Diagram(TypedDict):
    clean: int
    kmx: str
    diagram: str


class _Fixture(TypedDict):
    source_blob: str
    quantum_blob: str
    cases: list[_Case]
    diagrams: list[_Diagram]


_SOURCE: _Fixture = json.loads(
    Path(__file__)
    .with_name("fixtures")
    .joinpath("kickmix_specialized_comparison_file.json")
    .read_text()
)


def test_complete_specialized_physical_streams() -> None:
    assert _SOURCE["source_blob"] == "7de1b7a8bd5ba36226385d7c319a5260635ebd8e"
    assert _SOURCE["quantum_blob"] == "7c618915a2b61ea26472652cf88bc79fe17c4096"
    for case in _SOURCE["cases"]:
        mode, n, m, work, phase, ctrl, minimize, classical, eq, inv, size, digest = case
        lhs = tuple(reversed(range(n)))
        cursor = n + (m if mode == "qq" else 0)
        bit = m if classical else 0
        out = None if phase else cursor
        cursor += int(not phase)
        enable = cursor if ctrl else True
        cursor += ctrl

        def operand(kind: str) -> int | bool | Bit:
            nonlocal bit, cursor
            if kind == "q":
                cursor += 1
                return cursor - 1
            if kind == "b":
                bit += 1
                return Bit(bit - 1)
            return kind == "1"

        equal, inverted = operand(eq), operand(inv)
        clean = tuple(range(cursor, cursor + work))
        if mode == "qq":
            program, calls = physical_quantum_less_than(
                lhs,
                tuple(reversed(range(n, n + m))),
                out,
                clean,
                or_equal=equal,
                control=enable,
                minimize_qubits=bool(minimize),
            )
        else:
            right = tuple(
                Bit(i) if classical else i % 2 == 0 for i in reversed(range(m))
            )
            program, calls = physical_low_space_less_than(
                lhs,
                right,
                out,
                clean,
                or_equal=equal,
                inverted=inverted,
                control=enable,
            )
        text = encode_kmx_block_marks(program, calls)
        lines = text.splitlines()[1:]
        assert len(lines) == size, case
        assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == digest, case
        assert decode_kmx_block_marks(text) == (program, calls)


@pytest.mark.parametrize(
    "case", _SOURCE["diagrams"], ids=lambda case: f"clean-{case['clean']}"
)
def test_masked_prefix_original_diagrams(case: _Diagram) -> None:
    k = case["clean"]
    program, calls = physical_masked_prefix_phase(
        tuple(range(10)),
        tuple(Bit(i) for i in range(10)),
        clean=tuple(range(10, 10 + k)),
        dirty=tuple(range(10 + k, 17)),
    )
    declarations = (
        *KmxRegister("", tuple(("q", q) for q in range(10))).to_kmx(0),
        *KmxRegister("", tuple(("b", q) for q in range(10))).to_kmx(1),
    )
    _, _, body = encode_kmx_block_marks(program, calls).partition("\n")
    circuit = read_kmx_circuit("\n".join((*declarations, body)))
    assert circuit.to_kmx() == case["kmx"]
    assert render_circuit_text(circuit) == case["diagram"]
    counts = Counter(op.opcode for op in program.operations)
    assert counts["r"] == counts["hmr"] == k


@pytest.mark.parametrize("phase", (False, True))
def test_direct_quantum_comparator_all_physical_branches(phase: bool) -> None:
    program, _ = physical_quantum_less_than(
        (0, 1),
        (2, 3),
        None if phase else 4,
        (6, 7, 8),
        control=5,
        or_equal=True,
    )
    indices = torch.arange(64)
    amplitudes = torch.complex((indices + 1).double(), (7 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    initial[indices << 3] = amplitudes
    selected = ((indices >> 4) <= ((indices >> 2) & 3)) & ((indices & 1) == 1)
    expected = torch.zeros_like(initial)
    if phase:
        expected[indices << 3] = amplitudes * (1 - 2 * selected.int())
    else:
        expected[(indices ^ (selected.int() << 1)) << 3] = amplitudes
    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=program.n_bits + 1
    )
    assert sum(gate.name == "measure" for gate in gates) == 2
    for outcomes in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / 2,
            atol=5e-14,
            rtol=5e-14,
        )


def test_short_classical_rhs_and_inverted_output_all_branches() -> None:
    program, _ = physical_low_space_less_than(
        (0, 1),
        (True,),
        3,
        (4, 5, 6),
        inverted=True,
        control=2,
    )
    indices = torch.arange(16)
    amplitudes = torch.complex((indices + 1).double(), (7 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[indices << 3] = amplitudes
    selected = ((indices >> 2) >= 1) & ((indices & 2) != 0)
    expected = torch.zeros_like(initial)
    expected[(indices ^ selected.int()) << 3] = amplitudes
    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=program.n_bits + 1
    )
    events = sum(gate.name == "measure" for gate in gates)
    assert events == 6
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / 2 ** (events / 2),
            atol=5e-14,
            rtol=5e-14,
        )


def test_masked_prefix_restores_entangled_dirty_wire_in_every_branch() -> None:
    program, _ = physical_masked_prefix_phase(
        (0, 1, 2, 3, 4),
        (True,) * 5,
        clean=(6,),
        dirty=(5,),
    )
    indices = torch.arange(64)
    amplitudes = torch.complex((indices + 1).double(), (7 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[indices << 1] = amplitudes
    signs = torch.ones(64)
    for prefix in range(1, 6):
        ones = (indices >> (6 - prefix)) == ((1 << prefix) - 1)
        signs *= 1 - 2 * ones.int()
    expected = torch.zeros_like(initial)
    expected[indices << 1] = amplitudes * signs
    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=program.n_bits + 1
    )
    assert sum(gate.name == "measure" for gate in gates) == 2
    for outcomes in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / 2,
            atol=5e-14,
            rtol=5e-14,
        )
