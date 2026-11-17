"""Exercise the physical generator against an immutable locked CLI capture."""

import base64
import json
from itertools import product
from math import ceil, inf
from pathlib import Path
from typing import TypedDict
from xml.etree.ElementTree import Element, fromstring

import pytest
import torch

from flagquantum.compiler._component_trace.trace import ComponentTrace
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.arithmetic.physical_addition import (
    physical_classical_addition,
)
from flagquantum.compiler.registers.comparison.physical_comparison import (
    physical_less_than,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.modular.modular_doubling_dispatch import (
    modular_doubling_program,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.drawer.kmx_generate_command import main
from tests.unit.measurement_checks import expand_reset_branches, forced_branch


class _Case(TypedDict):
    argv: list[str]
    kmx: str
    kmb: str


_SOURCE = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_generation_command.json").read_text()
)
_CASES: list[_Case] = _SOURCE["cases"]


def _svg(node: Element) -> tuple[str, dict[str, str], str, list[object]]:
    return node.tag, node.attrib, (node.text or "").strip(), [_svg(x) for x in node]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: " ".join(case["argv"]))
def test_generated_bytes(case: _Case, tmp_path: Path) -> None:
    out = tmp_path / "circuit"
    argv = ["--kind", *case["argv"], "--out", str(out), "--validate"]
    assert main(argv) == 0
    assert out.read_bytes() == case["kmx"].encode()
    assert main([*argv, "--kmb"]) == 0
    assert out.read_bytes() == base64.b64decode(case["kmb"])


def test_diagram_precedes_kmb_and_analysis_survives_skip(tmp_path: Path) -> None:
    out, analysis = tmp_path / "circuit", tmp_path / "analysis"
    argv = ["--kind", "iadd", "--n", "3", "--out", str(out), "--kmb", "--diagram"]
    assert main(argv) == 0
    expected = _SOURCE["diagram"].encode()
    assert out.read_bytes() == expected
    assert main([*argv, "--skip_output", "--flame_out", str(analysis)]) == 0
    assert out.read_bytes() == expected
    assert _svg(fromstring(analysis.read_bytes())) == _svg(
        fromstring(_SOURCE["analysis"])
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--kind", "iadd"],
        ["--kind", "imul2_mod", "--mod", "5", "--n", "3"],
        ["--kind", "imul2_mod", "--n", "1"],
        ["--kind", "imul2_mod", "--mod", "8"],
        ["--kind", "imul2_mod", "--mod", "0b0101"],
        ["--kind", "imul2_mod", "--mod", "xyz"],
        ["--kind", "iadd_classical", "--n", "4", "--minimize_qubits"],
        ["--kind", "iadd_classical_simple", "--n", "2", "--clean", "1"],
        ["--kind", "flip_if_lt"],
        ["--kind", "flip_if_lt", "--n", "3", "--clean", "0"],
        ["--kind", "flip_if_lt", "--n", "2", "--clean", "0", "--btol", "2.1"],
        ["--kind", "flip_if_lt", "--n", "3", "--btol", "nan"],
        ["--kind", "iadd", "--n", "1000001"],
        ["--kind", "lookup", "--clean", "0"],
        ["--kind", "unlookup", "--controlled"],
        ["--kind", "lookup", "--n", "31"],
    ],
)
def test_rejection_preserves_output(tmp_path: Path, argv: list[str]) -> None:
    out = tmp_path / "existing"
    out.write_bytes(b"previous")
    assert main([*argv, "--out", str(out)]) == 1
    assert out.read_bytes() == b"previous"


@pytest.mark.parametrize(
    "width,clean_count,controlled,btol,minimize,measurements,resets",
    [
        (1, 0, True, inf, False, 0, 0),
        (2, 1, True, 2.0, False, 1, 1),
        (3, 3, True, inf, False, 3, 3),
        (3, 2, True, inf, False, 0, 0),
        (4, 3, False, inf, False, 1, 1),
        (4, 4, True, 2.1, True, 1, 1),
        (3, 0, False, 1.0, True, 0, 0),
    ],
)
def test_exported_comparison_preserves_every_complex_branch(
    width: int,
    clean_count: int,
    controlled: bool,
    btol: float,
    minimize: bool,
    measurements: int,
    resets: int,
) -> None:
    enable = 2 * width + clean_count
    target = enable + int(controlled)
    clean = tuple(range(2 * width, enable))
    program, _ = physical_less_than(
        tuple(reversed(range(width))),
        tuple(reversed(range(width, 2 * width))),
        target,
        clean,
        control=enable if controlled else True,
        btol=btol,
        minimize_qubits=minimize,
    )
    logical = (*range(2 * width), *range(enable, program.n_qubits))
    indices = torch.arange(1 << len(logical))
    basis = torch.zeros_like(indices)
    for i, q in enumerate(logical):
        basis |= ((indices >> i) & 1) << (program.n_qubits - 1 - q)

    psi = torch.complex((indices + 1).double(), (2 * indices - 3).double())
    psi /= psi.norm()
    state = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    state[basis] = psi
    retained = width if btol >= width else max(0, ceil(btol) - 1)
    drop = width - retained
    left = (indices % (1 << width)) >> drop
    right = (indices >> (width + drop)) % (1 << retained)
    enabled = ((indices >> (2 * width)) & 1).bool() if controlled else True
    output = basis ^ (
        (enabled & (left < right)).long() << (program.n_qubits - 1 - target)
    )
    expected = torch.zeros_like(state)
    expected[output] = psi

    assert sum(op.opcode == "hmr" for op in program.operations) == measurements
    assert sum(op.opcode == "r" for op in program.operations) == resets
    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=program.n_bits + 2
    )
    events = sum(gate.name == "measure" for gate in gates)
    assert events == resets + measurements
    for outcomes in product((0, 1), repeat=events):
        branch = forced_branch(state, gates, outcomes)
        torch.testing.assert_close(
            branch, expected / 2 ** (events / 2), atol=5e-14, rtol=5e-14
        )


@pytest.mark.parametrize("offset", [0, 3, 5])
@pytest.mark.parametrize(
    "width,clean_count,dirty_count,simple,controlled,measurements,resets",
    [
        (2, 2, 0, True, True, 1, 1),
        (3, 3, 0, True, True, 3, 3),
        (3, 2, 0, False, False, 2, 2),
        (4, 2, 2, False, True, 2, 2),
    ],
)
def test_classical_addition_export_restores_every_complex_branch(
    offset: int,
    width: int,
    clean_count: int,
    dirty_count: int,
    simple: bool,
    controlled: bool,
    measurements: int,
    resets: int,
) -> None:
    clean_start = width + int(simple)
    dirty_start = clean_start + clean_count
    enable = dirty_start + dirty_count
    clean = tuple(range(clean_start, dirty_start))
    dirty = tuple(range(dirty_start, enable))
    program, _ = physical_classical_addition(
        tuple(reversed(range(width))),
        tuple(ClassicalFieldCoefficient(bit) for bit in reversed(range(width))),
        clean=clean,
        dirty=dirty,
        control=enable if controlled else True,
        carry_in=width if simple else None,
        minimize_qubits=True,
        btol=inf,
    )
    logical = tuple(q for q in range(program.n_qubits) if q not in clean)
    data = torch.arange(1 << len(logical))
    basis = torch.zeros_like(data)
    for i, q in enumerate(logical):
        basis |= ((data >> i) & 1) << (program.n_qubits - 1 - q)
    psi = torch.complex((data + 1).double(), (2 * data - 3).double())
    psi /= psi.norm()
    state = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    state[basis] = psi
    size = 1 << width
    carry = ((data >> width) & 1) if simple else 0
    control = ((data >> logical.index(enable)) & 1) if controlled else 1
    word = data % size
    result = (word + (offset + carry) * control) % size
    output = basis.clone()
    for q in range(width):
        output ^= (((word ^ result) >> q) & 1) << (program.n_qubits - 1 - q)
    expected = torch.zeros_like(state)
    expected[output] = psi
    initial_classical = tuple(
        ClassicalXor((), bit, bit < width and bool(offset & (1 << bit))).instruction()
        for bit in program.input_bits
    )
    gates = (
        *initial_classical,
        *expand_reset_branches(
            bind_reversible_phase_execution(program), first_slot=program.n_bits + 2
        ),
    )
    assert sum(op.opcode == "hmr" for op in program.operations) == measurements
    assert sum(op.opcode == "r" for op in program.operations) == resets
    events = sum(gate.name == "measure" for gate in gates)
    assert events == measurements + resets
    for outcomes in product((0, 1), repeat=events):
        branch = forced_branch(state, gates, outcomes)
        torch.testing.assert_close(
            branch, expected / 2 ** (events / 2), atol=5e-14, rtol=5e-14
        )


@pytest.mark.parametrize(
    "modulus,register,controlled",
    [
        (5, False, False),
        (5, True, True),
        (7, False, True),
        (7, True, False),
    ],
)
def test_exported_modular_doubling_preserves_exact_complex_branches(
    modulus: int, register: bool, controlled: bool
) -> None:
    width = modulus.bit_length()
    control = width if controlled else None
    inputs = width + int(controlled)
    clean = tuple(range(inputs, inputs + width))
    middle = ClassicalFieldCoefficient(1) if register else bool(modulus & 2)
    program = modular_doubling_program(
        tuple(reversed(range(width))),
        (True, middle, True),
        clean=clean,
        control=control,
        physical_trace=ComponentTrace(),
        declared_input_width=width if register else 0,
    )
    outcome = width if register else 0
    assert {op.result_bit for op in program.operations if op.opcode == "hmr"} == {
        outcome
    }
    size = 1 << program.n_qubits
    state = torch.zeros(size, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for value, enable in product(range(modulus), range(1 + int(controlled))):
        physical = sum(
            ((value >> q) & 1) << (program.n_qubits - 1 - q) for q in range(width)
        )
        enabled = not controlled or bool(enable)
        if controlled:
            physical |= enable << (program.n_qubits - 1 - width)
        amplitude = torch.tensor(
            complex(value + 2, 3 - value + enable), dtype=torch.complex128
        )
        state[physical] = amplitude
        updated = 2 * value % modulus if enabled else value
        for q in range(width):
            physical ^= (((value ^ updated) >> q) & 1) << (program.n_qubits - 1 - q)
        expected[physical] = amplitude
    norm = state.norm()
    state /= norm
    expected /= norm
    setup = tuple(
        ClassicalXor((), bit, bit == 1 and bool(modulus & 2)).instruction()
        for bit in program.input_bits
    )
    gates = (
        *setup,
        *expand_reset_branches(
            bind_reversible_phase_execution(program), first_slot=program.n_bits + 2
        ),
    )
    events = sum(gate.name == "measure" for gate in gates)
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, gates, outcomes),
            expected / 2 ** (events / 2),
            atol=5e-14,
            rtol=5e-14,
        )
