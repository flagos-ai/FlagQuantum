"""Whole-file tests for the frozen Pauli SELECT tutorial and original grids."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.pauli_select_catalog import (
    PauliSelectExample,
    pauli_select_example_call_graph,
)
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.compiler.registers.lookup.unary_iteration import (
    unary_iteration_workspace,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source() -> Any:
    import kickmix
    from qualtran.bloqs.multiplexers import select_pauli_lcu

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    directory = Path(select_pauli_lcu.__file__).parent
    for suffix, digest in (
        (".ipynb", "6513d5faf7b227cd26534f30e05a7d3b55a29f5a"),
        (".py", "41e2f8ddb341fbbdd2d004ef533041bd24a5898c"),
        ("_test.py", "81ccdd543c60783cecee6a87ae238b050ed3b28b"),
    ):
        data = (directory / f"select_pauli_lcu{suffix}").read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    return select_pauli_lcu


def _compile(original: Any, words: tuple[str, ...]) -> tuple[CircuitIR, int]:
    width, target, polarity = (
        original.selection_bitsize,
        original.target_bitsize,
        original.control_val,
    )
    offset = int(polarity is not None)
    stop = offset + width + target
    work = unary_iteration_workspace(width, polarity)
    ports = {
        "selection": tuple(range(offset, offset + width)),
        "target": tuple(range(offset + width, stop)),
    }
    controls = (0,) if offset else ()
    if controls:
        ports["control"] = controls
    scratch = tuple(range(stop, stop + work))
    slots = tuple(range(work))
    op = SelectPauliLCU(width, target, words, polarity).make_bounded_on(
        selection=ports["selection"],
        target=ports["target"],
        control=controls,
        workspace=scratch,
        classical_bits=slots,
    )
    circuit = DynamicCircuit(stop + work)
    circuit.append(op)
    ir = circuit.to_ir()
    reference = reference_arithmetic_gates(
        original,
        ports,
        scratch,
        dict(zip(scratch, slots, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(ir.instructions, reference)
    assert_source_model(op.resource_model(), original)
    return ir, work


def _call_label(node: Any) -> str:
    if type(node).__name__ == "And":
        return f"And({node.cv1},{node.cv2})" + ("†" if node.uncompute else "")
    if type(node).__name__ == "Controlled":
        assert node.ctrl_spec.get_single_ctrl_val() == 1
        pauli = node.subbloq.gate
        assert pauli.coefficient == 1
        return f"C[{str(pauli)[1:]}]"
    return "X" if type(node).__name__ == "XGate" else type(node).__name__


@pytest.mark.parametrize("control", (None, 0, 1))
def test_notebook_signature_calls_and_all_measurements(
    control: int | None, source: Any
) -> None:
    import attrs
    import cirq
    from qualtran.resource_counting.generalizers import cirq_to_bloqs, ignore_split_join

    original = attrs.evolve(source._select_pauli_lcu.make(), control_val=control)
    op = PauliSelectExample(control)
    ir, work = _compile(original, tuple(op.select.select_unitaries))
    native = DynamicCircuit(op.qubit_count)
    native.append(op)
    assert native.to_ir().instructions == ir.instructions
    assert [(r.name, r.bitsize, r.side.value) for r in op.signature] == [
        (r.name, r.bitsize, r.side.value) for r in original.signature
    ]
    if control is None:
        _, children = original.call_graph(
            max_depth=1, generalizer=(cirq_to_bloqs, ignore_split_join)
        )
        assert pauli_select_example_call_graph().leaf_counts == {
            _call_label(child): n for child, n in children.items()
        }
    state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for bit, row in product(range(1 if control is None else 2), range(4)):
        start = ((bit << 6) | (row << 4)) << work
        addresses = torch.arange(start, start + (16 << work), 1 << work)
        s = torch.arange(1, 17, dtype=torch.float64)
        values = torch.complex(s + row, s * (bit + 1) - row)
        state[addresses] = values
        actual_pauli = torch.as_tensor(
            cirq.unitary(original.select_unitaries[row]), dtype=torch.complex128
        )
        expected[addresses] = (
            actual_pauli @ values if control is None or bit == control else values
        )
    norm = state.norm()
    state /= norm
    expected /= norm
    counts = Counter(g.name for g in ir.instructions)
    events = counts["measure"]
    assert (counts["t"] + counts["tdg"], counts["cx"], events) == (
        (8, 17, 2) if control is None else (12, 24, 3)
    )
    for bits in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, ir.instructions, bits),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=2e-13,
        )


@pytest.mark.parametrize("control", (0, 1))
def test_original_four_site_ising_grid(control: int, source: Any) -> None:
    import cirq
    from qualtran.bloqs.chemistry.ising import get_1d_ising_hamiltonian

    wires = cirq.LineQubit.range(4)
    rows = tuple(term.dense(wires) for term in get_1d_ising_hamiltonian(wires, 1, 1))
    original = source.SelectPauliLCU(3, 4, rows, control)
    words = tuple(str(row)[1:] for row in rows)
    ir, work = _compile(original, words)
    state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for bit, index in product(range(2), range(len(rows))):
        address = ((bit << 7) | (index << 4)) << work
        amp = torch.tensor(complex(index + 1, bit - index), dtype=torch.complex128)
        state[address] = amp
        output = (
            sum((letter == "X") << (3 - i) for i, letter in enumerate(words[index]))
            if bit == control
            else 0
        )
        expected[address | (output << work)] = amp
    norm = state.norm()
    state /= norm
    expected /= norm
    events = sum(g.name == "measure" for g in ir.instructions)
    for bits in ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events))):
        torch.testing.assert_close(
            forced_branch(state, ir.instructions, bits),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=2e-13,
        )


def test_original_three_site_ising_eigenstate_overlaps(source: Any) -> None:
    import cirq
    from qualtran.bloqs.chemistry.ising import (
        get_1d_ising_hamiltonian,
        get_1d_ising_lcu_coeffs,
    )

    wires = cirq.LineQubit.range(3)
    hamiltonian = get_1d_ising_hamiltonian(wires, 1, 1)
    rows = tuple(term.dense(wires) for term in hamiltonian)
    coefficients = get_1d_ising_lcu_coeffs(3, 1, 1)
    norm = coefficients.sum()
    weights = np.sqrt(coefficients / norm)
    original = source.SelectPauliLCU(3, 3, rows, 1)
    ir, work = _compile(original, tuple(str(row)[1:] for row in rows))
    eigenvalues, vectors = np.linalg.eigh(hamiltonian.matrix())
    events = sum(g.name == "measure" for g in ir.instructions)
    for eigenvalue, vector in zip(eigenvalues, vectors.T, strict=True):
        state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
        output = torch.zeros_like(state)
        for i, (weight, row) in enumerate(zip(weights, rows, strict=True)):
            start = ((1 << 6) | (i << 3)) << work
            addresses = torch.arange(start, start + (8 << work), 1 << work)
            state[addresses] = torch.as_tensor(weight * vector)
            output[addresses] = torch.as_tensor(weight * (cirq.unitary(row) @ vector))
        actual = forced_branch(state, ir.instructions, (0,) * events) * 2 ** (
            events / 2
        )
        torch.testing.assert_close(actual, output, rtol=0, atol=2e-13)
        assert complex(torch.vdot(state, actual)) == pytest.approx(
            eigenvalue / norm, abs=2e-13
        )


def test_source_coefficient_sign_and_nonunitary_boundary(source: Any) -> None:
    import cirq

    rows = (
        cirq.DensePauliString("Y", coefficient=-2 + 3j),
        cirq.DensePauliString("X", coefficient=1 + 2j),
    )
    original = source.SelectPauliLCU(1, 1, rows, 0)
    _compile(original, ("-Y", "X"))
    control, target = cirq.LineQubit.range(2)
    for coefficient in (0, 1j, -1j):
        nonunitary = source.SelectPauliLCU(
            1, 1, (cirq.DensePauliString("X", coefficient=coefficient), rows[0]), 1
        )
        leaf = nonunitary.nth_operation(None, 0, control, (target,))
        assert leaf.sub_operation.coefficient == 0 and not cirq.has_unitary(leaf)


def test_original_validation_and_control_metadata(source: Any) -> None:
    import cirq
    from qualtran import CtrlSpec

    with pytest.raises(ValueError, match="should contain 3"):
        source.SelectPauliLCU(2, 3, (cirq.DensePauliString("Y"),))
    with pytest.raises(ValueError, match="should be at-least 3"):
        source.SelectPauliLCU(1, 2, (cirq.DensePauliString("XX"),) * 5)
    for width, target, rows in ((2, 3, ("Y",)), (1, 2, ("XX",) * 5)):
        with pytest.raises(ValueError):
            SelectPauliLCU(width, target, rows).make_bounded_on(
                selection=range(width),
                target=range(width, width + target),
                workspace=(),
                classical_bits=(),
            ).emit()
    original = source._select_pauli_lcu.make()
    assert original.adjoint() is original and original._has_unitary_()
    for value in (0, 1):
        controlled = original.controlled(CtrlSpec(cvs=value))
        assert controlled.control_val == value
        assert controlled.adjoint() is controlled
        assert controlled.signature[0].name == "control"
