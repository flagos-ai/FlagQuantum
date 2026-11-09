"""Verify the bounded Majorana accumulator against the locked source."""

from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.lookup.bounded_majorana_select import (
    bounded_majorana_ir,
    bounded_majorana_workspace,
    source_bounded_majorana_cost,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "size,controlled,pauli", tuple(product((4, 6, 10), (False, True), "XY"))
)
def test_bounded_majorana_gate_stream(size, controlled, pauli):
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran import BQUInt, QBit, Register
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    width = (size - 1).bit_length()
    control = 0 if controlled else None
    start = int(controlled)
    selection = tuple(range(start, start + width))
    target = tuple(range(start + width, start + width + size))
    work = tuple(
        range(
            target[-1] + 1,
            target[-1] + 1 + bounded_majorana_workspace(size, controlled=controlled),
        )
    )
    bits = tuple(range(20, 20 + len(work) - 1))
    ports = {"orb": selection[:-1], "spin": selection[-1:], "target": target}
    if control is not None:
        ports["control"] = (control,)
    source = SelectedMajoranaFermion(
        (Register("orb", BQUInt(width - 1, size // 2)), Register("spin", BQUInt(1, 2))),
        (Register("control", QBit()),) if controlled else (),
        getattr(cirq, pauli),
    )
    expected = reference_arithmetic_gates(
        source,
        ports,
        work,
        dict(zip(work[1:], bits, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    actual = bounded_majorana_ir(selection, target, control, pauli, work, bits)
    assert_source(actual, expected)
    assert_source_model(
        source_bounded_majorana_cost(size, pauli, controlled=controlled), source
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("side,controlled", tuple(product((2, 3, 5), (False, True))))
def test_nested_majorana_preserves_original_axis_traversal(side, controlled):
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran import BQUInt, QBit, Register
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    extents = (side, side, 2)
    w = (side - 1).bit_length()
    start = int(controlled)
    size = 2 * side**2
    width = 2 * w + 1
    selection = tuple(range(start, start + width))
    target = tuple(range(start + width, start + width + size))
    work = tuple(
        range(
            target[-1] + 1,
            target[-1]
            + 1
            + bounded_majorana_workspace(
                size, controlled=controlled, axis_extents=extents
            ),
        )
    )
    bits = tuple(range(len(work) - 1))
    ports = {
        "x": selection[:w],
        "y": selection[w : 2 * w],
        "spin": selection[-1:],
        "target": target,
    }
    if controlled:
        ports["control"] = (0,)
    source = SelectedMajoranaFermion(
        (
            Register("x", BQUInt(w, side)),
            Register("y", BQUInt(w, side)),
            Register("spin", BQUInt(1, 2)),
        ),
        (Register("control", QBit()),) if controlled else (),
        cirq.Y,
    )
    expected = reference_arithmetic_gates(
        source,
        ports,
        work,
        dict(zip(work[1:], bits, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    actual = bounded_majorana_ir(
        selection,
        target,
        0 if controlled else None,
        "Y",
        work,
        bits,
        axis_extents=extents,
    )
    assert_source(actual, expected)


@pytest.mark.parametrize(
    "size,controlled,pauli",
    [*product((4,), (False, True), "XY"), (6, True, "Y")],
)
def test_majorana_exhaustive_branches_and_complex_phase(size, controlled, pauli):
    start = int(controlled)
    control = 0 if controlled else None
    width = (size - 1).bit_length()
    selection = tuple(range(start, start + width))
    target = tuple(range(start + width, start + width + size))
    work = tuple(range(target[-1] + 1, target[-1] + 1 + width + start))
    bits = tuple(range(len(work) - 1))
    gates = bounded_majorana_ir(selection, target, control, pauli, work, bits)
    events = sum(gate.name == "measure" for gate in gates)
    assert events == size - 2 + start
    n_qubits = work[-1] + 2
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for ctrl, address, data, spectator in product(
        range(1 + start), range(1 << width), range(1 << size), range(2)
    ):
        top = ((ctrl << width) | address) << size
        index = ((top | data) << (len(work) + 1)) | spectator
        coefficient = complex(index + 1, (index % 7) - 3)
        state[index] = coefficient
        if controlled and not ctrl:
            expected[index] = coefficient
        else:
            # Six-target source pruning maps the last padded orbital to the
            # previous populated orbital while keeping the spin bit.
            selected = address - 2 if address >= size else address
            bit = 1 << (size - 1 - selected)
            phase = (-1) ** ((data >> (size - selected)).bit_count())
            if pauli == "Y":
                phase *= -1j if data & bit else 1j
            expected[index ^ (bit << (len(work) + 1))] = phase * coefficient
    norm = state.norm()
    state /= norm
    expected /= norm * 2 ** (events / 2)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, gates, history), expected, atol=2e-12, rtol=0
        )
