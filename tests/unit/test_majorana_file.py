"""Compare the locked flat Majorana fixtures with their native bounded entry."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from flagquantum.algorithms.primitives.multiplexers.selected_majorana_fermion import (
    SelectedMajoranaFermion,
)
from flagquantum.compiler.registers.lookup.bounded_majorana_select import (
    bounded_majorana_workspace,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source() -> Any:
    import kickmix
    from qualtran.bloqs.multiplexers import selected_majorana_fermion

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    directory = Path(selected_majorana_fermion.__file__).parent
    for suffix, digest in (
        (".py", "656373799b28c6838bc3528733c26f0fba0adacf"),
        ("_test.py", "d0b964420757f48f6f600d3d8e555855c1b6843c"),
    ):
        data = (directory / f"selected_majorana_fermion{suffix}").read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    return selected_majorana_fermion.SelectedMajoranaFermion


@pytest.mark.parametrize(
    "width,targets", ((2, 3), (2, 4), (3, 5), (3, 8), (4, 9), (5, 9))
)
@pytest.mark.parametrize("pauli", ("X", "Y"))
def test_original_flat_stream_and_complex_state(
    width: int, targets: int, pauli: str, source: Any
) -> None:
    import cirq
    from qualtran import BQUInt, Register

    control = (0,)
    selection = tuple(range(1, width + 1))
    target = tuple(range(width + 1, width + targets + 1))
    logical = 1 + width + targets
    count = bounded_majorana_workspace(targets, controlled=True, selection_width=width)
    work = tuple(range(logical, logical + count))
    slots = tuple(range(count - 1))
    child = cirq.X if pauli == "X" else cirq.Y
    original = source(Register("selection", BQUInt(width, targets)), target_gate=child)
    ports = dict(control=control, selection=selection, target=target)
    named = {
        name: [cirq.LineQubit(wire) for wire in word] for name, word in ports.items()
    }
    inferred = source.make_on(target_gate=child, **named)
    assert inferred == original.on_registers(**named)
    assert inferred.qubits == tuple(cirq.LineQubit.range(logical))
    assert cirq.circuit_diagram_info(inferred.gate).wire_symbols == (
        "@",
        *("In",) * width,
        *(f"Z{pauli}",) * targets,
    )

    operation = SelectedMajoranaFermion.make_bounded_on(
        **ports, target_gate=pauli, workspace=work, classical_bits=slots
    )
    circuit = DynamicCircuit(logical + count)
    circuit.append(operation)
    stream = circuit.to_ir().instructions
    reference = reference_arithmetic_gates(
        inferred.gate,
        ports,
        work,
        dict(zip(work[1:], slots, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(stream, reference)
    assert_source_model(operation.resource_model(), inferred.gate)
    counts = Counter(g.name for g in stream)
    events = counts["measure"]
    assert (counts["t"] + counts["tdg"], events) == (4 * (targets - 1), targets - 1)

    # Each legal selector carries a coherent superposition of target words.
    state: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    for enable, row, word in product(
        range(2), range(targets), (0, 1, (1 << targets) - 1)
    ):
        prefix = ((enable << width) | row) << targets
        value = complex(1 + row, 2 * word - targets)
        state[(prefix | word) << count] = value
        mask = 1 << (targets - row - 1)
        sign = (-1) ** ((word >> (targets - row)).bit_count())
        phase = sign * complex(
            cirq.unitary(child)[int(not word & mask), int(bool(word & mask))]
        )
        if enable:
            expected[(prefix | (word ^ mask)) << count] = phase * value
        else:
            expected[(prefix | word) << count] = value
    norm = sum(abs(v) ** 2 for v in state.values()) ** 0.5
    state = {k: v / norm for k, v in state.items()}
    expected = {k: v / norm for k, v in expected.items()}
    histories = (
        product((0, 1), repeat=events)
        if targets <= 5
        else ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events)))
    )
    for history in histories:
        observed = _branch(
            stream, logical + count, history, dict(state), normalize_measurements=True
        )
        assert all(
            abs(observed.get(k, 0) - expected.get(k, 0)) < 2e-12
            for k in observed.keys() | expected.keys()
        )


@pytest.mark.parametrize("width,targets,pauli", ((1, 1, "Y"), (2, 1, "Z"), (3, 5, "Z")))
def test_flat_singleton_and_z_boundaries(
    width: int, targets: int, pauli: str, source: Any
) -> None:
    import cirq
    from qualtran import BQUInt, Register

    logical = 1 + width + targets
    n = bounded_majorana_workspace(targets, controlled=True, selection_width=width)
    workspace = tuple(range(logical, logical + n))
    slots = tuple(range(n - 1))
    ports = dict(
        control=(0,),
        selection=tuple(range(1, 1 + width)),
        target=tuple(range(1 + width, logical)),
    )
    original = source(
        Register("selection", BQUInt(width, targets)),
        target_gate=cirq.Y if pauli == "Y" else cirq.Z,
    )
    op = SelectedMajoranaFermion.make_bounded_on(
        **ports, target_gate=pauli, workspace=workspace, classical_bits=slots
    )
    reference = reference_arithmetic_gates(
        original,
        ports,
        workspace,
        dict(zip(workspace[1:], slots, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(op.emit(), reference)
    assert_source_model(op.resource_model(), original)


def test_original_empty_selection_is_outside_bounded_execution(source: Any) -> None:
    import cirq
    from qualtran import BQUInt, Register

    original = source(Register("selection", BQUInt(0, 1)), target_gate=cirq.Y)
    with pytest.raises(
        NotImplementedError, match="Selection register must not be empty"
    ):
        reference_arithmetic_gates(
            original,
            {"control": (0,), "selection": (), "target": (1,)},
            (2,),
            {},
            cirq_style=True,
            keep_pauli_leaves=True,
        )
    native = SelectedMajoranaFermion.make_bounded_on(
        control=(0,), selection=(), target=(1,), workspace=(2,), classical_bits=()
    )
    with pytest.raises(ValueError, match="concrete word"):
        native.emit()


def test_bounded_entry_checks_storage_and_preserves_default_operation() -> None:
    from flagquantum.errors import ValidationError

    ports = dict(control=(0,), selection=(1, 2), target=(3, 4, 5))
    with pytest.raises(ValidationError, match="at most one control"):
        SelectedMajoranaFermion.make_bounded_on(
            control=(0, 6),
            selection=(1, 2),
            target=(3, 4, 5),
            workspace=(7, 8, 9),
            classical_bits=(0, 1),
        )
    op = SelectedMajoranaFermion.make_bounded_on(
        **ports, target_gate="Y", workspace=(6, 7, 8), classical_bits=(0, 1)
    )
    with pytest.raises(ValidationError, match="exceed"):
        DynamicCircuit(8).append(op)
    with pytest.raises(ValueError, match="unary-tree|matching work"):
        SelectedMajoranaFermion.make_bounded_on(
            **ports, target_gate="Y", workspace=(6, 7), classical_bits=(0, 1)
        ).emit()
    direct = DynamicCircuit(6)
    direct.append(SelectedMajoranaFermion.make_on(**ports, target_gate="Y"))
    assert tuple(g.name for g in direct.to_ir().instructions) == (
        "selected_majorana_fermion",
    )
