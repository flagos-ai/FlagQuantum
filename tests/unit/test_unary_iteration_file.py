"""Whole-file correspondence for the original unary-iteration examples."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import prod
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.unary_iteration_catalog import (
    UnaryIntervalExample,
    UnaryXExample,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_ir,
    indexed_component_workspace,
)
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.compiler.registers.lookup.unary_iteration import (
    append_bounded_unary_tree,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source(pytestconfig: pytest.Config) -> Any:
    import kickmix
    from qualtran.bloqs.multiplexers import unary_iteration_bloq

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(unary_iteration_bloq.__file__).parent
    for name, digest in (
        ("unary_iteration.ipynb", "b0a18619c0b033d3870a24a4138c4b7400c70504"),
        ("unary_iteration_bloq.py", "d1f689a99ecccaf9a44983d074875f2dce64ccb6"),
        ("unary_iteration_bloq_test.py", "529640770dad7ee9ab564484c8cb725ad180e005"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    # The original test file supplies the two reference gate fixtures.
    pytestconfig.addinivalue_line("markers", "notebook: locked reference tutorial")
    return unary_iteration_bloq


def _check_complex_output(
    gates: tuple[Instruction, ...],
    width: int,
    rows: list[tuple[int, int]],
) -> None:
    state, expected = {}, {}
    norm = sum(abs(complex(i + 1, 3 - i)) ** 2 for i in range(len(rows))) ** 0.5
    for i, (initial, final) in enumerate(rows):
        amplitude = complex(i + 1, 3 - i) / norm
        state[initial] = amplitude
        expected[final] = amplitude
    events = sum(op.name == "measure" for op in gates)
    histories = (
        product((0, 1), repeat=events)
        if events <= 5
        else ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events)))
    )
    for history in histories:
        actual = _branch(
            gates, width, history, dict(state), normalize_measurements=True
        )
        assert actual.keys() == expected.keys()
        for address, amplitude in expected.items():
            assert actual[address] == pytest.approx(amplitude, abs=2e-12)


def _compare_source(
    original: Any,
    ports: dict[str, tuple[int, ...]],
    work: tuple[int, ...],
    gates: tuple[Instruction, ...],
    cost: ComponentCost,
) -> None:
    expected = reference_arithmetic_gates(
        original,
        ports,
        work,
        dict(zip(work, range(len(work)), strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(gates, expected)
    assert_source_model(cost, original)


@pytest.mark.parametrize(
    "width,targets,controls", ((3, 5, 1), (2, 4, 2), (1, 2, 3), (3, 4, 0))
)
def test_original_flat_width_control_and_target_order(
    width: int, targets: int, controls: int, source: Any
) -> None:
    from qualtran.bloqs.multiplexers.unary_iteration_bloq_test import ApplyXToLthQubit

    native = UnaryXExample(width, targets, controls)
    circuit = DynamicCircuit(native.qubit_count)
    circuit.append(native)
    gates = circuit.to_ir().instructions
    start = controls + width
    ports = {
        "selection": tuple(range(controls, start)),
        "target": tuple(range(start, native.logical_width)),
    }
    if controls:
        ports["control"] = tuple(range(controls))
    work = tuple(range(native.logical_width, native.qubit_count))
    _compare_source(
        ApplyXToLthQubit(width, targets, controls),
        ports,
        work,
        gates,
        native.resource_model(),
    )
    counts = Counter(op.name for op in gates)
    assert (counts["t"] + counts["tdg"], counts["measure"]) == (
        4 * (targets - 2 + controls),
        targets - 2 + controls,
    )
    rows = []
    for ctl, selected, data in product(
        range(1 << controls), range(targets), (0, (1 << targets) - 1)
    ):
        prefix = ((ctl << width) | selected) << targets
        output = data ^ (1 << selected) if ctl == (1 << controls) - 1 else data
        rows.append(((prefix | data) << len(work), (prefix | output) << len(work)))
    _check_complex_output(gates, native.qubit_count, rows)


@pytest.mark.parametrize("shape", ((2, 3, 2), (2, 2, 2)))
def test_original_multiple_axes(shape: tuple[int, int, int], source: Any) -> None:
    from qualtran.bloqs.multiplexers.unary_iteration_bloq_test import ApplyXToIJKthQubit

    widths = tuple((n - 1).bit_length() for n in shape)
    offset = 0
    axes = []
    targets = []
    for width in widths:
        axes.append(tuple(range(offset, offset + width)))
        offset += width
    for n in shape:
        targets.append(tuple(range(offset, offset + n)))
        offset += n
    count = indexed_component_workspace(widths, 0)
    work = tuple(range(offset, offset + count))
    rows = tuple(product(*(range(size) for size in shape)))

    def emit(index: int, predicate: int) -> tuple[Instruction, ...]:
        return tuple(
            Instruction("cx", (predicate, bank[digit]))
            for bank, digit in zip(targets, rows[index], strict=True)
        )

    gates = indexed_component_ir(
        tuple(axes),
        shape,
        emit,
        workspace=work,
        classical_bits=tuple(range(count)),
    )
    ports = dict(zip(("i", "j", "k", "t1", "t2", "t3"), (*axes, *targets), strict=True))
    model = indexed_component_cost(
        widths,
        shape,
        (ComponentCost("Three CNOT", 4, GateCost(clifford=3)),) * prod(shape),
        target_width=sum(shape),
        child_target_width=3,
    )
    _compare_source(ApplyXToIJKthQubit(shape), ports, work, gates, model)
    samples = []
    for address in rows:
        prefix = 0
        for digit, width in zip(address, widths, strict=True):
            prefix = (prefix << width) | digit
        mask = sum(
            1 << (offset - wire - 1)
            for bank, digit in zip(targets, address, strict=True)
            for wire in (bank[digit],)
        )
        initial = prefix << (sum(shape) + count)
        samples.append((initial, initial | (mask << count)))
    _check_complex_output(gates, offset + count, samples)


def test_original_nonzero_loop_and_empty_ranges(source: Any) -> None:
    import cirq
    from qualtran import GateWithRegisters, Signature

    class SourceIntervalLoop(GateWithRegisters):  # type: ignore[misc]
        @property
        def signature(self) -> Signature:
            return Signature.build(n=3, m=3, target=4)

        def decompose_from_registers(self, context: Any, **ports: Any) -> Any:
            tail: list[Any] = []
            for prefix, outer, i in source.unary_iteration(
                3, 5, tail, [], list(ports["n"]), context.qubit_manager
            ):
                yield prefix
                cleanup: list[Any] = []
                for body, predicate, j in source.unary_iteration(
                    6, 8, cleanup, [outer], list(ports["m"]), context.qubit_manager
                ):
                    yield body
                    yield cirq.CNOT(predicate, ports["target"][2 * (i - 3) + j - 6])
                yield cleanup
            yield tail

    op = UnaryIntervalExample()
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    gates = circuit.to_ir().instructions
    _compare_source(
        SourceIntervalLoop(),
        dict(n=(0, 1, 2), m=(3, 4, 5), target=(6, 7, 8, 9)),
        tuple(range(10, 15)),
        gates,
        op.resource_model(),
    )
    rows = []
    for i, j in product(range(3, 5), range(6, 8)):
        base = ((i << 3) | j) << 4
        rows.append((base << 5, (base | (1 << (3 - (2 * (i - 3) + j - 6)))) << 5))
    _check_complex_output(gates, op.qubit_count, rows)
    for upper in (3, 4):
        assert not list(
            source.unary_iteration(
                4, upper, [], [], [cirq.q("s")], cirq.SimpleQubitManager()
            )
        )
        empty: list[Instruction] = []

        def unexpected_child(_index: int, _wire: int) -> None:
            raise AssertionError("Empty traversal invoked the child")

        append_bounded_unary_tree(
            empty, None, (0,), upper, (), (), unexpected_child, lower=4
        )
        assert not empty


def test_offset_pruning_folds_only_contained_regions(source: Any) -> None:
    import cirq

    def can_fold(left: int, right: int) -> bool:
        return right - left == 2

    wires = cirq.LineQubit.range(4)
    original = source.unary_iteration(
        3, 7, [], [wires[0]], wires[1:], cirq.SimpleQubitManager(), can_fold
    )
    assert [index for _, _, index in original] == [3, 4, 6]
    visits: list[int] = []

    def record(index: int, _predicate: int) -> None:
        visits.append(index)

    for adjoint, order in ((False, [3, 4, 6]), (True, [6, 4, 3])):
        visits.clear()
        append_bounded_unary_tree(
            [],
            0,
            (1, 2, 3),
            7,
            (4, 5, 6),
            (0, 1, 2),
            record,
            lower=3,
            break_early=can_fold,
            adjoint=adjoint,
        )
        assert visits == order
