"""Literal duplicate graphs preserve source calls and retain erasure defects."""

from collections import Counter
from itertools import combinations_with_replacement, product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.compiler.registers.sorting.sorted_duplicates_component import (
    sorted_duplicates_program,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def bind_case(
    length: int, width: int, signed: bool, controlled: bool, *, spectator: bool = False
) -> tuple[BoundRegisterProgram, tuple[tuple[int, ...], ...], int]:
    dtype = signed_register_type(width) if signed else unsigned_register_type(width)
    logical = length * width + 1 + controlled
    ports = (
        tuple(tuple(range(i * width, (i + 1) * width)) for i in range(length))
        + ((length * width,),)
        + (((length * width + 1,),) if controlled else ())
    )
    graph = sorted_duplicates_program(length, dtype, controlled=controlled)
    pool = tuple(
        range(
            logical + spectator,
            logical + spectator + (length - 1) * (width + 2) + max(2, length - 1),
        )
    )
    return bind_register_program(graph, ports, clean=pool), ports, logical


def source_stream(
    source: Any, bound: BoundRegisterProgram, ports: tuple[tuple[int, ...], ...]
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.arithmetic import LinearDepthHalfLessThan
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.mcmt import MultiControlX

    length, width = int(source.l), int(source.dtype.bitsize)
    signed = type(source.dtype).__name__ == "QInt"
    inputs = {
        "xs": tuple(w for word in ports[:length] for w in word),
        "flag": ports[length],
    }
    if source.is_controlled:
        inputs["ctrl"] = ports[-1]
    outputs = {
        "xs": tuple(w for word in bound.outputs[:length] for w in word),
        "flag": bound.outputs[length],
    }
    if source.is_controlled:
        outputs["ctrl"] = bound.outputs[-1]
    cursor = 0
    slots = MeasurementSlots(bound.circuit.instructions)

    def leaf(bloq: Any, old: SourcePorts, desired: SourcePorts) -> SourceResult | None:
        nonlocal cursor
        # These accepted primitive lengths only delimit call-local allocations.
        # The unmodified source supplies every operation below.
        if isinstance(bloq, LinearDepthHalfLessThan):
            count = (12 if bloq.uncompute else 21) * width + 6 + 4 * signed
        elif isinstance(bloq, MultiControlX):
            count = 17 * (len(old["controls"]) - 1) + 1
        elif isinstance(bloq, (CNOT, XGate)):
            count = 1
        else:
            return None
        plan = tuple(
            w
            for event in bound.events
            if event.kind == "allocate"
            and event.instruction_offset is not None
            and cursor <= event.instruction_offset < cursor + count
            for w in event.wires
        )
        if isinstance(bloq, LinearDepthHalfLessThan):
            if bloq.uncompute:
                returned = {"a": old["b"], "b": old["a"]}
            else:
                assert len(plan) == width + 4
                returned = {**old, "c": plan[2 : width + 3][::-1], "target": plan[-1:]}
        else:
            returned = old
        expected: tuple[Instruction, ...]
        if isinstance(bloq, XGate):
            expected = (Instruction("x", old["q"]),)
        elif isinstance(bloq, CNOT):
            expected = (Instruction("cx", (*old["ctrl"], *old["target"])),)
        else:
            expected = reference_arithmetic_gates(
                bloq,
                old,
                plan,
                slots,
                expected_outputs=returned,
                reuse_allocations=False,
            )
        assert len(expected) == count
        assert all(returned[key] == value for key, value in desired.items())
        cursor += count
        return expected, returned

    gates, returned = walk_source_graph(source, inputs, outputs, leaf)
    assert cursor == len(bound.circuit.instructions) and returned == outputs
    return gates


@pytest.mark.parametrize(
    "length,width", ((1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 3))
)
@pytest.mark.parametrize("signed,controlled", tuple(product((False, True), repeat=2)))
def test_whole_source_calls_operands_feedback_outputs_and_resources(
    length: int, width: int, signed: bool, controlled: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = signed_register_type(width) if signed else unsigned_register_type(width)
    source = HasDuplicates(length, (QInt if signed else QUInt)(width), controlled)
    if length == 1 and not controlled:
        from qualtran import BloqError

        with pytest.raises(BloqError, match="controls"):
            source.decompose_bloq()
        with pytest.raises(ValueError, match="uncontrolled singleton"):
            sorted_duplicates_program(length, dtype)
        return
    graph = sorted_duplicates_program(length, dtype, controlled=controlled)
    assert verify_program_record(graph) == ()
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    bound, ports, logical = bind_case(length, width, signed, controlled)
    assert bound.outputs[:length] == (
        ports[:length] if length == 1 else (ports[length - 1], *ports[: length - 1])
    )
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == counts["reset"] == model.measurement
    active: set[int] = set()
    peak = 0
    for event in bound.events:
        if event.kind == "allocate":
            active.update(event.wires)
        else:
            active.difference_update(event.wires)
        peak = max(peak, len(active))
    assert not active
    assert logical + peak == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("signed,controlled", tuple(product((False, True), repeat=2)))
def test_two_word_valid_source_full_amplitudes_every_history_and_work_zero(
    signed: bool, controlled: bool
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound, _, logical = bind_case(2, 2, signed, controlled, spectator=True)
    width = max(w for g in bound.circuit.instructions for w in g.wires) + 1
    scratch = width - logical - 1
    domain = range(-2, 2) if signed else range(4)
    labels, mapped = [], []
    for words in combinations_with_replacement(domain, 2):
        for flag, ctrl, spectator in product(
            (0, 1), (0, 1) if controlled else (1,), (0, 1)
        ):
            data = (((words[0] & 3) << 2) | (words[1] & 3)) << 1 | flag
            output = (data & ~1) | (flag ^ ((words[0] == words[1]) and bool(ctrl)))
            if controlled:
                data, output = (data << 1) | ctrl, (output << 1) | ctrl
            labels.append(((data << 1) | spectator) << scratch)
            mapped.append(((output << 1) | spectator) << scratch)
    raw = torch.arange(1, len(labels) + 1, dtype=torch.float64)
    amps = torch.complex(raw, 7 - 3 * raw)
    amps /= amps.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[labels], expected[mapped] = amps, amps
    measured = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=measured):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** (measured / 2),
            rtol=0,
            atol=3e-12,
        )
    circuit = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(width, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), rtol=0, atol=3e-12
    )


def test_original_invalid_three_word_erasure_is_retained() -> None:
    np = pytest.importorskip("numpy")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    original = HasDuplicates(3, QUInt(2))
    values = dict(xs=np.array([0, 0, 1]), flag=0)
    assert original.on_classical_vals(**values)["flag"] == 1
    with pytest.raises(ValueError, match="Inconsistent.*And"):
        original.decompose_bloq().flatten().on_classical_vals(**values)
    bound, _, _ = bind_case(3, 2, False, False)
    width = max(w for g in bound.circuit.instructions for w in g.wires) + 1
    initial = (0, 0, 0, 0, 0, 1, *(0,) * (width - 6))
    with pytest.raises(AssertionError):
        certified_and_channel_map(bound.circuit.instructions, (initial,))
    assert_source(
        bound.circuit.instructions,
        source_stream(
            original,
            bound,
            tuple(tuple(range(2 * i, 2 * i + 2)) for i in range(3)) + ((6,),),
        ),
    )


def test_controlled_singleton_is_identity_on_dirty_coherent_data() -> None:
    bound, _, _ = bind_case(1, 2, False, True, spectator=True)
    assert Counter(g.name for g in bound.circuit.instructions) == {"cx": 2}
    raw = torch.arange(1, 33, dtype=torch.float64)
    state = torch.complex(raw, 7 - 3 * raw)
    state /= state.norm()
    torch.testing.assert_close(
        forced_branch(state, bound.circuit.instructions, ()), state, rtol=0, atol=3e-12
    )
    assert (
        CircuitIR.from_json(bound.circuit.to_json()).instructions
        == bound.circuit.instructions
    )


@pytest.mark.parametrize("controlled", (False, True))
def test_three_word_source_branch_channels_match_without_cleanliness_claim(
    controlled: bool,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    bound, ports, logical = bind_case(3, 1, False, controlled, spectator=True)
    reference = source_stream(HasDuplicates(3, QUInt(1), controlled), bound, ports)
    width = max(w for g in bound.circuit.instructions for w in g.wires) + 1
    scratch = width - logical - 1
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 3 * raw)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[:: 1 << scratch] = amplitudes
    measured = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=measured):
        torch.testing.assert_close(
            forced_branch(state, bound.circuit.instructions, history),
            forced_branch(state, reference, history),
            rtol=0,
            atol=3e-12,
        )


def test_actual_factory_native_existing_check_modes() -> None:
    from qualtran.bloqs.arithmetic.lists.has_duplicates import _has_duplicates

    from flagquantum.core.ir.program import ProgramRecord
    from tests.unit.register_example_checks import (
        CHECKS,
        CheckStatus,
        RegisterExample,
        check_example,
    )

    source = _has_duplicates.make()
    count, width = int(source.l), int(source.dtype.bitsize)
    graph = sorted_duplicates_program(count, unsigned_register_type(width))
    bound, ports, _ = bind_case(count, width, False, False)
    reference = source_stream(source, bound, ports)

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(
            record,
            ports,
            clean=tuple(
                range(
                    count * width + 1,
                    count * width + 1 + (count - 1) * (width + 2) + max(2, count - 1),
                )
            ),
        ).circuit

    for kind in CHECKS:
        result = check_example(
            RegisterExample(
                lambda: graph, lower, lambda: Counter(g.name for g in reference)
            ),
            kind,
        )
        assert result.status is CheckStatus.PASS, result.message


def test_complete_original_definition_dispositions() -> None:
    import ast
    from pathlib import Path

    from qualtran.bloqs.arithmetic.lists import has_duplicates

    audit = (
        Path(__file__).parents[2] / "docs/development/SORTED_DUPLICATES_FILE_AUDIT.md"
    ).read_text()
    count = 0
    for node in ast.parse(Path(has_duplicates.__file__).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            assert f"`{node.name}`" in audit
            count += 1
            for method in node.body if isinstance(node, ast.ClassDef) else ():
                if isinstance(method, ast.FunctionDef):
                    assert f"`{node.name}.{method.name}`" in audit
                    count += 1
    assert count == 11
