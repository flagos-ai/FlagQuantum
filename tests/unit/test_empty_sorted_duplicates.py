"""Degenerate duplicate graphs retain exact source calls and coherent flags."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_sorted_duplicates_factories import (
    sorted_duplicates_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.sorted_duplicates import (
    sorted_duplicate_requirements,
    sorted_duplicates_ir,
)
from flagquantum.compiler.registers.sorting.sorted_duplicates_component import (
    sorted_duplicates_program,
)
from flagquantum.compiler.registers.sorting.sorted_duplicates_resource_model import (
    source_sorted_duplicates_gate_cost,
    source_sorted_duplicates_resource_formula,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sorted_duplicates_component import source_stream

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "length,width,signed,controlled",
    [
        (length, width, signed, True)
        for length, width, signed in product((0, 1), (0, 1, 3), (False, True))
    ]
    + [
        (length, 0, False, controlled)
        for length, controlled in product((2, 3, 4), (False, True))
    ],
)
def test_complete_empty_domain_calls_amplitudes_branches_work_and_counts(
    length: int, width: int, signed: bool, controlled: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists import HasDuplicates
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = signed_register_type(width) if signed else unsigned_register_type(width)
    source = HasDuplicates(length, (QInt if signed else QUInt)(width), controlled)
    graph = sorted_duplicates_program(length, dtype, controlled=controlled)
    assert verify_program_record(graph) == ()
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(length))
    flag = length * width
    control = flag + 1 if controlled else None
    logical = flag + 1 + controlled
    spectator = logical
    q, c = sorted_duplicate_requirements(length, width, controlled=controlled)
    work = tuple(range(logical + 1, logical + 1 + q))
    ports = words + ((flag,),) + (((control,),) if control is not None else ())
    bound = bind_register_program(graph, ports, clean=work, borrowed=(spectator,))
    assert bound.outputs == ports
    reference = source_stream(source, bound, ports)
    assert_source(bound.circuit.instructions, reference)
    primitive, outputs = sorted_duplicates_ir(
        words,
        flag,
        work,
        tuple(range(c)),
        signed=signed,
        control=control,
        source_aliases=True,
    )
    # Binding may reuse a different scratch order. Both literal streams must
    # implement the same channel, while the source supplies exact bound operands.
    assert outputs == words
    constructor = resolve_component_program(
        _library(source, 0), 0, sorted_duplicates_factories()
    )
    assert (
        bind_register_program(constructor, ports, clean=work, borrowed=(spectator,))
        == bound
    )
    counts = Counter(gate.name for gate in reference)
    model = get_cost_value(source, QECGatesCost())
    assert (
        source_sorted_duplicates_gate_cost(
            length, width, signed=signed, controlled=controlled
        ).asdict()
        == model.asdict()
    )
    assert source_sorted_duplicates_resource_formula(
        length, width, signed=signed, controlled=controlled
    ).logical_qubits == get_cost_value(source, QubitCount())
    if length == 0:
        assert counts == {"cx": 2}
        assert model.clifford < 0
    else:
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert counts["measure"] == counts["reset"] == model.measurement
    if length <= 1:
        assert q == c == 0 and not bound.events
        assert counts == {"cx": 2}
    size = logical + 1 + q
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 5 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << size, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for label, amplitude in enumerate(amplitudes):
        initial[label << q] = amplitude
        enable = bool((label >> 1) & 1) if controlled else True
        toggle = length > 1 and enable
        output = label ^ ((1 << (1 + controlled)) if toggle else 0)
        expected[output << q] = amplitude
    for gates in (reference, primitive):
        measurements = sum(gate.name == "measure" for gate in gates)
        for history in product((0, 1), repeat=measurements):
            torch.testing.assert_close(
                forced_branch(initial, gates, history),
                expected / 2 ** (measurements / 2),
                atol=3e-12,
                rtol=0,
            )
    if not counts["measure"]:
        actual = (
            Circuit(size, inputs=initial, dtype=torch.complex128)
            .append(FieldOperation(size, bound.circuit.instructions))
            .state()
            .flatten()
        )
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=0)
    else:
        from flagquantum.dynamic import DynamicCircuit
        from flagquantum.runtime.dynamic.execution import run_dynamic

        circuit = DynamicCircuit(size, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(size, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=4, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(4, -1),
            expected.expand(4, -1),
            atol=3e-12,
            rtol=0,
        )


@pytest.mark.parametrize("length", (0, 1))
@pytest.mark.parametrize("width", (0, 2))
def test_uncontrolled_empty_controls_keep_original_failure_and_declaration(
    length: int, width: int
) -> None:
    from qualtran import BloqError, QUInt
    from qualtran.bloqs.arithmetic.lists import HasDuplicates

    source = HasDuplicates(length, QUInt(width))
    with pytest.raises(BloqError, match="controls"):
        source.decompose_bloq()
    program = resolve_component_program(
        _library(source, 0), 0, sorted_duplicates_factories()
    )
    assert not program.functions[0].blocks
    with pytest.raises(ValueError):
        sorted_duplicates_program(length, unsigned_register_type(width))


@pytest.mark.parametrize("controlled", (False, True))
def test_signed_empty_comparison_retains_source_sign_extension_failure(
    controlled: bool,
) -> None:
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.lists import HasDuplicates

    source = HasDuplicates(2, QInt(0), controlled)
    with pytest.raises(IndexError):
        source.decompose_bloq().flatten()
    program = resolve_component_program(
        _library(source, 0), 0, sorted_duplicates_factories()
    )
    assert not program.functions[0].blocks
    with pytest.raises(ValueError, match="positive"):
        sorted_duplicates_program(2, signed_register_type(0), controlled=controlled)
