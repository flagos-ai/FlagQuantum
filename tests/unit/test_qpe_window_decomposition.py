"""Stored phase-window calls and source-ordered Fourier adjoints."""

from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.luis_perina_state import (
    luis_perina_interim_program,
)
from flagquantum.algorithms.primitives.phase_estimation_window import (
    rectangular_window_program,
)
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.component_rebindings import (
    adjoint_word_rebinding,
    split_word_function,
)
from flagquantum.compiler.registers.phase.phase_gradient_component import (
    phase_gradient_unitary_program,
)
from flagquantum.compiler.registers.phase.textbook_qft_component import (
    textbook_qft_program,
)
from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_rz_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import Register, RegisterType
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_rotations_ui_export import originals as originals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _case(kind: str, width: int, adjoint: bool) -> tuple[Any, ProgramRecord]:
    from qualtran.bloqs.basic_gates.rotation import CRz
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPRSInterimPrep
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary

    source: Any
    if kind == "interim":
        source = LPRSInterimPrep(width)
        record = luis_perina_interim_program(width, adjoint=adjoint)
    elif kind == "crz":
        source = CRz(0.37)
        record = controlled_rz_program(0.37, adjoint=adjoint)
    elif kind.startswith("gradient"):
        controlled = kind == "gradient_controlled"
        source = PhaseGradientUnitary(width, 0.5, is_controlled=controlled)
        record = phase_gradient_unitary_program(
            width, 0.5, controlled=controlled, adjoint=adjoint
        )
    else:
        reverse = kind == "qft_reverse"
        source = QFTTextBook(width, reverse)
        record = textbook_qft_program(width, with_reverse=reverse, adjoint=adjoint)
    return (source.adjoint() if adjoint else source), record


@pytest.mark.parametrize(
    "kind,width,adjoint",
    tuple(
        product(
            ("interim", "gradient", "gradient_controlled", "qft", "qft_reverse", "crz"),
            (1, 2, 3, 4),
            (False, True),
        )
    ),
)
def test_source_calls_operands_and_exact_complex_operator(
    kind: str, width: int, adjoint: bool
) -> None:
    import cirq

    source, record = _case(kind, width, adjoint)
    assert not verify_program_record(record)
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    assert copy_component_program(record) == record
    ports = record.functions[0].registers or ()
    sizes = tuple(int(p.total_bits()) for p in ports)
    logical = sum(sizes)
    cursor = 0
    inputs = []
    for size in sizes:
        inputs.append(tuple(range(cursor, cursor + size)))
        cursor += size
    bound = bind_register_program(record, tuple(inputs), clean=(logical,))
    expected = torch.tensor(
        source.tensor_contract() if kind == "crz" else cirq.unitary(source),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=8e-12, rtol=0
    )


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_rectangular_output_ownership_calls_and_state(width: int) -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState

    record = rectangular_window_program(width)
    assert not verify_program_record(record)
    assert _native_trace(record) == _source_trace(
        SourceCalls, RectangularWindowState(width).decompose_bloq()
    )
    bound = bind_register_program(
        record, (), clean=tuple(range(width)), allow_workspace_outputs=True
    )
    actual = logical_operator(bound.circuit, width)[:, 0]
    expected = torch.full((1 << width,), 2 ** (-width / 2), dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, atol=8e-12, rtol=0)
    assert bound.circuit.n_wires == width


def test_symbolic_window_retains_a_declaration_and_malformed_views_refuse() -> None:
    import sympy
    from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState

    width = sympy.Symbol("n", positive=True, integer=True)
    record = rectangular_window_program(width)
    assert not verify_program_record(record)
    assert _native_trace(record) == _source_trace(
        SourceCalls, RectangularWindowState(width).decompose_bloq()
    )
    assert any(not f.blocks for f in record.functions)
    function = split_word_function("split", RegisterType("uint", 3))
    assert adjoint_word_rebinding(function, "join").name == "join"
    left, right = function.registers or ()
    with pytest.raises(ValueError, match="scalar word"):
        adjoint_word_rebinding(
            replace(
                function,
                registers=(
                    left,
                    Register(
                        right.name, RegisterType("uint", 2), right.shape, right.side
                    ),
                ),
            ),
            "invalid",
        )
    with pytest.raises(ValueError):
        adjoint_word_rebinding(
            replace(function, blocks=(replace(function.blocks[0], operations=()),)),
            "invalid",
        )


@pytest.mark.parametrize(
    "kind,width,adjoint",
    tuple(
        product(
            ("interim", "gradient_controlled", "qft_reverse"), (2, 3, 4), (False, True)
        )
    ),
)
def test_recursive_call_layers_preserve_original_operands(
    kind: str, width: int, adjoint: bool
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.mcmt.and_bloq import And

    source, record = _case(kind, width, adjoint)
    pending = [(record.functions[0].name, source)]
    functions = {f.name: f for f in record.functions}
    seen = set()
    while pending:
        name, original = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        function = functions[name]
        operations = function.blocks[0].operations
        if not operations or any(op.opcode != "call" for op in operations):
            continue
        try:
            graph = original.decompose_bloq()
        except DecomposeTypeError:
            assert isinstance(original, And)
            continue
        assert _native_trace(record, name) == _source_trace(SourceCalls, graph)
        for instance, _, _ in graph.iter_bloqsoqs():
            callee = operations[instance.i].callee
            assert callee is not None
            pending.append((callee, instance.bloq))


@pytest.mark.parametrize("width", [1, 2, 3])
def test_qpe_unfused_power_graph_full_output_amplitudes_and_clean_work(
    width: int,
) -> None:
    from math import sqrt

    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState
    from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.algorithms.primitives.textbook_phase_estimation import (
        textbook_z_phase_estimation_program,
    )
    from flagquantum.core.ir import CircuitIR
    from tests.unit.measurement_checks import forced_branch

    source = TextbookQPE(ZPowGate(0.468), RectangularWindowState(width))
    record = textbook_z_phase_estimation_program(width, 0.468)
    assert not verify_program_record(record)
    assert copy_component_program(record) == record
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    bound = bind_register_program(
        record,
        ((width,),),
        clean=(*range(width), width + 1),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == (tuple(range(width)), (width,))
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    expected = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    actual = logical_operator(bound.circuit, width + 1)[:, :2]
    torch.testing.assert_close(actual, expected, atol=8e-12, rtol=0)
    measurements = sum(i.name == "measure" for i in bound.circuit.instructions)
    assert measurements == 2**width - 1 + width * (width - 1) // 2
    # Both target eigenstates and coherent superpositions retain absolute phase.
    seed = torch.tensor([0.6, 0.8j], dtype=torch.complex128)
    circuit = Circuit(bound.circuit.n_wires, dtype=torch.complex128)
    circuit.append(FieldOperation(bound.circuit.n_wires, bound.circuit.instructions))
    assert len(circuit.to_ir().instructions) == len(bound.circuit.instructions)
    if width <= 2:
        initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
        initial[0], initial[2] = seed
        for history in product((0, 1), repeat=measurements):
            result = forced_branch(initial, bound.circuit.instructions, history) * sqrt(
                2**measurements
            )
            torch.testing.assert_close(result[::2], expected @ seed, atol=8e-12, rtol=0)
            torch.testing.assert_close(
                result[1::2], torch.zeros_like(result[1::2]), atol=8e-12, rtol=0
            )


@pytest.mark.parametrize(
    "identifier",
    [
        "rectangular_window_state_small",
        "rectangular_window_state_symbolic",
        "lprs_interim_prep",
        "textbook_qpe_small",
    ],
)
def test_complete_window_and_qpe_catalog_calls_and_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from examples.algorithms import rotations_ui_export as catalog
    from tests.unit.test_field_product_decomposition import source_walk

    rows = source_walk(identifier, originals[1][identifier], catalog=catalog)
    assert any(r["disposition"] == "exact_graph" for r in rows)


@pytest.mark.parametrize("width,adjoint", tuple(product((5, 6, 8, 12), (False, True))))
def test_wide_qft_source_moments_keep_inner_swap_order(
    width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook

    original = QFTTextBook(width)
    if adjoint:
        original = original.adjoint()
    record = textbook_qft_program(width, adjoint=adjoint)
    assert _native_trace(record) == _source_trace(
        SourceCalls, original.decompose_bloq()
    )
