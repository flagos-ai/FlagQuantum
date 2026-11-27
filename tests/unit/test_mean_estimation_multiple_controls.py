"""Typed mean controls preserve source predicates, complex branches and work."""

import sys
from collections import Counter
from itertools import product
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.mean_estimation_operator import (
    CodeForRandomVariable,
)
from flagquantum.compiler.formats.legacy_component_arguments import ControlSpecification
from flagquantum.compiler.registers.lookup.arctan_lookup_component import (
    arctan_lookup_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._mean_estimation_library import (
    controlled_mean_estimation_component_pair,
)
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_generalized_qsp_component_library import (
    _assert_source_call_operands,
)
from tests.unit.test_hamiltonian_gqsp_component_library import _capture
from tests.unit.test_mean_estimation_component import _registers
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def bind(program: ProgramRecord) -> tuple[BoundRegisterProgram, int, tuple[int, ...]]:
    offset = 0
    inputs = []
    for register in program.functions[0].registers or ():
        for _ in register.all_idxs():
            width = int(register.bitsize)
            inputs.append(tuple(range(offset, offset + width)))
            offset += width
    provisional = bind_register_program(
        program, tuple(inputs), clean=tuple(range(offset, offset + 40))
    )
    total = (
        max(
            (w for event in provisional.events for w in event.wires), default=offset - 1
        )
        + 1
    )
    clean = tuple(range(offset, total))
    bound = bind_register_program(program, tuple(inputs), clean=clean)
    assert bound.outputs == tuple(inputs)
    return bound, offset, clean


def context(
    precision: int, kind: str
) -> tuple[Any, tuple[ProgramRecord, ProgramRecord], tuple[int, ...]]:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QBit, QUInt
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
    )

    prep, encoder = BernoulliSynthesizer(0.37, 1), BernoulliEncoder(0.37, (0, 1), 1, 1)
    if kind == "collision":
        import attrs
        from qualtran._infra.registers import Register
        from qualtran.bloqs.multiplexers.select_base import SelectOracle

        def encode(self: Any, context: Any, q: Any, ctrl: Any) -> Any:
            try:
                import cirq
            except ModuleNotFoundError as _reference_import_error:
                unavailable_reference(_reference_import_error)

            yield cirq.CNOT(q[0], ctrl[0])

        encoder_class = attrs.frozen(
            type(
                "TargetNamedControl",
                (SelectOracle,),
                {
                    "__module__": __name__,
                    "control_registers": property(lambda _: ()),
                    "selection_registers": property(lambda _: prep.selection_registers),
                    "target_registers": property(lambda _: (Register("ctrl", QBit()),)),
                    "decompose_from_registers": encode,
                },
            )
        )
        encoder = encoder_class()
    bits: tuple[int, ...]
    if kind == "word":
        spec = ControlSpecification(
            (
                (unsigned_register_type(2), np.array(1)),
                (quantum_bit_type(), np.array(0)),
            )
        )
        original = CtrlSpec(qdtypes=(QUInt(2), QBit()), cvs=(np.array(1), np.array(0)))
        bits = (0, 1, 0)
    else:
        bits = {
            "positive": (1, 1),
            "mixed": (0, 1),
            "single": (1,),
            "negative": (0,),
            "collision": (0, 1),
        }[kind]
        values = np.array(bits if len(bits) > 1 else bits[0])
        spec = ControlSpecification(((quantum_bit_type(), values),))
        original = CtrlSpec(cvs=values)
    positive = encoder.get_ctrl_system(CtrlSpec())[0]
    native_code = CodeForRandomVariable(
        _registers(prep.selection_registers),
        _registers(encoder.selection_registers),
        _registers(encoder.target_registers),
    )
    pair = controlled_mean_estimation_component_pair(
        native_code,
        (_capture(prep, "prepare"), _capture(prep.adjoint(), "unprepare")),
        (_capture(encoder, "encoder"), _capture(encoder.adjoint(), "erase")),
        (
            _capture(positive, "positive_encoder"),
            _capture(positive.adjoint(), "positive_erase"),
        ),
        spec,
        arctan_bitsize=precision,
    )
    source = MeanEstimationOperator(
        SourceCode(prep, encoder), precision
    ).get_ctrl_system(original)[0]
    return source, pair, bits


@pytest.mark.parametrize(
    "kind", ("positive", "mixed", "word", "single", "negative", "collision")
)
@pytest.mark.parametrize("precision", (1, 3))
def test_actual_source_control_calls_full_complex_operator_all_histories_and_clean_work(
    kind: str, precision: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    source, pair, pattern = context(precision, kind)
    reference = torch.as_tensor(
        cirq.unitary(
            source.subbloq.controlled(num_controls=len(pattern), control_values=pattern)
        ),
        dtype=torch.complex128,
    )
    for inverse, graph in enumerate(pair):
        assert not verify_program_record(graph)
        assert _named_inputs(_native_trace(graph)) == _named_inputs(
            _source_trace(
                SourceCalls, (source.adjoint() if inverse else source).decompose_bloq()
            )
        )
        bound, logical, clean = bind(graph)
        events = certify_erasure_histories(
            bound.circuit.instructions, n_wires=bound.circuit.n_wires, zero_wires=clean
        )
        assert events == Counter(g.name for g in bound.circuit.instructions)["measure"]
        stride = 1 << len(clean)
        target = reference.mH if inverse else reference
        for column in range(1 << logical):
            initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
            initial[column * stride] = 1
            expected = torch.zeros_like(initial)
            expected[::stride] = target[:, column]
            for history in ((0,) * events, (1,) * events):
                torch.testing.assert_close(
                    forced_branch(initial, bound.circuit.instructions, history)
                    * 2 ** (events / 2),
                    expected,
                    atol=1e-11,
                    rtol=0,
                )
        weights = torch.arange(1, (1 << logical) + 1).to(torch.complex128)
        weights[1::2] *= 1j
        weights /= weights.norm()
        initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
        initial[::stride] = weights
        expected = torch.zeros_like(initial)
        expected[::stride] = target @ weights
        assert events <= 8
        with monkeypatch.context() as patch:
            for sdk in ("qualtran", "cirq", "qiskit"):
                patch.setitem(sys.modules, sdk, None)
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(initial, bound.circuit.instructions, history)
                    * 2 ** (events / 2),
                    expected,
                    atol=1e-11,
                    rtol=0,
                )
            circuit = DynamicCircuit(bound.circuit.n_wires).append(
                FieldOperation(bound.circuit.n_wires, bound.circuit.instructions)
            )
            assert circuit.to_ir().instructions == bound.circuit.instructions
            executable = DynamicCircuit(
                bound.circuit.n_wires, inputs=initial, dtype=initial.dtype
            )
            executable.append(
                FieldOperation(bound.circuit.n_wires, bound.circuit.instructions)
            )
            result = run_dynamic(executable, shots=2, seed=19, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(2, -1),
                expected.expand(2, -1),
                atol=1e-11,
                rtol=0,
            )


@pytest.mark.parametrize("source_width,precision", ((0, 1), (1, 0), (1, 1), (2, 3)))
def test_positive_arctan_lookup_all_dirty_columns_matches_original_cirq_unitary(
    source_width: int, precision: int
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    graph = arctan_lookup_program(source_width, precision, controlled=True)
    bound, logical, clean = bind(graph)
    matrix = cirq.unitary(ArcTan(source_width, precision).controlled())
    rows = tuple(product((0, 1), repeat=logical))
    inputs = tuple((*row, *(0 for _ in clean)) for row in rows)
    result, events = certified_and_channel_map(bound.circuit.instructions, inputs)
    assert events == Counter(g.name for g in bound.circuit.instructions)["measure"]
    for column, row in enumerate(result):
        value = sum(bit << (len(row) - i - 1) for i, bit in enumerate(row))
        expected = int(np.abs(matrix[:, column]).argmax()) << len(clean)
        assert value == expected
        assert matrix[expected >> len(clean), column] == 1


@pytest.mark.parametrize("precision", (1, 3))
def test_positive_mean_two_stage_calls_and_exact_integer_powers(precision: int) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.core.ir.program.signatures import Signature

    source, pair, _ = context(precision, "single")
    for graph, original in zip(pair, (source, source.adjoint()), strict=True):
        _assert_source_call_operands(original, graph)
    for exponent in (-2, 2):
        graph = pair[int(exponent < 0)]
        builder = ComponentProgramBuilder(Signature(graph.functions[0].registers or ()))
        state = builder.inputs
        for _ in range(abs(exponent)):
            state = builder.call_program(graph, state)
        bound, logical, clean = bind(builder.finalize(state))
        events = certify_erasure_histories(
            bound.circuit.instructions, n_wires=bound.circuit.n_wires, zero_wires=clean
        )
        reference = torch.tensor(
            cirq.unitary(source.subbloq**exponent), dtype=torch.complex128
        )
        target = torch.eye(2 * len(reference), dtype=torch.complex128)
        target[len(reference) :, len(reference) :] = reference
        stride = 1 << len(clean)
        for column in range(1 << logical):
            initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
            initial[column * stride] = 1
            expected = torch.zeros_like(initial)
            expected[::stride] = target[:, column]
            for history in ((0,) * events, (1,) * events):
                torch.testing.assert_close(
                    forced_branch(initial, bound.circuit.instructions, history)
                    * 2 ** (events / 2),
                    expected,
                    atol=1e-11,
                    rtol=0,
                )


@pytest.mark.parametrize("precision", (1, 3))
def test_positive_phase_recursive_stage_operands_preserve_bit_views(
    precision: int,
) -> None:
    from tests.unit.bit_view_call_checks import assert_source_bit_view_calls

    source, pair, _ = context(precision, "single")
    graph = pair[0]
    phase_entry = graph.functions[0].blocks[0].operations[0].callee
    assert phase_entry is not None
    source_phase = next(source.decompose_bloq().iter_bloqsoqs())[0].bloq
    assert_source_bit_view_calls(source_phase.decompose_bloq(), graph, phase_entry)
