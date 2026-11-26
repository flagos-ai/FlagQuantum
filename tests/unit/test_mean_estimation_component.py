"""Owned mean-estimation dependencies preserve source calls, phases and cleanup."""

import json
from collections import Counter
from collections.abc import Callable
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.mean_estimation_operator import (
    CodeForRandomVariable,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from flagquantum.compiler.formats.legacy_registers import deserialize_registers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._mean_estimation_library import (
    mean_estimation_component_pair,
)
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_complex_phase_component_library import (
    _private_source_encoder,
    _reference_operator,
)
from tests.unit.test_generalized_qsp_component_library import (
    _assert_source_call_operands,
)
from tests.unit.test_hamiltonian_gqsp_component_library import _capture
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _registers(source: Any) -> tuple[Register, ...]:
    return deserialize_registers(
        import_module("qualtran.serialization.registers")
        .registers_to_proto(source)
        .SerializeToString()
    )


def _build(
    source: Any, *, signal_junk_names: tuple[str, ...] | None = None
) -> tuple[ProgramRecord, ProgramRecord]:
    prep, encoder = source.code.synthesizer, source.code.encoder
    code = CodeForRandomVariable(
        _registers(prep.selection_registers),
        _registers(encoder.selection_registers),
        _registers(encoder.target_registers),
    )
    return mean_estimation_component_pair(
        code,
        (_capture(prep, "prepare"), _capture(prep.adjoint(), "unprepare")),
        (_capture(encoder, "encoder"), _capture(encoder.adjoint(), "erase_encoder")),
        arctan_bitsize=source.arctan_bitsize,
        signal_junk_names=signal_junk_names,
    )


def _check(
    source: Any,
    program: ProgramRecord,
    reference: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
    *,
    every_column_history: bool = True,
) -> Counter[str]:
    import cirq

    assert verify_program_record(program) == ()
    width = int(source.signature.n_qubits())
    targets = tuple(reversed(range(1, width + 1)))
    clean = tuple(range(width + 1, width + 7))
    bound = bind_register_program(program, (targets,), clean=clean, borrowed=(0,))
    assert bound.outputs == (targets,)
    expected = torch.tensor(
        cirq.Circuit(
            cirq.MatrixGate(reference.resolve_conj().numpy()).on(
                *(cirq.LineQubit(q) for q in targets)
            )
        ).unitary(qubit_order=cirq.LineQubit.range(width + 1)),
        dtype=torch.complex128,
    )
    counts = Counter(op.name for op in bound.circuit.instructions)
    events = counts["measure"]
    stride = 1 << len(clean)
    histories = tuple(product((0, 1), repeat=events))
    basis_histories = (
        histories if every_column_history else ((0,) * events, (1,) * events)
    )
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        ir = CircuitIR.from_json(bound.circuit.to_json())
        for column in range(1 << (width + 1)):
            state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            state[column * stride] = 1
            target = torch.zeros_like(state)
            target[::stride] = expected[:, column] / 2 ** (events / 2)
            for history in basis_histories:
                torch.testing.assert_close(
                    forced_branch(state, ir.instructions, history),
                    target,
                    atol=1e-11,
                    rtol=0,
                )
        values = torch.arange(1, (1 << (width + 1)) + 1).double()
        amplitudes = torch.complex(values, 2 - 3 * values)
        amplitudes /= amplitudes.norm()
        state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
        state[::stride] = amplitudes
        target = torch.zeros_like(state)
        target[::stride] = expected @ amplitudes
        for history in histories:
            torch.testing.assert_close(
                forced_branch(state, ir.instructions, history),
                target / 2 ** (events / 2),
                atol=1e-11,
                rtol=0,
            )
        circuit = DynamicCircuit(ir.n_wires, inputs=state, dtype=state.dtype)
        circuit.append(FieldOperation(ir.n_wires, ir.instructions))
        assert (
            json.loads(circuit.to_ir().to_json())["instructions"]
            == json.loads(ir.to_json())["instructions"]
        )
        result = run_dynamic(circuit, shots=3, seed=19, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(3, -1), target.expand(3, -1), atol=1e-11, rtol=0
        )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live
    return counts


@pytest.mark.parametrize("kind", ["bernoulli", "grover"])
@pytest.mark.parametrize("precision", [0, 1, 3])
def test_literal_mean_dependencies_and_both_exact_directions(
    kind: str, precision: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cirq
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
        GroverEncoder,
        GroverSynthesizer,
    )

    if kind == "bernoulli":
        prep, encoder = BernoulliSynthesizer(0.37, 1), BernoulliEncoder(
            0.37, (0, 1), 1, 1
        )
    else:
        prep, encoder = GroverSynthesizer(2), GroverEncoder(2, 1, 1)
    source = MeanEstimationOperator(SourceCode(prep, encoder), precision)
    for depth in (0, 1, 2):
        with pytest.raises(ValueError, match="Cannot serialize CodeForRandomVariable"):
            _library(source, depth)
    pair = _build(source)
    for program, original in zip(pair, (source, source.adjoint()), strict=True):
        _assert_source_call_operands(original, program)
        reference = torch.tensor(cirq.unitary(original), dtype=torch.complex128)
        _check(
            source, program, reference, monkeypatch, every_column_history=precision <= 1
        )


def _junk_synthesizer(*, entangled: bool = False) -> Any:
    import attrs
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    q = import_module("qualtran")
    gates = import_module("qualtran.bloqs.basic_gates")

    def build(self: Any, bb: Any, **soqs: Any) -> Any:
        selection = bb.split(soqs["selection"])
        selection[0] = bb.add(gates.Hadamard(), q=selection[0])
        if self.entangled:
            selection[0], junk = bb.add(
                gates.CNOT(), ctrl=selection[0], target=soqs["junk"]
            )
        else:
            selection[1] = bb.add(gates.SGate(), q=selection[1])
            junk = bb.add(gates.XGate(), q=soqs["junk"])
            junk = bb.add(gates.Hadamard(), q=junk)
        return {"selection": bb.join(selection, dtype=q.QAny(2)), "junk": junk}

    cls = attrs.frozen(
        type(
            "JunkSynthesizer",
            (PrepareOracle,),
            {
                "__module__": __name__,
                "__annotations__": {"entangled": bool},
                "selection_registers": property(
                    lambda self: (q.Register("selection", q.QAny(2)),)
                ),
                "junk_registers": property(
                    lambda self: (q.Register("junk", q.QBit()),)
                ),
                "build_composite_bloq": build,
            },
        )
    )
    return cls(entangled)


def test_measured_encoder_and_clean_synthesizer_junk_share_owned_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator,
    )

    source = MeanEstimationOperator(
        SourceCode(_junk_synthesizer(), _private_source_encoder()), 1
    )
    pair = _build(source)
    phase = _reference_operator(source.select, encoder_tensors=True)
    reflection = torch.tensor(
        source.reflect.decompose_bloq().tensor_contract(), dtype=torch.complex128
    )
    reference = reflection @ phase
    torch.testing.assert_close(
        reference.conj().T @ reference,
        torch.eye(4, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    for direction, program in enumerate(pair):
        _assert_source_call_operands(
            source if direction == 0 else source.adjoint(), program
        )
        _check(
            source,
            program,
            reference if direction == 0 else reference.conj().T,
            monkeypatch,
        )


def test_source_entangled_junk_counterexample_is_exposed_without_resetting_work() -> (
    None
):
    import cirq
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator,
    )

    from tests.unit.test_complex_phase_component_library import _source_operations

    source = MeanEstimationOperator(
        SourceCode(_junk_synthesizer(entangled=True), _private_source_encoder()), 1
    )
    program, _ = _build(source)
    clean = tuple(range(2, 8))
    bound = bind_register_program(program, ((0, 1),), clean=clean)
    phase_name = program.functions[0].blocks[0].operations[0].callee
    assert phase_name is not None
    phase_bound = bind_register_program(
        program, ((0, 1),), clean=clean, entry=phase_name
    )
    phase_end = len(phase_bound.circuit.instructions)
    junk_event = next(
        event
        for event in bound.events
        if event.kind == "allocate"
        and event.instruction_offset is not None
        and event.instruction_offset >= phase_end
    )
    (junk_wire,) = junk_event.wires
    ops, ports = _source_operations(source.reflect)
    public = tuple(q for r in source.reflect.signature for q in ports[r.name].flat)
    literal = cirq.Circuit(ops)
    hidden = tuple(sorted(literal.all_qubits() - set(public)))
    order = (*public, *hidden)
    unprepare = next(op for op in ops if op.gate == source.code.synthesizer.adjoint())
    (source_junk,) = tuple(q for q in unprepare.qubits if q not in public)
    phase = _reference_operator(source.select, encoder_tensors=True)
    state = torch.zeros(1 << len(order), dtype=torch.complex128)
    state[:: 1 << len(hidden)] = phase[:, 0]
    reference = (
        torch.tensor(literal.unitary(qubit_order=order), dtype=torch.complex128) @ state
    )
    target = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    for row, amplitude in enumerate(reference):
        if abs(amplitude) < 1e-12:
            continue
        digits = {q: (row >> (len(order) - 1 - i)) & 1 for i, q in enumerate(order)}
        assert all(digits[q] == 0 for q in hidden if q != source_junk)
        position = sum(
            digits[q] << (bound.circuit.n_wires - 1 - i) for i, q in enumerate(public)
        )
        position |= digits[source_junk] << (bound.circuit.n_wires - 1 - junk_wire)
        target[position] = amplitude
    initial = torch.zeros_like(target)
    initial[0] = 1
    events = sum(op.name == "measure" for op in bound.circuit.instructions)
    for history in product((0, 1), repeat=events):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, target / 2 ** (events / 2), atol=1e-11, rtol=0
        )
        assert float(actual[:: 1 << len(clean)].abs().square().sum()) < 1e-20
    # The original projected public tensor has lost the entire first column.
    # This is not an accepted clean-junk context, even though PREPARE has an inverse.
    projected = torch.tensor(
        source.reflect.decompose_bloq().tensor_contract(), dtype=torch.complex128
    )
    assert float(projected[:, 0].abs().square().sum()) < 1e-20


@pytest.mark.parametrize("kind", ["bernoulli", "grover"])
def test_exact_forward_inverse_roundtrip_every_joint_small_history(kind: str) -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
        GroverEncoder,
        GroverSynthesizer,
    )

    if kind == "bernoulli":
        prep, encoder = BernoulliSynthesizer(0.37, 1), BernoulliEncoder(
            0.37, (0, 1), 1, 1
        )
    else:
        prep, encoder = GroverSynthesizer(2), GroverEncoder(2, 1, 1)
    source = MeanEstimationOperator(SourceCode(prep, encoder), 1)
    pair = _build(source)
    width = int(source.signature.n_qubits())
    clean = tuple(range(width, width + 6))
    bounds = tuple(
        bind_register_program(p, (tuple(range(width)),), clean=clean) for p in pair
    )
    instructions = (*bounds[0].circuit.instructions, *bounds[1].circuit.instructions)
    raw = torch.arange(1, (1 << width) + 1).double()
    amplitudes = torch.complex(raw, 3 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (width + 6), dtype=torch.complex128)
    initial[::64] = amplitudes
    count = sum(op.name == "measure" for op in instructions)
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(initial, instructions, history),
            initial / 2 ** (count / 2),
            atol=1e-11,
            rtol=0,
        )


def test_source_encoder_control_missing_from_mean_signature_is_rejected() -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator,
    )

    from flagquantum.errors import SerializationError
    from tests.unit.test_complex_phase_component_library import _source_operations

    source = MeanEstimationOperator(
        SourceCode(_junk_synthesizer(), _private_source_encoder(controlled=True)), 1
    )
    with pytest.raises(KeyError, match="control"):
        _source_operations(source)
    with pytest.raises(SerializationError, match="encoder disagrees"):
        _build(source)


def test_pair_observer_keeps_complete_forward_definition_and_requires_exact_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from qualtran.bloqs.mean_estimation.complex_phase_oracle_test import ExampleSelect

    from flagquantum.compiler.formats.legacy_complex_phase_factories import (
        ComplexPhaseContext,
        complex_phase_factories,
    )
    from flagquantum.ecosystem.cirq._complex_phase_library import (
        complex_phase_library_factories,
    )
    from flagquantum.errors import SerializationError

    source = ExampleSelect(1)
    encoder = (_capture(source, "encoder"), _capture(source.adjoint(), "erase_encoder"))
    arguments: dict[str, ResolvedArgument] = {
        "encoder": encoder[0].functions[0],
        "arctan_bitsize": 1,
    }
    pairs: list[tuple[FunctionRecord, tuple[ProgramRecord, ProgramRecord]]] = []
    ordinary = next(
        iter(
            complex_phase_library_factories(
                encoder_owner=lambda _: encoder, target_names=lambda _: ("target",)
            ).values()
        )
    )
    observed = next(
        iter(
            complex_phase_library_factories(
                encoder_owner=lambda _: encoder,
                target_names=lambda _: ("target",),
                on_pair=lambda child, pair: pairs.append((child, pair)),
            ).values()
        )
    )
    forward = ordinary("phase_entry", None, arguments)
    result = observed("phase_entry", None, arguments)
    assert isinstance(forward, ProgramRecord)
    assert result == forward
    assert pairs[0][0] == forward.functions[0] and pairs[0][1][0] == forward
    assert pairs[0][1][1].functions[0].name == "phase_entry"
    module = import_module("flagquantum.ecosystem.cirq._complex_phase_library")
    contexts: list[ComplexPhaseContext] = []
    original = complex_phase_factories

    def capture(
        owner: Callable[[FunctionRecord, int], ComplexPhaseContext],
        *,
        on_pair: (
            Callable[[FunctionRecord, tuple[ProgramRecord, ProgramRecord]], None] | None
        ) = None,
    ) -> dict[str, ComponentFactory]:
        def own(child: FunctionRecord, precision: int) -> ComplexPhaseContext:
            context = owner(child, precision)
            contexts.append(context)
            return context

        return original(own, on_pair=on_pair)

    with monkeypatch.context() as patch:
        patch.setattr(module, "complex_phase_factories", capture)
        factory = next(
            iter(
                module.complex_phase_library_factories(
                    encoder_owner=lambda _: encoder, target_names=lambda _: ("target",)
                ).values()
            )
        )
        factory("main", None, arguments)
    missing = replace(contexts[0], inverse_phases=None)
    factory = next(
        iter(
            complex_phase_factories(
                lambda *_: missing, on_pair=lambda *_: None
            ).values()
        )
    )
    with pytest.raises(SerializationError, match="exact inverse phase owners"):
        factory("main", None, arguments)


def test_explicit_caller_junk_preserves_full_source_action_and_clean_private_work() -> (
    None
):
    import cirq
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator,
    )

    from tests.unit.test_complex_phase_component_library import _source_operations

    source = MeanEstimationOperator(
        SourceCode(_junk_synthesizer(entangled=True), _private_source_encoder()), 1
    )
    pair = _build(source, signal_junk_names=("state_junk",))
    assert tuple(r.name for r in pair[0].functions[0].registers or ()) == (
        "selection",
        "state_junk",
    )
    ops, ports = _source_operations(source.reflect)
    public = tuple(q for r in source.reflect.signature for q in ports[r.name].flat)
    literal = cirq.Circuit(ops)
    unprepare = next(op for op in ops if op.gate == source.code.synthesizer.adjoint())
    (junk,) = tuple(q for q in unprepare.qubits if q not in public)
    (flag,) = tuple(literal.all_qubits() - set(public) - {junk})
    matrix = torch.tensor(
        literal.unitary(qubit_order=(*public, junk, flag)), dtype=torch.complex128
    )
    torch.testing.assert_close(
        matrix[1::2, ::2],
        torch.zeros((8, 8), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    source_phase = _reference_operator(source.select, encoder_tensors=True)
    reference = matrix[::2, ::2] @ torch.kron(
        source_phase, torch.eye(2, dtype=torch.complex128)
    )
    for direction, program in enumerate(pair):
        bound = bind_register_program(program, ((0, 1), (2,)), clean=tuple(range(3, 9)))
        assert bound.outputs == ((0, 1), (2,))
        expected = reference if direction == 0 else reference.conj().T
        count = sum(op.name == "measure" for op in bound.circuit.instructions)
        for column in range(8):
            state = torch.zeros(1 << 9, dtype=torch.complex128)
            state[column << 6] = 1
            target = torch.zeros_like(state)
            target[::64] = expected[:, column] / 2 ** (count / 2)
            for history in product((0, 1), repeat=count):
                torch.testing.assert_close(
                    forced_branch(state, bound.circuit.instructions, history),
                    target,
                    atol=1e-11,
                    rtol=0,
                )
        if direction == 0:
            # On input00|junk0>, the source moves junk to1. That is an explicit
            # logical output here; every private physical wire still returns0.
            assert abs(reference[5, 0]) == pytest.approx(1, abs=1e-11)


def test_default32_full_gate_stream_with_sparse_verifier_avoids_dense_work_state() -> (
    None
):
    import cirq
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

    from tests.unit.sparse_register_execution import execute_sparse_register

    source_arctan = import_module(
        "qualtran.bloqs.arithmetic.trigonometric.arctan"
    ).ArcTan
    prep, encoder = BernoulliSynthesizer(0.37, 1), BernoulliEncoder(0.37, (0, 1), 1, 1)
    source = MeanEstimationOperator(SourceCode(prep, encoder))
    assert source.arctan_bitsize == 32
    # For this actual one-bit encoder, original apply produces exactly the same
    # signed fraction at precision1 and32. This permits an independent small
    # original SDK matrix reference without constructing its huge ancilla state.
    for value in (0, 1):
        _, sign, word = source_arctan(1, 32).apply(value, 0, 0)
        _, other_sign, other_word = source_arctan(1, 1).apply(value, 0, 0)
        assert sign == other_sign and word / 2**32 == other_word / 2
    ideal = torch.tensor(
        cirq.unitary(MeanEstimationOperator(SourceCode(prep, encoder), 1)),
        dtype=torch.complex128,
    )
    reflection = torch.tensor(cirq.unitary(source.reflect), dtype=torch.complex128)
    qubits = cirq.LineQubit.range(2)

    def literal_phase(value: int, *, adjoint: bool) -> complex:
        _, sign, word = source_arctan(1, 32).apply(value, 0, 0)
        amplitude = 1 + 0j
        for power in range(32):
            bit = (word >> (31 - power)) & 1
            for polarity, coefficient in ((0, 1), (1, -1)):
                gate = (cirq.Z ** (coefficient / 2 ** (1 + power))).controlled(
                    control_values=[polarity]
                )
                if adjoint:
                    gate = cirq.inverse(gate)
                # Independently execute the unchanged SDK's selected primitive
                # recipe, without importing it through FlagQuantum conversion.
                operations = cirq.decompose(
                    gate.on(*qubits),
                    keep=lambda op: len(op.qubits) <= 1
                    or op.gate == cirq.CNOT
                    or isinstance(op.gate, cirq.CZPowGate),
                )
                matrix = cirq.Circuit(operations).unitary(qubit_order=qubits)
                index = 2 * sign + bit
                amplitude *= complex(matrix[index, index])
        return amplitude

    phase = torch.diag(
        torch.tensor(
            [literal_phase(value, adjoint=False) for value in (0, 1)],
            dtype=torch.complex128,
        )
    )
    inverse_phase = torch.diag(
        torch.tensor(
            [literal_phase(value, adjoint=True) for value in (0, 1)],
            dtype=torch.complex128,
        )
    )
    references = (reflection @ phase, inverse_phase @ reflection)
    # Preserve the discovered source counterexample instead of calling source
    # stream parity mathematical accuracy. No global-phase alignment is used.
    assert torch.max(torch.abs(references[0] - ideal)).item() > 1e-5
    torch.testing.assert_close(inverse_phase, phase.conj().T, atol=1e-11, rtol=0)
    pair = _build(source)
    for direction, program in enumerate(pair):
        bound = bind_register_program(program, ((0,),), clean=tuple(range(1, 37)))
        expected = references[direction]
        for column in range(2):
            for seed in (0, 19, 31):
                actual = execute_sparse_register(
                    bound.circuit, {column << 36: 1 + 0j}, seed=seed
                )
                target = {
                    row << 36: complex(expected[row, column])
                    for row in range(2)
                    if abs(expected[row, column]) > 1e-12
                }
                assert actual == pytest.approx(target, abs=1e-11)
                assert all(key % 2**36 == 0 for key in actual)
