"""Compare explicit Hamiltonian GQSP constructor ownership to unchanged source."""

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_hamiltonian_gqsp_factories import (
    HamiltonianGQSPContext,
    hamiltonian_gqsp_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.phase.hamiltonian_gqsp_component import (
    hamiltonian_gqsp_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem._generalized_qsp_component_library import (
    generalized_qsp_library_factories,
)
from flagquantum.ecosystem.cirq._register_graph import _leaf_program
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_generalized_qsp_component_library import (
    _assert_source_call_operands,
)
from tests.unit.test_hamiltonian_gqsp import _source
from tests.unit.test_legacy_arithmetic_factories import _factories, _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_KEY = "qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp.HamiltonianSimulationByGQSP"


def _capture(source: Any, name: str) -> ProgramRecord:
    from flagquantum.compiler.formats.legacy_registers import deserialize_registers

    registers = deserialize_registers(
        import_module("qualtran.serialization.registers")
        .registers_to_proto(source.signature)
        .SerializeToString()
    )
    record = _leaf_program(source, name, registers)
    return record if isinstance(record, ProgramRecord) else ProgramRecord((record,))


def _context(source: Any) -> tuple[ProgramRecord, HamiltonianGQSPContext]:
    walk = _capture(source.walk_operator, "walk")
    inverse = _capture(source.walk_operator.adjoint(), "inverse_walk")
    positive = (
        _capture(source.walk_operator.controlled(), "positive_walk"),
        _capture(source.walk_operator.adjoint().controlled(), "positive_inverse_walk"),
    )
    factory = next(
        iter(
            generalized_qsp_library_factories(
                child_owner=lambda _: (walk, inverse),
                positive_child_owner=lambda _: positive,
            ).values()
        )
    )
    processor = factory(
        "processor",
        None,
        {
            "U": walk.functions[0],
            "P": source.gqsp.P,
            "Q": source.gqsp.Q,
            "negative_power": source.gqsp.negative_power,
            "precision": source.gqsp.precision,
        },
    )
    assert isinstance(processor, ProgramRecord)
    prepare = source.walk_operator.prepare
    return walk, HamiltonianGQSPContext(
        processor,
        (_capture(prepare, "prepare"), _capture(prepare.adjoint(), "unprepare")),
        tuple(r.name for r in prepare.junk_registers),
    )


def _bound_check(
    source: Any, program: ProgramRecord, monkeypatch: pytest.MonkeyPatch
) -> Counter[str]:
    import cirq

    assert verify_program_record(program) == ()
    program = _rename_program(program, "main")
    _assert_source_call_operands(source, program)
    # Reversed data placement and a coherent spectator expose register/phase errors.
    ports = ((0,), (3,), (2,))
    bound = bind_register_program(program, ports, clean=(4, 5, 6), borrowed=(1,))
    assert bound.outputs == ports
    expected = torch.tensor(
        cirq.Circuit(
            cirq.MatrixGate(source.decompose_bloq().tensor_contract()).on(
                *(cirq.LineQubit(q) for q in (0, 3, 2))
            )
        ).unitary(qubit_order=cirq.LineQubit.range(4)),
        dtype=torch.complex128,
    )
    counts = Counter(op.name for op in bound.circuit.instructions)
    stride = 1 << (bound.circuit.n_wires - 4)
    measured = counts["measure"]
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        ir = CircuitIR.from_json(bound.circuit.to_json())
        for column in range(16):
            state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            state[column * stride] = 1
            target = torch.zeros_like(state)
            target[::stride] = expected[:, column] / 2 ** (measured / 2)
            for history in product((0, 1), repeat=measured):
                torch.testing.assert_close(
                    forced_branch(state, ir.instructions, history),
                    target,
                    atol=1e-11,
                    rtol=0,
                )
        amplitudes = torch.complex(
            torch.arange(1, 17).double(), torch.arange(16).double() * -3
        )
        amplitudes /= amplitudes.norm()
        state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
        state[::stride] = amplitudes
        circuit = Circuit(ir.n_wires, inputs=state, dtype=state.dtype).append(
            FieldOperation(ir.n_wires, ir.instructions)
        )
        assert (
            json.loads(circuit.to_ir().to_json())["instructions"]
            == json.loads(ir.to_json())["instructions"]
        )
        target = torch.zeros_like(state)
        target[::stride] = expected @ amplitudes
        if measured:
            from flagquantum.dynamic import DynamicCircuit
            from flagquantum.runtime.dynamic.execution import run_dynamic

            dynamic = DynamicCircuit(ir.n_wires, inputs=state, dtype=state.dtype)
            dynamic.append(FieldOperation(ir.n_wires, ir.instructions))
            result = run_dynamic(dynamic, shots=3, seed=17, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(3, -1),
                target.expand(3, -1),
                atol=1e-11,
                rtol=0,
            )
        else:
            torch.testing.assert_close(
                circuit.state().flatten(), target, atol=1e-11, rtol=0
            )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live
    return counts


@pytest.mark.parametrize("time,precision", [(0.15, 0.05), (-0.15, 0.05), (0.75, 1e-5)])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_source_constructor_library_and_exact_cleanup(
    time: float, precision: float, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(time, precision)
    walk, context = _context(source)
    library = _library(source, depth)
    seen = []

    def owner(
        child: FunctionRecord, t: ResolvedArgument, eps: ResolvedArgument
    ) -> HamiltonianGQSPContext:
        assert child.registers == walk.functions[0].registers
        assert t == time and eps == precision
        seen.append((t, eps))
        return context

    # Exact source-selected children are captured at the SDK boundary, then replaced
    # by complete native definitions before resolving the source-free constructor.
    children = (
        source.walk_operator,
        source.walk_operator.block_encoding,
        source.walk_operator.prepare,
        source.walk_operator.block_encoding.select,
    )
    factories: dict[str, ComponentFactory] = dict(_factories())
    for child in children:
        key = type(child).__module__ + "." + type(child).__qualname__
        child_program = _capture(child, "owned_child")

        def bind(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
            record: ProgramRecord = child_program,
        ) -> ProgramRecord:
            assert registers == record.functions[0].registers
            return _rename_program(record, name)

        factories[key] = bind
    factories.update(hamiltonian_gqsp_factories(owner))
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        program = resolve_component_program(library, 0, factories)
    assert seen == [(time, precision)]
    counts = _bound_check(source, program, monkeypatch)
    assert counts["measure"] == 0


def test_constructor_context_replacement_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(0.15, 0.05)
    walk, context = _context(source)
    args: dict[str, ResolvedArgument] = {
        "walk_operator": walk.functions[0],
        "t": source.t,
        "precision": source.precision,
    }
    first = hamiltonian_gqsp_factories(lambda *_: context)[_KEY](
        "simulation", None, args
    )
    second = hamiltonian_gqsp_factories(lambda *_: replace(context))[_KEY](
        "simulation", None, args
    )
    assert first == second
    assert isinstance(first, ProgramRecord)
    _bound_check(source, first, monkeypatch)
    with pytest.raises(SerializationError, match="walk, time"):
        hamiltonian_gqsp_factories(lambda *_: context)[_KEY](
            "simulation", None, {"walk_operator": walk.functions[0]}
        )
    with pytest.raises(SerializationError, match="walk ports"):
        hamiltonian_gqsp_factories(
            lambda *_: replace(context, gqsp=context.prepare_pair[0])
        )[_KEY]("simulation", None, args)
    with pytest.raises(SerializationError, match="register signature"):
        hamiltonian_gqsp_factories(lambda *_: context)[_KEY]("simulation", (), args)
    for names in (("junk", "junk"), ("missing",), ("selection",)):
        with pytest.raises(ValueError):
            hamiltonian_gqsp_component_program(
                context.gqsp, *context.prepare_pair, junk_names=names
            )
    with pytest.raises(ValueError, match="complete logical"):
        hamiltonian_gqsp_component_program(
            ProgramRecord(()), *context.prepare_pair, junk_names=context.junk_names
        )


@pytest.mark.parametrize("time,precision", [(0.15, 0.05), (-0.15, 0.05), (0.75, 1e-5)])
def test_native_numerical_injection_matches_actual_source(
    time: float, precision: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
    from flagquantum.algorithms.primitives.hamiltonian_gqsp import HamiltonianGQSP
    from flagquantum.core.ir import Instruction
    from flagquantum.ecosystem._hamiltonian_gqsp_component_library import (
        HamiltonianGQSPBinding,
        hamiltonian_gqsp_library_factories,
    )

    source = _source(time, precision)
    walk, context = _context(source)

    def forbidden_flat_emitter(*args: object) -> tuple[Instruction, ...]:
        raise AssertionError("Typed component binding must not call flat emitters")

    prepare = BlackBoxPrepare(
        (("selection", 1),), (("junk", 1),), forbidden_flat_emitter, 2.0
    )
    model = HamiltonianGQSP(
        (("selection", 1), ("target", 1)),
        forbidden_flat_emitter,
        prepare,
        time,
        precision,
    )
    binding = HamiltonianGQSPBinding(
        model,
        (walk, _capture(source.walk_operator.adjoint(), "inverse_walk")),
        (
            _capture(source.walk_operator.controlled(), "positive_walk"),
            _capture(
                source.walk_operator.adjoint().controlled(), "positive_inverse_walk"
            ),
        ),
        context.prepare_pair,
        context.junk_names,
    )
    np.testing.assert_array_equal(model.coefficients, source.approx_cos)
    np.testing.assert_array_equal(model.gqsp.q, source.gqsp.Q)
    assert (
        model.degree == source.degree and model.gqsp.precision == source.gqsp.precision
    )
    arguments: dict[str, ResolvedArgument] = {
        "walk_operator": walk.functions[0],
        "t": time,
        "precision": precision,
    }
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        program = hamiltonian_gqsp_library_factories(lambda *_: binding)[_KEY](
            "main", None, arguments
        )
    assert isinstance(program, ProgramRecord)
    _bound_check(source, program, monkeypatch)
    wrong = replace(binding, model=replace(model, time=time + 1))
    with pytest.raises(SerializationError, match="numerical model context"):
        hamiltonian_gqsp_library_factories(lambda *_: wrong)[_KEY](
            "main", None, arguments
        )


def test_junk_free_source_matrix_context_and_original_codec_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        random_qubitization_walk_operator,
    )
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
    )

    walk, _ = random_qubitization_walk_operator(
        1, 1, random_state=np.random.RandomState(42)
    )
    source = HamiltonianSimulationByGQSP(walk, t=0.15, precision=0.05)
    assert not source.walk_operator.prepare.junk_registers
    # Keep the original serializer failure distinct from executable typed SDK views.
    with pytest.raises(ValueError, match="Object arrays cannot be saved"):
        _library(source, 0)
    _, context = _context(source)
    assert context.junk_names == ()
    program = hamiltonian_gqsp_component_program(
        context.gqsp, *context.prepare_pair, junk_names=()
    )
    _bound_check(source, program, monkeypatch)
