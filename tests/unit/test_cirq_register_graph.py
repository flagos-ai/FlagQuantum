"""Import unchanged typed source graphs without losing borrowed lifetimes."""

import builtins
from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_subtraction_foreign_file import native_signature
from tests.unit.test_unary_borrowed_child_calls import (
    borrowing_source as borrowing_source,
)
from tests.unit.test_unary_component_calls import source_slot_names
from tests.unit.test_unary_iteration_allocating_child import (
    allocating_source as allocating_source,
)
from tests.unit.test_unary_iteration_file import source as source
from tests.unit.test_unary_iteration_source_binding import (
    folded_source as folded_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("borrowed", "private", "nested"))
@pytest.mark.parametrize("controlled", (False, True))
def test_actual_graph_import_preserves_stream_branches_and_work_origin(
    borrowing_source: Any,
    allocating_source: Any,
    folded_source: Any,
    controlled: bool,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    factory = {
        "borrowed": borrowing_source,
        "private": allocating_source,
        "nested": folded_source,
    }[kind]
    original = factory(controlled)
    logical = {"borrowed": 3, "private": 5, "nested": 4}[kind] + int(controlled)
    clean_count = (1 if kind == "borrowed" else 2) + int(controlled)
    clean = tuple(range(logical, logical + clean_count))
    borrowed = (logical + clean_count,) if kind == "borrowed" else ()
    # One spectator sits outside all declared/source workspace.
    environment_count = 1 + len(borrowed)
    width = logical + clean_count + environment_count
    ports = {}
    inputs = []
    offset = 0
    for register in original.signature:
        first = offset
        for _ in register.all_idxs():
            size = int(register.bitsize)
            inputs.append(tuple(range(offset, offset + size)))
            offset += size
        ports[register.name] = tuple(range(first, offset))
    original_import = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError(f"Imported Program retained SDK dependency {name}")
        return original_import(name, *args, **kwargs)

    for adjoint in (False, True):
        direction = original.adjoint() if adjoint else original
        program = import_register_graph(direction)
        assert program.functions[0].registers == tuple(
            native_signature(direction.signature)
        )
        reference = reference_arithmetic_gates(
            direction,
            ports,
            clean,
            dict(zip(clean, range(clean_count), strict=True)),
            cirq_style=False,
            keep_pauli_leaves=True,
            borrowed_allocations=borrowed,
        )
        # The SDK is needed only for conversion. All subsequent construction,
        # binding, append and quantum trajectories use the owned native record.
        with monkeypatch.context() as context:
            context.setattr(builtins, "__import__", native_import)
            bound = bind_register_program(
                program, tuple(inputs), clean=clean, borrowed=borrowed
            )
            assert bound.outputs == tuple(inputs)
            assert_source_dependencies(
                source_slot_names(bound.circuit.instructions, clean), reference
            )
            circuit = Circuit(width)
            circuit.append(FieldOperation(width, bound.circuit.instructions))
            appended = circuit.to_ir()
            assert CircuitIR.from_json(appended.to_json()) == appended
            counts = Counter(gate.name for gate in appended.instructions)
            events = Counter((event.kind, event.dirty) for event in bound.events)
            assert events[("allocate", False)] == events[("free", False)]
            assert (
                events[("allocate", True)]
                == events[("free", True)]
                == (3 if borrowed else 0)
            )
            coherent = torch.zeros(1 << width, dtype=torch.complex128)
            states = []
            for basis in range(1 << (logical + environment_count)):
                logical_basis, environment = divmod(basis, 1 << environment_count)
                index = (
                    logical_basis << (clean_count + environment_count)
                ) | environment
                state = torch.zeros_like(coherent)
                amplitude = torch.tensor(
                    complex(basis + 1, 2 - basis), dtype=state.dtype
                )
                state[index] = amplitude
                coherent += state
                states.append(state)
            states.append(coherent / coherent.norm())
            for state in states:
                for history in product((0, 1), repeat=counts["measure"]):
                    actual = forced_branch(state, appended.instructions, history)
                    expected = forced_branch(state, reference, history)
                    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
                    assert (
                        torch.count_nonzero(
                            actual.reshape(
                                -1, 1 << clean_count, 1 << environment_count
                            )[:, 1:, :]
                        )
                        == 0
                    )
        model = get_cost_value(direction, QECGatesCost())
        assert counts["measure"] == model.measurement
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert get_cost_value(direction, QubitCount()) == logical + clean_count + len(
            borrowed
        )


def test_graph_import_stored_identity_and_invalid_external_objects() -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QUInt
    from qualtran._infra.registers import Register, Signature

    source_builder, inputs = BloqBuilder.from_signature(
        Signature((Register("word", QUInt(3)),))
    )
    graph = source_builder.finalize(**inputs)
    program = import_register_graph(graph)
    bound = bind_register_program(program, ((0, 1, 2),))
    assert not bound.circuit.instructions and not bound.events
    assert bound.outputs == ((0, 1, 2),)
    with pytest.raises(TypeError, match="live typed component"):
        import_register_graph(object())


@pytest.mark.parametrize(
    "controls,width,targets", ((1, 3, 5), (2, 2, 4), (3, 1, 2), (0, 3, 4))
)
def test_original_single_axis_consumers_retain_all_graph_primitives(
    source: Any, controls: int, width: int, targets: int
) -> None:
    from qualtran.bloqs.multiplexers.unary_iteration_bloq_test import ApplyXToLthQubit

    original = ApplyXToLthQubit(width, targets, controls)
    _assert_graph_stream(original, max(0, width + controls - 1))


@pytest.mark.parametrize("shape", ((2, 3, 2), (2, 2, 2)))
def test_original_three_axis_consumers_retain_all_graph_primitives(
    source: Any, shape: tuple[int, int, int]
) -> None:
    from qualtran.bloqs.multiplexers.unary_iteration_bloq_test import ApplyXToIJKthQubit

    original = ApplyXToIJKthQubit(shape)
    _assert_graph_stream(
        original, sum((extent - 1).bit_length() for extent in shape) - 1
    )


def _assert_graph_stream(original: Any, clean_count: int) -> None:
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    inputs = []
    ports = {}
    logical = 0
    for register in original.signature:
        first = logical
        for _ in register.all_idxs():
            width = int(register.bitsize)
            inputs.append(tuple(range(logical, logical + width)))
            logical += width
        ports[register.name] = tuple(range(first, logical))
    clean = tuple(range(logical, logical + clean_count))
    for adjoint in (False, True):
        direction = original.adjoint() if adjoint else original
        program = import_register_graph(direction)
        assert program.functions[0].registers == tuple(
            native_signature(direction.signature)
        )
        bound = bind_register_program(program, tuple(inputs), clean=clean)
        reference = reference_arithmetic_gates(
            direction,
            ports,
            clean,
            dict(zip(clean, range(clean_count), strict=True)),
            cirq_style=False,
            keep_pauli_leaves=True,
        )
        assert_source_dependencies(
            source_slot_names(bound.circuit.instructions, clean), reference
        )
        assert bound.outputs == tuple(inputs)
        counts = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(direction, QECGatesCost())
        assert counts["measure"] == model.measurement
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        live: set[int] = set()
        peak = logical
        for event in bound.events:
            if event.kind == "allocate":
                live.update(event.wires)
                peak = max(peak, logical + len(live))
            else:
                live.difference_update(event.wires)
        assert not live
        assert get_cost_value(direction, QubitCount()) == peak


def test_live_leaf_without_a_physical_protocol_is_rejected() -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.registers import Signature

    class UnserializedPhase(Bloq):
        def __init__(self, angle: float) -> None:
            self.angle = angle

        @property
        def signature(self) -> Any:
            return Signature.build(q=1)

    builder, inputs = BloqBuilder.from_signature(Signature.build(q=1))
    word = builder.add(UnserializedPhase(0.2), q=inputs["q"])
    word = builder.add(UnserializedPhase(0.7), q=word)
    graph = builder.finalize(q=word)
    with pytest.raises(ValueError, match="unsupported_operation"):
        import_register_graph(graph)


def test_distinct_live_phases_do_not_require_unique_constructor_wires() -> None:
    import cirq
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.registers import Signature

    class LivePhase(Bloq):
        def __init__(self, angle: float) -> None:
            self.angle = angle

        @property
        def signature(self) -> Any:
            return Signature.build(q=1)

        def as_cirq_op(self, qubit_manager: Any, **quregs: Any) -> Any:
            return cirq.rz(self.angle).on(quregs["q"][0]), quregs

    builder, inputs = BloqBuilder.from_signature(Signature.build(q=1))
    word = builder.add(LivePhase(0.2), q=inputs["q"])
    word = builder.add(LivePhase(0.7), q=word)
    bound = bind_register_program(
        import_register_graph(builder.finalize(q=word)), ((0,),)
    )
    assert [g.params["theta"] for g in bound.circuit.instructions] == pytest.approx(
        [0.2, 0.7], abs=1e-15, rel=0
    )
    state = torch.tensor([0.6, 0.8j], dtype=torch.complex128)
    expected = torch.as_tensor(cirq.unitary(cirq.rz(0.9)), dtype=state.dtype) @ state
    torch.testing.assert_close(
        forced_branch(state, bound.circuit.instructions, ()),
        expected,
        atol=2e-15,
        rtol=0,
    )
