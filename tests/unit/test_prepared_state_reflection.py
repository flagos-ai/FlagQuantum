"""Bind complete prepared-state reflections without changing append consumers."""

from cmath import exp
from hashlib import sha1
from itertools import product
from math import sqrt
from pathlib import Path

import pytest
import torch
from sympy import Symbol

from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.prepared_state_reflection import (
    PreparedStateReflection,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_requirements,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_controlled_unitary_synthesis import assert_source_stream
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_whole_sources() -> None:
    from qualtran.bloqs.reflections import reflection_using_prepare

    for name, blob in (
        ("reflection_using_prepare.py", "4a8467fde126bd596be69286d88c198726a4557a"),
        (
            "reflection_using_prepare_test.py",
            "df403c18060daf172e68c4a80099c306743a7a3a",
        ),
    ):
        raw = (Path(reflection_using_prepare.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def _with_clean_work(state: torch.Tensor, width: int) -> torch.Tensor:
    result = torch.zeros(state.numel() << width, dtype=state.dtype)
    result[:: 1 << width] = state
    return result


@pytest.mark.parametrize("width", range(4))
@pytest.mark.parametrize("value", (None, 0, 1))
@pytest.mark.parametrize("coefficient", (1, -1, 1j, -1j, exp(0.37j)))
def test_complete_bound_identity_reflection_and_all_inverse_histories(
    width: int, value: int | None, coefficient: complex
) -> None:
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    widths = () if not width else ((1, width - 1) if width > 1 else (1,))
    source = ReflectionUsingPrepare.reflection_around_zero(
        widths, control_val=value, global_phase=coefficient
    )
    native = PreparedStateReflection.around_zero(widths, value, coefficient)
    assert [(r.name, r.bitsize, r.side.value) for r in native.signature] == [
        (r.name, r.bitsize, r.side.value) for r in source.signature
    ]
    assert native.adjoint() is native and source.adjoint() is source
    data_width = width + int(value is not None)
    q, c = prepare_reflection_requirements(width, controlled=value is not None)
    work = tuple(range(1 + data_width, 1 + data_width + q))
    selection = {}
    offset = 1 + int(value is not None)
    for reg in native.selection_registers:
        selection[reg.name] = tuple(range(offset, offset + reg.bitsize))
        offset += reg.bitsize
    operation = native.make_on(
        selection=selection,
        reflection_workspace=work,
        reflection_classical_bits=tuple(range(c)),
        control=None if value is None else 1,
    )
    reverse = native.make_on(
        selection=selection,
        reflection_workspace=work,
        reflection_classical_bits=tuple(range(c)),
        control=None if value is None else 1,
        adjoint=True,
    )
    n = 1 + data_width + q
    logical = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(1 << (1 + data_width))],
        dtype=torch.complex128,
    )
    logical /= logical.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << data_width, 1 << data_width
    )
    target = (
        torch.kron(torch.eye(2, dtype=torch.complex128), matrix.contiguous()) @ logical
    )
    initial = _with_clean_work(logical, q)
    expected = _with_clean_work(target, q)
    circuit = DynamicCircuit(n, inputs=initial, dtype=torch.complex128).append(
        operation
    )
    gates = circuit.to_ir().instructions
    assert CircuitIR.from_json(circuit.to_ir().to_json()).instructions == gates
    events = sum(g.name == "measure" for g in gates)
    assert events == c
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / sqrt(2**events),
            rtol=0,
            atol=8e-12,
        )
    for column in range(1 << data_width):
        basis = torch.zeros_like(logical)
        basis[column] = 1
        column_expected = torch.zeros_like(logical)
        column_expected[: 1 << data_width] = matrix[:, column]
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(_with_clean_work(basis, q), gates, history),
                _with_clean_work(column_expected, q) / sqrt(2**events),
                rtol=0,
                atol=8e-12,
            )
    combined = (*gates, *reverse.instructions)
    for history in product((0, 1), repeat=2 * events):
        torch.testing.assert_close(
            forced_branch(initial, combined, history),
            initial / sqrt(2 ** (2 * events)),
            rtol=0,
            atol=1e-11,
        )
    identity = ComponentCost("IdentityPrepare", width)
    assert_source_model(native.resource_model(identity, identity), source)


def test_factories_symbolic_metadata_controls_and_source_declared_adjoint() -> None:
    import cirq
    from qualtran.bloqs.reflections import reflection_using_prepare

    source = reflection_using_prepare.ReflectionUsingPrepare.reflection_around_zero(
        (1, 2)
    )
    native = PreparedStateReflection.around_zero((1, 2))
    for value in (0, 1):
        expected = source.controlled(control_values=(value,))
        actual = native.controlled(value)
        assert actual.control_value == expected.control_val
        assert actual.wire_symbols() == tuple(
            cirq.circuit_diagram_info(expected).wire_symbols
        )
    width = Symbol("n", positive=True, integer=True)
    symbolic = PreparedStateReflection.around_zero((width,))
    original = reflection_using_prepare.ReflectionUsingPrepare.reflection_around_zero(
        (width,)
    )
    assert (
        symbolic.selection_registers[0].bitsize
        == original.selection_registers[0].bitsize
        == width
    )
    with pytest.raises(ValueError, match="concrete width"):
        symbolic.wire_symbols()
    broken = PreparedStateReflection.around_zero((1,), global_phase=1j)
    assert broken.adjoint() is broken
    source_broken = (
        reflection_using_prepare.ReflectionUsingPrepare.reflection_around_zero(
            (1,), global_phase=1j
        )
    )
    matrix = torch.as_tensor(source_broken.tensor_contract())
    torch.testing.assert_close(matrix @ matrix, -torch.eye(2, dtype=torch.complex128))


def test_actual_source_capture_replacement_with_named_directional_prepare() -> None:
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    source_prepare = PrepareIdentity.from_bitsizes((2,))
    captured = import_register_emitter(source_prepare)
    # The existing importer omits multiwire Identity no-ops. Keep that account separate.
    packed = BlackBoxPrepare((("reg0_", 2),), (), captured)
    native = PreparedStateReflection(packed, control_value=1, global_phase=1j)
    operation = native.make_on(
        selection={"selection": (1, 2)},
        reflection_workspace=(3,),
        reflection_classical_bits=(0,),
        control=0,
    )
    source = ReflectionUsingPrepare(source_prepare, control_val=1, global_phase=1j)
    snapshot = import_register_emitter(source)
    assert snapshot.requirements() == (0, 0)
    torch.testing.assert_close(
        get_unitary(snapshot.snapshots[0], dtype=torch.complex128),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        rtol=0,
        atol=5e-12,
    )
    for inverse in (False, True):
        # Original declared adjoint is the same nonreal reflection.
        assert_source_stream(snapshot.snapshots[int(inverse)], snapshot.snapshots[0])
    initial = _with_clean_work(torch.ones(8, dtype=torch.complex128) / sqrt(8), 1)
    target = _with_clean_work(
        torch.as_tensor(source.tensor_contract()) @ initial.reshape(8, 2)[:, 0], 1
    )
    for history in ((0,), (1,)):
        torch.testing.assert_close(
            forced_branch(initial, operation.instructions, history),
            target / sqrt(2),
            rtol=0,
            atol=4e-12,
        )


def test_complete_source_factories_and_named_provider_order() -> None:
    from collections.abc import Mapping

    from qualtran.bloqs.reflections import reflection_using_prepare as reference

    from flagquantum.algorithms.primitives.alias_preparation import (
        source_dense_alias_preparation_cost,
    )
    from flagquantum.algorithms.primitives.alias_sampling import alias_sampling_table
    from flagquantum.algorithms.primitives.identity_prepare import IdentityPrepare
    from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
        AliasPreparationShape,
    )
    from flagquantum.core.ir import Instruction

    table = alias_sampling_table([1] * 5, precision=0.01)
    shape = AliasPreparationShape.from_table(table)
    native = PreparedStateReflection(shape)
    original = reference._refl_using_prep.make()
    assert [(r.name, r.bitsize) for r in native.signature] == [
        (r.name, r.bitsize) for r in original.signature
    ]
    assert shape.l1_norm_of_coeffs == original.prepare_gate.l1_norm_of_coeffs
    cost = source_dense_alias_preparation_cost(table)
    assert_source_model(native.resource_model(cost, cost), original)
    zero = PreparedStateReflection.around_zero((1, 2, 3), 1, -1)
    source_zero = reference._refl_around_zero.make()
    assert zero.wire_symbols() == ("@", *("R_L",) * 6)
    assert_source_model(
        zero.resource_model(ComponentCost("I", 6), ComponentCost("I", 6)), source_zero
    )
    assert len(reference._REFL_USING_PREP_DOC.examples) == 2

    calls: list[bool] = []

    def provider(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert ports == {"reg0_": (1,)} and not work and not bits
        calls.append(adjoint)
        return (Instruction("h", ports["reg0_"]),)

    reflected = PreparedStateReflection(IdentityPrepare.from_bitsizes((1,)))
    operation = reflected.make_on(
        selection={"reg0_": (1,)}, reflection_workspace=(2,), prepare_emit=provider
    )
    assert calls == [True, False]
    # H(I-2|0><0|)H = -X, with a coherent spectator and clean mark target.
    initial = _with_clean_work(
        torch.tensor([1, 2j, -3j, 4], dtype=torch.complex128) / sqrt(30), 1
    )
    expected = _with_clean_work(
        -initial.reshape(4, 2)[:, 0].reshape(2, 2).flip(1).reshape(4), 1
    )
    torch.testing.assert_close(
        forced_branch(initial, operation.instructions, ()), expected, rtol=0, atol=2e-12
    )
    with pytest.raises(ValueError, match="control wire"):
        reflected.make_on(
            selection={"reg0_": (1,)}, reflection_workspace=(2,), control=0
        )
    with pytest.raises(ValueError, match="shared control binder"):
        reflected.controlled().controlled()
    with pytest.raises(ValueError, match="directional emitter"):
        native.make_on(
            selection={"selection": (0, 1, 2)}, reflection_workspace=(3, 4, 5)
        )


@pytest.mark.parametrize("existing", (None, 0, 1))
def test_shared_grouped_and_repeated_reflection_controls(existing: int | None) -> None:
    import numpy as np
    from qualtran import CtrlSpec
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.control.controlled_component import (
        controlled_component_ir,
        extend_controlled_component_ir,
    )

    base = PreparedStateReflection.around_zero((1,), existing)
    positive = PreparedStateReflection.around_zero((1,), 1)
    spec = CtrlSpec(cvs=np.array([0, 1]))
    source = ReflectionUsingPrepare.reflection_around_zero(
        (1,), control_val=existing
    ).controlled(spec)
    logical_width = 3 + int(existing is not None)
    selection = logical_width - 1
    work = tuple(range(logical_width, logical_width + 1 + int(existing is not None)))
    slots = tuple(range(len(work)))

    def emit(control: int) -> tuple[Instruction, ...]:
        return positive.make_on(
            selection={"reg0_": (selection,)}, reflection_workspace=(), control=control
        ).instructions

    registers = (ControlRegister(1, (0, 1)),)
    if existing is None:
        gates = controlled_component_ir(((0, 1),), registers, emit, work, slots)
    else:
        gates = extend_controlled_component_ir(
            ((0, 1),), registers, (2, existing), emit, work, slots
        )
    assert base.control_value == existing
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical_width, 1 << logical_width
    )
    state = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(1 << logical_width)],
        dtype=torch.complex128,
    )
    state /= state.norm()
    initial = _with_clean_work(state, len(work))
    expected = _with_clean_work(matrix @ state, len(work))
    circuit = DynamicCircuit(
        logical_width + len(work), inputs=initial, dtype=torch.complex128
    ).append(FieldOperation(logical_width + len(work), gates))
    assert circuit.to_ir().instructions == gates
    for history in product((0, 1), repeat=len(work)):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / sqrt(2 ** len(work)),
            rtol=0,
            atol=3e-12,
        )
