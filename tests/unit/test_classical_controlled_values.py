"""Original controlled callback delegation, mutation, sentinels and phase flow."""

from types import NotImplementedType

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_controlled_values import (
    controlled_basis_phase,
    controlled_classical_values,
)
from flagquantum.simulation.classical_pauli_values import (
    diagonal_classical_values,
    x_classical_value,
    z_basis_phase,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CONTROLS = [0, 1, 2, -1, np.int8(2), np.array(2), np.array([2]), np.array([0, 1]), None]
TARGETS = [0, 1, 2, -1, np.int8(1), np.array([0, 1]), None, sympy.Symbol("q")]


def source_child(kind):
    from qualtran.bloqs.basic_gates import Hadamard, XGate, ZGate

    return {"x": XGate(), "z": ZGate(), "h": Hadamard()}[kind]


def native_child(kind):
    def call(**values: object) -> dict[str, object] | NotImplementedType:
        if kind == "h":
            return NotImplemented
        if kind == "z":
            return diagonal_classical_values(**values)
        return {"q": x_classical_value(values["q"])}

    return call


def result_tuple(value):
    if value is NotImplemented:
        return (NotImplemented,)
    return (tuple(value), *value.values())


@pytest.mark.parametrize("kind", ["x", "z", "h"])
@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("target", TARGETS)
def test_original_controlled_raw_values_types_extras_and_not_implemented(
    kind, control, target
):
    from qualtran import Controlled, CtrlSpec, QUInt

    source = Controlled(source_child(kind), CtrlSpec(qdtypes=QUInt(2), cvs=2))
    names = source.ctrl_reg_names
    values = {names[0]: control, "q": target, "extra": np.array([7])}
    compare_calls(
        lambda v: result_tuple(source.on_classical_vals(**v)),
        lambda v: result_tuple(
            controlled_classical_values(
                v,
                names,
                ("q",),
                (np.array(2),),
                native_child(kind),
                symbolic_parameters=(2,),
            )
        ),
        (values,),
    )


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("target", TARGETS)
def test_original_controlled_z_phase_and_inactive_none(control, target):
    from qualtran import Controlled, CtrlSpec, QUInt

    source = Controlled(source_child("z"), CtrlSpec(qdtypes=QUInt(2), cvs=2))
    names = source.ctrl_reg_names
    values = {names[0]: control, "q": target}
    compare_calls(
        lambda v: (source.basis_state_phase(**v),),
        lambda v: (
            controlled_basis_phase(
                v,
                names,
                ("q",),
                (np.array(2),),
                lambda **args: z_basis_phase(args["q"]),
                symbolic_parameters=(2,),
            ),
        ),
        (values,),
    )


def test_active_mutable_child_output_and_opaque_phase_are_preserved_by_identity():
    from qualtran import Bloq, Controlled, CtrlSpec, QUInt, Signature

    source_output = {"q": 1}
    marker = object()

    class Callback(Bloq):
        @property
        def signature(self):
            return Signature.build(q=1)

        def on_classical_vals(self, **values):
            return source_output

        def basis_state_phase(self, **values):
            return marker

    source = Controlled(Callback(), CtrlSpec(qdtypes=QUInt(2), cvs=2))
    ctrl = np.int8(2)
    names = source.ctrl_reg_names
    values = {names[0]: ctrl, "q": 0}
    assert source.on_classical_vals(**values) is source_output
    native_output = {"q": 1}
    actual = controlled_classical_values(
        values, names, ("q",), (np.array(2),), lambda **args: native_output
    )
    assert actual is native_output
    assert native_output[names[0]] is source_output[names[0]] is ctrl
    assert list(native_output) == list(source_output) == ["q", names[0]]
    assert (
        controlled_basis_phase(
            values, names, ("q",), (np.array(2),), lambda **args: marker
        )
        is source.basis_state_phase(**values)
        is marker
    )


def test_inactive_output_is_kwargs_copy_and_missing_child_fields_fail_before_activation():
    from qualtran import Controlled, CtrlSpec, QUInt

    source = Controlled(source_child("x"), CtrlSpec(qdtypes=QUInt(2), cvs=2))
    names = source.ctrl_reg_names
    shared = np.array([3])
    values = {names[0]: 0, "q": 1, "extra": shared}
    actual = controlled_classical_values(
        values, names, ("q",), (np.array(2),), native_child("x")
    )
    original = source.on_classical_vals(**values)
    assert actual is not values and original is not values
    assert actual["extra"] is original["extra"] is shared
    for call in (
        lambda: source.on_classical_vals(**{names[0]: 0}),
        lambda: controlled_classical_values(
            {names[0]: 0}, names, ("q",), (np.array(2),), native_child("x")
        ),
    ):
        with pytest.raises(KeyError):
            call()


def test_non_thru_guard_precedes_reads_but_phase_has_no_such_guard():
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import ZeroState

    source = Controlled(ZeroState(), CtrlSpec())
    names = source.ctrl_reg_names
    with pytest.raises(ValueError):
        source.on_classical_vals()
    with pytest.raises(ValueError):
        controlled_classical_values(
            {}, names, ("q",), (np.array(1),), native_child("x"), thru_only=False
        )
    values = {names[0]: 0, "q": 0}
    assert (
        source.basis_state_phase(**values)
        is controlled_basis_phase(
            values, names, ("q",), (np.array(1),), lambda **args: None
        )
        is None
    )


def test_shared_activation_replacement_reaches_both_consumers(monkeypatch):
    import flagquantum.simulation.classical_controlled_values as module

    calls = []

    def active(values, patterns, *, symbolic_parameters):
        calls.append(tuple(values))
        return True

    monkeypatch.setattr(module, "classical_controls_active", active)
    values = {"ctrl": 0, "q": 1}
    assert controlled_classical_values(
        values, ("ctrl",), ("q",), (np.array(1),), native_child("x")
    ) == {"q": 0, "ctrl": 0}
    assert (
        controlled_basis_phase(
            values,
            ("ctrl",),
            ("q",),
            (np.array(1),),
            lambda **args: z_basis_phase(args["q"]),
        )
        == -1
    )
    assert calls == [(0,), (0,)]


def test_combined_word_controls_source_tensor_every_branch_inverse_and_models():
    from dataclasses import asdict
    from itertools import product

    import torch
    from qualtran import BloqBuilder, CtrlSpec, QBit, QUInt, Signature
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.classical_controlled_values import (
        WordControlledPaulis,
        x_child,
        z_child,
        z_phase,
    )
    from flagquantum.compiler._component_trace.compound_control_resource import (
        compound_control_key,
    )
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
    )
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    sources = [
        source_child(kind).controlled(CtrlSpec(qdtypes=QUInt(2), cvs=2))
        for kind in ("x", "z")
    ]
    builder, bound = BloqBuilder.from_signature(
        Signature.build_from_dtypes(ctrl=QUInt(2), q=QBit())
    )
    for source in sources:
        outputs = builder.add_d(
            source, **{source.ctrl_reg_names[0]: bound["ctrl"], "q": bound["q"]}
        )
        bound = {"ctrl": outputs[source.ctrl_reg_names[0]], "q": outputs["q"]}
    program = builder.finalize(**bound)
    circuit = DynamicCircuit(5)
    circuit.append(WordControlledPaulis())
    gates = tuple(circuit.to_ir().instructions)
    inverse_circuit = DynamicCircuit(5)
    inverse_circuit.append(WordControlledPaulis(inverse=True))
    inverse = tuple(inverse_circuit.to_ir().instructions)
    source_stream = reference_arithmetic_gates(
        program, {"ctrl": (0, 1), "q": (2,)}, (3,), {3: 0}
    )
    assert_source(gates, source_stream)
    grid = torch.arange(1, 17, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    patterns = (np.array(2),)
    for word, target, spectator in product(range(4), (0, 1), (0, 1)):
        moved = controlled_classical_values(
            {"ctrl": word, "q": target}, ("ctrl",), ("q",), patterns, x_child
        )
        phase = controlled_basis_phase(moved, ("ctrl",), ("q",), patterns, z_phase)
        output = controlled_classical_values(
            moved, ("ctrl",), ("q",), patterns, z_child
        )
        before = (word << 3) | (target << 2) | spectator
        after = (word << 3) | (output["q"] << 2) | spectator
        initial[before] = weights[(word << 2) | (target << 1) | spectator]
        expected[after] = (1 if phase is None else phase) * initial[before]
    matrix = torch.as_tensor(program.tensor_contract(), dtype=torch.complex128).reshape(
        8, 8
    )
    torch.testing.assert_close(
        matrix @ initial.reshape(8, 2, 2)[:, 0, :],
        expected.reshape(8, 2, 2)[:, 0, :],
        rtol=0,
        atol=2e-12,
    )
    for history in product((0, 1), repeat=2):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, expected / 2, rtol=0, atol=2e-12)
        residual = actual.reshape(8, 2, 2)[:, 1, :]
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), rtol=0, atol=2e-12
        )
    for history in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(initial, gates + inverse, history),
            initial / 4,
            rtol=0,
            atol=2e-12,
        )
    for legacy in (False, True):
        for kind, source in zip(("x", "z"), sources, strict=True):
            key = compound_control_key(CliffordCall(kind, 1), (1, 0))
            assert key is not None
            native = query_qec_gate_cost(key, legacy_shims=legacy)
            model = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
            assert {
                name: value for name, value in asdict(native).items() if value
            } == model.asdict()
        total = get_cost_value(program, QECGatesCost(legacy_shims=legacy))
        assert total.total_t_count() == sum(g.name in ("t", "tdg") for g in gates) == 8
        assert total.measurement == sum(g.name == "measure" for g in gates) == 2
    assert get_cost_value(program, QubitCount()) == 4
