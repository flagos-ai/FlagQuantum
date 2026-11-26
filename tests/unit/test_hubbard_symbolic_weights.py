"""Approved symbolic Hubbard weights preserve source calls, phases and cleanup."""

import sys
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.parameters import bind_parameter_value
from flagquantum.ecosystem._packed_oracle_component_library import (
    packed_oracle_library_factories,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_prepare_source_constructors import (
    source_module as source_module,
)
from tests.unit.test_hubbard_stored_graph import (
    original_hubbard_state,
    source_terminals,
)
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
sympy = pytest.importorskip("sympy")


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_approved_symbolic_constructor_retains_complete_source_calls(
    source_module: Any, dimension: int, inverse: bool, depth: int
) -> None:
    t, u = sympy.symbols("t u", real=True)
    source = source_module.PrepareHubbard(dimension, dimension, t, u)
    if inverse:
        source = source.adjoint()
    try:
        library = _library(source, depth)
    except NotImplementedError as error:
        assert (
            depth > 0 and "acos" in str(error) and "cannot be serialized" in str(error)
        )
        # The unmodified source codec cannot encode its deeper acos child.
        # Exercise the actual supported shallow constructor, retaining that refusal.
        library = _library(source, 0)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    record = resolve_component_program(library, 0, packed_oracle_library_factories())
    assert record.functions[0].blocks and verify_program_record(record) == ()
    pairs = compare_layers(record, source, source_terminals(record))
    for name, child in pairs.items():
        rotation = child.subbloq if type(child).__name__ == "Controlled" else child
        if type(rotation).__name__ != "Ry":
            continue
        count = len(next(f for f in record.functions if f.name == name).registers or ())
        emitted = bind_register_program(
            record, tuple((i,) for i in range(count)), entry=name
        ).circuit.instructions
        assert len(emitted) == 1
        angle = emitted[0].params["theta"]
        for tv, uv in ((1.0, 4.0), (0.37, 0.27), (0.0, 4.0), (1.0, 0.0)):
            actual = bind_parameter_value(angle, {"t": tv, "u": uv})
            expected = (
                rotation.angle.subs({t: tv, u: uv})
                if isinstance(rotation.angle, sympy.Expr)
                else rotation.angle
            )
            assert actual == pytest.approx(float(expected), abs=3e-15, rel=0)


@pytest.mark.parametrize(
    "t,u",
    (
        (1.0, 4.0),
        (0.37, 0.27),
        (0.0, 4.0),
        (1.0, 0.0),
        (0.37, 0.249999),
        (0.37, 0.25),
        (0.37, 0.250001),
        (-1.0, -4.0),
    ),
)
def test_symbolic_binding_matches_numeric_calls_and_all_complex_histories(
    source_module: Any, t: float, u: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum import Circuit

    ts, us = sympy.symbols("t u", real=True)
    source = source_module.PrepareHubbard(2, 2, ts, us)
    numeric_source = source_module.PrepareHubbard(2, 2, t, u)
    rng = np.random.default_rng(128)
    initial = rng.normal(size=1024) + 1j * rng.normal(size=1024)
    initial /= np.linalg.norm(initial)
    expected = original_hubbard_state(2, initial, t=t, u=u)
    inputs = ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8, 9))
    for inverse in (False, True):
        original = source.adjoint() if inverse else source
        numeric = numeric_source.adjoint() if inverse else numeric_source
        library = _library(original, 0)
        numeric_record = resolve_component_program(
            _library(numeric, 0), 0, packed_oracle_library_factories()
        )
        numeric_bound = bind_register_program(numeric_record, inputs, clean=(10, 11))
        with monkeypatch.context() as patch:
            for module in ("qualtran", "cirq", "kickmix", "quimb"):
                patch.setitem(sys.modules, module, None)
            record = resolve_component_program(
                library, 0, packed_oracle_library_factories()
            )
            bound = bind_register_program(record, inputs, clean=(10, 11))
            assert bound.outputs == inputs and bound.events == numeric_bound.events
            instructions = tuple(
                replace(op, params=bind_parameter_value(op.params, {"t": t, "u": u}))
                for op in bound.circuit.instructions
            )
            for actual, target in zip(
                instructions, numeric_bound.circuit.instructions, strict=True
            ):
                assert (actual.name, actual.wires, actual.metadata) == (
                    target.name,
                    target.wires,
                    target.metadata,
                )
                assert actual.params == pytest.approx(target.params, abs=3e-15, rel=0)
            count = sum(op.name == "measure" for op in instructions)
            assert count == 3
            state = torch.zeros(4096, dtype=torch.complex128)
            desired = torch.zeros_like(state)
            state[::4] = torch.as_tensor(expected if inverse else initial)
            desired[::4] = torch.as_tensor(initial if inverse else expected)
            circuit = Circuit(12, inputs=state, dtype=torch.complex128)
            circuit.append(FieldOperation(12, bound.circuit.instructions))
            restored = circuit.bind_parameters({"t": t, "u": u}).to_ir()
            for history in product((0, 1), repeat=count):
                branch_state = forced_branch(state, restored.instructions, history)
                torch.testing.assert_close(
                    branch_state, desired / 2 ** (count / 2), atol=3e-11, rtol=0
                )
                assert branch_state.reshape(-1, 4)[:, 1:].abs().max() < 3e-11


def test_symbolic_hubbard_rejects_invalid_binding_before_execution(
    source_module: Any,
) -> None:
    from flagquantum import Circuit

    t, u = sympy.symbols("t u", real=True)
    source = source_module.PrepareHubbard(2, 2, t, u)
    record = resolve_component_program(
        _library(source, 0), 0, packed_oracle_library_factories()
    )
    bound = bind_register_program(
        record, ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8, 9)), clean=(10, 11)
    )
    circuit = Circuit.from_ir(bound.circuit)
    with pytest.raises(KeyError, match="u"):
        circuit.bind_parameters({"t": 0.37})
    for values in ({"t": 0.0, "u": 0.0}, {"t": -1.0, "u": 8.0}):
        with pytest.raises(ValueError):
            circuit.bind_parameters(values)


def test_symbolic_hubbard_full_branch_gradient_matches_original_numeric_source(
    source_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum import Circuit

    ts, us = sympy.symbols("t u", real=True)
    source = source_module.PrepareHubbard(2, 2, ts, us)
    record = resolve_component_program(
        _library(source, 0), 0, packed_oracle_library_factories()
    )
    bound = bind_register_program(
        record, ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8, 9)), clean=(10, 11)
    )
    rng = np.random.default_rng(12801)
    initial = rng.normal(size=1024) + 1j * rng.normal(size=1024)
    initial /= np.linalg.norm(initial)
    probe = rng.normal(size=1024) + 1j * rng.normal(size=1024)
    probe /= np.linalg.norm(probe)
    step = 1e-6

    def score(t: float, u: float) -> float:
        return float(np.vdot(probe, original_hubbard_state(2, initial, t=t, u=u)).real)

    expected_t = (score(0.37 + step, 0.27) - score(0.37 - step, 0.27)) / (2 * step)
    expected_u = (score(0.37, 0.27 + step) - score(0.37, 0.27 - step)) / (2 * step)
    t = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    u = torch.tensor(0.27, dtype=torch.float64, requires_grad=True)
    state = torch.zeros(4096, dtype=torch.complex128)
    state[::4] = torch.as_tensor(initial)
    with monkeypatch.context() as patch:
        for module in ("qualtran", "cirq", "kickmix", "quimb", "sympy"):
            patch.setitem(sys.modules, module, None)
        circuit = Circuit(12, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(12, bound.circuit.instructions))
        instructions = circuit.bind_parameters({"t": t, "u": u}).to_ir().instructions
        branch = forced_branch(state, instructions, (0, 0, 0)) * 2**1.5
        value = torch.vdot(torch.as_tensor(probe), branch[::4]).real
        actual_t, actual_u = torch.autograd.grad(value, (t, u))
    assert actual_t.item() == pytest.approx(expected_t, abs=2e-9, rel=2e-6)
    assert actual_u.item() == pytest.approx(expected_u, abs=2e-9, rel=0)
