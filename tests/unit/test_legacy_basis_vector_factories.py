"""Actual computational basis libraries retain typed state/effect ownership."""

from collections import Counter
from itertools import product

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.legacy_basis_vector_factories import (
    basis_vector_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,state", tuple(product((1, 2, 3, 4, 8), (False, True))))
def test_all_small_integer_constructor_values_and_clean_owned_effects(
    width: int, state: bool
) -> None:
    from qualtran.bloqs.basic_gates import IntEffect, IntState

    for value in (range(1 << width) if width < 8 else (55, 255)):
        source = IntState(value, width) if state else IntEffect(value, width)
        records = tuple(
            resolve_component_program(
                _library(source, depth), 0, basis_vector_factories()
            )
            for depth in (0, 1, 2)
        )
        assert records[0] == records[1] == records[2]
        record = records[0]
        assert verify_program_record(record) == ()
        wires = tuple(reversed(range(width)))
        bound = bind_register_program(
            record,
            () if state else (wires,),
            clean=wires if state else (),
            borrowed=(width,),
            allow_workspace_outputs=state,
            input_ownership=() if state else tuple((wire, False) for wire in wires),
        )
        assert bound.outputs == ((wires,) if state else ())
        assert Counter(g.name for g in bound.circuit.instructions) == (
            {"x": value.bit_count()} if value else {}
        )
        # The coherent spectator is never part of the clean pool.
        initial = torch.zeros(1 << (width + 1), dtype=torch.complex128)
        expected = torch.zeros_like(initial)
        label = sum(
            ((value >> (width - 1 - i)) & 1) << (width - wire)
            for i, wire in enumerate(wires)
        )
        amplitudes = torch.tensor([1, 1j], dtype=torch.complex128) / (2**0.5)
        initial[(0 if state else label) : (0 if state else label) + 2] = amplitudes
        expected[(label if state else 0) : (label if state else 0) + 2] = amplitudes
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, ()), expected
        )
        circuit = Circuit(width + 1, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(width + 1, bound.circuit.instructions))
        torch.testing.assert_close(circuit.state().reshape(-1), expected)
        if not state and width:
            with pytest.raises(ValueError):
                bind_register_program(record, (wires,))


@pytest.mark.parametrize("kind", ("ZeroState", "OneState", "ZeroEffect", "OneEffect"))
def test_actual_atomic_constructors_and_source_width_refusal(kind: str) -> None:
    from qualtran.bloqs.basic_gates import z_basis

    source = getattr(z_basis, kind)()
    record = resolve_component_program(_library(source, 0), 0, basis_vector_factories())
    state, bit = kind.endswith("State"), int(kind.startswith("One"))
    bound = bind_register_program(
        record,
        () if state else ((0,),),
        clean=(0,) if state else (),
        allow_workspace_outputs=state,
        input_ownership=() if state else ((0, False),),
    )
    assert Counter(g.name for g in bound.circuit.instructions) == (
        {"x": 1} if bit else {}
    )
    factory = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis." + kind]
    with pytest.raises(NotImplementedError, match="Come back"):
        factory("bad", None, {"n": 2})
    with pytest.raises(NotImplementedError, match="Come back"):
        getattr(z_basis, kind)(2)


def test_symbolic_declarations_and_bad_metadata() -> None:
    from qualtran.bloqs.basic_gates import IntEffect, IntState

    for cls in (IntState, IntEffect):
        for value, width in (
            (sympy.Symbol("v"), 3),
            (0, sympy.Symbol("n", positive=True)),
        ):
            record = resolve_component_program(
                _library(cls(value, width), 0), 0, basis_vector_factories()
            )
            assert not record.functions[0].blocks
    factory = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.IntState"]
    for data in (
        {},
        {"val": -1, "bitsize": 3},
        {"val": 8, "bitsize": 3},
        {"val": 0, "bitsize": 3, "extra": 1},
    ):
        with pytest.raises(SerializationError):
            factory("bad", None, data)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"val": 1, "bitsize": 3})


@pytest.mark.parametrize("state", (False, True))
def test_empty_source_failures_remain_declarations(state: bool) -> None:
    from qualtran import BloqError
    from qualtran.bloqs.basic_gates import IntEffect, IntState

    source = IntState(0, 0) if state else IntEffect(0, 0)
    record = resolve_component_program(_library(source, 0), 0, basis_vector_factories())
    assert not record.functions[0].blocks
    with pytest.raises(BloqError if state else IndexError):
        _library(source, 1)
