"""Actual conditional-arithmetic source captures retain branches and work cleanup."""

from collections import Counter
from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest
import torch
from sympy import Symbol, simplify

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR
from flagquantum.core.value_math import SymbolicInt
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.arithmetic import controlled_add_or_subtract

    for name, blob in (
        ("controlled_add_or_subtract.py", "ad4ab25546702e3b891fb9995b2668de03fcd3cb"),
        (
            "controlled_add_or_subtract_test.py",
            "b0bc8c8a0a72e643bfa6453e1bcd594457a214fc",
        ),
    ):
        raw = (Path(controlled_add_or_subtract.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize(
    "name", ("_ctrl_add_or_sub_unsigned", "_ctrl_add_or_sub_signed")
)
def test_complete_eight_bit_factories_capture_both_original_directions(
    name: str,
) -> None:
    from qualtran.bloqs.arithmetic import controlled_add_or_subtract as original
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    factory = getattr(original, name)
    source = factory.make()
    assert factory in original._CONTROLLED_ADD_OR_SUBTRACT_DOC.examples
    assert source.b_dtype == source.a_dtype
    assert source.adjoint().adjoint() == source
    emitter = import_register_emitter(source)
    assert emitter.layout == (("ctrl", 1), ("a", 8), ("b", 8))
    assert emitter.requirements() == (7, 7) and emitter.binding_swaps == (0, 0)
    ports = {"ctrl": (0,), "a": tuple(range(1, 9)), "b": tuple(range(9, 17))}
    work, bits = tuple(range(17, 24)), tuple(range(40, 47))
    for inverse in (False, True):
        root = source.adjoint() if inverse else source
        gates = emitter(ports, work, bits, inverse)
        slots = {
            g.wires[0]: g.metadata["classical_bit"]
            for g in gates
            if g.name == "measure"
        }
        reference = reference_arithmetic_gates(
            root, ports, work, slots, cirq_style=True
        )
        assert_source(gates, reference)
        counts = Counter(g.name for g in gates)
        model = get_cost_value(root, QECGatesCost())
        assert counts["t"] + counts["tdg"] == model.total_t_count() == 28
        assert counts["measure"] == counts["reset"] == model.measurement == 7
        assert get_cost_value(root, QubitCount()) == 24
        assert counts["x"] == (0 if inverse else 2)
        ir = Circuit(24).append(FieldOperation(24, gates)).to_ir()
        assert CircuitIR.from_json(ir.to_json()).instructions == gates


def test_symbolic_factory_keeps_declaration_and_physical_admission_separate() -> None:
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        _ctrl_add_or_sub_signed_symb,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = _ctrl_add_or_sub_signed_symb.make()
    assert source.b_dtype == source.a_dtype
    assert source.adjoint().add_when_ctrl_is_on is False
    assert len(source.decompose_bloq().bloq_instances) == 5
    assert get_cost_value(source, QECGatesCost()).total_t_and_ccz_count() == {
        "n_t": 0,
        "n_ccz": source.a_dtype.bitsize - 1,
    }
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(source)


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("add_on", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_actual_captured_small_source_complete_amplitudes_and_all_round_trip_histories(
    kind: str, add_on: bool, inverse: bool
) -> None:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )

    dtype = {
        "unsigned": QUInt,
        "signed": QInt,
        "montgomery": QMontgomeryUInt,
        "mixed": QUInt,
    }[kind]
    source = ControlledAddOrSubtract(
        dtype(2), QInt(3) if kind == "mixed" else dtype(3), add_when_ctrl_is_on=add_on
    )
    emitter = import_register_emitter(source)
    assert emitter.requirements() == (2, 2) and emitter.binding_swaps == (0, 0)
    ports = {"ctrl": (2,), "a": (1, 0), "b": (5, 4, 3)}
    work, bits = (7, 8), (19, 20)
    gates = emitter(ports, work, bits, inverse)
    other = emitter(ports, work, bits, not inverse)
    root = source.adjoint() if inverse else source
    slots = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    assert_source(
        gates, reference_arithmetic_gates(root, ports, work, slots, cirq_style=True)
    )
    data = torch.tensor(
        [complex(i + 1, i % 7 - 3) for i in range(128)], dtype=torch.complex128
    )
    data /= data.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    initial[::4] = data
    matrix = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
    order = (2, 1, 0, 5, 4, 3, 6, 7, 8)
    reverse_order = tuple(order.index(q) for q in range(9))
    expected = matrix @ initial.reshape((2,) * 9).permute(order).reshape(64, 8)
    expected = expected.reshape((2,) * 9).permute(reverse_order).flatten()
    histories = tuple(product((0, 1), repeat=2))
    for first in histories:
        branch = forced_branch(initial, gates, first)
        torch.testing.assert_close(branch, expected / 2, atol=3e-12, rtol=0)
        assert torch.count_nonzero(branch.reshape(-1, 4)[:, 1:]) == 0
        for second in histories:
            restored = forced_branch(branch, other, second)
            torch.testing.assert_close(restored, initial / 4, atol=3e-12, rtol=0)
    ir = Circuit(9).append(FieldOperation(9, gates)).to_ir()
    assert CircuitIR.from_json(ir.to_json()).instructions == gates


@pytest.mark.parametrize(
    "left_width,right_width",
    (
        (1, 1),
        (2, 3),
        (8, 8),
        (10, 10),
        (
            Symbol("n", positive=True, integer=True),
            Symbol("n", positive=True, integer=True),
        ),
        (
            Symbol("n", positive=True, integer=True),
            Symbol("n", positive=True, integer=True) + 1,
        ),
    ),
)
@pytest.mark.parametrize("add_on", (False, True))
def test_existing_independent_formula_all_categories_and_peak(
    left_width: SymbolicInt, right_width: SymbolicInt, add_on: bool
) -> None:
    from dataclasses import fields

    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.resources.symbolic_resource_model import (
        source_controlled_add_or_subtract_resource_formula,
    )

    source = ControlledAddOrSubtract(
        QUInt(left_width), QUInt(right_width), add_when_ctrl_is_on=add_on
    )
    actual = source_controlled_add_or_subtract_resource_formula(
        left_width, right_width, add_when_control_is_on=add_on
    )
    expected = get_cost_value(source, QECGatesCost())
    for field in fields(actual):
        value = (
            get_cost_value(source, QubitCount())
            if field.name == "logical_qubits"
            else getattr(expected, field.name)
        )
        assert simplify(getattr(actual, field.name) - value) == 0
