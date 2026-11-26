"""Complete controlled-adder source captures, factories and native score."""

import sys
from collections import Counter
from dataclasses import replace
from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.arithmetic.controlled_addition import (
    controlled_add_adjoint_ir,
    controlled_add_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.drawer.addition_symbols import addition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_controlled_addition import layout
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_register_tensor import native_signature
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_complete_source() -> None:
    import csv

    from qualtran.bloqs.arithmetic import controlled_addition

    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        blobs = {row["path"]: row["git_blob"] for row in csv.DictReader(stream)}
    for name in ("controlled_addition.py", "controlled_addition_test.py"):
        raw = (Path(controlled_addition.__file__).parent / name).read_bytes()
        assert (
            sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == blobs["qualtran/bloqs/arithmetic/" + name]
        )


def test_complete_small_factory_capture_binding_json_and_resource() -> None:
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic import controlled_addition as source
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    root = source._cadd_small.make()
    assert source._CADD_DOC.examples == (source._cadd_small, source._cadd_large)
    assert root.a_dtype == root.b_dtype == QUInt(3) and root.cv == 1
    emitter = import_register_emitter(root)
    assert emitter.requirements() == (5, 5) and emitter.binding_swaps == (0, 0)
    ports = {"ctrl": (0,), "a": (3, 2, 1), "b": (6, 5, 4)}
    work, bits = tuple(range(7, 12)), tuple(range(40, 45))
    for inverse in (False, True):
        gates = emitter(ports, work, bits, inverse)
        actual_root = root.adjoint() if inverse else root
        slots = {
            g.wires[0]: g.metadata["classical_bit"]
            for g in gates
            if g.name == "measure"
        }
        assert_source(
            gates,
            reference_arithmetic_gates(actual_root, ports, work, slots),
        )
        counts = Counter(g.name for g in gates)
        model = get_cost_value(actual_root, QECGatesCost())
        assert counts["t"] + counts["tdg"] == model.total_t_count() == 20
        assert counts["measure"] == counts["reset"] == model.measurement == 5
        assert get_cost_value(actual_root, QubitCount()) == 12
        ir = Circuit(12).append(FieldOperation(12, gates)).to_ir()
        assert CircuitIR.from_json(ir.to_json()).instructions == gates


def test_large_factory_and_symbolic_boundary_keep_distinct_admission() -> None:
    from qualtran._infra.bloq import DecomposeTypeError
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd, _cadd_large
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount
    from sympy import Symbol

    source = _cadd_large.make()
    assert source.a_dtype == source.b_dtype == QUInt(1000)
    assert get_cost_value(source, QECGatesCost()).total_toffoli_only() == 1999
    previous_limit = sys.getrecursionlimit()
    try:
        # Recursion exhaustion is an environment boundary, not a universal source result.
        sys.setrecursionlimit(1000)
        with pytest.raises(RuntimeError, match="maximum recursion depth") as failure:
            get_cost_value(source, QubitCount())
        cause = failure.value.__cause__
        assert isinstance(cause, RuntimeError)
        assert isinstance(cause.__cause__, RecursionError)
        sys.setrecursionlimit(3000)
        assert get_cost_value(source, QubitCount()) == 4000
    finally:
        sys.setrecursionlimit(previous_limit)
    from flagquantum.compiler.registers.arithmetic.controlled_add_resource_model import (
        source_controlled_add_resource_formula,
    )

    assert source_controlled_add_resource_formula(1000, 1000).logical_qubits == 4000
    a, b, loaded, carry, load_bits, carry_bits = layout(1000, 1000)
    gates = controlled_add_ir(0, a, b, loaded, carry, load_bits, carry_bits)
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == 7996
    assert counts["measure"] == counts["reset"] == 1999
    n = Symbol("n")
    symbolic = CAdd(QUInt(n))
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        symbolic.decompose_bloq()
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(symbolic)
    assert get_cost_value(symbolic, QECGatesCost()).total_t_and_ccz_count() == {
        "n_t": 0,
        "n_ccz": 2 * n - 1,
    }
    controlled = source.controlled()
    assert controlled.bloq_counts() == {And(): 1, And().adjoint(): 1, source: 1}


@pytest.mark.parametrize("dtype", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("cv", (0, 1))
def test_actual_capture_complete_amplitudes_and_all_inverse_branches(
    dtype: str, cv: int
) -> None:
    from qualtran._infra.data_types import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    kinds: dict[str, type[QUInt] | type[QInt] | type[QMontgomeryUInt]] = {
        "unsigned": QUInt,
        "signed": QInt,
        "montgomery": QMontgomeryUInt,
    }
    kind = kinds[dtype]
    root = CAdd(kind(1), kind(2), cv=cv)
    emitter = import_register_emitter(root)
    ports = {"ctrl": (0,), "a": (1,), "b": (2, 3)}
    work, bits = (5, 6), (40, 41)
    assert emitter.requirements() == (2, 2)
    initial = torch.zeros(128, dtype=torch.complex128)
    data = torch.tensor(
        [complex(i + 1, i % 7 - 3) for i in range(32)], dtype=torch.complex128
    )
    data /= data.norm()
    initial[::4] = data
    for inverse in (False, True):
        gates = emitter(ports, work, bits, inverse)
        other = emitter(ports, work, bits, not inverse)
        native = controlled_add_adjoint_ir if inverse else controlled_add_ir
        # Named capture assigns classical storage by first measurement encounter.
        slots = {
            g.wires[0]: g.metadata["classical_bit"]
            for g in gates
            if g.name == "measure"
        }
        expected_gates = native(
            0, (1,), (2, 3), (5,), (6,), (slots[5],), (slots[6],), control_value=cv
        )
        assert_source(gates, expected_gates)
        actual_root = root.adjoint() if inverse else root
        matrix = torch.as_tensor(actual_root.tensor_contract(), dtype=torch.complex128)
        expected = (matrix @ initial.reshape(16, 8)).flatten()
        for first in product((0, 1), repeat=2):
            branch = forced_branch(initial, gates, first)
            torch.testing.assert_close(branch, expected / 2, rtol=0, atol=2e-12)
            assert torch.count_nonzero(branch.reshape(-1, 4)[:, 1:]) == 0
            for second in product((0, 1), repeat=2):
                restored = forced_branch(branch, other, second)
                torch.testing.assert_close(restored, initial / 4, rtol=0, atol=2e-12)


@pytest.mark.parametrize("cv", (0, 1))
def test_native_score_retains_direct_source_labels_and_broken_drawer_boundary(
    cv: int,
) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.drawing.musical_score import get_musical_score_data

    source = CAdd(QUInt(3), cv=cv)
    signature = native_signature(source.signature)
    names = tuple(r.name for r in signature)
    step = RegisterStep(source.short_name(), names, names)
    step = replace(step, gate="controlled_add")
    score = layout_register_score(
        {r.name: int(r.bitsize) for r in signature}, names, (step,)
    )
    builder, inputs = BloqBuilder.from_signature(source.signature)
    composite = builder.finalize(**builder.add_d(source, **inputs))
    # The source drawing API changed, while CAdd retains its old Soquet signature.
    with pytest.raises(TypeError, match="positional arguments"):
        get_musical_score_data(composite)
    from types import SimpleNamespace

    data = register_score_data(score, wire_symbol=addition_wire_symbol)
    assert len(data["hlines"]) == 3 and len(data["vlines"]) == 1
    for reg in source.signature:
        symbol = getattr(source, "wire_symbol")(SimpleNamespace(reg=reg))
        assert addition_wire_symbol(step, reg.name, "THRU") == (
            type(symbol).__name__,
            {"text": symbol.text},
        )
    assert addition_wire_symbol(step, None, None) == (
        "Text",
        {"text": source.short_name(), "fontsize": 10},
    )
    with pytest.raises(ValueError, match="not declared"):
        addition_wire_symbol(step, "unknown", "THRU")
