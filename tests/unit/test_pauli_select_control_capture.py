"""Capture actual source control wrappers without their failing object codec."""

import builtins
from collections import Counter
from collections.abc import Callable
from itertools import product
from types import ModuleType
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._pauli_select_component_library import (
    capture_pauli_select_pair,
)
from flagquantum.errors import SerializationError
from tests.unit.pauli_select_reference import assert_all_branch_operators
from tests.unit.pauli_select_reference import source as source
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = (
    "zero",
    "one",
    "array",
    "word",
    "separate",
    "singleton",
    "matrix",
    "nested",
    "extended",
)


def controlled_source(source: Any, cv: int | None, case: str, target: int = 1) -> Any:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import CtrlSpec, QBit, QUInt
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    base = source.SelectPauliLCU(
        1,
        target,
        (
            cirq.DensePauliString("X" * target, coefficient=-1),
            cirq.DensePauliString("Y" * target),
        ),
        cv,
    )
    specs = {
        "zero": CtrlSpec(cvs=0),
        "one": CtrlSpec(cvs=1),
        "array": CtrlSpec(cvs=(1, 0)),
        "word": CtrlSpec(qdtypes=QUInt(2), cvs=1),
        "separate": CtrlSpec(qdtypes=(QBit(), QBit()), cvs=(np.array(0), np.array(1))),
        "singleton": CtrlSpec(cvs=np.array([0])),
        "matrix": CtrlSpec(cvs=np.array([[0, 1]])),
    }
    if case == "nested":
        return ControlledViaAnd(ControlledViaAnd(base, specs["zero"]), specs["array"])
    if case == "extended":
        first = base.get_ctrl_system(specs["zero"])[0]
        return first.get_ctrl_system(specs["one"])[0]
    return base.get_ctrl_system(specs[case])[0]


@pytest.mark.parametrize("cv,case", tuple(product((None, 0, 1), CASES)))
@pytest.mark.parametrize("target", (0, 1))
def test_actual_wrappers_full_calls_branches_and_private_storage(
    source: Any, cv: int | None, case: str, target: int
) -> None:
    from qualtran import Adjoint
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.serialization.bloq import bloqs_to_proto

    original = controlled_source(source, cv, case, target)
    if target:
        with pytest.raises(ValueError, match="Object arrays"):
            bloqs_to_proto(original, max_depth=0)
    else:
        assert bloqs_to_proto(original, max_depth=0).SerializeToString()
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    for inverse, record in enumerate(capture_pauli_select_pair(original)):
        selected = Adjoint(original) if inverse else original
        terminals = atomic_leaves(record) | {
            f.name for f in record.functions if "pauli_leaf_" in f.name
        }
        compare_layers(record, selected, terminals)
        bound, logical = bind(record)
        assert bound.circuit.n_wires == get_cost_value(selected, QubitCount())
        ports = {}
        offset = 0
        for reg in record.functions[0].registers or ():
            width = int(reg.total_qubits())
            ports[reg.name] = tuple(range(offset, offset + width))
            offset += width
        measurements = {
            gate.wires[0]: int(gate.metadata["classical_bit"])
            for gate in bound.circuit.instructions
            if gate.name == "measure"
        }
        # Each fixture's private bit is measured once; source keys bind to
        # those explicit SSA result slots rather than unrelated pool indices.
        assert len(measurements) == sum(
            g.name == "measure" for g in bound.circuit.instructions
        )
        reference = reference_arithmetic_gates(
            selected,
            ports,
            tuple(range(logical, bound.circuit.n_wires)),
            measurements,
            keep_pauli_leaves=True,
        )
        assert_source_dependencies(bound.circuit.instructions, reference)
        assert_all_branch_operators(
            bound.circuit, logical, expected.conj().T if inverse else expected
        )
        assert (
            CircuitIR.from_json(bound.circuit.to_json()).instructions
            == bound.circuit.instructions
        )
        active: set[int] = set()
        for event in bound.events:
            (
                active.update(event.wires)
                if event.kind == "allocate"
                else active.difference_update(event.wires)
            )
        assert not active
        if not inverse:
            actual_counts = Counter(g.name for g in bound.circuit.instructions)
        else:
            assert Counter(g.name for g in bound.circuit.instructions) == actual_counts


@pytest.mark.parametrize("cv", (None, 0, 1))
def test_owned_control_replacement_runs_normal_append_without_source_imports(
    source: Any, cv: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = controlled_source(source, cv, "array")
    records = capture_pauli_select_pair(original)
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    imported: Callable[..., ModuleType] = builtins.__import__

    def native_only(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError(f"Native control execution imported source SDK {name}")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", native_only)
    for inverse, record in enumerate(records):
        bound, logical = bind(record)
        operation = _BoundBlockOperation(
            bound.circuit.instructions, bound.circuit.n_wires
        )
        circuit = DynamicCircuit(bound.circuit.n_wires).append(operation)
        assert circuit.to_ir().instructions == bound.circuit.instructions
        assert_all_branch_operators(
            circuit.to_ir(), logical, expected.conj().T if inverse else expected
        )


def test_original_specialized_extension_callback_failure_is_not_repaired(
    source: Any,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    helper = controlled_source(source, 1, "zero")
    original = ControlledViaAnd(helper, CtrlSpec(cvs=(1, 0)))
    with pytest.raises(TypeError, match="integer scalar arrays"):
        original.decompose_bloq()
    with pytest.raises(SerializationError, match="control callback"):
        capture_pauli_select_pair(original)
