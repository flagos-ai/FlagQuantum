"""Complete chemistry placeholder and controlled-phase external consumers."""

import builtins
from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.algorithms.primitives.chemistry_number_phase import (
    ChemistryNumberPhase,
)
from flagquantum.algorithms.primitives.chemistry_qroam_model import (
    chemistry_qroam_log_block_size,
    chemistry_qroam_signature,
    source_chemistry_qroam_cost,
    source_chemistry_qroam_two_axis_cost,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_unary_component_calls import source_slot_names

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_clean_block_exponent_query_matches_original(adjoint: bool) -> None:
    from qualtran.bloqs.chemistry.black_boxes import (
        get_optimal_log_block_size_clean_ancilla,
        get_qroam_cost_clean_ancilla,
    )

    for rows, width, block in product(
        (8, 17, 64, 100, 1000), (1, 3, 8), (None, 1, 3, 8)
    ):
        assert chemistry_qroam_log_block_size(
            rows, width, block_size=block, adjoint=adjoint
        ) == get_optimal_log_block_size_clean_ancilla(rows, width, adjoint, block)
        assert source_chemistry_qroam_cost(
            rows, width, block_size=block, adjoint=adjoint
        ).gates.toffoli == get_qroam_cost_clean_ancilla(rows, width, adjoint, block)
    # The source breaks an exact two-power tie in favor of the smaller exponent.
    assert chemistry_qroam_log_block_size(8, 1, adjoint=adjoint) == 1


def test_block_query_refusal_and_existing_cost_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.chemistry.black_boxes import (
        get_optimal_log_block_size_clean_ancilla,
    )

    import flagquantum.algorithms.primitives.chemistry_qroam_model as module

    with pytest.raises(AssertionError):
        get_optimal_log_block_size_clean_ancilla(2, 3)
    with pytest.raises(ValueError, match="rows >= target width"):
        chemistry_qroam_log_block_size(2, 3)
    for block in (0, -1):
        with pytest.raises(ValueError, match="positive block"):
            chemistry_qroam_log_block_size(8, 2, block_size=block)
    calls = []

    def replacement(
        rows: int, width: int, *, block_size: int | None = None, adjoint: bool = False
    ) -> int:
        calls.append((rows, width, block_size, adjoint))
        return 2

    monkeypatch.setattr(module, "chemistry_qroam_log_block_size", replacement)
    assert module.source_chemistry_qroam_cost(16, 2).gates.toffoli == 10
    assert calls == [(16, 2, None, False)]


@pytest.mark.parametrize(
    "depth,adjoint,kind", tuple(product((0, 1, 2), (False, True), ("one", "two")))
)
def test_actual_placeholder_library_preserves_signature_and_model_only_semantics(
    depth: int, adjoint: bool, kind: str
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.black_boxes import QROAM, QROAMTwoRegs
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = (
        QROAM(51, 20, is_adjoint=adjoint)
        if kind == "one"
        else QROAMTwoRegs(5, 10, 3, 2, 4, is_adjoint=adjoint)
    )
    proto = bloqs_to_proto(source, max_depth=depth)
    raw = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(raw)
    assert serialize_component_library(library) == raw
    assert bloqs_from_proto(proto)[0] == source
    root = library.entries[0]
    arguments = {arg.name: arg.value for arg in root.declaration.arguments}
    assert all(type(value) is int for value in arguments.values())

    def integer(name: str) -> int:
        value = arguments[name]
        assert isinstance(value, int)
        return value

    width = integer("target_bitsize")
    inverse = bool(integer("is_adjoint"))
    if kind == "one":
        rows = integer("data_size")
        model = source_chemistry_qroam_cost(rows, width, adjoint=inverse)
    else:
        rows = integer("data_a_size")
        model = source_chemistry_qroam_two_axis_cost(
            rows,
            integer("data_b_size"),
            integer("data_a_block_size"),
            integer("data_b_block_size"),
            width,
            adjoint=inverse,
        )
    assert tuple(chemistry_qroam_signature(rows, width)) == root.declaration.registers
    assert model.qubits == get_cost_value(source, QubitCount())
    assert model.gates.toffoli == get_cost_value(source, QECGatesCost()).toffoli
    assert not root.connections
    if depth:
        assert sum(count for _, count in root.counts) == model.gates.toffoli
    assert source.adjoint().adjoint() == source
    assert str(source) == ("QROAM†" if adjoint else "QROAM")
    with pytest.raises(
        DecomposeNotImplementedError, match="does not declare a decomposition"
    ):
        source.decompose_bloq()


@pytest.mark.parametrize("values", ((0,), (1,), (0, 1), (1, 0, 1)))
def test_actual_stored_phase_graph_preserves_recipe_all_branches_and_native_append(
    values: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    system_width = 2
    source = ApplyControlledZs(values, system_width)
    graph = source.decompose_bloq().flatten()
    emitter = import_register_emitter(graph)
    work_count = len(values) - 1
    logical = len(values) + system_width
    clean = tuple(range(logical, logical + work_count))
    slots = tuple(range(work_count))
    width = logical + work_count + 1
    ports = {
        "ctrls": tuple(range(len(values))),
        "system": tuple(range(len(values), logical)),
    }
    assert emitter.layout == (("ctrls", len(values)), ("system", system_width))
    assert emitter.requirements() == (work_count, work_count)
    assert emitter.binding_swaps == (0, 0)
    native_model = ChemistryNumberPhase(values, system_width).resource_model()
    assert native_model.qubits == get_cost_value(source, QubitCount())
    assert native_model.gates.toffoli == get_cost_value(source, QECGatesCost()).toffoli
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    joint = torch.complex(raw, 2 - raw)
    joint /= joint.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial.reshape(-1, 1 << work_count, 2)[:, 0, :] = joint.reshape(-1, 2)
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError(
                f"Owned chemistry capture retained SDK dependency {name}"
            )
        return original_import(name, *args, **kwargs)

    for adjoint in (False, True):
        direction = source.adjoint() if adjoint else source
        expected = reference_arithmetic_gates(
            direction,
            ports,
            clean,
            dict(zip(clean, slots, strict=True)),
            keep_pauli_leaves=True,
        )
        with monkeypatch.context() as context:
            context.setattr(builtins, "__import__", owned_import)
            gates = emitter(ports, clean, slots, adjoint)
            assert_source_dependencies(source_slot_names(gates, clean), expected)
            counts = Counter(gate.name for gate in gates)
            assert counts["measure"] == work_count
            assert counts["t"] + counts["tdg"] == 4 * work_count
            circuit = Circuit(width)
            circuit.append(FieldOperation(width, gates))
            restored = CircuitIR.from_json(circuit.to_ir().to_json())
            for history in product((0, 1), repeat=work_count):
                actual = forced_branch(initial, restored.instructions, history)
                reference = forced_branch(initial, expected, history)
                torch.testing.assert_close(actual, reference, atol=2e-12, rtol=0)
                assert (
                    torch.count_nonzero(
                        actual.reshape(-1, 1 << work_count, 2)[:, 1:, :]
                    )
                    == 0
                )


def test_original_flat_graph_serialization_defect_is_not_a_decomposition_failure() -> (
    None
):
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
    from qualtran.serialization.bloq import bloqs_to_proto

    source = ApplyControlledZs((0, 1), 2)
    graph = source.decompose_bloq().flatten()
    assert graph.bloq_instances
    with pytest.raises(ValueError, match="numpy.uint8"):
        bloqs_to_proto(graph, max_depth=1, pred=lambda _: False)
    # The original outer component remains serializable before recursive flattening.
    assert bloqs_to_proto(source, max_depth=2).table
