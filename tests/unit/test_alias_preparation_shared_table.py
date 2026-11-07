"""Alias consumers use shared QROM layout, workspace, binding and model profiles."""

import pytest

from flagquantum.algorithms.primitives.alias_preparation import (
    alias_preparation_ir,
    source_dense_alias_preparation_cost,
    source_sparse_alias_preparation_cost,
)
from flagquantum.algorithms.primitives.alias_sampling import (
    alias_sampling_table,
    sparse_alias_sampling_table,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.qrom_table import QromTable
from flagquantum.core.ir import Instruction
from tests.unit.test_alias_preparation import layout

pytestmark = pytest.mark.unit


def alias_case(mode):
    if mode == "dense":
        return alias_sampling_table((0, 1, 2), precision=0.5)
    if mode == "sparse":
        return sparse_alias_sampling_table({3: 1, 0: 3}, 4, precision=0.25)
    if mode == "dense_singleton":
        return alias_sampling_table((1,), precision=1)
    if mode == "sparse_singleton":
        return sparse_alias_sampling_table({3: 1}, 4, precision=1)
    if mode == "sparse_unit_domain":
        return sparse_alias_sampling_table({0: 1}, 1, precision=1)
    raise AssertionError(mode)


@pytest.mark.parametrize(
    "mode",
    ["dense", "sparse", "dense_singleton", "sparse_singleton", "sparse_unit_domain"],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_replacement_lookup_workspace_and_body_flow_through_alias(
    monkeypatch, mode, adjoint
):
    table = alias_case(mode)
    monkeypatch.setattr(QromTable, "workspace_size", property(lambda self: 23))
    calls = []

    def replacement(self, ports, work, slots):
        calls.append((self, ports, work, slots))
        return (Instruction("z", (ports[-1],)),)

    monkeypatch.setattr(QromTable, "bind_packed", replacement)
    bound, sparse = layout(table)
    selection, sigma, alt, keep, flag, work, slots = bound
    assert len(work) == len(slots) == 23
    gates = alias_preparation_ir(table, *bound, sparse_index=sparse, adjoint=adjoint)
    assert len(calls) == 1
    lookup, packed, actual_work, actual_slots = calls[0]
    address = sparse if table.indices is not None else selection
    targets = (
        (keep,)
        if table.selection_size == 1
        else ((selection, alt, keep) if table.indices is not None else (alt, keep))
    )
    assert packed == (*address, *(w for word in targets for w in word))
    assert actual_work == work and actual_slots == slots
    sentinel = Instruction("z", (packed[-1],))
    assert sum(g == sentinel for g in gates) == 1
    assert lookup.num_controls == 0 and lookup.shape == (len(table.keep),)
    if table.selection_size == 1:
        # The pre-existing native singleton execution omits source zero-width alt.
        assert lookup.datasets == (table.keep,)
        assert lookup.target_bitsizes == (table.mu,)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("mode", ["dense", "sparse", "sparse_singleton"])
def test_original_qrom_child_metadata_is_preserved(monkeypatch, mode):
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
        StatePreparationAliasSampling,
    )

    if mode == "dense":
        source = StatePreparationAliasSampling.from_probabilities(
            (0, 1, 2), precision=0.5
        )
    elif mode == "sparse":
        source = SparseStatePreparationAliasSampling.from_sparse_dict(
            {3: 1, 0: 3}, 4, precision=0.25
        )
    else:
        source = SparseStatePreparationAliasSampling.from_sparse_dict(
            {3: 1}, 4, precision=1
        )
    table = alias_case(mode)
    captured = []
    original = QromTable.bind_packed

    def recording(self, *args):
        captured.append(self)
        return original(self, *args)

    monkeypatch.setattr(QromTable, "bind_packed", recording)
    bound, sparse = layout(table)
    alias_preparation_ir(table, *bound, sparse_index=sparse)
    lookup = captured[0]
    upstream = source.qrom_bloq
    assert lookup.shape == upstream.data_shape
    assert lookup.selection_bitsizes == upstream.selection_bitsizes
    assert lookup.target_bitsizes == upstream.target_bitsizes
    assert lookup.target_shapes == upstream.target_shapes
    assert lookup.datasets == tuple(tuple(map(int, a.flat)) for a in upstream.data)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("legacy", [False, True])
def test_replacement_qrom_model_keeps_profile_and_bystander_ownership(
    monkeypatch, mode, legacy
):
    table = alias_case(mode)
    replacement = ComponentCost("replacement_qrom", 37, GateCost(t=19, measurement=7))
    calls = []

    def model(self, *, legacy_shims=False):
        calls.append((self, legacy_shims))
        return replacement

    monkeypatch.setattr(QromTable, "resource_model", model)
    factory = (
        source_dense_alias_preparation_cost
        if mode == "dense"
        else source_sparse_alias_preparation_cost
    )
    parent = factory(table, legacy_shims=legacy)
    assert len(calls) == 1 and calls[0][1] is legacy
    assert parent.calls[1].component is replacement
    assert parent.calls[1].bystander_qubits == table.mu + 1
