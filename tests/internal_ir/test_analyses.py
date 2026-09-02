from __future__ import annotations

import pytest

from flagquantum._compiler.analyses.base import AnalysisManager
from flagquantum._compiler.analyses.def_use import DefUseAnalysis
from flagquantum._compiler.analyses.qubit_lifetime import QubitLifetimeAnalysis
from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum._compiler.ir.modules import QuantumModule
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _module() -> QuantumModule:
    source = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("x", (1,)),
        ),
    )
    return import_circuit_ir(source).imported.module


def test_def_use_analysis_records_linear_chain_deterministically() -> None:
    module = _module()
    block = module.body.blocks[0]
    first, second, third = block.operations
    result = DefUseAnalysis().run(module)

    assert [str(item.value) for item in result.definitions[:2]] == [
        "%entry.0",
        "%entry.1",
    ]
    assert all(item.block_argument for item in result.definitions[:2])
    assert result.uses_of(first.results[0].id)[0].site.operation_index == 1
    assert result.uses_of(second.results[1].id)[0].site.operation_index == 2
    assert result.definitions_of(third.results[0].id)[0].result_index == 0


def test_qubit_lifetime_maps_each_operand_to_its_corresponding_result() -> None:
    module = _module()
    first, second, _ = module.body.blocks[0].operations
    result = QubitLifetimeAnalysis().run(module)

    q0 = result.lifetime_of(first.results[0].id)
    q1 = result.lifetime_of(second.operands[1].id)
    assert q0 is not None and q0.successor_values == (second.results[0].id,)
    assert q1 is not None and q1.successor_values == (second.results[1].id,)
    assert q0.consumed_at is not None
    assert q0.consumed_at.operation_index == 1


def test_analysis_cache_is_revision_aware_and_instance_local() -> None:
    module = _module()
    revised = QuantumModule(module.body, revision=1)
    analysis = DefUseAnalysis()
    first_manager = AnalysisManager()
    second_manager = AnalysisManager()

    first = first_manager.get(analysis, module)
    assert first_manager.get(analysis, module) is first
    assert first_manager.get(analysis, revised) is not first
    assert second_manager.get(analysis, module) is not first
    assert first_manager.entry_count == 2


def test_only_explicitly_preserved_analysis_is_carried_to_new_revision() -> None:
    module = _module()
    revised = QuantumModule(module.body, revision=1)
    manager = AnalysisManager()
    def_use = DefUseAnalysis()
    lifetime = QubitLifetimeAnalysis()
    old_def_use = manager.get(def_use, module)
    old_lifetime = manager.get(lifetime, module)

    manager.carry_preserved(module, revised, frozenset({"def_use"}))

    assert manager.get(def_use, revised) is old_def_use
    assert manager.get(lifetime, revised) is not old_lifetime
