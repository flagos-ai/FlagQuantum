from __future__ import annotations

from dataclasses import dataclass

import pytest

from flagquantum._compiler.analyses.base import AnalysisManager
from flagquantum._compiler.analyses.def_use import DefUseAnalysis
from flagquantum._compiler.analyses.qubit_lifetime import QubitLifetimeAnalysis
from flagquantum._compiler.diagnostics import Diagnostic, DiagnosticCode
from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum._compiler.ir.modules import Block, QuantumModule, Region
from flagquantum._compiler.ir.operations import Operation
from flagquantum._compiler.passes.base import PassDescriptor, PassResult
from flagquantum._compiler.passes.canonicalize import CanonicalizeAttributesPass
from flagquantum._compiler.passes.manager import PassManager
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _module() -> QuantumModule:
    return import_circuit_ir(CircuitIR(1, (Instruction("h", (0,)),))).imported.module


@dataclass(frozen=True)
class _RevisionPass:
    descriptor: PassDescriptor = PassDescriptor("revision_only", "1.0")

    def run(self, module: QuantumModule) -> PassResult:
        return PassResult(
            QuantumModule(module.body, revision=module.revision + 1),
            changed=True,
            preserved_analyses=frozenset({"def_use"}),
        )


def test_pipeline_digest_binds_order_version_options_and_seed() -> None:
    first = _RevisionPass(PassDescriptor("first", "1", {"level": 1}, seed=7))
    second = _RevisionPass(PassDescriptor("second", "1"))

    digest = PassManager((first, second)).pipeline_digest

    assert digest == PassManager((first, second)).pipeline_digest
    assert digest != PassManager((second, first)).pipeline_digest
    assert (
        digest
        != PassManager(
            (_RevisionPass(PassDescriptor("first", "2", {"level": 1}, seed=7)), second)
        ).pipeline_digest
    )


def test_canonicalization_is_a_semantic_and_object_no_op() -> None:
    module = _module()
    result = PassManager((CanonicalizeAttributesPass(),)).run(module)

    assert result.ok
    assert result.module is module
    assert result.pass_results[0].changed is False
    assert result.pass_results[0].statistics["operations_scanned"] == 1


def test_pass_manager_carries_only_declared_preserved_analysis() -> None:
    module = _module()
    analyses = AnalysisManager()
    def_use = DefUseAnalysis()
    lifetime = QubitLifetimeAnalysis()
    old_def_use = analyses.get(def_use, module)
    old_lifetime = analyses.get(lifetime, module)

    result = PassManager((_RevisionPass(),)).run(module, analysis_manager=analyses)

    assert result.ok and result.module.revision == 1
    assert analyses.get(def_use, result.module) is old_def_use
    assert analyses.get(lifetime, result.module) is not old_lifetime


def test_semantic_change_fails_closed_with_structured_diagnostic() -> None:
    module = _module()

    class RoguePass:
        descriptor = PassDescriptor("rogue", "1")

        def run(self, current: QuantumModule) -> PassResult:
            changed = QuantumModule(
                Region((Block(operations=(Operation("test.changed"),)),)),
                revision=current.revision + 1,
            )
            return PassResult(changed, changed=True)

    result = PassManager((RoguePass(),)).run(module)

    assert not result.ok
    assert result.module is module
    assert result.pass_results == ()
    assert "semantic program identity" in result.diagnostics[0].message


def test_changed_pass_must_advance_revision() -> None:
    module = _module()

    class BadRevisionPass:
        descriptor = PassDescriptor("bad_revision", "1")

        def run(self, current: QuantumModule) -> PassResult:
            return PassResult(current, changed=True)

    result = PassManager((BadRevisionPass(),)).run(module)

    assert not result.ok
    assert "advance module revision" in result.diagnostics[0].message


def test_pass_diagnostic_stops_pipeline_without_accepting_output() -> None:
    module = _module()

    class FailingPass:
        descriptor = PassDescriptor("failing", "1")

        def run(self, current: QuantumModule) -> PassResult:
            return PassResult(
                QuantumModule(current.body, revision=current.revision + 1),
                changed=True,
                diagnostics=(
                    Diagnostic(
                        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                        "intentional test failure",
                    ),
                ),
            )

    result = PassManager((FailingPass(), _RevisionPass())).run(module)

    assert not result.ok
    assert result.module is module
    assert len(result.pass_results) == 1
    assert result.diagnostics[0].message == "intentional test failure"
