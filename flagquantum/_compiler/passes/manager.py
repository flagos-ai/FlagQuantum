"""Deterministic, contract-checked private pass pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..analyses.base import AnalysisManager
from ..diagnostics import Diagnostic, DiagnosticCode, DiagnosticSeverity
from ..ir.modules import QuantumModule
from .base import CompilerPass, PassResult


@dataclass(frozen=True)
class PipelineResult:
    module: QuantumModule
    pass_results: tuple[PassResult, ...]
    diagnostics: tuple[Diagnostic, ...]
    pipeline_digest: str

    @property
    def ok(self) -> bool:
        return not any(
            item.severity is DiagnosticSeverity.ERROR for item in self.diagnostics
        )


class PassManager:
    def __init__(self, passes: tuple[CompilerPass, ...] = ()) -> None:
        self._passes = tuple(passes)

    @property
    def pipeline_digest(self) -> str:
        payload = json.dumps(
            [item.descriptor.canonical() for item in self._passes],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _contract_error(message: str) -> Diagnostic:
        return Diagnostic(DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH, message)

    def run(
        self,
        module: QuantumModule,
        *,
        analysis_manager: AnalysisManager | None = None,
    ) -> PipelineResult:
        current = module
        results: list[PassResult] = []
        diagnostics: list[Diagnostic] = []
        analyses = analysis_manager or AnalysisManager()
        for compiler_pass in self._passes:
            result = compiler_pass.run(current)
            violation = self._validate_result(compiler_pass, current, result)
            if violation is not None:
                diagnostics.append(violation)
                break
            results.append(result)
            diagnostics.extend(result.diagnostics)
            if any(
                item.severity is DiagnosticSeverity.ERROR for item in result.diagnostics
            ):
                break
            if result.changed:
                analyses.carry_preserved(
                    current, result.module, result.preserved_analyses
                )
                current = result.module
        return PipelineResult(
            current, tuple(results), tuple(diagnostics), self.pipeline_digest
        )

    def _validate_result(
        self,
        compiler_pass: CompilerPass,
        before: QuantumModule,
        result: PassResult,
    ) -> Diagnostic | None:
        label = compiler_pass.descriptor.name
        if not compiler_pass.descriptor.preserves_semantics:
            return self._contract_error(
                f"Phase 1 pass {label!r} must declare semantic preservation"
            )
        identity_changed = result.module.program_identity != before.program_identity
        if (
            compiler_pass.descriptor.program_identity_policy == "preserve"
            and identity_changed
        ):
            return self._contract_error(
                f"pass {label!r} changed semantic program identity"
            )
        if (
            compiler_pass.descriptor.program_identity_policy == "transform"
            and result.changed
            and not identity_changed
        ):
            return self._contract_error(
                f"transform pass {label!r} reported a change without deriving "
                "a new program identity"
            )
        if result.changed and result.module.revision <= before.revision:
            return self._contract_error(
                f"changed pass {label!r} must advance module revision"
            )
        if not result.changed and result.module != before:
            return self._contract_error(
                f"unchanged pass {label!r} returned a different module"
            )
        return None


__all__ = ["PassManager", "PipelineResult"]
