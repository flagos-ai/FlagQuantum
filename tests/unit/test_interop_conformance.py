from __future__ import annotations

from typing import Any

import pytest

import flagquantum as fq
from flagquantum.interop import (
    INTEROP_API_VERSION,
    InteropConversionError,
    InteropConversionIssue,
    InteropConversionReport,
    InteropExportResult,
    InteropImportResult,
    InteropRejectionCase,
    InteropRoundTripCase,
    run_adapter_conformance,
    semantic_fingerprint,
)

pytestmark = pytest.mark.unit


class ReferenceAdapter:
    name = "reference"
    api_version = INTEROP_API_VERSION
    dependency_extra = "reference"

    def import_program(
        self, artifact: Any, *, allow_lossy: bool = False
    ) -> InteropImportResult:
        if artifact == "unsupported":
            issue = InteropConversionIssue(
                "unsupported_operation", "operation is not representable", "warning"
            )
            report = InteropConversionReport(
                self.name, "from_reference", "1.0", (issue,)
            )
            if not allow_lossy:
                raise InteropConversionError("conversion would be lossy", report)
            return InteropImportResult(fq.Circuit(1).to_ir(), report)
        source = artifact["program"]
        imported = fq.CircuitIR(
            source.n_wires,
            source.instructions,
            dtype=source.dtype,
            shape=source.shape,
            observables=source.observables,
            measurements=source.measurements,
            metadata={"interop": {"source": "reference"}},
        )
        return InteropImportResult(
            imported,
            InteropConversionReport(self.name, "from_reference", "1.0"),
        )

    def export_program(
        self, program: Any, *, allow_lossy: bool = False
    ) -> InteropExportResult:
        return InteropExportResult(
            {"program": fq.Circuit.from_ir(program).to_ir()},
            InteropConversionReport(self.name, "to_reference", "1.0"),
        )


def test_common_suite_certifies_round_trip_and_explicit_loss() -> None:
    result = run_adapter_conformance(
        ReferenceAdapter(),
        (
            InteropRoundTripCase(
                "bell",
                fq.Circuit(2).h(0).cx(0, 1).to_ir(),
            ),
        ),
        rejection_cases=(
            InteropRejectionCase(
                "unsupported_import",
                "import",
                "unsupported",
                "unsupported_operation",
            ),
        ),
    )

    assert result.passed
    assert [case.kind for case in result.cases] == ["round_trip", "rejection"]
    payload = result.to_dict()
    assert payload["schema"] == "flagquantum_interop_conformance_v1"
    assert payload["adapter"] == "reference"
    assert payload["passed"] is True


def test_semantic_fingerprint_ignores_transport_metadata_not_program() -> None:
    source = fq.Circuit(2).h(0).cx(0, 1).to_ir()
    transported = fq.CircuitIR(
        source.n_wires,
        source.instructions,
        dtype=source.dtype,
        shape=source.shape,
        metadata={"interop": {"source": "external"}},
    )
    changed = fq.Circuit(2).x(0).cx(0, 1).to_ir()

    assert semantic_fingerprint(source) == semantic_fingerprint(transported)
    assert semantic_fingerprint(source) != semantic_fingerprint(changed)
    assert semantic_fingerprint(
        source, semantic_metadata={"phase": 0.0}
    ) != semantic_fingerprint(source, semantic_metadata={"phase": 0.5})


def test_common_suite_reports_adapter_contract_drift() -> None:
    class DriftedAdapter(ReferenceAdapter):
        api_version = "2.0"

        def export_program(
            self, program: Any, *, allow_lossy: bool = False
        ) -> InteropExportResult:
            result = super().export_program(program, allow_lossy=allow_lossy)
            return InteropExportResult(
                result.artifact,
                InteropConversionReport("wrong", "to_reference", "1.0"),
            )

    result = run_adapter_conformance(
        DriftedAdapter(),
        (InteropRoundTripCase("identity", fq.Circuit(1).x(0).to_ir()),),
    )

    assert not result.passed
    assert [item.code for item in result.violations] == ["api_version_mismatch"]
    assert "report_adapter_mismatch" in {
        item.code for item in result.cases[0].violations
    }


def test_common_suite_rejects_ambiguous_case_names() -> None:
    case = InteropRoundTripCase("duplicate", fq.Circuit(1).to_ir())
    with pytest.raises(ValueError, match="must be unique"):
        run_adapter_conformance(ReferenceAdapter(), (case, case))


def test_common_suite_detects_silent_loss_acceptance() -> None:
    class UnsafeAdapter(ReferenceAdapter):
        def import_program(
            self, artifact: Any, *, allow_lossy: bool = False
        ) -> InteropImportResult:
            if artifact == "unsupported":
                return InteropImportResult(
                    fq.Circuit(1).to_ir(),
                    InteropConversionReport(self.name, "from_reference", "1.0"),
                )
            return super().import_program(artifact, allow_lossy=allow_lossy)

    result = run_adapter_conformance(
        UnsafeAdapter(),
        (),
        rejection_cases=(
            InteropRejectionCase(
                "silent_loss", "import", "unsupported", "unsupported_operation"
            ),
        ),
    )

    assert not result.passed
    assert "strict_rejection_accepted" in {
        item.code for item in result.cases[0].violations
    }


def test_common_suite_reports_non_adapter_without_invoking_it() -> None:
    result = run_adapter_conformance(
        object(),
        (InteropRoundTripCase("unused", fq.Circuit(1).to_ir()),),
    )

    assert not result.passed
    assert result.cases == ()
    assert {item.code for item in result.violations} == {
        "invalid_adapter",
        "api_version_mismatch",
        "invalid_adapter_identity",
    }
