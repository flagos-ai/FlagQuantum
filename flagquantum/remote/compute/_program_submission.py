"""Direct Circuit submission layered over the Jiuding script-job transport."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ._program_job import decode_program_result, write_program_bundle

if TYPE_CHECKING:
    from ...observables import OutputRequest


class _ProgramSubmissionMixin:
    """Add program jobs without expanding the low-level transport module."""

    def submit_program(
        self,
        program: Any,
        *,
        target: str,
        image: str,
        receipt: str | Path,
        outputs: OutputRequest | Sequence[OutputRequest] | None = None,
        shots: int | None = None,
        image_region: str = "PRIVATE",
        python: str = sys.executable,
        pythonpath: str | Path | None = None,
        cpus: int = 2,
        memory_gib: int = 2,
    ) -> dict[str, Any]:
        """Submit a Circuit directly without requiring a user-authored script."""

        from ...core.ir import ensure_circuit_ir
        from ...observables import lower_outputs

        ir = ensure_circuit_ir(program)
        if outputs is None:
            if shots is not None:
                raise TypeError(
                    "shots requires fq.samples(...) or fq.counts(...) output"
                )
            ir = replace(ir, measurements=())
            operation = "statevector"
        else:
            measurements = lower_outputs(outputs, n_wires=ir.n_wires, shots=shots)
            assert measurements is not None
            ir = replace(ir, measurements=measurements)
            operation = "measurements"
        _, model = self._resolve_compute_target(target)
        selected_target = f"jiuding:gpu/{model}" if model else "jiuding:cpu"
        receipt_path = Path(receipt).resolve()
        if receipt_path.exists():
            raise FileExistsError(f"receipt already exists: {receipt_path}")
        if not receipt_path.parent.is_dir():
            raise ValueError("receipt directory must exist on shared storage")
        request_path, runner_path = write_program_bundle(
            ir,
            target=selected_target,
            operation=operation,
            receipt=receipt_path,
        )
        try:
            record = self.submit(
                runner_path,
                target=target,
                image=image,
                receipt=receipt_path,
                image_region=image_region,
                python=python,
                pythonpath=pythonpath,
                cpus=cpus,
                memory_gib=memory_gib,
            )
        except BaseException:
            if not receipt_path.exists():
                request_path.unlink(missing_ok=True)
                runner_path.unlink(missing_ok=True)
            raise
        return {
            **record,
            "submission_kind": "flagquantum_program",
            "program_request": str(request_path),
            "requested_target": target,
            "selected_target": selected_target,
            "operation": operation,
        }


def decode_job_result(value: Any, receipt: dict[str, Any]) -> Any:
    """Decode program responses while preserving arbitrary script results."""

    if not (
        isinstance(value, dict)
        and value.get("schema") == "flagquantum.jiuding.workspace_executor"
        and value.get("request_id") == "batch-program"
    ):
        return value
    resources = receipt.get("resources", {})
    evidence = value.get("evidence", {})
    requested_target = receipt.get("requested_target") or resources.get("target")
    selected_target = receipt.get("selected_target") or evidence.get("target")
    if not isinstance(requested_target, str) or not requested_target:
        raise RuntimeError("Jiuding program receipt has no requested target")
    if not isinstance(selected_target, str) or not selected_target:
        raise RuntimeError("Jiuding program result has no selected target")
    return decode_program_result(
        value,
        requested_target=requested_target,
        selected_target=selected_target,
    )


__all__ = ("_ProgramSubmissionMixin", "decode_job_result")
