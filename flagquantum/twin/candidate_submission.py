"""Persistent bindings for prospectively submitted Twin candidate trials."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from ..remote.qpu import DeploymentResult, ProviderTaskHandle
from .candidate import (
    TwinCandidateEvaluation,
    TwinCandidateTrial,
)
from .submission import TwinSubmission, _require_fields

_SUBMISSION_SCHEMA = "flagquantum.twin_candidate_submission.v1"


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TwinCandidateSubmission:
    """A candidate trial bound to one already submitted remote task.

    Examples:
        submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
        fq.twin.dump_candidate_submission(
            submission,
            "twin-candidate-submission.json",
        )
    """

    trial: TwinCandidateTrial
    submission: TwinSubmission
    schema: str = _SUBMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SUBMISSION_SCHEMA:
            raise ValueError("unsupported Twin candidate-submission schema")
        if not isinstance(self.trial, TwinCandidateTrial):
            raise TypeError("trial must be a TwinCandidateTrial")
        if not isinstance(self.submission, TwinSubmission):
            raise TypeError("submission must be a TwinSubmission")
        if self.submission.experiment != self.trial.experiment:
            raise ValueError(
                "candidate submission does not match the frozen candidate trial"
            )

    @classmethod
    def from_receipt(
        cls,
        trial: TwinCandidateTrial,
        receipt: ProviderTaskHandle,
    ) -> TwinCandidateSubmission:
        """Bind a receipt without submitting or contacting a provider."""

        if not isinstance(trial, TwinCandidateTrial):
            raise TypeError("trial must be a TwinCandidateTrial")
        return cls(
            trial=trial,
            submission=TwinSubmission.from_receipt(trial.experiment, receipt),
        )

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def receipt(self) -> ProviderTaskHandle:
        return self.submission.receipt

    def validate_result(
        self,
        result: DeploymentResult,
        *,
        circuit: Any,
        confidence_level: float = 0.95,
    ) -> TwinCandidateEvaluation:
        """Validate a fetched result without submitting another task."""

        return self.trial.validate_result(
            result,
            receipt=self.receipt,
            circuit=circuit,
            confidence_level=confidence_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trial": self.trial.to_dict(),
            "submission": self.submission.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinCandidateSubmission:
        """Restore a candidate submission from strict version-1 data."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin candidate submission must be a mapping")
        _require_fields(
            payload,
            {"schema", "trial", "submission"},
            name="Twin candidate submission",
        )
        trial = payload["trial"]
        submission = payload["submission"]
        if not isinstance(trial, Mapping) or not isinstance(submission, Mapping):
            raise ValueError("Twin candidate submission members must be JSON objects")
        try:
            return cls(
                schema=str(payload["schema"]),
                trial=TwinCandidateTrial.from_dict(trial),
                submission=TwinSubmission.from_dict(submission),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin candidate submission") from error


def load_candidate_submission(
    path: str | PathLike[str],
) -> TwinCandidateSubmission:
    """Load a candidate submission offline without polling or resubmitting.

    Examples:
        submission = fq.twin.load_candidate_submission(
            "twin-candidate-submission.json"
        )
        result = provider.fetch_result(submission.receipt)
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin candidate submission from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin candidate submission file must contain a JSON object")
    return TwinCandidateSubmission.from_dict(payload)


def dump_candidate_submission(
    submission: TwinCandidateSubmission,
    path: str | PathLike[str],
) -> None:
    """Write a private candidate-submission file without replacing evidence."""

    if not isinstance(submission, TwinCandidateSubmission):
        raise TypeError("submission must be a TwinCandidateSubmission")
    destination = Path(path)
    encoded = (
        json.dumps(
            submission.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        try:
            existing = load_candidate_submission(destination)
        except ValueError as error:
            raise ValueError(
                "Refusing to replace invalid Twin candidate submission "
                f"at {destination}"
            ) from error
        if existing.identity != submission.identity:
            raise ValueError(
                "Refusing to replace different Twin candidate submission "
                f"at {destination}"
            )
    except OSError as error:
        raise ValueError(
            f"Cannot write Twin candidate submission to {destination}"
        ) from error


__all__ = (
    "dump_candidate_submission",
    "load_candidate_submission",
    "TwinCandidateSubmission",
)
