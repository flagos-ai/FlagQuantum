# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 testing.py.

"""Reusable checks for owned register examples and their lowered circuits."""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_compatibility import (
    RegisterTypeCheckingSeverity,
    check_register_types_consistent,
)
from flagquantum.core.ir.program.registers import RegisterType

CheckKind = Literal["make", "decompose", "counts", "serialize", "qtyping"]
CHECKS: tuple[CheckKind, ...] = ("make", "decompose", "counts", "serialize", "qtyping")


class CheckStatus(Enum):
    PASS = auto()
    FAIL = auto()
    MISSING = auto()
    NA = auto()
    UNVERIFIED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    message: str = ""


@dataclass(frozen=True)
class RegisterExample:
    make: Callable[[], object]
    lower: Callable[[ProgramRecord], CircuitIR] | None = None
    declared_counts: Callable[[], Mapping[str, int]] | None = None
    type_edges: tuple[tuple[RegisterType, RegisterType], ...] = ()
    atomic: bool = False


def check_example(example: RegisterExample, kind: CheckKind) -> CheckResult:
    """Check the explicit factory and independently supplied declaration."""
    try:
        record = example.make()
        if not isinstance(record, ProgramRecord):
            return CheckResult(CheckStatus.FAIL, "Factory must return a ProgramRecord")
        if kind == "make":
            return CheckResult(CheckStatus.PASS)

        diagnostics = verify_program_record(record)
        if diagnostics:
            return CheckResult(
                CheckStatus.FAIL, "; ".join(d.message for d in diagnostics)
            )
        if kind == "qtyping":
            return _check_types(example)
        if kind == "counts":
            return _check_counts(example, record)

        if example.lower is None:
            status = CheckStatus.NA if example.atomic else CheckStatus.MISSING
            return CheckResult(status, "No physical decomposition is declared")
        circuit = example.lower(record).validate()
        if kind == "serialize":
            encoded = circuit.to_dict()
            decoded = CircuitIR.from_dict(json.loads(json.dumps(encoded)))
            if decoded.to_dict() != encoded:
                return CheckResult(
                    CheckStatus.FAIL,
                    "Circuit wire or operation data changed on roundtrip",
                )
        return CheckResult(CheckStatus.PASS)
    except AssertionError as error:
        return CheckResult(CheckStatus.FAIL, str(error))
    except Exception as error:
        return CheckResult(CheckStatus.ERROR, f"{type(error).__name__}: {error}")


def _check_types(example: RegisterExample) -> CheckResult:
    if example.lower is None:
        return CheckResult(CheckStatus.NA, "Types require a decomposition")
    for severity in RegisterTypeCheckingSeverity:
        for producer, consumer in example.type_edges:
            if check_register_types_consistent(producer, consumer, severity):
                continue
            status = (
                CheckStatus.FAIL
                if severity is RegisterTypeCheckingSeverity.LOOSE
                else CheckStatus.UNVERIFIED
            )
            return CheckResult(
                status,
                f"{severity.name} register type mismatch: {producer} -> {consumer}",
            )
    return CheckResult(CheckStatus.PASS)


def _check_counts(example: RegisterExample, record: ProgramRecord) -> CheckResult:
    if example.lower is None and example.declared_counts is None:
        return CheckResult(
            CheckStatus.MISSING, "Neither declaration nor decomposition is available"
        )
    if example.lower is None or example.declared_counts is None:
        return CheckResult(
            CheckStatus.UNVERIFIED, "A second independent count is unavailable"
        )
    declared = dict(example.declared_counts())
    observed = Counter(
        instruction.name
        for instruction in example.lower(record).validate().instructions
    )
    if declared != observed:
        return CheckResult(
            CheckStatus.FAIL,
            f"Declared instruction counts {declared} != {dict(observed)}",
        )
    return CheckResult(CheckStatus.PASS)
