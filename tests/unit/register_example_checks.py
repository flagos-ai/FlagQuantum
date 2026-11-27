# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 testing.py and conftest.py.

"""Pytest and classical-action adapters for shared register example checks."""

from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest

from flagquantum.testing.example_checks import (
    CHECKS as CHECKS,
)
from flagquantum.testing.example_checks import (
    CheckKind as CheckKind,
)
from flagquantum.testing.example_checks import (
    CheckResult as CheckResult,
)
from flagquantum.testing.example_checks import (
    CheckStatus as CheckStatus,
)
from flagquantum.testing.example_checks import (
    RegisterExample as RegisterExample,
)
from flagquantum.testing.example_checks import (
    check_example as check_example,
)


def apply_pytest_result(kind: CheckKind, result: CheckResult) -> None:
    """Preserve source skips for unavailable or unverified checks.

    Invalid owned fixtures fail. Source-only expected failures remain in the
    locked reference suite and do not exempt the native checks.
    """
    status = result.status
    if status is CheckStatus.PASS:
        return
    if (
        (kind == "decompose" and status in (CheckStatus.NA, CheckStatus.MISSING))
        or (
            kind == "counts"
            and status in (CheckStatus.NA, CheckStatus.MISSING, CheckStatus.UNVERIFIED)
        )
        or (kind == "qtyping" and status in (CheckStatus.NA, CheckStatus.UNVERIFIED))
    ):
        pytest.skip(result.message)
    pytest.fail(result.message, pytrace=False)


def assert_classical_action(
    declared: Callable[..., Any],
    decomposed: Callable[..., Any],
    **inputs: Sequence[Any],
) -> None:
    """Compare both classical paths over their complete Cartesian fixture."""
    names = tuple(inputs)
    for values in product(*inputs.values()):
        arguments = dict(zip(names, values, strict=True))
        np.testing.assert_equal(
            declared(**arguments), decomposed(**arguments), err_msg=f"{arguments=}"
        )
