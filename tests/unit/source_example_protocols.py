"""Observe every frozen source example protocol before pytest skip policy."""

from collections.abc import Callable

import qualtran.testing as checks
from qualtran import BloqExample


def source_example_protocols(example: BloqExample) -> dict[str, tuple[str, str]]:
    """Keep raw status and diagnostic without swallowing unexpected failures."""
    protocols: dict[str, Callable[[BloqExample], object]] = {
        "make": checks.assert_bloq_example_make,
        "decompose": checks.assert_bloq_example_decompose,
        "counts": checks.assert_equivalent_bloq_example_counts,
        "serialize": checks.assert_bloq_example_serializes,
        "qtyping": checks.assert_bloq_example_qtyping,
    }
    results = {}
    for mode, check in protocols.items():
        try:
            check(example)
        except checks.BloqCheckException as error:
            results[mode] = (error.check_result.name, error.msg)
        else:
            results[mode] = ("PASS", "")
    return results
