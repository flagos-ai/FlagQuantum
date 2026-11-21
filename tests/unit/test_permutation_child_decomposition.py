"""Permutation labels require owning cycle metadata and unchanged measured feedback."""

from dataclasses import replace

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.permutation_child_decomposition import (
    permutation_child_decomposition,
)

pytestmark = pytest.mark.unit


def test_cycle_and_equality_labels_reject_changed_feedback_and_unrelated_constants() -> (
    None
):
    parameters = native.attributes("sparse_permutation")
    record = native.decomposition("sparse_permutation")
    assert record is not None
    changed = replace(
        record,
        functions=tuple(
            replace(
                function,
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(
                                    operation,
                                    condition_clauses=tuple(
                                        tuple(
                                            replace(c, equals=not c.equals)
                                            for c in clause
                                        )
                                        for clause in operation.condition_clauses
                                    ),
                                )
                                if operation.condition_clauses
                                else operation
                            )
                            for operation in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
            for function in record.functions
        ),
    )
    assert changed != record
    rejected = set()
    for function in record.functions:
        qualified = permutation_child_decomposition(record, function.name, parameters)
        if qualified is None or qualified[0] not in {
            "PermutationCycle",
            "EqualsAConstant",
        }:
            continue
        assert (
            permutation_child_decomposition(changed, function.name, parameters) is None
        )
        assert (
            permutation_child_decomposition(
                record, function.name, {**parameters, "cycles": ((5, 6, 9),)}
            )
            is None
        )
        rejected.add(qualified[0])
    assert rejected == {"PermutationCycle", "EqualsAConstant"}
