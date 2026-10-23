"""Guard the decision that takes a parity row out of scope.

`out_of_scope` is the only status in the parity matrix that makes the programme's
target smaller rather than the gap narrower, so it is the only status that must
be earned by a record rather than by an edit. These tests hold three things
together: the token, the record that declares it, and the tree the record makes a
claim about. A row that moves without its record, a record that stops naming its
row, and a record whose measured claim the tree no longer supports all fail here.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from tools.parity_matrix import load_inputs, validate_contract

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _declined_rows(contract: dict) -> list[dict]:
    return [
        row
        for domain in contract["domains"]
        for row in domain["capabilities"]
        if row["status"] == "out_of_scope"
    ]


@pytest.fixture()
def contract_and_registry() -> tuple[dict, dict]:
    return load_inputs()


def test_every_declined_row_names_a_record_that_names_it(
    contract_and_registry: tuple[dict, dict],
) -> None:
    contract, _ = contract_and_registry
    declined = _declined_rows(contract)
    assert declined, "no row is out of scope, so this guard is measuring nothing"
    for row in declined:
        record = ROOT / row["decision_ref"]
        assert record.is_file(), f"{row['id']} cites a missing record"
        assert f"`{row['id']}`" in record.read_text(encoding="utf-8")


def test_a_declined_row_is_not_also_scheduled(
    contract_and_registry: tuple[dict, dict],
) -> None:
    for row in _declined_rows(contract_and_registry[0]):
        assert row["priority"] == "excluded", row["id"]


def test_the_contract_rejects_a_declined_row_without_a_record(
    contract_and_registry: tuple[dict, dict],
) -> None:
    contract = copy.deepcopy(contract_and_registry[0])
    _declined_rows(contract)[0].pop("decision_ref")
    errors = validate_contract(contract, contract_and_registry[1])
    assert any("'out_of_scope' requires a decision_ref" in error for error in errors)


def test_the_contract_rejects_a_record_that_does_not_name_the_row(
    contract_and_registry: tuple[dict, dict],
) -> None:
    contract = copy.deepcopy(contract_and_registry[0])
    _declined_rows(contract)[0]["decision_ref"] = "docs/roadmap/README.md"
    errors = validate_contract(contract, contract_and_registry[1])
    assert any("have drifted apart" in error for error in errors)


def test_the_contract_rejects_a_missing_record(
    contract_and_registry: tuple[dict, dict],
) -> None:
    contract = copy.deepcopy(contract_and_registry[0])
    _declined_rows(contract)[0]["decision_ref"] = "docs/roadmap/NEVER_WRITTEN.md"
    errors = validate_contract(contract, contract_and_registry[1])
    assert any("decision_ref does not exist" in error for error in errors)


def test_the_contract_rejects_out_of_scope_that_is_still_scheduled(
    contract_and_registry: tuple[dict, dict],
) -> None:
    contract = copy.deepcopy(contract_and_registry[0])
    _declined_rows(contract)[0]["priority"] = "next"
    errors = validate_contract(contract, contract_and_registry[1])
    assert any(
        "'out_of_scope' requires 'excluded' priority" in error for error in errors
    )


def test_the_declined_type_is_still_absent_from_the_tree(
    contract_and_registry: tuple[dict, dict],
) -> None:
    """The record's central claim is an absence, so the tree must keep it.

    If a matrix-valued operator type is ever introduced, the decision has been
    reversed by implementation and the record must be revisited rather than left
    describing a tree that no longer exists.
    """

    pattern = re.compile(r"MatrixOperator|matrix_operator")
    found = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "flagquantum").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert not found, f"a matrix-valued operator type now exists: {found}"


def test_the_declined_type_is_not_a_stable_export() -> None:
    import flagquantum as fq

    assert "MatrixOperator" not in fq.__all__
    assert not any("Matrix" in name and "Operator" in name for name in fq.__all__)


# The record quotes outputs, so every one of them is a claim a test has to falsify.


def test_the_four_routes_the_record_quotes_are_real() -> None:
    import torch

    import flagquantum as fq
    from flagquantum.noise.channels import amplitude_damping_channel
    from flagquantum.operators import SuperOperator
    from flagquantum.simulation.density_matrix import (
        apply_unitary_density,
        expand_operator,
    )

    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    identity = torch.eye(2, dtype=torch.complex128)

    program = fq.Circuit(1).any(0, matrix=pauli_x).to_ir().instructions
    assert [(item.name, item.wires) for item in program] == [("any", (0,))]

    term = SuperOperator.left_right_multiply(pauli_x, identity)
    assert tuple(term.dense().shape) == (4, 4)

    channel = amplitude_damping_channel(0.3)
    assert len(channel.kraus) == 2
    assert type(channel.kraus[0]).__name__ == "Tensor"

    assert callable(apply_unitary_density)
    assert callable(expand_operator)


def test_the_record_quotes_the_refusals_this_tree_raises() -> None:
    import torch

    import flagquantum as fq

    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    quoted = {
        "X(0) + matrix": "unsupported operand type(s) for +: "
        "'Observable' and 'Tensor'",
        "X(0) @ matrix": "unsupported operand type(s) for @: "
        "'Observable' and 'Tensor'",
        "matrix + X(0)": "unsupported operand type(s) for +: "
        "'Tensor' and 'Observable'",
    }
    calls = {
        "X(0) + matrix": lambda: fq.X(0) + pauli_x,
        "X(0) @ matrix": lambda: fq.X(0) @ pauli_x,
        "matrix + X(0)": lambda: pauli_x + fq.X(0),
    }
    for label, call in calls.items():
        with pytest.raises(TypeError) as raised:
            call()
        assert str(raised.value) == quoted[label]
    assert type(fq.X(0) @ fq.Y(1)).__name__ == "Observable"


def test_the_record_quotes_the_pauli_projection_this_tree_produces() -> None:
    import torch

    from flagquantum.simulation.pauli import normalized_pauli_basis, pauli_words

    basis = normalized_pauli_basis(
        1, dtype=torch.complex128, device=torch.device("cpu")
    )
    hadamard = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / 2**0.5
    coefficients = [torch.trace(b.mH @ hadamard) for b in basis]
    assert pauli_words(1) == ("I", "X", "Y", "Z")
    assert [complex(round(float(c.real), 6)) for c in coefficients] == [
        0j,
        (1 + 0j),
        0j,
        (1 + 0j),
    ]
    rebuilt = sum(c * b for c, b in zip(coefficients, basis, strict=True))
    assert bool(torch.allclose(rebuilt, hadamard, atol=1e-12))


def test_the_record_quotes_the_real_dtype_refusal_this_tree_raises() -> None:
    import torch

    from flagquantum.operators import SuperOperator

    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    with pytest.raises(ValueError) as raised:
        SuperOperator.left_right_multiply(pauli_x, torch.eye(2))
    assert "an operator factor must have a complex dtype" in str(raised.value)
    assert "torch.float32" in str(raised.value)


def test_a_wrapped_matrix_operator_joins_the_superoperator_algebra() -> None:
    """The record's reason 2, measured: the factor contract is duck-typed."""

    import torch

    from flagquantum.operators import SuperOperator

    class Wrapped:
        def __init__(self, matrix):
            self._matrix = matrix

        dtype = property(lambda self: self._matrix.dtype)
        dimension = property(lambda self: int(self._matrix.shape[0]))

        def __matmul__(self, state):
            return self._matrix @ state

        def __rmatmul__(self, state):
            return state @ self._matrix

        def dense(self):
            return self._matrix

        def adjoint(self):
            return Wrapped(self._matrix.mH)

    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    term = SuperOperator.left_right_multiply(
        Wrapped(pauli_x), torch.eye(2, dtype=torch.complex128)
    )
    assert term.dimension == 2
    assert tuple(term.dense().shape) == (4, 4)
    assert tuple(term.adjoint().dense().shape) == (4, 4)
