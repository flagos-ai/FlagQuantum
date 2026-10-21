"""The keyword that carries a custom operation's matrix is named ``matrix``.

``Circuit.any`` used to call it ``unitary``, which was a false claim about the
value: :func:`flagquantum.expectation` builds its circuit by handing the *observable*
to this keyword, and an observable is Hermitian rather than unitary. The keyword is
now ``matrix``, the same word :meth:`flagquantum.Circuit.gate` already uses and the
word the executors read, and ``unitary`` stays as a deprecated alias so a caller
written against 0.3.x keeps working through the migration window.

These tests hold three things apart, because they can fail independently:

- the new spelling records and executes the operator;
- the old spelling records the *same* instruction and warns, so the alias is a
  forwarding name rather than a second code path that could drift;
- supplying neither or both is refused by name, and the refusal for "neither" keeps
  the text CPython produced when the keyword was required.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import flagquantum as fq
from flagquantum.circuit import expectation

pytestmark = pytest.mark.unit

#: The ``x`` gate, spelled as a matrix, so a custom operation and a registered
#: opcode can be compared through the same statevector.
X_MATRIX = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)


def _state(circuit: fq.Circuit) -> list[float]:
    return [round(float(value), 6) for value in circuit.state().reshape(-1).real]


def _instructions(circuit: fq.Circuit) -> list[dict[str, object]]:
    return list(circuit.to_ir().to_dict()["instructions"])


def test_the_matrix_keyword_records_an_operation_the_registry_does_not_know() -> None:
    circuit = fq.Circuit(2).x(0).any(0, 1, matrix=torch.kron(X_MATRIX, X_MATRIX))
    assert [item.name for item in circuit.to_ir().instructions] == ["x", "any"]
    # The opcode is unknown to the registry, so the matrix is what executes; ``x`` on
    # qubit 0 followed by ``x`` on both qubits leaves qubit 1 excited.
    assert _state(circuit) == [0.0, 1.0, 0.0, 0.0]


def test_the_deprecated_keyword_records_the_same_instruction() -> None:
    matrix = torch.kron(X_MATRIX, X_MATRIX)
    with pytest.warns(DeprecationWarning):
        through_alias = fq.Circuit(2).x(0).any(0, 1, matrix=None, unitary=matrix)
    through_matrix = fq.Circuit(2).x(0).any(0, 1, matrix=matrix)
    assert _instructions(through_alias) == _instructions(through_matrix)
    assert _state(through_alias) == _state(through_matrix)


def test_the_deprecated_keyword_warns_once_with_the_replacement_and_the_removal() -> (
    None
):
    circuit = fq.Circuit(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        circuit.any(0, unitary=X_MATRIX)
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    message = str(caught[0].message)
    assert "any(unitary=...) is deprecated; use any(matrix=...)" in message
    assert "0.4.0" in message
    # The warning must point at the caller's line, not at the library's own frame,
    # or a user reading a warning-filtered test run cannot find the call to change.
    assert caught[0].filename == __file__


def test_the_public_alias_method_accepts_the_matrix_keyword() -> None:
    # ``Circuit.unitary`` is the older method name for the same operation and is
    # not deprecated; it must follow the keyword rename because it is ``any``.
    circuit = fq.Circuit(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        circuit.unitary(0, matrix=X_MATRIX)
    assert [item.name for item in circuit.to_ir().instructions] == ["any"]
    assert _state(circuit) == [0.0, 1.0]


def test_supplying_neither_keyword_is_refused_by_name() -> None:
    with pytest.raises(TypeError) as caught:
        fq.Circuit(1).any(0)
    # The message is the one CPython produced while the keyword was required, with
    # the new name in it, so a caller matching on the old text still matches the
    # shape and is told which keyword to write.
    assert str(caught.value) == (
        "Circuit.any() missing 1 required keyword-only argument: 'matrix'"
    )


def test_supplying_both_keywords_is_refused_rather_than_ordered() -> None:
    with pytest.raises(TypeError) as caught:
        fq.Circuit(1).any(0, matrix=X_MATRIX, unitary=X_MATRIX)
    assert str(caught.value) == (
        "Circuit.any() received both 'matrix' and its deprecated alias 'unitary'; "
        "supply exactly one"
    )


def test_a_refused_call_leaves_the_circuit_unchanged() -> None:
    circuit = fq.Circuit(1).h(0)
    before = _instructions(circuit)
    for kwargs in ({}, {"matrix": X_MATRIX, "unitary": X_MATRIX}):
        with pytest.raises(TypeError):
            circuit.any(0, **kwargs)
    assert _instructions(circuit) == before


def test_the_dense_expectation_route_passes_an_observable_that_is_not_unitary() -> None:
    # This is the reason the keyword could not keep the name ``unitary``: the value
    # handed to it here is an observable, and an observable is not unitary.
    observable = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
    ket = torch.tensor([1.0, 1.0], dtype=torch.complex128) / 2**0.5
    value = expectation((observable, (0,)), ket=ket)
    assert torch.allclose(value, torch.zeros(1, dtype=torch.complex128))
