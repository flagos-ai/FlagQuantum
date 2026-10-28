"""Approved finite M-P CH feasibility, not production or performance acceptance."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.stabilizer._phase_overlap import (
    _CliffordState,
    _overlap,
    _unit_phase,
    _validate_word,
)
from tests.integration import test_ftqc_coherent_reference as scalar_reference

pytestmark = pytest.mark.integration
_one_cpu_thread = scalar_reference._one_cpu_thread
references = scalar_reference.references
_ATOL = 1e-12
_NAMES = tuple(f"n{n}t{t}" for n, t in scalar_reference._MATRIX) + ("hth", "bell")
_Word = tuple[tuple[str, tuple[int, ...]], ...]


def _branch_words(instructions: _Word) -> tuple[_Word, ...]:
    words: list[_Word] = []
    for choices in itertools.product(
        (0, 1), repeat=sum(g == "t" for g, _ in instructions)
    ):
        selected = iter(choices)
        word: list[tuple[str, tuple[int, ...]]] = []
        for gate, wires in instructions:
            if gate == "t":
                if next(selected):
                    word.append(("s", wires))
            else:
                word.append((gate, wires))
        words.append(tuple(word))
    return tuple(words)


def _public_values(
    case: scalar_reference._Reference,
) -> tuple[torch.Tensor, torch.Tensor]:
    circuit = fq.Circuit(case.n)
    for gate, wires in case.instructions:
        circuit.gate(gate, wires)
    result = fq.run(
        circuit,
        outputs=fq.probabilities(list(range(case.n))),
        options=fq.ExecutionOptions(
            mode="statevector",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
    )
    assert result.probabilities is not None
    return result.to_statevector().reshape(-1), result.probabilities


@dataclass(frozen=True)
class _Capture:
    states: dict[str, tuple[_CliffordState, ...]]
    bases: dict[int, tuple[_CliffordState, ...]]
    grams: dict[str, torch.Tensor]
    amplitudes: dict[str, torch.Tensor]
    public: dict[str, tuple[torch.Tensor, torch.Tensor]]
    counts: tuple[int, int, int]


@pytest.fixture(scope="module")
def capture(references: dict[str, scalar_reference._Reference]) -> _Capture:
    states = {
        name: tuple(
            _CliffordState.from_word(case.n, word)
            for word in _branch_words(case.instructions)
        )
        for name, case in references.items()
    }
    bases = {
        n: tuple(
            _CliffordState.from_word(
                n,
                tuple(
                    ("x", (wire,)) for wire in range(n) if index & (1 << (n - 1 - wire))
                ),
            )
            for index in range(1 << n)
        )
        for n in (1, 2, 3)
    }
    state_count = sum(map(len, states.values())) + sum(map(len, bases.values()))
    grams: dict[str, torch.Tensor] = {}
    amplitudes: dict[str, torch.Tensor] = {}
    gram_queries = basis_queries = 0
    for name, branches in states.items():
        grams[name] = torch.tensor(
            [[_overlap(bra, ket) for ket in branches] for bra in branches],
            dtype=torch.complex128,
        )
        gram_queries += len(branches) ** 2
        amplitudes[name] = torch.tensor(
            [
                [_overlap(bra, ket) for ket in branches]
                for bra in bases[references[name].n]
            ],
            dtype=torch.complex128,
        )
        basis_queries += len(branches) * (1 << references[name].n)
    assert (state_count, gram_queries, basis_queries) == (64, 562, 288)
    public = {name: _public_values(case) for name, case in references.items()}
    assert len(public) == 10
    return _Capture(states, bases, grams, amplitudes, public, (64, 850, 10))


def _finite_complex(values: torch.Tensor, shape: tuple[int, ...]) -> None:
    if (
        values.shape != shape
        or values.dtype != torch.complex128
        or values.device.type != "cpu"
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("require finite ordered CPU complex128 capture")


def _reconstruct(
    gram: torch.Tensor, amplitudes: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    count = coefficients.numel()
    _finite_complex(gram, (count, count))
    _finite_complex(coefficients, (count,))
    _finite_complex(amplitudes, (amplitudes.shape[0], count))
    norm = torch.vdot(coefficients, gram @ coefficients)
    if abs(complex(norm) - 1) > _ATOL:
        raise ValueError("coherent norm must be one before normalization")
    state = amplitudes @ coefficients
    if abs(float(state.abs().square().sum()) - 1) > _ATOL:
        raise ValueError("basis amplitudes lost coherent norm")
    return state


@pytest.mark.parametrize("name", _NAMES)
def test_canonical_amplitudes_and_complex_gram_are_independent(
    name: str, capture: _Capture, references: dict[str, scalar_reference._Reference]
) -> None:
    case = references[name]
    expected = torch.stack(case.branches, dim=1)
    torch.testing.assert_close(capture.amplitudes[name], expected, atol=_ATOL, rtol=0)
    independent_gram = expected.conj().T @ expected
    gram = capture.grams[name]
    torch.testing.assert_close(gram, independent_gram, atol=_ATOL, rtol=0)
    torch.testing.assert_close(gram, gram.conj().T, atol=_ATOL, rtol=0)
    torch.testing.assert_close(
        gram.diag(),
        torch.ones(len(case.branches), dtype=torch.complex128),
        atol=_ATOL,
        rtol=0,
    )
    for state in capture.states[name]:
        assert (
            len(state.f) == len(state.g) == len(state.m) == len(state.gamma) == case.n
        )
        assert len(state.word) <= 22
        assert all(
            type(row) is int for rows in (state.f, state.g, state.m) for row in rows
        )


@pytest.mark.parametrize("name", _NAMES)
def test_coherent_outputs_norm_and_public_execution(
    name: str, capture: _Capture, references: dict[str, scalar_reference._Reference]
) -> None:
    case = references[name]
    coefficients = torch.tensor(case.coefficients, dtype=torch.complex128)
    state = _reconstruct(capture.grams[name], capture.amplitudes[name], coefficients)
    torch.testing.assert_close(state, case.direct, atol=_ATOL, rtol=0)
    density = scalar_reference._unit_density(state, 1 << case.n)
    public_state, probabilities = capture.public[name]
    torch.testing.assert_close(
        scalar_reference._unit_density(public_state, 1 << case.n),
        density,
        atol=_ATOL,
        rtol=0,
    )
    assert probabilities.shape == (1, 1 << case.n)
    assert probabilities.dtype == torch.float64 and probabilities.device.type == "cpu"
    assert bool(torch.isfinite(probabilities).all())
    torch.testing.assert_close(
        probabilities[0], state.abs().square(), atol=_ATOL, rtol=0
    )
    assert abs(float(probabilities.sum()) - 1) <= _ATOL


def test_rephase_compensation_and_wrong_phase_are_discriminating(
    capture: _Capture, references: dict[str, scalar_reference._Reference]
) -> None:
    name = "bell"
    coefficients = torch.tensor(references[name].coefficients, dtype=torch.complex128)
    amplitudes = capture.amplitudes[name]
    gram = capture.grams[name]
    phases = torch.tensor([1j, 1], dtype=torch.complex128)
    rephased_amplitudes = amplitudes * phases
    rephased_gram = phases.conj()[:, None] * gram * phases[None, :]
    expected = references[name].direct
    compensated = _reconstruct(
        rephased_gram, rephased_amplitudes, coefficients / phases
    )
    torch.testing.assert_close(compensated, expected, atol=_ATOL, rtol=0)
    assert not torch.allclose(
        rephased_amplitudes @ coefficients, expected, atol=_ATOL, rtol=0
    )
    assert not torch.allclose(gram, gram.conj(), atol=_ATOL, rtol=0)
    for wrong in (
        gram.conj(),
        gram.abs().to(torch.complex128),
        torch.diag(gram.diag()),
    ):
        with pytest.raises(ValueError, match="norm"):
            _reconstruct(wrong, amplitudes, coefficients)
    mixture = (amplitudes.abs().square() * coefficients.abs().square()).sum(dim=1)
    mixture /= coefficients.abs().square().sum()
    assert not torch.allclose(mixture, expected.abs().square(), atol=_ATOL, rtol=0)


@pytest.mark.parametrize(
    "damage", ["shape", "dtype", "nan", "infinity", "zero", "scale"]
)
def test_bad_capture_is_not_normalized_into_agreement(
    damage: str, capture: _Capture
) -> None:
    gram = capture.grams["bell"].clone()
    amplitudes = capture.amplitudes["bell"].clone()
    # Use the captured Gram with a literal normalized one-branch coefficient.
    coefficients = torch.tensor([1 + 0j, 0j], dtype=torch.complex128)
    if damage == "shape":
        gram = gram[:1]
    elif damage == "dtype":
        gram = gram.to(torch.complex64)
    elif damage == "nan":
        gram[0, 0] = torch.tensor(complex(float("nan"), 0), dtype=torch.complex128)
    elif damage == "infinity":
        amplitudes[0, 0] = torch.tensor(
            complex(float("inf"), 0), dtype=torch.complex128
        )
    elif damage == "zero":
        coefficients.zero_()
    else:
        amplitudes *= 0.5
    with pytest.raises(ValueError):
        _reconstruct(gram, amplitudes, coefficients)


@pytest.mark.parametrize(
    ("n", "word"),
    [
        (0, ()),
        (4, ()),
        (True, ()),
        (2, (("t", (0,)),)),
        (2, (("p0", (0,)),)),
        (2, (("cz", (0, 1)),)),
        (2, (("h", (2,)),)),
        (2, (("h", (True,)),)),
        (2, (("cx", (0, 0)),)),
        (2, (("h", (0,)),) * 23),
    ],
)
def test_outside_input_domain_refused_before_construction(n: int, word: _Word) -> None:
    with pytest.raises(ValueError):
        _validate_word(n, word)


@pytest.mark.parametrize("phase", [0j, 2 + 0j, complex(float("nan"), 0)])
def test_nonunit_phase_refused_without_query(phase: complex) -> None:
    with pytest.raises(ValueError):
        _unit_phase(phase)


def test_unequal_widths_refused_before_overlap_scratch(capture: _Capture) -> None:
    with pytest.raises(ValueError, match="equal widths"):
        _overlap(capture.bases[1][0], capture.bases[2][0])
    assert capture.counts == (64, 850, 10)
