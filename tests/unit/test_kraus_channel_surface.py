"""Pin the general Kraus channel surface the parity contract claims.

`contracts/cudaq-parity-matrix.toml` states five things about
`flagquantum.noise.KrausChannel`: it accepts an arbitrary operator list, refuses
a set that is not trace-preserving with completeness compared at
``atol = rtol = 1e-6``, refuses a dimension that is not a positive power of two,
serializes through the ``flagquantum.kraus_channel.v1`` schema, and reaches
simulation through `NoiseModel.add`. Three of the five had no test: the
tolerance was covered only by a channel that misses by a factor of two (so any
tolerance between 1e-6 and 0.5 would have passed), the schema had no round trip
at all, and the dimension rule was untested.

The tolerance assertions are written as a pair that brackets the documented
limit rather than as a single refusal. A completeness error of 1.9e-6 is
accepted and 2.1e-6 is refused, which is what ``atol = rtol = 1e-6`` means for an
identity target and nothing else: an ``atol``-only rule at 1e-6 would refuse the
first, and any tolerance of 1e-5 or wider would accept the second.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.noise import KrausChannel, NoiseModel

pytestmark = pytest.mark.unit

_COMPLEX = torch.complex128


def _scaled_identity(delta: float) -> torch.Tensor:
    """Return the single Kraus operator whose completeness error is ``delta``."""

    scale: torch.Tensor = torch.tensor((1.0 - delta) ** 0.5, dtype=torch.float64)
    eye: torch.Tensor = torch.eye(2, dtype=_COMPLEX)
    return scale * eye


def _pauli_channel(probability: float) -> KrausChannel:
    """Return the four-operator channel ``(1-p) I + p/3 (X + Y + Z)``."""

    paulis = (
        torch.tensor([[0, 1], [1, 0]], dtype=_COMPLEX),
        torch.tensor([[0, -1j], [1j, 0]], dtype=_COMPLEX),
        torch.tensor([[1, 0], [0, -1]], dtype=_COMPLEX),
    )
    operators = ((1.0 - probability) ** 0.5 * torch.eye(2, dtype=_COMPLEX),) + tuple(
        (probability / 3.0) ** 0.5 * pauli for pauli in paulis
    )
    return KrausChannel("pauli", operators)


def test_an_arbitrary_operator_count_is_accepted() -> None:
    """The constructor is not restricted to the two-operator form."""

    channel = _pauli_channel(0.4)

    assert len(channel.kraus) == 4
    assert channel.n_qubits == 1


def test_the_completeness_error_is_compared_at_the_documented_tolerance() -> None:
    """1.9e-6 is accepted and 2.1e-6 is refused, which brackets 2e-6."""

    accepted = KrausChannel("near", (_scaled_identity(1.9e-6),))
    assert accepted.n_qubits == 1

    with pytest.raises(ValueError, match="trace-preserving"):
        KrausChannel("far", (_scaled_identity(2.1e-6),))


def test_the_refusal_reports_the_completeness_error_it_measured() -> None:
    """The message carries the number, so a near miss is diagnosable."""

    with pytest.raises(ValueError, match=r"max completeness error=2\.100e-06"):
        KrausChannel("far", (_scaled_identity(2.1e-6),))


@pytest.mark.parametrize("size", [3, 5, 6])
def test_a_dimension_that_is_not_a_power_of_two_is_refused(size: int) -> None:
    with pytest.raises(ValueError, match="positive power of two"):
        KrausChannel("bad", (torch.eye(size, dtype=_COMPLEX),))


def test_a_two_qubit_dimension_is_accepted() -> None:
    """Four is a power of two, so a two-wire operator set is in scope."""

    channel = KrausChannel("identity", (torch.eye(4, dtype=_COMPLEX),))

    assert channel.n_qubits == 2


def test_a_zero_width_operator_is_refused() -> None:
    """A 0x0 tensor passes the square and finiteness checks, so only the size
    check refuses it; accepting it would give a channel of ``n_qubits == -1``."""

    with pytest.raises(ValueError, match="positive power of two"):
        KrausChannel("zero", (torch.empty((0, 0), dtype=_COMPLEX),))


def test_an_operator_after_the_first_with_the_wrong_shape_is_refused() -> None:
    """The shape rule applies to the whole list, not just its head."""

    with pytest.raises(ValueError, match="same square shape"):
        KrausChannel(
            "mismatch", (torch.eye(2, dtype=_COMPLEX), torch.eye(4, dtype=_COMPLEX))
        )


def test_a_non_finite_operator_after_the_first_is_refused() -> None:
    """A NaN in a later branch would poison every expectation silently."""

    nan = torch.tensor([[float("nan"), 0], [0, float("nan")]], dtype=_COMPLEX)
    with pytest.raises(ValueError, match="finite values"):
        KrausChannel("nonfinite", (torch.eye(2, dtype=_COMPLEX), nan))


def test_the_serialized_schema_round_trips_complex_operators_exactly() -> None:
    """Every operator entry survives, including the imaginary parts."""

    operators = (
        torch.tensor([[1, 0], [0, 1j]], dtype=_COMPLEX) / 2**0.5,
        torch.tensor([[1j, 0], [0, 1]], dtype=_COMPLEX) / 2**0.5,
    )
    channel = KrausChannel(
        "imaginary", operators, (("theta", 0.25), ("label", "amplitude"))
    )

    payload = channel.to_dict()
    restored = KrausChannel.from_dict(payload)

    assert payload["schema"] == "flagquantum.kraus_channel.v1"
    assert restored.name == channel.name
    assert len(restored.kraus) == len(channel.kraus)
    for before, after in zip(channel.kraus, restored.kraus, strict=True):
        assert torch.equal(before, after)


def test_a_payload_from_another_schema_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported Kraus channel schema"):
        KrausChannel.from_dict({"schema": "flagquantum.kraus_channel.v2", "kraus": []})


def test_the_channel_reaches_density_matrix_execution() -> None:
    """The four-operator channel applies as the Pauli channel it describes.

    ``X`` maps ``|0>`` to ``|1>`` before the channel runs, and both the ``X``
    and ``Y`` branches flip it back, so the population of ``|0>`` is ``2p/3``.
    The density-matrix path is complex64, and the measured deviation from the
    analytic value is 1.6e-8, so the tolerance is 1e-6: comfortably above that
    rounding and far below the 0.267 the assertion would have to accept if the
    channel were dropped or applied to the wrong state.
    """

    probability = 0.4
    circuit = fq.Circuit(1).x(0)
    model = NoiseModel().add("x", _pauli_channel(probability))

    density = circuit.noisy_density_matrix(model)

    assert float(density[0, 0, 0].real) == pytest.approx(2 * probability / 3, abs=1e-6)
    assert float(density[0, 1, 1].real) == pytest.approx(
        1 - 2 * probability / 3, abs=1e-6
    )
