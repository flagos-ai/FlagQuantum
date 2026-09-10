import pytest

from flagquantum.qec import PauliFrame, SyndromeRound

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "bits", [(0.2, 1.8), (0.0, 1.0), ("0", "1"), (-1, 1), (0, 2), ()]
)
def test_syndrome_rejects_nonbinary_input(bits):
    with pytest.raises(ValueError, match="binary integers"):
        SyndromeRound(0, bits)


def test_boolean_measurements_are_normalized_to_integer_bits():
    record = SyndromeRound(0, (False, True))
    assert record.bits == (0, 1)
    assert all(type(bit) is int for bit in record.bits)


def test_pauli_frame_does_not_truncate_readout_values():
    with pytest.raises(ValueError, match="binary integers"):
        PauliFrame((0,)).apply((0.2, 1, 0))
