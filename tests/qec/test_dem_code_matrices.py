"""The code-record bridge and the matrix route into a detector error model.

A model can be read off two descriptions of an experiment. The circuit route
forces one mechanism at a time through a lowered program and reads the flip set
off the circuit's layouts; the matrix route reads a parity-check matrix and a
logical-operator matrix and derives the same signatures combinatorially. The
tests here pin the second route: what a code record contributes to it, which
rows and columns the matrices are indexed by, what the detector geometry is, and
which inputs are refused.

The stim comparison lives beside this file, because a model built from matrices
is a claim about an experiment and stim is the independent side that can say the
claim is the one its author meant.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    CodeCheck,
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    Pauli,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    code_matrices,
)

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Code:
    """A code record, so the bridge is exercised on layouts the codes miss.

    The two codes the package ships declare their data wires from zero and carry
    both check types. A record written here reaches a data-wire offset and an
    X-type-only code, and it satisfies the ``StabilizerCode`` protocol by
    declaring the same properties.
    """

    data: tuple[int, ...]
    ancillas: tuple[int, ...]
    checks: tuple[CodeCheck, ...]
    observables: tuple[Pauli, ...]

    @property
    def distance(self) -> int:
        return len(self.data)

    @property
    def num_data_qubits(self) -> int:
        return len(self.data)

    @property
    def num_ancilla_qubits(self) -> int:
        return len(self.ancillas)

    @property
    def data_wires(self) -> tuple[int, ...]:
        return self.data

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return self.ancillas

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return self.observables


def _z_check(index: int, ancilla: int, support: tuple[int, ...]) -> CodeCheck:
    """Return a Z-type check coupling ``support`` into ``ancilla``."""

    return CodeCheck(
        index=index,
        stabilizer=Pauli(z_wires=support),
        ancilla_wire=ancilla,
        cnot_wires=tuple((wire, ancilla) for wire in support),
    )


def _x_check(index: int, ancilla: int, support: tuple[int, ...]) -> CodeCheck:
    """Return an X-type check coupling ``ancilla`` out into ``support``."""

    return CodeCheck(
        index=index,
        stabilizer=Pauli(x_wires=support),
        ancilla_wire=ancilla,
        cnot_wires=tuple((ancilla, wire) for wire in support),
    )


def _noise(**rates: float) -> PhenomenologicalNoise:
    return PhenomenologicalNoise(**rates)


def _signatures(model: DetectorErrorModel) -> list[tuple[float, tuple[int, ...]]]:
    """Return every mechanism as its rate and its flipped detectors."""

    return [
        (error.probability, tuple(sorted(error.detectors))) for error in model.errors
    ]


# --- the code-record bridge ------------------------------------------------


def test_code_matrices_read_the_z_type_support_of_a_repetition_code() -> None:
    """Check ``c`` of the distance-3 repetition code measures ``Z_c Z_{c+1}``."""

    hz, lz = code_matrices(RepetitionCode(distance=3))

    assert hz.tolist() == [[1, 1, 0], [0, 1, 1]]
    assert lz.tolist() == [[1, 1, 1]]


def test_code_matrices_index_columns_by_the_declared_data_wires() -> None:
    """A code may put its data qubits on any wires, so a column is a wire."""

    code = _Code(
        data=(10, 11, 12),
        ancillas=(20,),
        checks=(_z_check(0, 20, (10, 12)),),
        observables=(Pauli(z_wires=(10,)),),
    )

    hz, lz = code_matrices(code)

    # Wire 11 is outside the check and outside the logical operator, so column 1
    # is the wire the code did not use rather than a wire it silently shifted.
    assert hz.tolist() == [[1, 0, 1]]
    assert lz.tolist() == [[1, 0, 0]]


def test_code_matrices_keep_only_the_z_type_rows() -> None:
    """An X-type check and an X-type observable contribute no row at all."""

    code = _Code(
        data=(0, 1),
        ancillas=(2, 3),
        checks=(_z_check(0, 2, (0, 1)), _x_check(1, 3, (0, 1))),
        observables=(Pauli(z_wires=(0,)), Pauli(x_wires=(1,))),
    )

    hz, lz = code_matrices(code)

    # The X-type check is absent from ``hz``. The X-type observable would be an
    # all-zero row, and an all-zero row declares an observable no mechanism ever
    # flips, so it is dropped rather than reported as always zero.
    assert hz.tolist() == [[1, 1]]
    assert lz.tolist() == [[1, 0]]


def test_code_matrices_measure_the_surface_patch_checks() -> None:
    """The rotated patch contributes one row per Z-type check, in code order."""

    code = RotatedSurfaceCode(distance=3)

    hz, lz = code_matrices(code)

    z_checks = [check for check in code.checks if check.stabilizer.z_wires]
    assert hz.shape == (len(z_checks), code.num_data_qubits)
    expected = [
        [1 if wire in check.stabilizer.support else 0 for wire in code.data_wires]
        for check in z_checks
    ]
    assert hz.tolist() == expected
    # The patch declares Z memory only, and its logical observable is Z on the
    # first ``distance`` data wires rather than on every wire of the patch.
    observable = code.logical_observables[0]
    assert lz.tolist() == [
        [1 if wire in observable.z_wires else 0 for wire in code.data_wires]
    ]


def test_code_matrices_refuse_a_code_without_a_z_type_check() -> None:
    """Read as Z memory, an X-only code has no detector for a fault to flip."""

    code = _Code(
        data=(0, 1),
        ancillas=(2,),
        checks=(_x_check(0, 2, (0, 1)),),
        observables=(Pauli(z_wires=(0,)),),
    )

    with pytest.raises(ValueError, match="declares no Z-type check"):
        code_matrices(code)


def test_code_matrices_refuse_a_check_outside_the_data_wires() -> None:
    """A support wire the code never declares is a broken record, not a column."""

    code = _Code(
        data=(0, 1),
        ancillas=(2,),
        checks=(_z_check(0, 2, (0, 9)),),
        observables=(),
    )

    with pytest.raises(ValueError, match="does not declare as a data wire"):
        code_matrices(code)


def test_code_matrices_reject_a_non_code() -> None:
    with pytest.raises(TypeError, match="must be a StabilizerCode"):
        code_matrices("repetition")  # type: ignore[arg-type]


# --- the matrix route ------------------------------------------------------


def test_from_code_matrices_declares_the_code_capacity_geometry() -> None:
    """The detector count is one band per round and never a terminal readout."""

    hz = torch.tensor([[1, 1, 0], [0, 1, 1]])
    lz = torch.tensor([[1, 1, 1]])

    model = DetectorErrorModel.from_code_matrices(
        hz=hz, lz=lz, noise=_noise(data_flip=0.05), num_rounds=4
    )

    assert model.num_detectors == 4 * 2
    assert model.num_observables == 1


def test_from_code_matrices_pairs_a_fault_with_the_band_after_it() -> None:
    """A data fault in round ``r`` reaches band ``r`` and band ``r + 1``."""

    hz = torch.tensor([[1, 1, 0], [0, 1, 1]])
    lz = torch.tensor([[1, 1, 1]])

    model = DetectorErrorModel.from_code_matrices(
        hz=hz, lz=lz, noise=_noise(data_flip=0.1), num_rounds=2
    )

    # Two bands of two checks. Data fault on wire 0 flips check 0 in both bands;
    # wire 2 flips check 1 in both bands; wire 1 flips check 0 and check 1, and
    # in each band it reaches both rows, so its shared signature is one mechanism
    # per band pair.
    assert _signatures(model) == [
        (0.1, (0, 1, 2, 3)),
        (0.1, (0, 2)),
        (0.1, (1, 3)),
        (0.1, (2,)),
        (0.1, (2, 3)),
        (0.1, (3,)),
    ]
    assert {tuple(sorted(error.observables)) for error in model.errors} == {(0,)}


def test_from_code_matrices_lets_the_last_band_stand_alone() -> None:
    """The final round has no band after it, so its faults flip one band only."""

    hz = torch.tensor([[1, 1, 0], [0, 1, 1]])

    model = DetectorErrorModel.from_code_matrices(
        hz=hz, noise=_noise(data_flip=0.1), num_rounds=1
    )

    assert model.num_detectors == 2
    assert _signatures(model) == [
        (0.1, (0,)),
        (0.1, (0, 1)),
        (0.1, (1,)),
    ]


def test_from_code_matrices_omits_a_qubit_outside_every_check() -> None:
    """A fault that flips no detector and no observable is not a mechanism."""

    # Wire 2 is in no check and in no logical operator, so no rate reaches it and
    # it contributes nothing; wires 0 and 1 both flip the one detector and are
    # told apart by the observable.
    model = DetectorErrorModel.from_code_matrices(
        hz=torch.tensor([[1, 1, 0]]),
        lz=torch.tensor([[1, 0, 0]]),
        noise=_noise(data_flip=0.1),
        num_rounds=1,
    )

    assert model.num_observables == 1
    assert model.num_errors == 2
    assert all(error.detectors or error.observables for error in model.errors)
    assert sorted(_signatures(model)) == [(0.1, (0,)), (0.1, (0,))]


def test_from_code_matrices_merges_a_shared_signature_by_independent_parity() -> None:
    """Two faults a decoder cannot tell apart are one mechanism, not two."""

    # Both data qubits flip the same single check, and a measurement fault on
    # that check flips it too. The data mechanisms carry the observable and the
    # measurement mechanism does not, so the shared detector signature is two
    # mechanisms, each an independent-parity aggregate of the rates behind it.
    hz = torch.tensor([[1, 1]])
    lz = torch.tensor([[1, 1]])

    model = DetectorErrorModel.from_code_matrices(
        hz=hz,
        lz=lz,
        noise=_noise(data_flip=0.1, measurement_flip=0.2),
        num_rounds=1,
    )

    assert model.mechanisms_are_unique()
    assert sorted(_signatures(model)) == [(0.1 + 0.1 - 2 * 0.01, (0,)), (0.2, (0,))]
    with_observable = [error for error in model.errors if error.observables == (0,)]
    without = [error for error in model.errors if error.observables == ()]
    assert [error.detectors for error in with_observable] == [(0,)]
    assert [error.detectors for error in without] == [(0,)]
    assert with_observable[0].probability == pytest.approx(0.18)
    assert without[0].probability == pytest.approx(0.2)


def test_from_code_matrices_round_trips_through_dem_text() -> None:
    """A model read from matrices states itself in DEM text like any other."""

    hz = torch.tensor([[1, 1, 0], [0, 1, 1]])
    lz = torch.tensor([[1, 1, 1]])

    model = DetectorErrorModel.from_code_matrices(
        hz=hz, lz=lz, noise=_noise(data_flip=0.1, measurement_flip=0.05), num_rounds=2
    )

    assert DetectorErrorModel.from_stim_text(model.to_stim_text()) == model


def test_from_code_matrices_reaches_the_matching_decoder() -> None:
    """One round of a distance-3 repetition code is a graphlike model."""

    hz, lz = code_matrices(RepetitionCode(distance=3))

    model = DetectorErrorModel.from_code_matrices(
        hz=hz,
        lz=lz,
        noise=_noise(data_flip=0.1, measurement_flip=0.1),
        num_rounds=1,
    )
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)

    assert model.num_detectors == 2
    # A fault on the middle data wire flips both checks and the logical
    # observable; either single check is explained by the boundary.
    assert decoder.decode((0, 1)).observables == (0,)
    assert decoder.decode((0,)).observables == ()
    assert decoder.decode((1,)).observables == ()


def test_from_code_matrices_refuses_a_hyperedge_before_decoding() -> None:
    """Two rounds of the repetition code are not a graph, and matching says so."""

    hz, lz = code_matrices(RepetitionCode(distance=3))

    model = DetectorErrorModel.from_code_matrices(
        hz=hz, lz=lz, noise=_noise(data_flip=0.1), num_rounds=2
    )

    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(model)


def test_from_code_matrices_refuses_a_round_count_below_one() -> None:
    hz = torch.tensor([[1, 1]])

    with pytest.raises(ValueError, match="at least one"):
        DetectorErrorModel.from_code_matrices(
            hz=hz, noise=_noise(data_flip=0.1), num_rounds=0
        )
    with pytest.raises(TypeError, match="num_rounds must be an integer"):
        DetectorErrorModel.from_code_matrices(
            hz=hz, noise=_noise(data_flip=0.1), num_rounds=1.5  # type: ignore[arg-type]
        )


def test_from_code_matrices_refuses_a_matrix_that_is_not_binary() -> None:
    """A rate matrix is not a support, and rounding it would pick the checks."""

    with pytest.raises(TypeError, match="cannot carry a torch.float32 element"):
        DetectorErrorModel.from_code_matrices(
            hz=torch.tensor([[1.0, 1.0]]), noise=_noise(data_flip=0.1)
        )
    with pytest.raises(ValueError, match="only zeros and ones"):
        DetectorErrorModel.from_code_matrices(
            hz=torch.tensor([[1, 2]]), noise=_noise(data_flip=0.1)
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        DetectorErrorModel.from_code_matrices(
            hz=torch.tensor([1, 1]), noise=_noise(data_flip=0.1)
        )
    with pytest.raises(TypeError, match="hz must be a torch.Tensor"):
        DetectorErrorModel.from_code_matrices(
            hz=[[1, 1]], noise=_noise(data_flip=0.1)  # type: ignore[arg-type]
        )


def test_from_code_matrices_refuses_a_matrix_pair_about_different_qubits() -> None:
    with pytest.raises(ValueError, match="a logical operator matrix must be indexed"):
        DetectorErrorModel.from_code_matrices(
            hz=torch.tensor([[1, 1, 0]]),
            lz=torch.tensor([[1, 1]]),
            noise=_noise(data_flip=0.1),
        )


def test_from_code_matrices_refuses_a_checkless_matrix() -> None:
    """A model needs a detector, and a matrix with no check declares none."""

    with pytest.raises(ValueError, match="but no check"):
        DetectorErrorModel.from_code_matrices(
            hz=torch.zeros((0, 3), dtype=torch.int64),
            noise=_noise(data_flip=0.1),
        )
    with pytest.raises(ValueError, match="declares no data qubit"):
        DetectorErrorModel.from_code_matrices(
            hz=torch.zeros((2, 0), dtype=torch.int64),
            noise=_noise(data_flip=0.1),
        )


def test_from_code_matrices_accepts_a_matrix_of_booleans() -> None:
    """A support is a support whatever integer type a caller reached for."""

    model = DetectorErrorModel.from_code_matrices(
        hz=torch.tensor([[True, True, False], [False, True, True]]),
        noise=_noise(data_flip=0.1),
    )

    assert model.num_detectors == 2
    assert model.num_errors == 3
