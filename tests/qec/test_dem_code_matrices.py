"""The code-record bridge and the matrix route into a detector error model.

A model can be read off two descriptions of an experiment. The circuit route
forces one mechanism at a time through a lowered program and reads the flip set
off the circuit's layouts; the matrix route reads a Calderbank-Shor-Steane code's
generator matrices and derives the same signatures combinatorially. The tests
here pin the second route: what a code record contributes to it, which rows and
columns the matrices are indexed by, where each of the two fault families lands,
what the detector geometry is, and which inputs are refused.

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
    CssCodeMatrices,
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    Pauli,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    SteaneCode,
    ancilla_bands,
    css_code_matrices,
)

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Code:
    """A code record, so the bridge is exercised on layouts the codes miss.

    The codes the package ships declare their data wires from zero. A record
    written here reaches a data-wire offset, an X-type-only code, and a code
    whose logical observable is neither pure type, and it satisfies the
    ``StabilizerCode`` protocol by declaring the same properties.
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
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

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


def _repetition_matrices() -> CssCodeMatrices:
    return CssCodeMatrices(
        hz=torch.tensor([[1, 1, 0], [0, 1, 1]]),
        lz=torch.tensor([[1, 1, 1]]),
    )


# --- the code-record bridge ------------------------------------------------


def test_css_code_matrices_read_the_z_half_of_a_repetition_code() -> None:
    """Check ``c`` of the distance-3 repetition code measures ``Z_c Z_{c+1}``.

    The code declares no X-type check and no X-type logical operator, so those
    two blocks are empty rather than absent: the record still has four matrices,
    and the empty ones say the code has none of that type.
    """

    matrices = css_code_matrices(RepetitionCode(distance=3))

    assert matrices.hz.tolist() == [[1, 1, 0], [0, 1, 1]]
    assert matrices.lz.tolist() == [[1, 1, 1]]
    assert matrices.hx.shape == (0, 3)
    assert matrices.lx.shape == (0, 3)
    assert (matrices.num_z_checks, matrices.num_x_checks) == (2, 0)
    assert matrices.num_observables == 1


def test_css_code_matrices_read_both_halves_of_the_steane_code() -> None:
    """A CSS code's Z-type and X-type checks are the same supports by definition."""

    matrices = css_code_matrices(SteaneCode())

    # The Steane checks come in pairs with one support, which is what makes the
    # code Calderbank-Shor-Steane, so the two check blocks are equal.
    assert matrices.hz.tolist() == matrices.hx.tolist()
    assert matrices.hz.shape == (3, 7)
    assert matrices.lz.tolist() == [[1, 1, 1, 0, 0, 0, 0]]
    assert matrices.lx.tolist() == [[1, 1, 1, 0, 0, 0, 0]]
    assert (matrices.num_checks, matrices.num_observables) == (6, 2)


def test_css_code_matrices_keep_the_declaration_order_of_each_type() -> None:
    """The Steane checks are declared Z first, so ``hz`` is the first three."""

    code = SteaneCode()

    matrices = css_code_matrices(code)

    z_checks = [check for check in code.checks if check.stabilizer.z_wires]
    x_checks = [check for check in code.checks if check.stabilizer.x_wires]
    assert matrices.hz.tolist() == [
        [1 if wire in check.stabilizer.support else 0 for wire in code.data_wires]
        for check in z_checks
    ]
    assert matrices.hx.tolist() == [
        [1 if wire in check.stabilizer.support else 0 for wire in code.data_wires]
        for check in x_checks
    ]


def test_css_code_matrices_index_columns_by_the_declared_data_wires() -> None:
    """A code may put its data qubits on any wires, so a column is a wire."""

    code = _Code(
        data=(10, 11, 12),
        ancillas=(20,),
        checks=(_z_check(0, 20, (10, 12)),),
        observables=(Pauli(z_wires=(10,)),),
    )

    matrices = css_code_matrices(code)

    # Wire 11 is outside the check and outside the logical operator, so column 1
    # is the wire the code did not use rather than a wire it silently shifted.
    assert matrices.hz.tolist() == [[1, 0, 1]]
    assert matrices.lz.tolist() == [[1, 0, 0]]


def test_css_code_matrices_read_an_x_only_code() -> None:
    """A code with X-type checks alone is an X-check block and an empty Z one."""

    code = _Code(
        data=(0, 1),
        ancillas=(2,),
        checks=(_x_check(0, 2, (0, 1)),),
        observables=(Pauli(x_wires=(1,)),),
    )

    matrices = css_code_matrices(code)

    assert matrices.hz.shape == (0, 2)
    assert matrices.hx.tolist() == [[1, 1]]
    assert matrices.lz.shape == (0, 2)
    assert matrices.lx.tolist() == [[0, 1]]
    # One check of the X type is still one detector per round, so a model reads
    # it rather than refusing the code for having no Z-type check.
    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(phase_flip=0.1), num_rounds=1
    )
    assert model.num_detectors == 1
    assert _signatures(model) == [(0.1, (0,)), (0.1, (0,))]


def test_css_code_matrices_measure_the_surface_patch_checks() -> None:
    """The rotated patch contributes one Z row per Z-type check, in code order."""

    code = RotatedSurfaceCode(distance=3)

    matrices = css_code_matrices(code)

    z_checks = [check for check in code.checks if check.stabilizer.z_wires]
    assert matrices.hz.shape == (len(z_checks), code.num_data_qubits)
    expected = [
        [1 if wire in check.stabilizer.support else 0 for wire in code.data_wires]
        for check in z_checks
    ]
    assert matrices.hz.tolist() == expected
    # The patch declares Z memory only, and its logical observable is Z on the
    # first ``distance`` data wires rather than on every wire of the patch.
    observable = code.logical_observables[0]
    assert matrices.lz.tolist() == [
        [1 if wire in observable.z_wires else 0 for wire in code.data_wires]
    ]
    assert matrices.lx.shape == (0, code.num_data_qubits)


def test_css_code_matrices_refuse_a_mixed_type_logical_operator() -> None:
    """An operator that is neither pure X nor pure Z is not a CSS logical."""

    code = _Code(
        data=(0, 1, 2),
        ancillas=(3,),
        checks=(_z_check(0, 3, (0, 1)),),
        observables=(Pauli(x_wires=(0,), z_wires=(1,)),),
    )

    with pytest.raises(ValueError, match="not a CSS logical operator"):
        css_code_matrices(code)


def test_css_code_matrices_refuse_a_check_outside_the_data_wires() -> None:
    """A support wire the code never declares is a broken record, not a column."""

    code = _Code(
        data=(0, 1),
        ancillas=(2,),
        checks=(_z_check(0, 2, (0, 9)),),
        observables=(),
    )

    with pytest.raises(ValueError, match="does not declare as a data wire"):
        css_code_matrices(code)


def test_css_code_matrices_report_the_offending_logical_observable() -> None:
    """The refusal names the observable, so a caller knows which row is wrong."""

    code = _Code(
        data=(0, 1),
        ancillas=(2,),
        checks=(_z_check(0, 2, (0, 1)),),
        observables=(Pauli(z_wires=(0,)), Pauli(z_wires=(0, 5))),
    )

    with pytest.raises(ValueError, match="logical observable 1"):
        css_code_matrices(code)


def test_css_code_matrices_reject_a_non_code() -> None:
    with pytest.raises(TypeError, match="must be a StabilizerCode"):
        css_code_matrices("repetition")  # type: ignore[arg-type]


# --- the matrix record -----------------------------------------------------


def test_css_code_matrix_record_refuses_blocks_about_different_qubits() -> None:
    """Every non-empty block is indexed by the same data qubits."""

    with pytest.raises(ValueError, match="indexed by the same data qubits"):
        CssCodeMatrices(hz=torch.tensor([[1, 1, 0]]), hx=torch.tensor([[1, 1]]))


def test_css_code_matrix_record_fills_a_missing_block() -> None:
    """An omitted block is empty, and it is empty at the width the others state."""

    matrices = CssCodeMatrices(hz=torch.tensor([[1, 1, 0], [0, 1, 1]]))

    assert matrices.hx.shape == (0, 3)
    assert matrices.lz.shape == (0, 3)
    assert matrices.lx.shape == (0, 3)
    assert matrices.num_qubits == 3


def test_css_code_matrix_record_refuses_a_block_with_rows_but_no_column() -> None:
    """A check over no data qubit is malformed rather than trivially empty."""

    with pytest.raises(ValueError, match="row\\(s\\) but no column"):
        CssCodeMatrices(hz=torch.zeros((2, 0), dtype=torch.int64))


def test_css_code_matrix_record_refuses_a_block_that_is_not_binary() -> None:
    """A rate matrix is not a support, and rounding it would pick the checks."""

    with pytest.raises(TypeError, match="cannot carry a torch.float32 element"):
        CssCodeMatrices(hz=torch.tensor([[1.0, 1.0]]))
    with pytest.raises(ValueError, match="only zeros and ones"):
        CssCodeMatrices(hz=torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        CssCodeMatrices(hz=torch.tensor([1, 1]))
    with pytest.raises(TypeError, match="hz must be a torch.Tensor"):
        CssCodeMatrices(hz=[[1, 1]])  # type: ignore[arg-type]
    # The same refusal reaches the blocks a caller states by keyword.
    with pytest.raises(TypeError, match="lx must be a torch.Tensor"):
        CssCodeMatrices(hz=torch.tensor([[1, 1]]), lx="logical")  # type: ignore[arg-type]


def test_css_code_matrix_record_accepts_a_matrix_of_booleans() -> None:
    """A support is a support whatever integer type a caller reached for."""

    matrices = CssCodeMatrices(
        hz=torch.tensor([[True, True, False], [False, True, True]])
    )

    model = DetectorErrorModel.from_code_matrices(matrices, noise=_noise(data_flip=0.1))

    assert model.num_detectors == 2
    assert model.num_errors == 3


# --- the matrix route ------------------------------------------------------


def test_from_code_matrices_declares_the_code_capacity_geometry() -> None:
    """The detector count is one band per round and never a terminal readout."""

    model = DetectorErrorModel.from_code_matrices(
        _repetition_matrices(), noise=_noise(data_flip=0.05), num_rounds=4
    )

    assert model.num_detectors == 4 * 2
    assert model.num_observables == 1


def test_from_code_matrices_pairs_a_fault_with_the_band_after_it() -> None:
    """A data fault in round ``r`` reaches band ``r`` and band ``r + 1``."""

    model = DetectorErrorModel.from_code_matrices(
        _repetition_matrices(), noise=_noise(data_flip=0.1), num_rounds=2
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

    model = DetectorErrorModel.from_code_matrices(
        CssCodeMatrices(hz=torch.tensor([[1, 1, 0], [0, 1, 1]])),
        noise=_noise(data_flip=0.1),
        num_rounds=1,
    )

    assert model.num_detectors == 2
    assert _signatures(model) == [
        (0.1, (0,)),
        (0.1, (0, 1)),
        (0.1, (1,)),
    ]


def test_from_code_matrices_put_the_x_detectors_after_the_z_detectors() -> None:
    """One band is the Z-type checks first and then the X-type checks."""

    # Three Z-type checks and two X-type checks, so the X block sits at offsets 3
    # and 4 inside every band of five and a Z fault reaches it instead of the
    # three Z rows.
    matrices = CssCodeMatrices(
        hz=torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        hx=torch.tensor([[1, 0, 0], [0, 1, 1]]),
    )

    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(phase_flip=0.1), num_rounds=2
    )

    assert model.num_detectors == 10
    # Wires 1 and 2 share an X-type check, so their faults are one mechanism in
    # each band and the merged rate is the odd-parity aggregate of the two.
    assert sorted(_signatures(model)) == [
        (0.1, (3, 8)),
        (0.1, (8,)),
        (0.1 + 0.1 - 2 * 0.01, (4, 9)),
        (0.1 + 0.1 - 2 * 0.01, (9,)),
    ]
    # A Z fault cannot reach a logical operator this code does not declare, and
    # the code declares none of the X type, so every mechanism is detector-only.
    assert {tuple(error.observables) for error in model.errors} == {()}


def test_from_code_matrices_reads_a_y_fault_as_both_families_at_once() -> None:
    """A Y fault is an X fault and a Z fault, so it reaches both detector blocks."""

    matrices = CssCodeMatrices(
        hz=torch.tensor([[1, 1, 0], [0, 1, 1]]),
        hx=torch.tensor([[1, 0, 1]]),
        lz=torch.tensor([[1, 1, 1]]),
        lx=torch.tensor([[0, 1, 0]]),
    )

    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(both_flip=0.1), num_rounds=1
    )

    assert model.num_detectors == 3
    assert model.num_observables == 2
    assert sorted(_signatures(model)) == [
        (0.1, (0, 1)),
        (0.1, (0, 2)),
        (0.1, (1, 2)),
    ]
    # The observable rows are the two logical operators, in the order the record
    # states them: the Z-type one read by an X fault and the X-type one read by
    # the same fault's Z half. Only the wire both of them hold reaches both.
    assert sorted(
        (tuple(sorted(error.detectors)), tuple(sorted(error.observables)))
        for error in model.errors
    ) == [
        ((0, 1), (0, 1)),
        ((0, 2), (0,)),
        ((1, 2), (0,)),
    ]


def test_from_code_matrices_index_the_x_logicals_after_the_z_logicals() -> None:
    """Observable row ``kz`` is the first X-type logical operator."""

    matrices = CssCodeMatrices(
        hz=torch.tensor([[1, 1]]),
        hx=torch.tensor([[1, 1]]),
        lz=torch.tensor([[1, 0]]),
        lx=torch.tensor([[0, 1]]),
    )

    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(data_flip=0.1, phase_flip=0.2), num_rounds=1
    )

    assert model.num_observables == 2
    assert sorted(
        (error.probability, error.detectors, error.observables)
        for error in model.errors
    ) == [
        (0.1, (0,), ()),
        (0.1, (0,), (0,)),
        (0.2, (1,), ()),
        (0.2, (1,), (1,)),
    ]


def test_from_code_matrices_let_a_measurement_fault_flip_no_observable() -> None:
    """A measurement fault is a syndrome error, so no logical operator sees it."""

    matrices = CssCodeMatrices(
        hz=torch.tensor([[1, 1]]),
        hx=torch.tensor([[0, 1]]),
        lz=torch.tensor([[1, 1]]),
    )

    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(measurement_flip=0.1), num_rounds=2
    )

    # One measurement fault per check per round, each on its own check's row in
    # its own band and in the band after it, and never on an observable row. The
    # final round has no band after it, so its two faults stand alone.
    assert model.num_detectors == 4
    assert _signatures(model) == [
        (0.1, (0, 2)),
        (0.1, (1, 3)),
        (0.1, (2,)),
        (0.1, (3,)),
    ]
    assert {tuple(error.observables) for error in model.errors} == {()}


def test_from_code_matrices_omit_a_qubit_outside_every_check() -> None:
    """A fault that flips no detector and no observable is not a mechanism."""

    # Wire 2 is in no check and in no logical operator, so no rate reaches it and
    # it contributes nothing; wires 0 and 1 both flip the one detector and are
    # told apart by the observable.
    matrices = CssCodeMatrices(
        hz=torch.tensor([[1, 1, 0]]), lz=torch.tensor([[1, 0, 0]])
    )

    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(data_flip=0.1), num_rounds=1
    )

    assert model.num_observables == 1
    assert model.num_errors == 2
    assert all(error.detectors or error.observables for error in model.errors)
    assert sorted(_signatures(model)) == [(0.1, (0,)), (0.1, (0,))]


def test_from_code_matrices_merge_a_shared_signature_by_independent_parity() -> None:
    """Two faults a decoder cannot tell apart are one mechanism, not two."""

    # Both data qubits flip the same single check, and a measurement fault on
    # that check flips it too. The data mechanisms carry the observable and the
    # measurement mechanism does not, so the shared detector signature is two
    # mechanisms, each an independent-parity aggregate of the rates behind it.
    matrices = CssCodeMatrices(hz=torch.tensor([[1, 1]]), lz=torch.tensor([[1, 1]]))

    model = DetectorErrorModel.from_code_matrices(
        matrices,
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


def test_from_code_matrices_do_not_conflate_the_two_fault_families() -> None:
    """An X fault and a Z fault on one qubit are two mechanisms, not one rate."""

    matrices = CssCodeMatrices(
        hz=torch.tensor([[1, 1]]),
        hx=torch.tensor([[1, 1]]),
        lz=torch.tensor([[1, 0]]),
        lx=torch.tensor([[0, 1]]),
    )

    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(data_flip=0.02, phase_flip=0.03), num_rounds=1
    )

    # Both families reach the same pair of detectors here, one on each detector,
    # and they stay apart because each carries its own observable row. A model
    # that summed them into one rate would lose which logical operator moved.
    assert sorted(
        (error.probability, error.detectors, error.observables)
        for error in model.errors
    ) == [
        (0.02, (0,), ()),
        (0.02, (0,), (0,)),
        (0.03, (1,), ()),
        (0.03, (1,), (1,)),
    ]


def test_from_code_matrices_round_trips_through_dem_text() -> None:
    """A model read from matrices states itself in DEM text like any other."""

    model = DetectorErrorModel.from_code_matrices(
        _repetition_matrices(),
        noise=_noise(data_flip=0.1, measurement_flip=0.05),
        num_rounds=2,
    )

    assert DetectorErrorModel.from_stim_text(model.to_stim_text()) == model


def test_from_code_matrices_round_trip_an_x_detector_block_through_dem_text() -> None:
    """The extended shape survives the text route like the Z-only shape does."""

    model = DetectorErrorModel.from_code_matrices(
        css_code_matrices(SteaneCode()),
        noise=_noise(data_flip=0.01, phase_flip=0.02, both_flip=0.03),
        num_rounds=2,
    )

    restored = DetectorErrorModel.from_stim_text(model.to_stim_text())
    assert restored == model
    assert restored.num_detectors == 12
    assert restored.num_observables == 2


def test_from_code_matrices_reach_the_matching_decoder() -> None:
    """One round of a distance-3 repetition code is a graphlike model."""

    model = DetectorErrorModel.from_code_matrices(
        _repetition_matrices(),
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


def test_from_code_matrices_refuse_a_hyperedge_before_decoding() -> None:
    """Two rounds of the repetition code are not a graph, and matching says so."""

    model = DetectorErrorModel.from_code_matrices(
        _repetition_matrices(), noise=_noise(data_flip=0.1), num_rounds=2
    )

    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(model)


def test_from_code_reaches_a_model_in_one_call() -> None:
    """A code record is the shortest route: no circuit and no manual matrices."""

    model = DetectorErrorModel.from_code(
        RepetitionCode(distance=3), noise=_noise(data_flip=0.1), num_rounds=1
    )

    assert model == DetectorErrorModel.from_code_matrices(
        css_code_matrices(RepetitionCode(distance=3)),
        noise=_noise(data_flip=0.1),
        num_rounds=1,
    )


def test_from_code_reads_a_steane_logical_operator_from_both_bases() -> None:
    """Both of the Steane code's logical operators reach a model as observables."""

    model = DetectorErrorModel.from_code(
        SteaneCode(), noise=_noise(data_flip=0.1, phase_flip=0.1), num_rounds=1
    )

    assert model.num_detectors == 6
    assert model.num_observables == 2
    # An X fault reads the Z logical operator and a Z fault reads the X one, and
    # the two supports coincide here, so each fault family reports observable 0
    # or observable 1 respectively. A fault on a wire outside the logical
    # support still moves a detector and reports no observable at all.
    observable_rows = {tuple(sorted(error.observables)) for error in model.errors}
    assert observable_rows == {(), (0,), (1,)}


def test_from_code_passes_the_round_count_through() -> None:
    """The one-call route states the same experiment as the matrix route."""

    code = SteaneCode()

    model = DetectorErrorModel.from_code(
        code, noise=_noise(measurement_flip=0.01), num_rounds=3
    )

    assert model.num_detectors == 3 * 6


def test_from_code_matrices_refuse_a_round_count_below_one() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DetectorErrorModel.from_code_matrices(
            _repetition_matrices(), noise=_noise(data_flip=0.1), num_rounds=0
        )
    with pytest.raises(TypeError, match="num_rounds must be an integer"):
        DetectorErrorModel.from_code_matrices(
            _repetition_matrices(),
            noise=_noise(data_flip=0.1),
            num_rounds=1.5,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="num_rounds must be an integer"):
        DetectorErrorModel.from_code(
            SteaneCode(), noise=_noise(data_flip=0.1), num_rounds=True
        )


def test_from_code_matrices_refuse_a_checkless_matrix() -> None:
    """A model needs a detector, and a matrix set with no check declares none."""

    with pytest.raises(ValueError, match="but no check"):
        DetectorErrorModel.from_code_matrices(
            CssCodeMatrices(hz=torch.zeros((0, 3), dtype=torch.int64)),
            noise=_noise(data_flip=0.1),
        )
    with pytest.raises(ValueError, match="but no check"):
        DetectorErrorModel.from_code_matrices(
            CssCodeMatrices(hz=torch.zeros((0, 3), dtype=torch.int64)),
            noise=_noise(measurement_flip=0.1),
        )


def test_from_code_matrices_reject_a_noise_record_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="noise must be a PhenomenologicalNoise"):
        DetectorErrorModel.from_code_matrices(
            _repetition_matrices(), noise=0.1  # type: ignore[arg-type]
        )


def test_from_code_matrices_reject_a_matrix_record_that_is_not_one() -> None:
    """The keyword is a record now, so a bare matrix is refused by name."""

    with pytest.raises(TypeError, match="matrices must be a CssCodeMatrices"):
        DetectorErrorModel.from_code_matrices(
            torch.tensor([[1, 1]]), noise=_noise(data_flip=0.1)  # type: ignore[arg-type]
        )
