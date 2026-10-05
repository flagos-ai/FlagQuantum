# -*- coding: utf-8 -*-
"""Tests for Clifford data regression."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import (
    CDR_ASSUMPTIONS,
    CDR_LIMITATIONS,
    CDR_SNAP_OPCODES,
    CdrResult,
    CliffordFit,
    CliffordTrainingPoint,
    CliffordVariant,
    Hamiltonian,
    HamiltonianTerm,
    clifford_variants,
    run_cdr,
)
from flagquantum.algorithms import cdr as cdr_module
from flagquantum.core import OPERATOR_SCHEMAS
from flagquantum.core.ir import (
    CircuitIR,
    Instruction,
    IRValidationError,
    MeasurementNode,
    ObservableNode,
)
from flagquantum.errors import CapabilityError
from flagquantum.noise import (
    NoiseModel,
    NoiseRule,
    ReadoutError,
    ReadoutRule,
    depolarizing_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.simulation.density_matrix import density_matrix_from_ir
from flagquantum.simulation.stabilizer import require_clifford_program
from flagquantum.simulation.stabilizer.engine import CLIFFORD_GATE_NAMES
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

HALF_PI = math.pi / 2.0
QUARTER_TURN_WORDS = cdr_module._QUARTER_TURN_WORDS
SEPARATION_TOLERANCE = cdr_module._SEPARATION_TOLERANCE
MAXIMUM_TRAINING_POINTS = cdr_module._MAXIMUM_TRAINING_POINTS

Z0 = Hamiltonian([HamiltonianTerm(1.0, {0: "z"})])
Z1 = Hamiltonian([HamiltonianTerm(1.0, {1: "z"})])
ZX = Hamiltonian([HamiltonianTerm(1.0, {0: "z"}), HamiltonianTerm(1.0, {1: "x"})])


def _target(*, first: float = 0.6, second: float | None = None) -> fq.Circuit:
    """Return ``h(0) ry(0, first) [ry(1, second)] cx(0, 1)``."""

    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.ry(0, first)
    if second is not None:
        circuit.ry(1, second)
    circuit.cx(0, 1)
    return circuit


def _noise(probability: float = 0.06) -> NoiseModel:
    return NoiseModel(
        rules=[
            NoiseRule(
                gate_names=("cx",),
                channel=two_qubit_depolarizing_channel(probability),
            ),
            NoiseRule(gate_names=("h",), channel=depolarizing_channel(0.03)),
        ]
    )


def _ideal(circuit, hamiltonian) -> float:
    density = density_matrix_from_ir(
        circuit, bsz=1, device="cpu", dtype=torch.complex128
    )
    return float(torch.as_tensor(hamiltonian.expectation(density)).reshape(-1)[0])


def _lowered_copy(circuit, model: NoiseModel):
    """Return ``circuit`` with ``model``'s channels inserted, without running it."""

    from flagquantum.compiler.noise import lower_noise_model

    return lower_noise_model(circuit.to_ir(), model)


def _phase_aligned_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the largest entry of ``left - phase * right`` over every phase."""

    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    index = int(torch.argmax(flat_right.abs()))
    if float(flat_right[index].abs()) == 0.0:
        return float(flat_left.abs().max())
    phase = flat_left[index] / flat_right[index]
    phase = phase / phase.abs()
    return float((flat_left - phase * flat_right).abs().max())


# --- the quarter-turn ladder -------------------------------------------------


@pytest.mark.parametrize("opcode", sorted(QUARTER_TURN_WORDS))
def test_the_ladder_reproduces_every_rotation_it_claims(opcode: str) -> None:
    """Each word must realize its rotation to the complex128 floor."""

    words = QUARTER_TURN_WORDS[opcode]
    assert len(words) == 4, opcode
    for residue, word in enumerate(words):
        rotated = fq.Circuit(1)
        rotated.gate(opcode, (0,), params={"theta": residue * HALF_PI})
        named = fq.Circuit(1)
        for name in word:
            named.gate(name, (0,))
        difference = _phase_aligned_difference(
            get_unitary(rotated, dtype=torch.complex128),
            get_unitary(named, dtype=torch.complex128),
        )
        assert difference < 1e-14, (opcode, residue, word, difference)


@pytest.mark.parametrize("opcode", ("phase", "u1"))
def test_the_diagonal_ladders_need_no_global_phase(opcode: str) -> None:
    """A diagonal rotation's word is the rotation itself, not only up to a phase."""

    for residue in range(4):
        rotated = fq.Circuit(1)
        rotated.gate(opcode, (0,), params={"theta": residue * HALF_PI})
        named = fq.Circuit(1)
        for name in QUARTER_TURN_WORDS[opcode][residue]:
            named.gate(name, (0,))
        assert torch.allclose(
            get_unitary(rotated, dtype=torch.complex128),
            get_unitary(named, dtype=torch.complex128),
        ), (opcode, residue)


def test_every_snap_opcode_is_a_single_wire_theta_rotation() -> None:
    """The ladder's domain must be exactly the schema's, not a private list."""

    for opcode in CDR_SNAP_OPCODES:
        schema = OPERATOR_SCHEMAS[opcode]
        assert schema.arity == 1, opcode
        assert schema.parameters == ("theta",), opcode
        assert schema.unitary, opcode


def test_a_rotation_already_on_the_grid_emits_no_gate_at_all() -> None:
    """Residue zero is the empty word for every ladder, not an explicit identity.

    The distinction is stated in the capability registry and in the guide -- a
    rotation already on the grid is dropped rather than emitted as an identity --
    and it is the difference between a variant that carries the operation and one
    that does not. An explicit ``i`` would leave every unitary and every
    expectation unchanged, so only this test can tell the two apart.
    """

    assert [words[0] for words in QUARTER_TURN_WORDS.values()] == [()] * len(
        QUARTER_TURN_WORDS
    )
    for opcode in QUARTER_TURN_WORDS:
        circuit = fq.Circuit(1)
        circuit.gate(opcode, (0,), params={"theta": 0.0})
        assert clifford_variants(circuit)[0].circuit.instructions == ()


def test_every_word_names_gates_the_stabilizer_engine_executes() -> None:
    """A ladder entry outside the Clifford vocabulary would make a variant unusable."""

    for opcode, words in QUARTER_TURN_WORDS.items():
        for residue, word in enumerate(words):
            for name in word:
                assert name in CLIFFORD_GATE_NAMES, (opcode, residue, name)


def test_the_snap_opcode_roster_is_sorted_and_declared() -> None:
    assert tuple(sorted(QUARTER_TURN_WORDS)) == CDR_SNAP_OPCODES
    assert CDR_SNAP_OPCODES == ("phase", "rx", "ry", "rz", "u1")


# --- the training set --------------------------------------------------------


def test_the_default_training_set_snaps_every_rotation_to_its_nearest_turn() -> None:
    variants = clifford_variants(_target())
    nearest = variants[0]
    assert nearest.name == "nearest"
    # 0.6 rad is under a quarter turn, so its nearest turn is the identity and the
    # rotation disappears; the h and the cx are untouched.
    assert [instruction.name for instruction in nearest.circuit.instructions] == [
        "h",
        "cx",
    ]


def test_the_default_training_set_names_one_variant_per_rotation() -> None:
    variants = clifford_variants(_target(first=0.6, second=0.9))
    assert [variant.name for variant in variants] == [
        "nearest",
        "ry-at-instruction-1",
        "ry-at-instruction-2",
    ]


def test_each_variant_moves_exactly_one_site_to_its_other_turn() -> None:
    """The whole instruction sequence is the specification, spelled out here.

    ``h(0) ry(0, 0.6) ry(1, 0.9) cx(0, 1)``: 0.6 is under a quarter turn so its
    nearest turn is the identity, and 0.9 is over one so its nearest turn is a
    quarter turn. The three variants therefore place the two sites at residues
    ``(0, 1)``, ``(1, 1)`` and ``(0, 0)``, and the ladder table -- verified
    against the gate matrices above -- says what each residue emits.
    """

    circuit = _target(first=0.6, second=0.9)
    variants = clifford_variants(circuit)
    residues = ((0, 1), (1, 1), (0, 0))
    assert len(variants) == len(residues)
    for variant, placement in zip(variants, residues, strict=True):
        expected = ["h"]
        for opcode, residue in zip(("ry", "ry"), placement, strict=True):
            expected.extend(QUARTER_TURN_WORDS[opcode][residue])
        expected.append("cx")
        assert [
            instruction.name for instruction in variant.circuit.instructions
        ] == expected, variant.name
    # Each further variant moves exactly one site away from the all-nearest one,
    # which is what makes the training circuits differ one site at a time.
    for placement in residues[1:]:
        moved = [
            index
            for index, (near, other) in enumerate(
                zip(residues[0], placement, strict=True)
            )
            if near != other
        ]
        assert len(moved) == 1, placement


def test_every_variant_is_a_clifford_circuit_the_engine_accepts() -> None:
    for variant in clifford_variants(_target(first=0.6, second=0.9)):
        # The construction already called this, so a second call is the claim
        # restated as the engine's own answer rather than as this module's.
        assert isinstance(require_clifford_program(variant.circuit), CircuitIR)


def test_the_angle_shift_is_the_largest_angle_the_variant_moved() -> None:
    """0.6 sits 0.600 from turn 0; 0.9 sits 0.671 from the turn above it.

    So the all-nearest variant moved 0.671, the variant that flips the first site
    moved 0.971, and the one that flips the second moved 0.900. The shift is the
    distance from the target, which is what makes "near-Clifford" a number.
    """

    variants = clifford_variants(_target(first=0.6, second=0.9))
    assert variants[0].angle_shift == pytest.approx(HALF_PI - 0.9)
    assert variants[0].angle_shift == pytest.approx(0.6707963267948965)
    assert variants[1].angle_shift == pytest.approx(HALF_PI - 0.6)
    assert variants[2].angle_shift == pytest.approx(0.9)
    assert variants[1].angle_shift > variants[2].angle_shift > variants[0].angle_shift


def test_the_training_set_does_not_modify_the_target() -> None:
    circuit = _target(first=0.6, second=0.9)
    before = list(circuit.to_qir())
    width = circuit.n_wires
    clifford_variants(circuit)
    assert list(circuit.to_qir()) == before
    assert circuit.n_wires == width


def test_building_the_training_set_is_deterministic() -> None:
    first = clifford_variants(_target(first=0.6, second=0.9))
    second = clifford_variants(_target(first=0.6, second=0.9))
    assert [variant.name for variant in first] == [variant.name for variant in second]
    assert [
        [instruction.name for instruction in variant.circuit.instructions]
        for variant in first
    ] == [
        [instruction.name for instruction in variant.circuit.instructions]
        for variant in second
    ]


def test_a_rotation_exactly_between_two_turns_goes_to_the_lower_one() -> None:
    """A half-turn boundary is equidistant, and the tie rule is stated in the unit.

    ``pi/4`` sits exactly between turn 0 and turn 1. Taking the upper turn there
    would make the nearest turn depend on the direction of a comparison that is
    equal in both, so the rule is fixed as the lower turn and this is the test
    that fixes it: the all-nearest variant drops the rotation entirely, and the
    variant that flips it emits the quarter-turn word for turn 1.
    """

    circuit = fq.Circuit(1)
    circuit.rx(0, math.pi / 4)
    variants = clifford_variants(circuit)
    assert [instruction.name for instruction in variants[0].circuit.instructions] == []
    assert variants[0].angle_shift == pytest.approx(HALF_PI / 2.0)
    assert [instruction.name for instruction in variants[1].circuit.instructions] == [
        "sx"
    ]
    assert variants[1].angle_shift == variants[0].angle_shift


def test_a_rotation_already_on_the_grid_reappears_as_its_named_gate() -> None:
    circuit = fq.Circuit(1)
    circuit.rz(0, 3 * HALF_PI)
    variants = clifford_variants(circuit)
    assert [instruction.name for instruction in variants[0].circuit.instructions] == [
        "sdg"
    ]
    assert variants[0].angle_shift == 0.0


# --- refusals ----------------------------------------------------------------


def test_a_program_with_no_snappable_rotation_is_refused() -> None:
    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    with pytest.raises(ValueError, match="it contains no operation from phase, rx"):
        clifford_variants(circuit)


def test_a_rotation_carrying_its_own_matrix_is_refused() -> None:
    ir = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                "ry",
                (0,),
                params={"theta": 0.6},
                matrix=torch.eye(2, dtype=torch.complex64),
            ),
        ),
    )
    with pytest.raises(ValueError, match="carries its own matrix"):
        clifford_variants(ir)


def test_the_ir_owns_the_refusal_of_an_angle_that_is_not_a_real_number() -> None:
    """The quarter-turn grid never sees such an angle, because the IR refuses it.

    This is the boundary's check, restated here so that a future slice which
    loosens it fails next to the code that relies on it.
    """

    with pytest.raises(IRValidationError, match="must be a finite real number"):
        CircuitIR(
            n_wires=1,
            instructions=(Instruction("ry", (0,), params={"theta": float("inf")}),),
        )
    with pytest.raises(TypeError, match="must be a real number"):
        CircuitIR(
            n_wires=1,
            instructions=(Instruction("ry", (0,), params={"theta": "0.6"}),),
        )


def test_an_unbound_parameter_has_no_nearest_quarter_turn() -> None:
    """A traced angle is the one angle the IR stores without a number."""

    ir = CircuitIR(
        n_wires=1,
        instructions=(Instruction("ry", (0,), params={"theta": fq.Parameter("t")}),),
    )
    with pytest.raises(TypeError, match="not defined on an unbound parameter"):
        clifford_variants(ir)


def test_a_measurement_request_is_refused() -> None:
    ir = CircuitIR(
        n_wires=1,
        instructions=(Instruction("ry", (0,), params={"theta": 0.6}),),
        measurements=(MeasurementNode("sample", (0,)),),
    )
    with pytest.raises(ValueError, match="declares measurement request"):
        clifford_variants(ir)


def test_an_observable_request_is_refused() -> None:
    ir = CircuitIR(
        n_wires=1,
        instructions=(Instruction("ry", (0,), params={"theta": 0.6}),),
        observables=(ObservableNode("z", (0,)),),
    )
    with pytest.raises(ValueError, match="declares observable node"):
        clifford_variants(ir)


def test_the_training_set_ceiling_is_refused() -> None:
    circuit = fq.Circuit(1)
    for _ in range(MAXIMUM_TRAINING_POINTS):
        circuit.ry(0, 0.6)
    with pytest.raises(ValueError, match="evaluates at most"):
        clifford_variants(circuit)


def test_the_training_set_at_the_ceiling_is_accepted() -> None:
    circuit = fq.Circuit(1)
    for _ in range(MAXIMUM_TRAINING_POINTS - 1):
        circuit.ry(0, 0.6)
    variants = clifford_variants(circuit)
    assert len(variants) == MAXIMUM_TRAINING_POINTS


def test_a_model_naming_a_rewritten_rotation_is_refused() -> None:
    model = NoiseModel(
        rules=[
            NoiseRule(gate_names=("ry",), channel=depolarizing_channel(0.02)),
            NoiseRule(gate_names=("cx",), channel=two_qubit_depolarizing_channel(0.06)),
        ]
    )
    with pytest.raises(ValueError, match="attaches noise after ry"):
        run_cdr(_target(), ZX, noise_model=model)


def test_a_model_declaring_a_readout_rule_is_refused() -> None:
    model = NoiseModel(
        rules=[
            NoiseRule(gate_names=("cx",), channel=two_qubit_depolarizing_channel(0.06))
        ],
        readout_rules=[
            ReadoutRule((0,), ReadoutError(((0.9, 0.1), (0.1, 0.9)))),
        ],
    )
    with pytest.raises(ValueError, match="declares a readout rule"):
        run_cdr(_target(), ZX, noise_model=model)


def test_a_model_that_is_not_a_model_is_refused() -> None:
    with pytest.raises(TypeError, match="needs a noise model"):
        run_cdr(_target(), ZX, noise_model={"cx": 0.05})


def test_an_observable_that_is_not_a_hamiltonian_is_refused() -> None:
    with pytest.raises(TypeError, match="needs a Hamiltonian observable"):
        run_cdr(_target(), "zz", noise_model=_noise())


def test_a_supplied_training_circuit_outside_the_clifford_vocabulary_is_refused() -> (
    None
):
    not_clifford = fq.Circuit(3)
    not_clifford.h(0)
    not_clifford.ccx(0, 1, 2)
    with pytest.raises(CapabilityError, match="is not a Clifford gate"):
        run_cdr(
            _target(),
            ZX,
            noise_model=_noise(),
            variants=(_target(), not_clifford),
        )


def test_an_empty_training_set_is_refused() -> None:
    with pytest.raises(ValueError, match="training set is empty"):
        run_cdr(_target(), ZX, noise_model=_noise(), variants=())


def test_a_training_set_that_is_not_a_sequence_is_refused() -> None:
    with pytest.raises(TypeError, match="must be a sequence of circuits"):
        run_cdr(_target(), ZX, noise_model=_noise(), variants="variants")


def test_a_single_point_training_set_is_refused() -> None:
    with pytest.raises(ValueError, match="needs at least 2 training circuits"):
        run_cdr(_target(), ZX, noise_model=_noise(), variants=(_target(),))


def test_an_oversized_supplied_training_set_is_refused() -> None:
    variants = tuple(
        _target(first=0.6 + 0.01 * index)
        for index in range(MAXIMUM_TRAINING_POINTS + 1)
    )
    with pytest.raises(ValueError, match="evaluates at most"):
        run_cdr(_target(), ZX, noise_model=_noise(), variants=variants)


def test_a_training_set_that_does_not_separate_is_refused() -> None:
    """Both variants of ``h(0) ry(1, 0.9) cx(0, 1)`` leave ``Z0`` at zero.

    The rotation is on wire 1 and the observable acts on wire 0, so the noise the
    model applies cannot move the measured value and no slope is determined. The
    refusal says so instead of returning a slope fitted to rounding.
    """

    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.ry(1, 0.9)
    circuit.cx(0, 1)
    with pytest.raises(ValueError, match="do not separate under the noise model"):
        run_cdr(circuit, Z0, noise_model=_noise(), dtype=torch.complex128)


# --- the fit -----------------------------------------------------------------


def _points(*pairs: tuple[float, float]) -> tuple[CliffordTrainingPoint, ...]:
    return tuple(
        CliffordTrainingPoint(name=f"p{index}", ideal=ideal, noisy=noisy)
        for index, (ideal, noisy) in enumerate(pairs)
    )


def test_the_fit_is_the_least_squares_line() -> None:
    points = _points((1.0, 0.9), (2.0, 1.9), (4.0, 3.9))
    fit = cdr_module._fit(points, spread=3.0)
    noisy = torch.tensor([0.9, 1.9, 3.9], dtype=torch.float64)
    ideal = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    expected_slope = float(
        ((noisy - noisy.mean()) * (ideal - ideal.mean())).sum()
        / ((noisy - noisy.mean()) ** 2).sum()
    )
    assert fit.slope == pytest.approx(expected_slope)
    assert fit.intercept == pytest.approx(
        float(ideal.mean()) - expected_slope * float(noisy.mean())
    )
    assert fit.degrees_of_freedom == 1
    assert fit.max_residual is not None and fit.max_residual < 1e-15


def test_a_two_point_fit_reports_its_residual_as_absent() -> None:
    fit = cdr_module._fit(_points((1.0, 0.9), (0.0, 0.0)), spread=0.9)
    assert fit.degrees_of_freedom == 0
    assert fit.max_residual is None


def test_a_fit_over_coincident_points_reports_a_residual_at_the_floor() -> None:
    fit = cdr_module._fit(_points((1.0, 0.9), (0.0, 0.0), (1.0, 0.9)), spread=0.9)
    assert fit.degrees_of_freedom == 1
    assert fit.max_residual is not None and fit.max_residual < 1e-15


def test_predict_is_the_fitted_line() -> None:
    fit = cdr_module._fit(_points((1.0, 0.9), (2.0, 1.9), (4.0, 3.9)), spread=3.0)
    assert fit.predict(2.4) == pytest.approx(fit.slope * 2.4 + fit.intercept)


def test_the_fit_refuses_fewer_than_two_points() -> None:
    with pytest.raises(ValueError, match="an affine fit needs at least 2"):
        CliffordFit(
            slope=1.0,
            intercept=0.0,
            points=_points((1.0, 0.9)),
            noisy_spread=0.0,
            max_residual=0.0,
            degrees_of_freedom=0,
        )


def test_the_fit_refuses_a_residual_where_it_has_no_degree_of_freedom() -> None:
    with pytest.raises(ValueError, match="must not report one"):
        CliffordFit(
            slope=1.0,
            intercept=0.0,
            points=_points((1.0, 0.9), (0.0, 0.0)),
            noisy_spread=0.9,
            max_residual=0.0,
            degrees_of_freedom=0,
        )


def test_the_fit_refuses_a_degree_count_that_is_not_the_point_count_less_two() -> None:
    with pytest.raises(ValueError, match="degree of freedom per point"):
        CliffordFit(
            slope=1.0,
            intercept=0.0,
            points=_points((1.0, 0.9), (0.0, 0.0), (2.0, 1.8)),
            noisy_spread=1.8,
            max_residual=0.0,
            degrees_of_freedom=5,
        )


def test_the_fit_refuses_a_non_finite_coefficient() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        CliffordFit(
            slope=float("nan"),
            intercept=0.0,
            points=_points((1.0, 0.9), (0.0, 0.0)),
            noisy_spread=0.9,
            max_residual=None,
            degrees_of_freedom=0,
        )


def test_a_training_point_refuses_a_non_finite_expectation() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        CliffordTrainingPoint(name="p", ideal=float("inf"), noisy=0.0)


def test_a_training_point_needs_a_name() -> None:
    with pytest.raises(ValueError, match="must be named"):
        CliffordTrainingPoint(name="", ideal=0.0, noisy=0.0)


def test_a_variant_needs_a_name_and_a_circuit() -> None:
    with pytest.raises(ValueError, match="must be named"):
        CliffordVariant(name="", circuit=_target().to_ir(), angle_shift=0.0)
    with pytest.raises(TypeError, match="needs the IR"):
        CliffordVariant(name="v", circuit="ir", angle_shift=0.0)


# --- the end-to-end mitigation ----------------------------------------------


def test_the_result_carries_the_target_value_the_estimate_and_the_fit() -> None:
    circuit = _target(first=0.6, second=0.9)
    result = run_cdr(circuit, ZX, noise_model=_noise(), dtype=torch.complex128)
    assert isinstance(result, CdrResult)
    assert result.estimate == pytest.approx(result.fit.predict(result.unmitigated))
    assert result.correction == pytest.approx(result.estimate - result.unmitigated)
    assert result.variants == tuple(point.name for point in result.fit.points)
    assert result.assumptions == CDR_ASSUMPTIONS
    assert result.limitations == CDR_LIMITATIONS


def test_the_mitigation_moves_the_expectation_toward_the_noiseless_value() -> None:
    """The measured claim: with three separating points, the fit corrects the value.

    These are measurements, not tolerances chosen to fit. The target is
    ``h(0) ry(0, 0.7) ry(1, 0.4) cx(0, 1)`` observed as ``Z0 + X1`` under
    two-qubit depolarizing noise at 0.06 on ``cx`` and one-qubit depolarizing
    noise at 0.03 on ``h``, in complex128.
    """

    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.ry(0, 0.7)
    circuit.ry(1, 0.4)
    circuit.cx(0, 1)
    result = run_cdr(circuit, ZX, noise_model=_noise(), dtype=torch.complex128)
    ideal = _ideal(circuit, ZX)
    assert ideal == pytest.approx(-0.254799345, abs=1e-8)
    assert result.unmitigated == pytest.approx(-0.214372680, abs=1e-8)
    assert abs(result.unmitigated - ideal) == pytest.approx(4.043e-02, rel=1e-2)
    assert abs(result.estimate - ideal) == pytest.approx(7.523e-03, rel=1e-2)
    assert abs(result.estimate - ideal) < abs(result.unmitigated - ideal)
    # Three distinct noisy values, so the fit has one degree of freedom and its
    # residual is a real diagnostic rather than an identity.
    noisy = [point.noisy for point in result.fit.points]
    assert len({round(value, 9) for value in noisy}) == 3
    assert result.fit.degrees_of_freedom == 1


def test_the_residual_exposes_an_affine_premise_that_does_not_hold() -> None:
    """A non-zero residual is the only diagnostic of a violated premise.

    The same target and model as the test above leave 1.4e-2 of residual, while a
    training set whose three points fall exactly on a line leaves none. Reporting
    both is what separates a fit that was checked from one that was assumed.
    """

    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.ry(0, 0.7)
    circuit.ry(1, 0.4)
    circuit.cx(0, 1)
    checked = run_cdr(circuit, ZX, noise_model=_noise(), dtype=torch.complex128)
    assert checked.fit.max_residual == pytest.approx(1.360e-02, rel=1e-2)

    exact = cdr_module._fit(_points((1.0, 0.9), (2.0, 1.9), (4.0, 3.9)), spread=3.0)
    assert exact.max_residual is not None and exact.max_residual < 1e-15
    assert checked.fit.max_residual > 1e-3


def test_the_slope_is_the_reciprocal_of_the_shrinkage_the_model_applied() -> None:
    """One noise source scales one measured value, so the slope has a meaning.

    The default training set of ``h(0) ry(0, 0.6) cx(0, 1)`` observed as ``Z0``
    reaches its two points at ideal values 0 and -1; the model shrinks the second
    to -0.9467, so the fitted slope is 1 / 0.9467 and the intercept is zero. A
    slope of one would mean the fit found no relation at all.
    """

    circuit = _target(first=0.6)
    model = NoiseModel(
        rules=[
            NoiseRule(gate_names=("cx",), channel=two_qubit_depolarizing_channel(0.05))
        ]
    )
    result = run_cdr(circuit, Z0, noise_model=model, dtype=torch.complex128)
    assert [round(point.ideal, 9) for point in result.fit.points] == [0.0, -1.0]
    assert result.fit.points[1].noisy == pytest.approx(-0.946666620, abs=1e-8)
    assert result.fit.slope == pytest.approx(1 / 0.946666620, rel=1e-8)
    assert abs(result.fit.intercept) < 1e-15
    ideal = _ideal(circuit, Z0)
    assert result.unmitigated == pytest.approx(-0.534528182, abs=1e-8)
    assert abs(result.unmitigated - ideal) == pytest.approx(3.011e-02, rel=1e-2)
    assert abs(result.estimate - ideal) < 1e-14


def test_supplying_variants_takes_over_the_training_set_and_the_noise_claim() -> None:
    """The refusal on a rewritten rotation is the default set's, not the unit's."""

    model = NoiseModel(
        rules=[
            NoiseRule(gate_names=("ry",), channel=depolarizing_channel(0.02)),
            NoiseRule(gate_names=("cx",), channel=two_qubit_depolarizing_channel(0.06)),
        ]
    )
    circuit = _target(first=0.6, second=0.9)
    # The default set is refused first, so the caller knows what they are taking on.
    with pytest.raises(ValueError, match="attaches noise after ry"):
        run_cdr(circuit, ZX, noise_model=model)
    supplied = clifford_variants(circuit)
    result = run_cdr(
        circuit,
        ZX,
        noise_model=model,
        variants=tuple(variant.circuit for variant in supplied),
        dtype=torch.complex128,
    )
    assert result.variants == ("variant-0", "variant-1", "variant-2")
    assert result.unmitigated == pytest.approx(
        _ideal(_lowered_copy(circuit, model), ZX), abs=1e-8
    )
    # Only the default set's names are the rotation sites; a supplied set is named
    # by position, because this unit did not build it and cannot describe it.
    assert not any(name.startswith("ry-") for name in result.variants)


def test_the_result_refuses_a_variant_name_count_that_is_not_the_point_count() -> None:
    fit = cdr_module._fit(_points((1.0, 0.9), (0.0, 0.0)), spread=0.9)
    with pytest.raises(ValueError, match="one training circuit per point"):
        CdrResult(
            estimate=1.0,
            unmitigated=0.9,
            fit=fit,
            variants=("only-one",),
        )


def test_the_result_refuses_repeated_variant_names() -> None:
    fit = cdr_module._fit(_points((1.0, 0.9), (0.0, 0.0), (2.0, 1.8)), spread=1.8)
    with pytest.raises(ValueError, match="named distinctly"):
        CdrResult(
            estimate=1.0,
            unmitigated=0.9,
            fit=fit,
            variants=("same", "same", "same"),
        )


def test_the_result_refuses_to_carry_no_assumptions_or_limitations() -> None:
    fit = cdr_module._fit(_points((1.0, 0.9), (0.0, 0.0)), spread=0.9)
    with pytest.raises(ValueError, match="must carry the assumptions"):
        CdrResult(
            estimate=1.0, unmitigated=0.9, fit=fit, variants=("a", "b"), assumptions=()
        )
    with pytest.raises(ValueError, match="must carry its limitations"):
        CdrResult(
            estimate=1.0, unmitigated=0.9, fit=fit, variants=("a", "b"), limitations=()
        )


def test_the_assumptions_and_limitations_name_what_the_unit_does_not_do() -> None:
    text = " ".join(CDR_ASSUMPTIONS + CDR_LIMITATIONS)
    assert "same noise" in text
    assert "affine" in text
    assert "no confidence interval" in text
    # Readout-error mitigation now exists as its own unit, so the sentence this
    # asserts is that it is separate rather than that it is missing; asserting the
    # old wording would pin the unit's removal instead of its boundary.
    assert "Readout-error mitigation is a separate unit beside this one" in text
    assert "circuit folding" in text
    assert CDR_LIMITATIONS and all(statement for statement in CDR_LIMITATIONS)


def test_a_run_is_reproducible() -> None:
    circuit = _target(first=0.6, second=0.9)
    first = run_cdr(circuit, ZX, noise_model=_noise(), dtype=torch.complex128)
    second = run_cdr(circuit, ZX, noise_model=_noise(), dtype=torch.complex128)
    assert first.estimate == second.estimate
    assert [point.noisy for point in first.fit.points] == [
        point.noisy for point in second.fit.points
    ]


def test_the_separation_floor_is_the_declared_tolerance() -> None:
    """The refusal is a declared floor, not a value reached by tuning."""

    assert SEPARATION_TOLERANCE == 1e-9
    assert SEPARATION_TOLERANCE < 1e-7


# --- the surface -------------------------------------------------------------


def test_the_module_exports_exactly_the_names_it_declares() -> None:
    assert sorted(cdr_module.__all__) == [
        "CDR_ASSUMPTIONS",
        "CDR_LIMITATIONS",
        "CDR_SNAP_OPCODES",
        "CdrResult",
        "CliffordFit",
        "CliffordTrainingPoint",
        "CliffordVariant",
        "clifford_variants",
        "run_cdr",
    ]
    for name in cdr_module.__all__:
        assert hasattr(cdr_module, name), name


def test_the_package_reexports_the_unit() -> None:
    import flagquantum.algorithms as algorithms

    assert algorithms.run_cdr is run_cdr
    assert algorithms.clifford_variants is clifford_variants
    assert algorithms.CdrResult is CdrResult
    assert algorithms.cdr is cdr_module
    for name in cdr_module.__all__:
        assert name in algorithms.__all__, name
