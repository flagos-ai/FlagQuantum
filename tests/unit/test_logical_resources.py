"""The logical-layer resource report: its reuse, its classes, and its arithmetic.

Three properties are asserted here rather than described, because each is a
reason this module exists at all.

**The gate-level tally is the compiler's, not a second count.**  Every test that
reads a count reads the same program twice -- once through
:func:`flagquantum.compiler.resource_estimation.estimate_resources` and once
through :func:`estimate_logical_resources` -- and asserts the two agree.  The
T-family table is compared against the compiler's own, so a T rule cannot drift
away from the one a caller reads directly.  A test that only asserted a T-count
of two would pass against a second implementation of what a T-depth means, which
is exactly the duplicate this slice exists to avoid.

**The Clifford+T classification is total.**  A logical resource estimate is
defined over Clifford+T programs, so the class of accepted opcodes has to be a
closed set rather than a list that may fall behind a new opcode.  The tests
partition the operator schema: every declared unitary opcode is either inside the
Clifford+T families or in one of the two refused families, every declared channel
is refused, and a circuit applying every admitted opcode is accepted.  A new
opcode added to the schema without a decision here fails the partition test
rather than being silently counted as nothing.

**A Toffoli is not a Clifford gate.**  It is a compound operation whose T-count is
whatever decomposition the caller's synthesis pipeline picked, so counting it as
one Clifford operation would understate the T-count of every logical program that
contains one -- the single number a logical resource estimate exists to report.
The tests assert the refusal by name, and assert that it is the same kind of
refusal a rotation gets: an owned absence rather than a silent guess.

The tolerances are not numerical ones.  Every count in this module is an integer,
so every assertion on a count is exact equality; there is no floating-point
comparison anywhere below.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import tomllib

import flagquantum as fq
import flagquantum.algorithms as algorithms
from flagquantum.algorithms.logical_resources import (
    CLIFFORD_OPCODES,
    CLIFFORD_T_OPCODES,
    DECOMPOSITION_OPCODES,
    LOGICAL_RESOURCE_BASIS,
    PARAMETRIC_OPCODES,
    SURFACE_CODE_MODEL,
    T_FAMILY_OPCODES,
    LogicalResourceReport,
    estimate_logical_resources,
    surface_code_qubits_per_logical,
)
from flagquantum.compiler.resource_estimation import (
    _T_FAMILY_OPCODES,
    estimate_resources,
)
from flagquantum.core.ir import Instruction, MeasurementNode, ensure_circuit_ir
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MATURITY = ROOT / "capability-maturity.toml"
CAPABILITY = "algorithms_logical_resource_estimation"

CHANNEL_OPCODES = frozenset(
    name
    for name, schema in OPERATOR_SCHEMAS.items()
    if schema.semantic_kind == "channel"
)
UNITARY_OPCODES = frozenset(
    name
    for name, schema in OPERATOR_SCHEMAS.items()
    if schema.semantic_kind == "unitary"
)
REFUSED_OPCODES = DECOMPOSITION_OPCODES | PARAMETRIC_OPCODES


def _mixed() -> object:
    """Return a three-wire Clifford+T program with a T at two different depths."""

    return fq.Circuit(3).h(0).cz(0, 1).t(0).cx(0, 2).t(1).x(2).tdg(0)


def _apply(opcode: str) -> object:
    """Apply one declared opcode once through the generic gate path."""

    schema = OPERATOR_SCHEMAS[opcode]
    width = max(schema.arity, 1)
    circuit = fq.Circuit(width)
    return circuit.gate(
        opcode,
        tuple(range(width)),
        **dict.fromkeys(schema.parameters, 0.5),
    )


def _with_measurement(program: object, count: int = 1) -> object:
    """Return the same program carrying ``count`` measurement records."""

    ir = ensure_circuit_ir(program)
    records = tuple(
        MeasurementNode("counts", (wire,), shots=None) for wire in range(count)
    )
    return dataclasses.replace(ir, measurements=records)


def _maturity_contract() -> dict:
    with MATURITY.open("rb") as handle:
        return tomllib.load(handle)


def _parity_rows() -> dict[str, dict]:
    """Return the parity contract's rows keyed by capability id."""

    with (ROOT / "contracts/cudaq-parity-matrix.toml").open("rb") as handle:
        contract = tomllib.load(handle)
    return {
        capability["id"]: capability
        for domain in contract["domains"]
        for capability in domain["capabilities"]
    }


def test_the_tally_is_the_compilers_own_record() -> None:
    """A second count would pass a T-count assertion and fail this one."""

    program = _mixed()
    report = estimate_logical_resources(program, distance=5)
    direct = estimate_resources(program)
    assert report.estimate == direct
    assert report.t_count == direct.t_count
    assert report.t_depth == direct.t_depth
    assert report.estimate.operation_counts == direct.operation_counts
    assert report.estimate.depth == direct.depth


def test_the_clifford_count_completes_the_compilers_operation_counts() -> None:
    """Every operation the compiler counted is in exactly one of the two families."""

    report = estimate_logical_resources(_mixed(), distance=5)
    counted = report.estimate.operation_counts
    assert sum(counted.values()) == report.estimate.n_operations
    assert report.clifford_count + report.t_count == report.n_clifford_t
    assert report.n_clifford_t == report.estimate.n_operations
    assert set(counted) <= CLIFFORD_T_OPCODES
    assert report.t_count == sum(counted.get(name, 0) for name in T_FAMILY_OPCODES)
    assert report.clifford_count == sum(
        counted.get(name, 0) for name in CLIFFORD_OPCODES
    )
    assert report.clifford_count == 4
    assert report.t_count == 3


def test_the_t_family_table_is_the_compilers_own_table() -> None:
    """The classification is restated here; the counts are not, and the table is."""

    assert T_FAMILY_OPCODES == _T_FAMILY_OPCODES


def test_the_families_are_pairwise_disjoint() -> None:
    assert CLIFFORD_OPCODES.isdisjoint(T_FAMILY_OPCODES)
    assert CLIFFORD_T_OPCODES == CLIFFORD_OPCODES | T_FAMILY_OPCODES
    assert DECOMPOSITION_OPCODES.isdisjoint(PARAMETRIC_OPCODES)
    assert DECOMPOSITION_OPCODES.isdisjoint(CLIFFORD_T_OPCODES)
    assert PARAMETRIC_OPCODES.isdisjoint(CLIFFORD_T_OPCODES)
    assert CHANNEL_OPCODES.isdisjoint(REFUSED_OPCODES)


def test_no_compound_or_parametric_opcode_is_counted_as_clifford() -> None:
    """The error this catches is silent, and it flatters the T-count."""

    assert frozenset({"ccx", "cswap"}) == DECOMPOSITION_OPCODES
    assert CLIFFORD_T_OPCODES.isdisjoint({"ccx", "cswap"})
    assert "ccx" not in CLIFFORD_OPCODES
    assert "cswap" not in CLIFFORD_OPCODES


def test_every_declared_opcode_is_classified_or_refused() -> None:
    """The partition, so a new schema opcode cannot be silently uncounted."""

    covered = CLIFFORD_T_OPCODES | REFUSED_OPCODES | CHANNEL_OPCODES
    assert covered == set(OPERATOR_SCHEMAS)
    assert len(covered) == len(OPERATOR_SCHEMAS)
    assert CLIFFORD_T_OPCODES <= UNITARY_OPCODES
    assert REFUSED_OPCODES <= UNITARY_OPCODES
    assert CLIFFORD_T_OPCODES | REFUSED_OPCODES == UNITARY_OPCODES
    assert (
        frozenset(
            {
                "cphase",
                "crx",
                "cry",
                "crz",
                "phase",
                "rx",
                "rxx",
                "ry",
                "ryy",
                "rz",
                "rzz",
                "u1",
                "u2",
                "u3",
            }
        )
        == PARAMETRIC_OPCODES
    )


def test_every_clifford_and_t_opcode_is_accepted_together() -> None:
    """The accepted class is closed under the schema rather than a sample of it."""

    program = fq.Circuit(3)
    for opcode in sorted(CLIFFORD_T_OPCODES):
        schema = OPERATOR_SCHEMAS[opcode]
        program.gate(opcode, tuple(range(max(schema.arity, 1))))
    report = estimate_logical_resources(program, distance=3)
    assert report.estimate.n_operations == len(CLIFFORD_T_OPCODES)
    assert report.clifford_count == len(CLIFFORD_OPCODES)
    assert report.t_count == len(T_FAMILY_OPCODES)
    assert set(report.estimate.operation_counts) == set(CLIFFORD_T_OPCODES)


@pytest.mark.parametrize("opcode", sorted(PARAMETRIC_OPCODES))
def test_a_parametric_rotation_is_refused_by_name(opcode: str) -> None:
    """Angle synthesis is absent, so the input has to already be Clifford+T."""

    with pytest.raises(CapabilityError, match=repr(opcode)):
        estimate_logical_resources(_apply(opcode), distance=3)


@pytest.mark.parametrize("opcode", sorted(DECOMPOSITION_OPCODES))
def test_a_compound_operation_is_refused_by_name(opcode: str) -> None:
    """Its T-count is its decomposition's, and the decompositions disagree."""

    with pytest.raises(CapabilityError, match=repr(opcode)):
        estimate_logical_resources(_apply(opcode), distance=3)


def test_the_compound_refusal_says_why_a_toffoli_is_not_clifford() -> None:
    """The refusal is the whole answer a caller costing an adder needs."""

    with pytest.raises(CapabilityError) as caught:
        estimate_logical_resources(fq.Circuit(3).h(0).ccx(0, 1, 2), distance=3)
    message = str(caught.value)
    assert "'ccx'" in message
    assert "'h'" not in message
    assert "compound operation" in message
    assert "Clifford+T gate" in message
    assert "decomposition" in message
    assert "Clifford+T programs" in message
    assert "Decompose the gate into the Clifford and T gates you mean" in message


def test_the_rotation_refusal_names_every_offending_opcode() -> None:
    """A caller who wrote several rotations needs the whole list to act on."""

    program = fq.Circuit(2).h(0).rz(0, 0.3).rx(1, 0.4).t(1)
    with pytest.raises(CapabilityError) as caught:
        estimate_logical_resources(program, distance=3)
    message = str(caught.value)
    assert "'rx', 'rz'" in message
    assert "'h'" not in message and "'t'" not in message
    assert "angle synthesis" in message
    assert "which is not implemented" in message
    assert "Decompose the rotation into the Clifford+T gates you mean" in message


def test_the_compound_refusal_is_reached_before_the_rotation_refusal() -> None:
    """A program carrying both is reported by its compound operation."""

    program = fq.Circuit(3).ccx(0, 1, 2).rz(0, 0.3)
    with pytest.raises(CapabilityError) as caught:
        estimate_logical_resources(program, distance=3)
    message = str(caught.value)
    assert "'ccx'" in message
    assert "'rz'" not in message


def test_the_two_refusals_are_distinguishable_by_their_reason() -> None:
    """One message for two absences would leave a caller unable to act."""

    with pytest.raises(CapabilityError) as decomposition:
        estimate_logical_resources(_apply("ccx"), distance=3)
    with pytest.raises(CapabilityError) as parametric:
        estimate_logical_resources(_apply("rz"), distance=3)
    assert str(decomposition.value) != str(parametric.value)
    assert "angle synthesis" not in str(decomposition.value)
    assert "compound operation" not in str(parametric.value)


def test_the_trotter_circuit_is_refused_until_angle_synthesis_exists() -> None:
    """The gate-level estimator reads it; this one cannot, and says which blocker."""

    from flagquantum.algorithms.core import transverse_field_ising
    from flagquantum.algorithms.trotter import trotter_circuit

    hamiltonian = transverse_field_ising(3, coupling=0.7, field=0.5)
    program = trotter_circuit(hamiltonian, 0.4, steps=1, order=2)
    assert estimate_resources(program).t_count == 0
    with pytest.raises(CapabilityError, match="angle synthesis"):
        estimate_logical_resources(program, distance=5)


def test_a_channel_is_refused_before_the_families_are_read() -> None:
    """A lowered noise model adds layers no logical operation corresponds to."""

    program = fq.Circuit(2).h(0).t(0).gate("depolarizing", (0,), probability=0.1)
    with pytest.raises(CapabilityError) as caught:
        estimate_logical_resources(program, distance=3)
    message = str(caught.value)
    assert "1 channel instruction(s)" in message
    assert "physical-layer model" in message


def test_a_channel_is_refused_even_when_every_opcode_is_clifford_t() -> None:
    """The channel refusal is not a side effect of the opcode classification."""

    program = fq.Circuit(2).h(0).gate("bit_flip", (0,), probability=0.1)
    assert set(estimate_resources(program).operation_counts) == {"h", "bit_flip"}
    with pytest.raises(CapabilityError, match="channel"):
        estimate_logical_resources(program, distance=3)


@pytest.mark.parametrize("metadata", [{"is_dynamic": True}, {"condition_clauses": ()}])
def test_a_data_dependent_program_is_refused_by_the_compiler(metadata: dict) -> None:
    """The refusal comes from the estimator this unit reads, and is not re-decided."""

    program = ensure_circuit_ir(fq.Circuit(1).h(0).t(0))
    program = dataclasses.replace(
        program, instructions=(Instruction("x", (0,), metadata=metadata),)
    )
    with pytest.raises(CapabilityError) as caught:
        estimate_logical_resources(program, distance=3)
    assert str(caught.value) == str(
        pytest.raises(CapabilityError, estimate_resources, program).value
    )


@pytest.mark.parametrize(
    ("distance", "physical"),
    [(3, 17), (5, 49), (7, 97), (9, 161), (11, 241), (101, 20401)],
)
def test_the_patch_size_is_the_rotated_models_own_arithmetic(
    distance: int, physical: int
) -> None:
    assert surface_code_qubits_per_logical(distance) == 2 * distance**2 - 1
    assert surface_code_qubits_per_logical(distance) == physical


@pytest.mark.parametrize("distance", [1, 2, 0, -3])
def test_a_distance_below_three_is_refused(distance: int) -> None:
    with pytest.raises(ValueError, match="at least 3"):
        surface_code_qubits_per_logical(distance)


@pytest.mark.parametrize("distance", [4, 6, 100])
def test_an_even_distance_is_refused(distance: int) -> None:
    with pytest.raises(ValueError, match="must be odd"):
        surface_code_qubits_per_logical(distance)


def test_the_distance_floor_gives_the_reason_for_the_floor() -> None:
    """A floor with no reason invites a caller to lower it and expect a patch.

    The refusal has to say what a distance-1 and a distance-2 patch are, because
    the caller's next move is to ask why two is not enough; a message that only
    states the floor leaves that question open.
    """

    with pytest.raises(ValueError) as caught:
        surface_code_qubits_per_logical(2)
    message = str(caught.value)
    assert "distance-1 patch carries no redundancy and corrects nothing" in message
    assert "distance-2 patch detects an error without correcting one" in message


@pytest.mark.parametrize("distance", [3.0, "5", None, True, False])
def test_a_non_integer_distance_is_refused(distance: object) -> None:
    """``True`` is refused as a bool rather than accepted as the integer one."""

    with pytest.raises(TypeError, match="must be an integer"):
        surface_code_qubits_per_logical(distance)  # type: ignore[arg-type]


def test_the_distance_refusal_reaches_the_reporting_entry_point() -> None:
    for bad in (2, 4, 3.0):
        with pytest.raises((ValueError, TypeError)):
            estimate_logical_resources(fq.Circuit(1).h(0), distance=bad)  # type: ignore[arg-type]


def test_a_bad_distance_is_reported_before_the_program_is_read() -> None:
    """A bad argument is reported as a bad argument, not as a bad program."""

    with pytest.raises(ValueError, match="at least 3"):
        estimate_logical_resources(object(), distance=1)  # type: ignore[arg-type]


def test_a_program_that_is_not_a_program_is_refused() -> None:
    with pytest.raises(TypeError):
        estimate_logical_resources(42, distance=3)  # type: ignore[arg-type]


def test_the_footprint_is_the_product_of_the_patch_count_and_the_layers() -> None:
    report = estimate_logical_resources(_mixed(), distance=7)
    assert report.physical_qubits_per_logical == 2 * 7**2 - 1 == 97
    assert report.surface_code_cycles == report.logical_depth * 7
    assert report.physical_qubits == report.physical_qubits_per_logical * (
        report.n_qubits
    )
    assert (
        report.spacetime_volume == report.physical_qubits * report.surface_code_cycles
    )
    assert report.logical_depth == report.estimate.depth + report.n_measurements


def test_a_measurement_is_charged_one_logical_layer() -> None:
    """A measurement is an IR record the schedule does not cover, so it is added."""

    program = _mixed()
    bare = estimate_logical_resources(program, distance=5)
    with_records = estimate_logical_resources(_with_measurement(program, 2), distance=5)
    assert bare.n_measurements == 0
    assert with_records.n_measurements == 2
    assert with_records.logical_depth == bare.logical_depth + 1
    assert with_records.surface_code_cycles == bare.surface_code_cycles + 5
    assert with_records.physical_qubits == bare.physical_qubits
    assert with_records.spacetime_volume == bare.spacetime_volume + 5 * (
        bare.physical_qubits
    )
    assert with_records.clifford_count == bare.clifford_count
    assert with_records.t_count == bare.t_count


def test_the_default_logical_count_is_the_declared_register() -> None:
    """The declared width is charged, including wires no operation touches."""

    program = fq.Circuit(4).h(0).t(0)
    report = estimate_logical_resources(program, distance=3)
    assert report.n_qubits == report.estimate.n_qubits == 4
    assert report.estimate.used_wires == 1
    assert report.physical_qubits == 17 * 4


def test_a_caller_may_charge_fewer_logical_qubits_than_the_register() -> None:
    """A register holding ancillas is charged for the logical count alone."""

    report = estimate_logical_resources(fq.Circuit(4).h(0).t(0), distance=3, n_qubits=1)
    assert report.n_qubits == 1
    assert report.physical_qubits == 17
    assert report.spacetime_volume == 17 * report.surface_code_cycles


def test_a_caller_may_charge_more_logical_qubits_than_the_register() -> None:
    """A code block laid out for a larger algorithm is a legal request."""

    report = estimate_logical_resources(fq.Circuit(1).h(0), distance=3, n_qubits=8)
    assert report.n_qubits == 8
    assert report.physical_qubits == 17 * 8


@pytest.mark.parametrize("count", [0, -1, -17])
def test_a_logical_count_below_one_is_refused(count: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        estimate_logical_resources(fq.Circuit(1).h(0), distance=3, n_qubits=count)


@pytest.mark.parametrize("count", [1.0, "1", True])
def test_a_non_integer_logical_count_is_refused(count: object) -> None:
    """``None`` means the register; anything else has to be an integer."""

    program = fq.Circuit(2).h(0).t(0)
    with pytest.raises(TypeError, match="must be an integer"):
        estimate_logical_resources(program, distance=3, n_qubits=count)  # type: ignore[arg-type]


def test_the_no_override_means_the_register() -> None:
    program = fq.Circuit(2).h(0).t(0)
    report = estimate_logical_resources(program, distance=3, n_qubits=None)
    assert report.n_qubits == 2


def test_the_report_states_its_basis_and_its_code_model() -> None:
    report = estimate_logical_resources(_mixed(), distance=5)
    assert report.basis == LOGICAL_RESOURCE_BASIS
    payload = report.to_dict()
    assert payload["kind"] == "flagquantum.logical_resource_report"
    assert payload["basis"] == LOGICAL_RESOURCE_BASIS
    assert payload["surface_code_model"] == SURFACE_CODE_MODEL


def test_the_record_is_json_compatible_and_agrees_with_the_report() -> None:
    import json

    report = estimate_logical_resources(_with_measurement(_mixed()), distance=3)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["clifford_count"] == report.clifford_count
    assert payload["t_count"] == report.t_count
    assert payload["t_depth"] == report.t_depth
    assert payload["n_clifford_t"] == report.n_clifford_t
    assert payload["n_measurements"] == report.n_measurements
    assert payload["code_distance"] == report.code_distance
    assert payload["n_qubits"] == report.n_qubits
    assert payload["logical_depth"] == report.logical_depth
    assert payload["physical_qubits"] == report.physical_qubits
    assert payload["surface_code_cycles"] == report.surface_code_cycles
    assert payload["spacetime_volume"] == report.spacetime_volume
    assert payload["physical_qubits_per_logical"] == (
        report.physical_qubits_per_logical
    )
    assert payload["estimate"] == report.estimate.to_dict()
    assert payload["assumptions"] == list(report.assumptions)

    # The serialized count is the one that was charged for, not the register
    # width the nested estimate reports.  An override is the only case that
    # separates the two, so the record is checked again with one.
    charged = estimate_logical_resources(_mixed(), distance=3, n_qubits=1)
    assert charged.estimate.n_qubits == 3
    assert charged.n_qubits == 1
    serialized = json.loads(json.dumps(charged.to_dict()))
    assert serialized["n_qubits"] == 1
    assert serialized["n_qubits"] != serialized["estimate"]["n_qubits"]


def test_the_assumptions_name_the_numbers_the_arithmetic_used() -> None:
    """Each assumption has to be checkable against the report it travels with."""

    report = estimate_logical_resources(_with_measurement(_mixed(), 1), distance=7)
    assert len(report.assumptions) == 5
    joined = " ".join(report.assumptions)
    assert "7 surface-code cycles" in joined
    assert "distance-7 patch" in joined
    assert "one round of syndrome extraction on a distance-7 patch" in joined
    assert f"{report.physical_qubits_per_logical} physical qubits" in joined
    assert SURFACE_CODE_MODEL in joined
    assert "1 measurement record(s)" in joined
    assert "are outside the instruction sequence the schedule covers" in joined
    assert "each is charged one logical layer" in joined
    assert "a program with no measurement is charged none" in joined
    assert f"{report.n_qubits} logical qubit(s)" in joined
    assert f"{report.surface_code_cycles} surface-code cycles" in joined
    assert "n_qubits" in joined
    assert all(assumption.strip() for assumption in report.assumptions)


def test_the_report_carries_a_limitations_statement() -> None:
    """The field the maturity schema requires is present and says the absences."""

    report = estimate_logical_resources(_mixed(), distance=5)
    assert report.limitations
    assert report.to_dict()["capability_evidence"] == {
        "limitations": report.limitations
    }
    for absence in (
        "no logical error rate",
        "angle synthesis",
        "distillation",
        "placement",
        "rotated surface code",
        "Toffoli",
    ):
        assert absence in report.limitations, absence


def test_the_evidence_block_uses_only_maturity_field_names() -> None:
    """The registry's own vocabulary, checked against the registry's own tables."""

    document = _maturity_contract()
    capability = document["capabilities"][CAPABILITY]
    required = set(document["levels"][capability["level"]]["required_evidence"])
    block = estimate_logical_resources(_mixed(), distance=5).to_dict()[
        "capability_evidence"
    ]
    assert block, "the evidence block must not be empty"
    assert set(block) <= required, set(block) - required
    assert "limitations" in block


def test_the_maturity_entry_names_the_model_the_report_names() -> None:
    """The registry may not describe a different code model than the report costs."""

    capability = _maturity_contract()["capabilities"][CAPABILITY]
    assert capability["category"] == "simulation_and_training"
    assert capability["level"] == "development_evidence"
    assert capability["runtime_modes"] == ["not_applicable"]
    assert capability["hardware"] == ["cpu"]
    assert capability["gradient_support"] == "not_applicable"
    assert capability["distribution_semantics"] == "not_applicable"
    assert "rotated surface code" in capability["limitations"]
    assert "logical error rate" in capability["limitations"]
    assert "Toffoli" in capability["limitations"]
    assert capability["focused_tests"] == "tests/unit/test_logical_resources.py"
    assert capability["documentation"] == "docs/guides/ALGORITHMS.md"


def test_the_maturity_entry_names_only_paths_that_exist() -> None:
    capability = _maturity_contract()["capabilities"][CAPABILITY]
    for field in (
        "quick_start",
        "documentation",
        "focused_tests",
        "development_artifact",
    ):
        value = capability.get(field)
        assert isinstance(value, str) and value, field
        assert (ROOT / value).exists(), f"{field} does not exist: {value}"
    assert capability["public_apis"]
    for name in capability["public_apis"]:
        assert name.startswith("flagquantum.")


def test_the_parity_row_cites_this_capability() -> None:
    """The row cites the entry it is closed against, so the claim is checkable."""

    row = _parity_rows()["logical_resource_estimation"]
    assert row["status"] == "partial"
    assert row["dependency_class"] == "B_open_neutral"
    assert row["maturity_ref"] == CAPABILITY
    for evidence in row["evidence"]:
        if not evidence.startswith("search:"):
            assert (ROOT / evidence).exists(), evidence
    assert "logical_resources" in row["reason"]


def test_the_parity_row_no_longer_claims_the_capability_is_absent() -> None:
    """A closed row whose reason still says "absent" is a stale claim."""

    row = _parity_rows()["logical_resource_estimation"]
    assert not row["reason"].startswith("Absent")
    assert "Qualtran" not in row["reason"]
    assert row["priority"] == "next"


def test_the_module_declares_its_own_surface() -> None:
    from flagquantum.algorithms import logical_resources as module

    assert set(module.__all__) == {
        "CLIFFORD_OPCODES",
        "CLIFFORD_T_OPCODES",
        "DECOMPOSITION_OPCODES",
        "LOGICAL_RESOURCE_BASIS",
        "PARAMETRIC_OPCODES",
        "SURFACE_CODE_MODEL",
        "T_FAMILY_OPCODES",
        "LogicalResourceReport",
        "estimate_logical_resources",
        "surface_code_qubits_per_logical",
    }
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_the_algorithms_package_exposes_the_logical_resource_surface() -> None:
    from flagquantum.algorithms import logical_resources as module

    assert algorithms.logical_resources is module
    assert algorithms.estimate_logical_resources is estimate_logical_resources
    assert algorithms.surface_code_qubits_per_logical is (
        surface_code_qubits_per_logical
    )
    assert algorithms.LogicalResourceReport is LogicalResourceReport
    assert algorithms.LOGICAL_RESOURCE_BASIS == LOGICAL_RESOURCE_BASIS
    assert algorithms.SURFACE_CODE_MODEL == SURFACE_CODE_MODEL
    for name in (
        "LogicalResourceReport",
        "estimate_logical_resources",
        "logical_resources",
        "surface_code_qubits_per_logical",
    ):
        assert name in algorithms.__all__, name


def test_the_report_is_frozen_and_compares_by_value() -> None:
    first = estimate_logical_resources(_mixed(), distance=5)
    second = estimate_logical_resources(_mixed(), distance=5)
    assert isinstance(first, LogicalResourceReport)
    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.t_count = 0  # type: ignore[misc]


def test_the_report_nests_the_compiler_record_rather_than_copying_it() -> None:
    """A reader who wants per-wire depths reads them from the record that owns them."""

    report = estimate_logical_resources(_mixed(), distance=5)
    assert isinstance(report.estimate, type(estimate_resources(_mixed())))
    assert report.estimate.per_wire_depth == estimate_resources(_mixed()).per_wire_depth
    assert report.estimate.max_operation_width == 2
    assert report.estimate.channel_count == 0


def test_the_logic_and_the_report_read_the_same_program_object() -> None:
    """An IR and a Circuit reach the same numbers."""

    circuit = _mixed()
    from_ir = estimate_logical_resources(ensure_circuit_ir(circuit), distance=5)
    from_circuit = estimate_logical_resources(circuit, distance=5)
    assert from_ir == from_circuit


def test_the_surface_code_model_is_the_rotated_one() -> None:
    assert SURFACE_CODE_MODEL == "rotated_surface_code_2d"
    assert LOGICAL_RESOURCE_BASIS == "clifford_t_tally_times_surface_code_distance"
