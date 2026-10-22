"""Native Core construction, immutable identity and continuous-path semantics."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace
from typing import get_type_hints

import pytest
import torch

import flagquantum as fq
from flagquantum.core.geometry import cartesian_pair, same_time
from flagquantum.core.ir import IRValidationError, ensure_circuit_ir
from flagquantum.core.ir.levels import (
    INVALID_INPUT,
    SUPPORTED_EXACT,
    UNSUPPORTED_WITH_DIAGNOSTICS,
    lower_to_level,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.model import BlockRecord, FunctionRecord, Operation
from flagquantum.core.ir.program.physical import (
    RydbergContext,
    RydbergPulse,
    RydbergTransport,
)
from flagquantum.errors import CapabilityError
from flagquantum.experimental.simulation import RydbergProgram

pytestmark = pytest.mark.unit


def pair(**overrides: object) -> RydbergProgram:
    """Feed deliberately malformed arguments through the real public boundary."""
    arguments: dict[str, object] = {
        "atoms": (("a", "left"), ("b", "near")),
        "sites": {
            "left": (0.0, 0.0),
            "near": (2.0, 0.0),
            "far": (4.0, 0.0),
            "top": (0.0, 2.0),
            "across": (-2.0, 0.0),
        },
        "length_unit": "um",
        "time_unit": "us",
        "c6": 64.0,
        "loss_rate": 0.2,
        "minimum_separation": 1.0,
        "transport_channels": ("m0", "m1"),
        "drive_channels": ("d0", "d1"),
    }
    arguments.update(overrides)
    constructor: Callable[..., RydbergProgram] = RydbergProgram
    return constructor(**arguments)


def test_frontend_records_one_program_and_preserves_snapshot_identity() -> None:
    program = pair()
    before = program.to_ir()
    old_hash = hash(before)
    assert (
        program.move(
            "b", source="near", destination="far", start=0, duration=0.5, channel="m0"
        )
        is program
    )
    after = program.to_ir()
    assert before.functions[0].blocks[0].operations == ()
    assert hash(before) == old_hash and before != after
    assert len(after.functions[0].blocks[0].operations) == 1
    assert isinstance(
        after.functions[0].blocks[0].operations[0].physical_data, RydbergTransport
    )
    assert program.commanded_position("b", 0.25) == (3.0, 0.0)
    assert program.commanded_position("b", 0.5) == (4.0, 0.0)
    assert program.commanded_position("b", 3.0) == (4.0, 0.0)
    assert program.atom_ids == ("a", "b") and program.dimensions == (3, 3)
    with pytest.raises(FrozenInstanceError):
        setattr(after.physical_context, "loss_rate", 4)


def test_input_edits_do_not_mutate_captured_sites_or_controls() -> None:
    atoms = [("a", "left"), ("b", "near")]
    sites = {"left": (0.0, 0.0), "near": (2.0, 0.0), "far": (4.0, 0.0)}
    program = pair(atoms=atoms, sites=sites)
    program.move(
        "b", source="near", destination="far", start=0, duration=1, channel="m0"
    )
    record = program.to_ir()
    atoms.reverse()
    sites["far"] = (100, 100)
    assert program.to_ir() == record
    assert program.commanded_position("b", 0.5) == (3.0, 0.0)


def test_timestamp_history_does_not_depend_on_append_order() -> None:
    forward, reverse = pair(), pair()
    for program, order in ((forward, (0, 1)), (reverse, (1, 0))):
        for index in order:
            source, destination = (("near", "far"), ("far", "top"))[index]
            program.move(
                "b",
                source=source,
                destination=destination,
                start=index * 0.5,
                duration=0.5,
                channel="m0",
            )
    assert forward.to_ir() == reverse.to_ir()
    assert hash(forward.to_ir()) == hash(reverse.to_ir())
    assert reverse.commanded_position("b", 0.75) == (2, 1)


def test_parallel_groups_read_one_prior_state_and_canonicalize_by_actor_order() -> None:
    left, right = pair(), pair()
    left.move(
        ("a", "b"),
        source=("left", "near"),
        destination=("top", "far"),
        start=0,
        duration=1,
        channel="m0",
    )
    right.move(
        ("b", "a"),
        source=("near", "left"),
        destination=("far", "top"),
        start=0,
        duration=1,
        channel="m0",
    )
    assert left.to_ir() == right.to_ir()
    assert left.commanded_position("a", 0.5) == (0, 1)
    assert left.commanded_position("b", 0.5) == (3, 0)


@pytest.mark.parametrize("loss", [0.0, 0.2, 100.0])
def test_crossing_and_holding_atoms_remain_obstacles_despite_loss(loss: float) -> None:
    crossing = pair(loss_rate=loss)
    crossing.move(
        ("a", "b"),
        source=("left", "near"),
        destination=("near", "left"),
        start=0,
        duration=1,
        channel="m0",
    )
    with pytest.raises(IRValidationError, match="physical.path_collision"):
        crossing.to_ir()
    holding = pair(loss_rate=loss)
    holding.move(
        "b", source="near", destination="across", start=0, duration=1, channel="m0"
    )
    with pytest.raises(IRValidationError, match="physical.path_collision"):
        holding.to_ir()


def test_exact_geometry_handles_large_coordinates_without_overflow_or_lost_offsets() -> (
    None
):
    program = pair(sites={"left": (-1e308, 0), "near": (0, 1), "far": (1e308, 0)})
    program.move(
        "a", source="left", destination="far", start=0, duration=1, channel="m0"
    )
    assert program.commanded_position("a", 0.5) == (0, 0)
    unsafe = pair(
        sites={"left": (-1e308, 0), "near": (0, 1), "far": (1e308, 0)},
        minimum_separation=1.1,
    )
    unsafe.move(
        "a", source="left", destination="far", start=0, duration=1, channel="m0"
    )
    with pytest.raises(IRValidationError, match="physical.path_collision"):
        unsafe.to_ir()


def test_source_history_and_actor_move_overlap_refuse_with_occupancy_context() -> None:
    program = pair().move(
        "b", source="left", destination="far", start=0, duration=1, channel="m0"
    )
    with pytest.raises(IRValidationError, match="physical.occupancy_conflict.*source"):
        program.to_ir()
    overlap = pair().move(
        "b", source="near", destination="far", start=0, duration=1, channel="m0"
    )
    overlap.move(
        "b", source="far", destination="top", start=0.5, duration=1, channel="m1"
    )
    with pytest.raises(IRValidationError, match="overlapping moves"):
        overlap.to_ir()


def test_channel_and_drive_conflicts_are_distinct_from_allowed_simultaneity() -> None:
    resource = pair().move(
        "a", source="left", destination="top", start=0, duration=1, channel="m0"
    )
    resource.move(
        "b", source="near", destination="far", start=0, duration=1, channel="m0"
    )
    with pytest.raises(IRValidationError, match="physical.channel_conflict.*m0"):
        resource.to_ir()
    pulses = pair().pulse("a", start=0, duration=1, amplitude=1, channel="d0")
    pulses.pulse("a", start=0.5, duration=1, amplitude=1, channel="d1")
    with pytest.raises(IRValidationError, match="physical.channel_conflict"):
        pulses.to_ir()
    allowed = pair().move(
        "b", source="near", destination="far", start=0, duration=1, channel="m0"
    )
    allowed.pulse("a", start=0, duration=1, amplitude=1, channel="d0")
    allowed.pulse("b", start=0, duration=1, amplitude=1, channel="d1")
    assert verify_program_record(allowed.to_ir()) == ()


def test_abutting_moves_use_the_shared_eight_ulp_boundary_policy() -> None:
    assert same_time(0.1 + 0.2, 0.3)
    assert not same_time(0.3, 0.3000001)
    program = pair().move(
        "b", source="near", destination="far", start=0, duration=0.1 + 0.2, channel="m0"
    )
    program.move(
        "b", source="far", destination="top", start=0.3, duration=0.2, channel="m0"
    )
    assert verify_program_record(program.to_ir()) == ()
    assert program.commanded_position("b", 0.3) == (4.0, 0.0)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_boundary_comparison_uses_the_requested_real_precision(precision: str) -> None:
    dtype = torch.float32 if precision == "float32" else torch.float64
    start = torch.tensor(1.0, dtype=dtype)
    eight = start.clone()
    for _ in range(8):
        eight = torch.nextafter(eight, torch.tensor(math.inf, dtype=dtype))
    nine = torch.nextafter(eight, torch.tensor(math.inf, dtype=dtype))
    compare: Callable[..., bool] = same_time
    assert compare(start.item(), eight.item(), precision=precision)
    assert not compare(start.item(), nine.item(), precision=precision)


def test_atom_site_namespaces_are_separate_and_large_ids_are_not_split() -> None:
    program = pair(atoms=(("left", "left"), ("atom_b", "near")))
    program.move(
        "atom_b", source="near", destination="far", start=0, duration=1, channel="m0"
    )
    assert program.atom_ids == ("left", "atom_b")
    assert program.commanded_position("atom_b", 0.5) == (3, 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("atoms", ()),
        ("atoms", (("a", "left"), ("a", "near"))),
        ("atoms", (("a", "left"), ("b", "left"))),
        ("atoms", (("a", "unknown"),)),
        ("sites", {"left": (0, 0), "near": (0, 0)}),
        ("minimum_separation", 0),
        ("loss_rate", -1),
        ("c6", math.inf),
        ("length_unit", "cm"),
        ("time_unit", "ns"),
        ("coordinate_frame", ""),
        ("geometry_sampling", "automatic"),
        ("drive_channels", ("m0",)),
        ("transport_channels", ("m0", "m0")),
    ],
)
def test_invalid_context_values_are_not_silently_repaired(
    field: str, value: object
) -> None:
    with pytest.raises(IRValidationError):
        pair(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("c6", True),
        ("loss_rate", torch.tensor(0.2, requires_grad=True)),
        ("atoms", "ab"),
        ("sites", [(0, 0)]),
        ("transport_channels", "m0"),
    ],
)
def test_wrong_argument_types_refuse_without_detaching(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError):
        pair(**{field: value})


def test_scalar_controls_reuse_core_expressions_and_refuse_tensor_source_leaves() -> (
    None
):
    amplitude = fq.Parameter("omega")
    program = pair().pulse(
        "b", start=0, duration=1, amplitude=0.5 * amplitude + 1, phase=0.7, channel="d0"
    )
    data = program.to_ir().functions[0].blocks[0].operations[0].physical_data
    assert isinstance(data, RydbergPulse)
    assert data.amplitude == 0.5 * amplitude + 1
    tensor_control: Callable[..., RydbergProgram] = program.pulse
    with pytest.raises(TypeError, match="ordinary real"):
        tensor_control(
            "a",
            start=0,
            duration=1,
            amplitude=torch.tensor(1.0, requires_grad=True),
            channel="d1",
        )


def test_tensor_helpers_keep_order_presence_and_detector_effects_separate() -> None:
    program = pair()
    assert program.basis_state(("r", "lost")).abs().argmax().item() == 5
    assert program.basis_state(("lost", "r")).abs().argmax().item() == 7
    assert torch.equal(program.projector({}), torch.eye(9, dtype=torch.complex128))
    expected = torch.zeros(9, dtype=torch.complex128)
    expected[[1, 4, 7]] = 1
    assert torch.equal(program.projector({"b": "r"}), torch.diag(expected))
    assert torch.equal(program.no_flag_operator(0), program.projector({}))
    assert torch.equal(program.no_flag_operator(1), program.survival_operator())
    local = torch.diag(torch.tensor([1, 1, 0.2], dtype=torch.complex128))
    torch.testing.assert_close(
        program.no_flag_operator(0.8), torch.kron(local, local), atol=1e-15, rtol=0
    )
    for tensor in (
        program.basis_state(("g", "r")),
        program.projector({}),
        program.survival_operator(),
    ):
        assert tensor.dtype == torch.complex128 and tensor.device.type == "cpu"
    with pytest.raises(IRValidationError):
        program.projector({"missing": "r"})


def test_large_geometry_roster_does_not_admit_exponential_tensor_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = pair(atoms=(("a", "left"), ("b", "near"), ("c", "far")))
    assert program.dimensions == (3, 3, 3)
    assert program.commanded_position("c", 1) == (4, 0)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("tensor allocation occurred before capability refusal")

    for name in ("zeros", "ones", "eye", "tensor", "kron"):
        monkeypatch.setattr(torch, name, forbidden)
    for call in (
        lambda: program.basis_state(("g", "g", "g")),
        lambda: program.projector({}),
        program.survival_operator,
        lambda: program.no_flag_operator(0.8),
    ):
        with pytest.raises(CapabilityError, match="before allocation"):
            call()


def test_profile_dispatch_and_front_door_refusal_use_existing_ir_boundaries() -> None:
    program = pair()
    for source in (program, program.to_ir()):
        lowered = lower_to_level(source, "program")
        assert lowered.state == SUPPORTED_EXACT and lowered.record == program.to_ir()
        assert lowered.canonical_payload is None
        assert lower_to_level(source, "quantum").state == UNSUPPORTED_WITH_DIAGNOSTICS
        for entry in (ensure_circuit_ir, fq.plan, fq.run):
            with pytest.raises(CapabilityError, match="lindblad"):
                entry(source)
    assert program_to_circuit(program.to_ir()).state == UNSUPPORTED_WITH_DIAGNOSTICS


def test_malformed_and_unsupported_records_retain_distinct_diagnostics() -> None:
    record = pair().to_ir()
    assert record.physical_context is not None
    invalid = replace(
        record, physical_context=replace(record.physical_context, loss_rate=-1)
    )
    assert lower_to_level(invalid, "program").state == INVALID_INPUT
    assert verify_program_record(invalid)[0].code == "physical.invalid_context"
    unsupported = replace(
        record, functions=(FunctionRecord("other", (BlockRecord("entry"),)),)
    )
    result = lower_to_level(unsupported, "program")
    assert result.state == UNSUPPORTED_WITH_DIAGNOSTICS
    assert result.diagnostics[0].code == "physical.unsupported_structure"
    wrong_data = replace(
        record,
        functions=(
            FunctionRecord(
                "main", (BlockRecord("entry", operations=(Operation("rydberg.move"),)),)
            ),
        ),
    )
    assert verify_program_record(wrong_data)[0].code == "physical.invalid_operation"
    empty = replace(record, functions=())
    assert lower_to_level(empty, "program").state == INVALID_INPUT
    assert verify_program_record(empty)[0].code == "program.function_arity"
    moving = (
        pair()
        .move("b", source="near", destination="far", start=0, duration=1, channel="m0")
        .to_ir()
    )
    missing_context = replace(moving, physical_context=None)
    rejected = lower_to_level(missing_context, "program")
    assert rejected.state == INVALID_INPUT
    assert rejected.diagnostics[0].code == "physical.invalid_context"


def test_approved_record_fields_export_and_public_hints_are_exact() -> None:
    assert [field.name for field in fields(RydbergTransport)] == [
        "atoms",
        "source",
        "destination",
        "start",
        "duration",
        "channel",
    ]
    assert [field.name for field in fields(RydbergPulse)] == [
        "atoms",
        "start",
        "duration",
        "amplitude",
        "detuning",
        "phase",
        "channel",
    ]
    assert [field.name for field in fields(RydbergContext)] == [
        "atoms",
        "sites",
        "length_unit",
        "time_unit",
        "c6",
        "loss_rate",
        "minimum_separation",
        "coordinate_frame",
        "transport_channels",
        "drive_channels",
        "geometry_sampling",
    ]
    assert get_type_hints(RydbergProgram.basis_state)["return"] is torch.Tensor
    assert list(inspect.signature(RydbergProgram).parameters) == [
        "atoms",
        "sites",
        "length_unit",
        "time_unit",
        "c6",
        "loss_rate",
        "minimum_separation",
        "coordinate_frame",
        "transport_channels",
        "drive_channels",
        "geometry_sampling",
    ]
    assert "RydbergProgram" not in fq.__all__
    assert cartesian_pair((2, 4), "test") == (2.0, 4.0)
