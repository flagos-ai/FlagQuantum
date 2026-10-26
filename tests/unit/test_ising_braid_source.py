"""Native Ising source semantics, canonical encoding and explicit route refusals."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, patch

import pytest
import torch

import flagquantum as fq
import flagquantum.deployment as deployment
import flagquantum.lindblad as fql
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.core.ir import IRValidationError, ensure_circuit_ir
from flagquantum.core.ir.levels import (
    SUPPORTED_EXACT,
    UNSUPPORTED_WITH_DIAGNOSTICS,
    lower_to_level,
    program_to_circuit,
)
from flagquantum.core.ir.program.model import Operation, ProgramRecord, ValueRef
from flagquantum.core.ir.program.physical import (
    IsingContext,
    IsingExchange,
    RydbergPulse,
)
from flagquantum.core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from flagquantum.core.ir.program.verifier import verify_program_record
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import CapabilityError, SerializationError
from flagquantum.experimental.simulation import (
    IsingBraidProgram as ExperimentalIsingBraidProgram,
)
from flagquantum.lindblad._plan import _identity
from flagquantum.remote.compute._native_job import NativeJiudingJobClient
from flagquantum.remote.compute.jiuding import JiudingClient

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("initial", ["1", "psi"])
def test_ordered_word_and_fusion_coordinates_are_source_not_qubits(
    initial: Literal["1", "psi"],
) -> None:
    program = IsingBraidProgram(initial_channel=initial)
    for left, clockwise in ((1, False), (0, True), (2, False)):
        assert program.exchange(left, clockwise=clockwise) is program
    record = program.to_ir()
    assert isinstance(record, ProgramRecord)
    assert record.physical_context == IsingContext(
        ("a", "b", "c", "d"), "1", "left_associated", "nayak_ising", initial
    )
    operations = record.functions[0].blocks[0].operations
    assert tuple(op.opcode for op in operations) == ("ising.exchange",) * 3
    assert tuple(op.physical_data for op in operations) == (
        IsingExchange(1, False),
        IsingExchange(0, True),
        IsingExchange(2, False),
    )
    order = ["a", "b", "c", "d"]
    for op in operations:
        data = op.physical_data
        assert isinstance(data, IsingExchange)
        order[data.left], order[data.left + 1] = order[data.left + 1], order[data.left]
    assert order == ["c", "a", "d", "b"]
    assert all(
        not op.operands and not op.results and not op.parameters for op in operations
    )
    assert record.canonical_payload is None and verify_program_record(record) == ()
    assert program.basis_labels == (("1", "sigma", "1"), ("psi", "sigma", "1"))


def test_projectors_are_fresh_complete_cpu_mathematical_inputs() -> None:
    assert ExperimentalIsingBraidProgram is IsingBraidProgram
    program = IsingBraidProgram()
    one, psi = program.projector("1"), program.projector("psi")
    assert one.dtype == psi.dtype == torch.complex128
    assert one.device.type == psi.device.type == "cpu"
    assert one.shape == psi.shape == (2, 2)
    assert torch.equal(one + psi, torch.eye(2, dtype=torch.complex128))
    assert torch.equal(one @ psi, torch.zeros((2, 2), dtype=torch.complex128))
    one.zero_()
    assert program.projector("1")[0, 0] == 1


def test_snapshots_and_identity_keep_word_direction_and_initial_channel() -> None:
    program = IsingBraidProgram().exchange(1)
    source = program.to_ir()
    before = native_program_to_dict(source)
    original_hash = hash(source)
    program.exchange(1, clockwise=True)
    assert hash(source) == original_hash and native_program_to_dict(source) == before
    assert source != program.to_ir()
    assert program.to_ir() != IsingBraidProgram().to_ir()
    assert (
        IsingBraidProgram().exchange(0).to_ir()
        != IsingBraidProgram().exchange(2).to_ir()
    )
    assert source != IsingBraidProgram().exchange(1, clockwise=True).to_ir()
    assert source != IsingBraidProgram(initial_channel="psi").exchange(1).to_ir()
    assert source != IsingBraidProgram().exchange(0).exchange(1).to_ir()
    assert source.physical_context is not None
    with pytest.raises(FrozenInstanceError):
        setattr(source.physical_context, "initial_channel", "psi")


@pytest.mark.parametrize("index", [True, False, 1.0, "1", None, [], torch.tensor(1)])
def test_wrong_exchange_index_type_is_atomic(index: object) -> None:
    program = IsingBraidProgram().exchange(0)
    before = program.to_ir()
    exchange: Callable[..., IsingBraidProgram] = program.exchange
    with pytest.raises(TypeError):
        exchange(index)
    assert program.to_ir() == before


@pytest.mark.parametrize("index", [-1, 3, 9])
def test_out_of_range_exchange_is_atomic(index: int) -> None:
    program = IsingBraidProgram()
    before = program.to_ir()
    with pytest.raises(ValueError):
        program.exchange(index)
    assert program.to_ir() == before


@pytest.mark.parametrize("clockwise", [None, 0, 1, "false"])
def test_direction_is_a_boolean_not_a_truthy_control(clockwise: object) -> None:
    program = IsingBraidProgram()
    exchange: Callable[..., IsingBraidProgram] = program.exchange
    with pytest.raises(TypeError):
        exchange(1, clockwise=clockwise)
    assert not program.to_ir().functions[0].blocks[0].operations


@pytest.mark.parametrize("channel", [None, False, 1, ["1"], "sigma", "", "vacuum"])
def test_channel_errors_do_not_infer_or_project_a_sector(channel: object) -> None:
    error = ValueError if isinstance(channel, str) else TypeError
    constructor: Callable[..., IsingBraidProgram] = IsingBraidProgram
    projector: Callable[..., torch.Tensor] = IsingBraidProgram().projector
    with pytest.raises(error):
        constructor(initial_channel=channel)
    with pytest.raises(error):
        projector(channel)


@pytest.mark.parametrize("initial", ["1", "psi"])
def test_native_source_1_1_round_trip_preserves_full_context_and_word(
    initial: Literal["1", "psi"],
) -> None:
    original = (
        IsingBraidProgram(initial_channel=initial)
        .exchange(1)
        .exchange(0, clockwise=True)
        .to_ir()
    )
    payload = native_program_to_dict(original)
    assert payload["schema"] == "flagquantum.program_ir" and payload["version"] == "1.1"
    assert payload["physical_context"] == {
        "kind": "ising_braid",
        "anyon_ids": ["a", "b", "c", "d"],
        "total_charge": "1",
        "fusion_tree": "left_associated",
        "gauge": "nayak_ising",
        "initial_channel": initial,
    }
    restored = native_program_from_dict(json.loads(json.dumps(payload)))
    assert restored == original and hash(restored) == hash(original)
    assert native_program_to_dict(restored) == payload


@pytest.mark.parametrize(
    "location", ["program", "context", "function", "block", "operation", "data"]
)
def test_unknown_source_fields_at_every_depth_fail_closed(location: str) -> None:
    payload = native_program_to_dict(IsingBraidProgram().exchange(1).to_ir())
    nodes = {
        "program": payload,
        "context": payload["physical_context"],
        "function": payload["functions"][0],
        "block": payload["functions"][0]["blocks"][0],
    }
    nodes["operation"] = nodes["block"]["operations"][0]
    nodes["data"] = nodes["operation"]["physical_data"]
    nodes[location]["unreviewed"] = True
    with pytest.raises(SerializationError, match="exactly fields"):
        native_program_from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_charge", "psi"),
        ("fusion_tree", "right_associated"),
        ("gauge", "other"),
        ("initial_channel", "sigma"),
        ("initial_channel", 1),
        ("anyon_ids", ["a", "b", "c"]),
        ("anyon_ids", ["a", "b", "b", "d"]),
    ],
)
def test_context_validation_precedes_source_admission(
    field: str, value: object
) -> None:
    payload = native_program_to_dict(IsingBraidProgram().to_ir())
    payload["physical_context"][field] = value
    with pytest.raises(SerializationError, match="physical.invalid_context"):
        native_program_from_dict(payload)


@pytest.mark.parametrize(
    "field,value", [("left", True), ("left", 1.0), ("left", 3), ("clockwise", 0)]
)
def test_source_decoder_does_not_coerce_exchange_fields(
    field: str, value: object
) -> None:
    payload = native_program_to_dict(IsingBraidProgram().exchange(1).to_ir())
    payload["functions"][0]["blocks"][0]["operations"][0]["physical_data"][
        field
    ] = value
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


@pytest.mark.parametrize("version", ["1.0", "1.2", 1.1, True])
def test_ising_source_cannot_borrow_another_source_version(version: object) -> None:
    payload = native_program_to_dict(IsingBraidProgram().to_ir())
    payload["version"] = version
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


def test_mixed_profiles_and_ssa_payloads_are_not_a_braid_word() -> None:
    record = IsingBraidProgram().exchange(1).to_ir()
    function, block = record.functions[0], record.functions[0].blocks[0]
    for operation in (
        Operation("ising.exchange"),
        Operation("rydberg.pulse", physical_data=IsingExchange(1, False)),
        Operation(
            "ising.exchange", physical_data=RydbergPulse((), 0, 1, 1, 0, 0, "laser")
        ),
        Operation(
            "ising.exchange",
            (ValueRef("entry", 0),),
            physical_data=IsingExchange(1, False),
        ),
    ):
        broken = replace(
            record,
            functions=(
                replace(function, blocks=(replace(block, operations=(operation,)),)),
            ),
        )
        assert verify_program_record(broken)[0].code == "physical.invalid_operation"
        with pytest.raises(SerializationError):
            native_program_to_dict(broken)
    unsupported = replace(record, canonical_payload={})
    assert (
        verify_program_record(unsupported)[0].code == "physical.unsupported_structure"
    )


def test_existing_structure_verifier_still_precedes_profile_admission() -> None:
    record = IsingBraidProgram().exchange(1).to_ir()
    function = record.functions[0]
    block = function.blocks[0]
    for broken, code in (
        (replace(record, functions=()), "program.function_arity"),
        (
            replace(record, functions=(replace(function, entry="absent"),)),
            "program.entry_missing",
        ),
        (
            replace(record, functions=(replace(function, name="other"),)),
            "physical.unsupported_structure",
        ),
        (
            replace(record, functions=(function, replace(function, name="other"))),
            "physical.unsupported_structure",
        ),
        (
            replace(
                record,
                functions=(
                    replace(
                        function, entry="other", blocks=(replace(block, name="other"),)
                    ),
                ),
            ),
            "physical.unsupported_structure",
        ),
    ):
        assert code in {item.code for item in verify_program_record(broken)}
        with pytest.raises(SerializationError):
            native_program_to_dict(broken)


def test_context_extensions_are_refused_before_an_unreadable_source_is_emitted() -> (
    None
):
    @dataclass(frozen=True)
    class ExtendedContext(IsingContext):
        unreviewed: str = "extra"

    program = IsingBraidProgram()
    source = program.to_ir()
    context = source.physical_context
    assert isinstance(context, IsingContext)
    extension = ExtendedContext(
        context.anyon_ids,
        context.total_charge,
        context.fusion_tree,
        context.gauge,
        context.initial_channel,
    )
    broken = replace(source, physical_context=extension)
    with pytest.raises(SerializationError, match="physical.invalid_context"):
        native_program_to_dict(broken)
    program._record = broken
    with pytest.raises(IRValidationError, match="physical.invalid_context"):
        program.to_ir()


def test_a_duck_typed_source_does_not_gain_native_frontend_authority() -> None:
    class ForeignSource:
        def to_ir(self) -> ProgramRecord:
            return IsingBraidProgram().to_ir()

    source = ForeignSource()
    assert lower_to_level(source, "program").state != SUPPORTED_EXACT
    with pytest.raises(CapabilityError, match="Ising"):
        ensure_circuit_ir(source)


def test_existing_internal_contract_reader_requires_both_approved_profiles() -> None:
    from tools.check_multi_level_ir_contract import contract_errors

    contract_path = (
        Path(__file__).parents[2]
        / "contracts/multi-level-ir-internal-v1-candidate.json"
    )
    contract = json.loads(contract_path.read_text())
    assert contract["version"] == "0.5"
    assert contract_errors(contract) == ()
    removed = deepcopy(contract)
    removed["physical_profiles"] = [contract["physical_profiles"][0]]
    assert any(
        "physical_profiles differ" in error for error in contract_errors(removed)
    )
    unresolved = deepcopy(contract)
    unresolved["physical_profiles"][1][
        "context_type"
    ] = "flagquantum.core.absent.IsingContext"
    assert any("must resolve" in error for error in contract_errors(unresolved))


@pytest.mark.parametrize("as_record", [False, True])
def test_existing_level_boundary_refuses_circuit_and_unbuilt_levels(
    as_record: bool,
) -> None:
    program = IsingBraidProgram().exchange(1)
    source = program.to_ir() if as_record else program
    converted = lower_to_level(source, "program")
    assert converted.state == SUPPORTED_EXACT and converted.record == program.to_ir()
    for level in ("quantum", "target"):
        assert lower_to_level(source, level).state == UNSUPPORTED_WITH_DIAGNOSTICS
    assert program_to_circuit(program.to_ir()).state == UNSUPPORTED_WITH_DIAGNOSTICS
    for entry in (ensure_circuit_ir, compile_program, emit_openqasm):
        with pytest.raises(CapabilityError, match="Ising"):
            entry(source)
    if as_record:
        for entry in (fq.plan, fq.run):
            with pytest.raises(CapabilityError, match="Ising"):
                entry(source)


@pytest.mark.parametrize("as_record", [False, True])
def test_lindblad_facades_reject_discrete_source_before_rydberg_access(
    as_record: bool,
) -> None:
    program = IsingBraidProgram().exchange(1)
    source = program.to_ir() if as_record else program
    state = program.projector("1")
    for entry in (fql.plan, fql.run):
        with pytest.raises(CapabilityError, match="Hamiltonian"):
            entry(source, state, [0.0, 1.0])
    with pytest.raises(CapabilityError, match="Hamiltonian"):
        fql.adjoint_gradient(
            source, state, [0.0, 1.0], lambda result: result.real.sum()
        )


def test_malformed_source_keeps_ir_diagnostics_before_lindblad_refusal() -> None:
    program = IsingBraidProgram()
    record = program.to_ir()
    context = record.physical_context
    assert isinstance(context, IsingContext)
    broken = replace(record, physical_context=replace(context, anyon_ids=("a", "b")))
    with pytest.raises(IRValidationError, match="physical.invalid_context"):
        fql.plan(broken, program.projector("1"), [0.0, 1.0])


def test_old_native_plan_cannot_seal_an_ising_source_as_rydberg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures/ising_braid_legacy_plans.json"
    payload = deepcopy(json.loads(fixture.read_text())["plans"]["native_rydberg"])
    payload["program"] = native_program_to_dict(IsingBraidProgram().exchange(1).to_ir())
    payload["identity"] = _identity(payload)
    numerical = Mock(
        side_effect=AssertionError("discrete source reached Rydberg numerical work")
    )
    monkeypatch.setattr(
        "flagquantum.lindblad._native_plan._plan_density_matrix_evolution", numerical
    )
    with pytest.raises(SerializationError, match="timed Rydberg"):
        fql.LindbladPlan.from_dict(payload)
    assert not numerical.called


@pytest.mark.parametrize("as_record", [False, True])
def test_deployment_and_remote_reject_before_side_effects(as_record: bool) -> None:
    program = IsingBraidProgram()
    source = program.to_ir() if as_record else program
    with patch("flagquantum.deployment.cloud.compile_program") as compiler:
        entry: Callable[..., object] = deployment.create_deployment_package
        with pytest.raises(CapabilityError, match="Ising"):
            entry(source)
        assert not compiler.called
    for client in (
        JiudingClient(workspace="test"),
        NativeJiudingJobClient(project="set.project"),
    ):
        guards = {
            name: Mock(side_effect=AssertionError("unexpected remote side effect"))
            for name in (
                "workspace",
                "_resolve_compute_target",
                "submit",
                "_request",
                "_auth",
            )
        }
        with (
            patch.multiple(client, **guards),
            pytest.raises(CapabilityError, match="Ising"),
        ):
            client.submit_program(source, target="jiuding:cpu", image="unused")
        assert not any(spy.called for spy in guards.values())


def test_physical_opcode_does_not_reserve_a_circuit_matrix_name() -> None:
    circuit = fq.Circuit(1)
    circuit.any(0, matrix=torch.eye(2, dtype=torch.complex64), name="ising.exchange")
    assert torch.equal(
        fq.run(circuit, outputs=fq.probabilities()).probabilities,
        torch.tensor([[1.0, 0.0]]),
    )
    assert "IsingBraidProgram" not in fq.__all__
