"""The tensor-network release producer is machine-checked before it is trusted.

The producer is what turns a measured distributed run into a payload the release
gate reads, so the parts of it that decide *what a payload may claim* are tested
here rather than left to the campaign that runs it: the ladder that the frozen
manifest implies, the timing protocol that may not be chosen by argument, the
ownership map that every rank has to agree on, and the projection that lifts a
role document onto the contract the sealer reads. A producer that got any of
these wrong would seal a run as evidence for something it did not measure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks import tensor_network_release_evidence as producer
from benchmarks.internal.evidence.speedup import bootstrap_ratio_interval

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]

RELEASE_MANIFEST = Path("benchmarks/manifests/tensor_network_release_v1.json")
FROZEN = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
SPEED = FROZEN["speed_workload"]


def _ladder() -> list[dict]:
    return producer._speed_ladder(SPEED)


def test_the_frozen_ladder_contains_the_acceptance_configuration() -> None:
    names = [entry["name"] for entry in _ladder()]
    assert SPEED["acceptance_configuration"] in names
    # The widest point of each frozen multiset is the acceptance configuration,
    # so the reported speedup is the one a reader would look for first.
    accepted = next(
        entry
        for entry in _ladder()
        if entry["name"] == SPEED["acceptance_configuration"]
    )
    assert accepted["qubits"] == max(SPEED["qubits"])
    assert accepted["layers"] == max(SPEED["layers"])
    assert accepted["slice_count"] == max(SPEED["slice_label_counts"])


def test_the_ladder_refuses_an_acceptance_configuration_it_does_not_contain() -> None:
    broken = copy.deepcopy(SPEED)
    broken["acceptance_configuration"] = "matched_speed_999q_l1_s1"
    with pytest.raises(SystemExit, match="acceptance configuration"):
        producer._speed_ladder(broken)


def test_the_ladder_refuses_parallel_multisets_of_different_lengths() -> None:
    broken = copy.deepcopy(SPEED)
    broken["layers"] = [6, 8]
    with pytest.raises(SystemExit, match="different lengths"):
        producer._speed_ladder(broken)


def test_the_timed_circuit_carries_one_parameter_per_wire_and_layer() -> None:
    circuit, parameters = producer._speed_circuit(
        14, 6, device=torch.device("cpu"), dtype=torch.complex64
    )
    assert len(parameters) == 14 * 6
    assert all(parameter.requires_grad for parameter in parameters)
    # The parameter count is the point of the workload, so it is the circuit's
    # own structure that has to carry it rather than a number beside the circuit.
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert circuit.n_wires == 14


def test_the_ownership_projection_groups_parameters_by_their_owner_rank() -> None:
    owned = producer._owned_parameters(
        (
            {"rank": 1, "parameter_indices": [3, 1]},
            {"rank": 0, "parameter_indices": [0]},
            {"rank": 1, "parameter_indices": [2]},
        )
    )
    assert owned == {
        "rank:0": ["parameter:0"],
        "rank:1": ["parameter:1", "parameter:2", "parameter:3"],
    }


def test_ownership_must_agree_across_the_ranks_that_published_it() -> None:
    shared = {"rank:0": ["parameter:0"], "rank:1": ["parameter:1"]}
    assert producer._agreed_ownership([shared, shared], context="c") == shared
    # A payload that concatenated the ranks would list each parameter once per
    # rank, and one that read only rank 0 would accept a leg whose other ranks
    # published a different map. Both are the same failure and both fail closed.
    with pytest.raises(SystemExit, match="disagree"):
        producer._agreed_ownership(
            [shared, {"rank:0": ["parameter:1"], "rank:1": ["parameter:0"]}],
            context="c",
        )
    with pytest.raises(SystemExit, match="no parameter ownership"):
        producer._agreed_ownership([{}], context="c")


def test_the_bootstrap_interval_brackets_a_ratio_the_samples_imply() -> None:
    baseline = [4.0, 4.1, 3.9, 4.05, 3.95]
    sharded = [1.0, 1.02, 0.98, 1.01, 0.99]
    lower, upper = bootstrap_ratio_interval(baseline, sharded)
    assert lower < 4.0 < upper
    # The interval is a percentile interval of the ratio, so it can never
    # exclude the point estimate the medians themselves give.
    assert lower <= 4.0 <= upper


def test_the_timing_protocol_is_read_from_the_manifest_not_from_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_arguments = [
        "--role",
        "matched-speed",
        "--release-manifest",
        str(RELEASE_MANIFEST),
        "--warmup",
        str(int(SPEED["warmup_steps"]) + 1),
        "--iterations",
        str(int(SPEED["measured_steps"])),
    ]
    with pytest.raises(SystemExit, match="frozen matched-speed protocol requires"):
        producer.main(parser_arguments)
    # Omitting the counts entirely is the same refusal: the protocol is not
    # something a caller may leave to a default.
    with pytest.raises(SystemExit, match="frozen matched-speed protocol requires"):
        producer.main(
            [
                "--role",
                "matched-speed",
                "--release-manifest",
                str(RELEASE_MANIFEST),
            ]
        )

    # The same call with the frozen counts gets past the protocol check and
    # fails later, on the environment a rank has to be launched with.
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(KeyError):
        producer.main(
            [
                "--role",
                "matched-speed",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--warmup",
                str(SPEED["warmup_steps"]),
                "--iterations",
                str(SPEED["measured_steps"]),
            ]
        )


def test_the_release_payload_role_lifts_the_contract_out_of_a_document(
    tmp_path: Path,
) -> None:
    contract = {
        "acceptance_case": "matched_speed",
        "world_size": 2,
        "state_mode": FROZEN["runtime"]["state_mode"],
    }
    document = tmp_path / "role.json"
    document.write_text(json.dumps({"measurements": contract}), encoding="utf-8")
    output = tmp_path / "evidence.json"
    assert (
        producer.main(
            [
                "--role",
                "release-payload",
                "--document",
                str(document),
                "--measurements",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == contract

    # A role document without a measurements block has nothing to project, and
    # a projection that silently produced an empty payload would seal a run as
    # evidence for a contract nobody assembled.
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"role": "matched-speed"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no measurements block"):
        producer.main(
            [
                "--role",
                "release-payload",
                "--document",
                str(empty),
                "--measurements",
                str(tmp_path / "never.json"),
            ]
        )


def test_the_protocol_digest_identifies_the_manifest_that_froze_the_ladder() -> None:
    # The matched-speed ladder is frozen inline in the manifest, so the manifest
    # is what identifies the timed protocol. Hashing the capacity workload here
    # would attribute a speed run to a circuit the run never touched.
    assert (
        producer._protocol_digest(RELEASE_MANIFEST)
        == hashlib.sha256(RELEASE_MANIFEST.read_bytes()).hexdigest()
    )
    assert producer._protocol_digest(RELEASE_MANIFEST) != producer._workload_digest(
        FROZEN["capacity_workload"]
    )


def test_the_checkpoint_budget_is_absent_unless_the_run_was_launched_with_one() -> None:
    # A measurement taken without a bounded tape must say so, rather than
    # carrying a zero that reads as an empty budget.
    assert producer._checkpoint_budget_bytes(argparse.Namespace()) is None
    bounded = argparse.Namespace(checkpoint_budget_bytes=4 * 1024**3)
    assert producer._checkpoint_budget_bytes(bounded) == 4 * 1024**3


def test_the_launcher_hands_the_checkpoint_budget_to_the_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The budget changes the memory profile of a step, so it is the one input a
    # pair leg and a single-device leg have to share for their peaks to be
    # comparable. It travels as a recorded input rather than as an assumption.
    captured: list[argparse.Namespace] = []

    def record(arguments: argparse.Namespace) -> int:
        captured.append(arguments)
        return 0

    monkeypatch.setattr(producer, "_role_capacity_failure", record)
    budget = 6 * 1024**3
    assert (
        producer.main(
            [
                "--role",
                "capacity-failure",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--checkpoint-budget-bytes",
                str(budget),
            ]
        )
        == 0
    )
    assert producer._checkpoint_budget_bytes(captured[0]) == budget


def _frozen_state_mode() -> str:
    return str(FROZEN["runtime"]["state_mode"])


def _configuration(name: str, seconds: float) -> dict:
    return {
        "name": name,
        "qubits": 18,
        "layers": 10,
        "slice_count": 6,
        "steps": 1,
        "parameter_count": 180,
        "seconds": [seconds] * 10,
        "median_seconds": seconds,
        "minimum_seconds": seconds,
        "communication_bytes": 4096,
        # The matched-speed role records the collective time the runtime reported
        # for the configuration it timed. That duration is what the release
        # contract's communication fraction is measured from, so a leg that did
        # not carry it is not a leg a fraction can be read from.
        "collective_seconds": 0.25,
        "training_step_count": 1,
        "parameter_ownership": {"rank:0": ["parameter:0"], "rank:1": ["parameter:1"]},
        "optimizer_update_ownership": {
            "rank:0": ["parameter:0"],
            "rank:1": ["parameter:1"],
        },
        "parameter_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_ownership_semantics": "sharded_across_ranks",
    }


def _leg(*, world: int, seconds: float) -> dict:
    """Return one timed leg of the matched-speed ladder as the role records it.

    The two legs differ in the world they ran at, which is the whole comparison,
    and the sharded leg carries one record per rank because the ownership maps
    are compared across the ranks rather than read from rank 0.
    """

    configurations = [_configuration(entry["name"], seconds) for entry in _ladder()]
    record = {
        "schema": "flagquantum.tensor_network_release_measurements.v1",
        "role": "matched-speed",
        "rank": 0,
        "world_size": world,
        "local_world_size": 1,
        "node_count": world,
        "state_mode": _frozen_state_mode(),
        "commit": "a" * 40,
        "measurements": {"configurations": configurations},
        "timings": [seconds],
        "warmup": int(SPEED["warmup_steps"]),
        "iterations": int(SPEED["measured_steps"]),
        "measured_peak_memory_bytes": 1024,
        "device_name": "NVIDIA A800-SXM4-80GB",
        "workload_sha256": "b" * 64,
        # A leg launched without the bound records none, and the summary role
        # compares the two legs' budgets before it reports a ratio.
        "checkpoint_budget_bytes": None,
        "software": {"torch": "2.9.0"},
    }
    return {**record, "ranks": [dict(record) for _ in range(world)]}


def test_the_producer_states_the_state_mode_the_manifest_froze(tmp_path: Path) -> None:
    """The mode is the frozen contract's, not a second copy the producer owns."""

    from flagquantum.runtime.audit.vocabulary import TENSOR_NETWORK_STATE_MODES

    assert producer._stated_state_mode(RELEASE_MANIFEST) == _frozen_state_mode()
    assert _frozen_state_mode() in TENSOR_NETWORK_STATE_MODES

    # A manifest that named another capability's mode, or none at all, has to
    # refuse the producer: a payload sealed under it would be read by the release
    # gate as evidence for a capability that was never measured.
    drifted = copy.deepcopy(FROZEN)
    drifted["runtime"]["state_mode"] = "distributed_mps"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not recognize"):
        producer._stated_state_mode(manifest)


def test_every_rank_record_states_the_state_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-rank records the capacity roles seal carry the mode themselves."""

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        producer.torch.cuda,
        "get_device_properties",
        lambda device: argparse.Namespace(name="NVIDIA A800-SXM4-80GB"),
    )
    arguments = argparse.Namespace(
        role="capacity-completion",
        release_manifest=RELEASE_MANIFEST,
        node_count=2,
        slice_count=4,
        steps=2,
        checkpoint_budget_bytes=None,
    )
    record = producer._rank_record(
        arguments,
        device=torch.device("cpu"),
        contract=FROZEN["capacity_workload"],
        digest="c" * 64,
        shape=(4, 6, 2),
    )
    assert record["state_mode"] == _frozen_state_mode()


def test_a_role_document_that_does_not_state_the_frozen_mode_is_refused(
    tmp_path: Path,
) -> None:
    """Every role writes through one boundary, so the requirement is checked once.

    A document that stated another capability's mode, or none, would be sealed as
    evidence the release gate reads as foreign, so the producer refuses to
    serialise it rather than leaving the mistake to the gate.
    """

    output = tmp_path / "role.json"
    arguments = argparse.Namespace(
        role="capacity-failure",
        release_manifest=RELEASE_MANIFEST,
        measurements=output,
    )
    with pytest.raises(SystemExit, match="must state the frozen state mode"):
        producer._write(arguments, {"role": "capacity-failure"})
    with pytest.raises(SystemExit, match="must state the frozen state mode"):
        producer._write(
            arguments, {"role": "capacity-failure", "state_mode": "distributed_mps"}
        )
    assert not output.exists()

    producer._write(
        arguments, {"role": "capacity-failure", "state_mode": _frozen_state_mode()}
    )
    assert json.loads(output.read_text(encoding="utf-8"))["state_mode"] == (
        _frozen_state_mode()
    )


def test_the_assembled_speed_summary_states_the_state_mode(tmp_path: Path) -> None:
    """A role document assembled from real legs carries the mode, top to bottom.

    The speed-summary role is the one role the pair can assemble without an
    accelerator, and its document is the one the sealer projects onto the release
    payload, so the mode has to survive both the document and the projection.
    """

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_leg(world=1, seconds=4.0)), encoding="utf-8")
    sharded = tmp_path / "sharded.json"
    sharded.write_text(json.dumps(_leg(world=2, seconds=2.0)), encoding="utf-8")
    summary = tmp_path / "summary.json"
    assert (
        producer.main(
            [
                "--role",
                "speed-summary",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--baseline",
                str(baseline),
                "--sharded",
                str(sharded),
                "--measurements",
                str(summary),
            ]
        )
        == 0
    )
    document = json.loads(summary.read_text(encoding="utf-8"))
    assert document["state_mode"] == _frozen_state_mode()
    assert document["measurements"]["state_mode"] == _frozen_state_mode()

    # The sealer reads the projected measurements as the envelope's evidence, so
    # the mode has to be inside that block and not only beside it.
    evidence = tmp_path / "evidence.json"
    assert (
        producer.main(
            [
                "--role",
                "release-payload",
                "--document",
                str(summary),
                "--measurements",
                str(evidence),
            ]
        )
        == 0
    )
    assert json.loads(evidence.read_text(encoding="utf-8"))["state_mode"] == (
        _frozen_state_mode()
    )


def test_the_communication_fraction_is_measured_from_the_timed_call(
    tmp_path: Path,
) -> None:
    """The fraction is a ratio of two durations the runtime reported itself.

    The frozen contract requires a measured communication fraction, and a
    fraction the producer invented -- or divided bytes by a step time to obtain
    -- would put two different units in one ratio and let the payload disagree
    with the run it describes. The value is therefore read from the timed call's
    own step records, and a leg that did not report those seconds is refused
    rather than recorded with a zero.
    """

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_leg(world=1, seconds=4.0)), encoding="utf-8")

    silent = _leg(world=2, seconds=2.0)
    for configuration in silent["measurements"]["configurations"]:
        del configuration["collective_seconds"]
    unmeasured = tmp_path / "unmeasured.json"
    unmeasured.write_text(json.dumps(silent), encoding="utf-8")
    with pytest.raises(SystemExit, match="collective seconds"):
        producer.main(
            [
                "--role",
                "speed-summary",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--baseline",
                str(baseline),
                "--sharded",
                str(unmeasured),
                "--measurements",
                str(tmp_path / "never.json"),
            ]
        )

    sharded = tmp_path / "sharded.json"
    sharded.write_text(json.dumps(_leg(world=2, seconds=2.0)), encoding="utf-8")
    summary = tmp_path / "summary.json"
    assert (
        producer.main(
            [
                "--role",
                "speed-summary",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--baseline",
                str(baseline),
                "--sharded",
                str(sharded),
                "--measurements",
                str(summary),
            ]
        )
        == 0
    )
    measured = json.loads(summary.read_text(encoding="utf-8"))["measurements"]
    acceptance = next(
        item
        for item in measured["configurations"]
        if item["name"] == SPEED["acceptance_configuration"]
    )
    assert measured["communication_fraction"] == pytest.approx(
        acceptance["sharded_collective_seconds"] / acceptance["sharded_median_seconds"]
    )
    # The definition travels with the number, because the same name is used
    # elsewhere in this repository for a byte share and a reader has to be able
    # to tell which ratio a payload reports.
    assert "collective seconds" in measured["communication_fraction_definition"]


def test_a_matched_speed_pair_must_share_one_checkpoint_budget(
    tmp_path: Path,
) -> None:
    """Two legs under different budgets are not one comparison of one device.

    The budget bounds the reverse tape, so it changes the memory profile of the
    step both legs timed while leaving the arithmetic identical. A ratio between
    a bounded leg and an unbounded one would therefore report the effect of the
    bound as though it were the effect of the second device, which is the one
    number a matched-speed ladder exists to isolate.
    """

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_leg(world=1, seconds=4.0)), encoding="utf-8")

    budget = 4 * 1024**3
    bounded = tmp_path / "bounded.json"
    bounded.write_text(
        json.dumps({**_leg(world=2, seconds=2.0), "checkpoint_budget_bytes": budget}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="different checkpoint budgets"):
        producer.main(
            [
                "--role",
                "speed-summary",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--baseline",
                str(baseline),
                "--sharded",
                str(bounded),
                "--measurements",
                str(tmp_path / "never.json"),
            ]
        )

    # Both legs under one budget assemble, and the payload states the budget
    # rather than leaving a reader to assume the unbounded path completed it.
    shared = tmp_path / "bounded_baseline.json"
    shared.write_text(
        json.dumps({**_leg(world=1, seconds=4.0), "checkpoint_budget_bytes": budget}),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    assert (
        producer.main(
            [
                "--role",
                "speed-summary",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--baseline",
                str(shared),
                "--sharded",
                str(bounded),
                "--measurements",
                str(summary),
            ]
        )
        == 0
    )
    measured = json.loads(summary.read_text(encoding="utf-8"))["measurements"]
    assert measured["checkpoint_budget_bytes"] == budget
