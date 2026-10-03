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

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks import tensor_network_release_evidence as producer

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
    lower, upper = producer._bootstrap_ratio_interval(baseline, sharded)
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
    contract = {"acceptance_case": "matched_speed", "world_size": 2}
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
