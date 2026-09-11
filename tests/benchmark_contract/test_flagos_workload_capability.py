"""Fail-closed contracts for the FlagOS F4 workload capability matrix."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flagquantum.runtime.distributed.workload_capability import (
    FlagOSWorkloadCapabilityError,
    build_flagos_workload_capability_matrix,
)
from tools.audit_flagos_workload_capability import (
    DEFAULT_CONFORMANCE,
    DEFAULT_OUTPUT,
    DEFAULT_SCALE,
    DEFAULT_TRAINING,
    load_evidence,
    main,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.distributed_cpu]


def _evidence():
    loaded = [
        load_evidence(path)
        for path in (DEFAULT_CONFORMANCE, DEFAULT_SCALE, DEFAULT_TRAINING)
    ]
    return tuple(item[0] for item in loaded), tuple(item[1] for item in loaded)


def _build(payloads=None):
    original, descriptors = _evidence()
    conformance, scale, training = payloads or original
    return build_flagos_workload_capability_matrix(
        conformance,
        scale,
        training,
        evidence_artifacts=descriptors,
    ).to_dict()


def test_checked_in_evidence_has_exact_workload_classification():
    payload = _build()
    capabilities = payload["capabilities"]
    assert (
        capabilities["flagos.statevector.forward"]["status"] == "development_verified"
    )
    assert (
        capabilities["flagos.statevector.training"]["status"] == "development_verified"
    )
    full = capabilities["flagos.collectives.full"]
    assert full["status"] == "unsupported"
    assert {
        (item["primitive"], item["dtype"])
        for item in full["details"]["unsupported_checks"]
    } == {
        ("reduce_scatter_tensor", "complex64"),
        ("reduce_scatter_tensor", "complex128"),
    }
    assert payload["inner_communication_route"] == "unattributed"
    assert payload["flagcx_route_verified"] is False
    assert payload["production_support_claim_allowed"] is False
    assert "flagcx_provider_identity_unavailable" not in payload["blockers"]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda items: items[1].__setitem__("world_sizes", [2, 4]),
        lambda items: items[1].__setitem__("scalability_claim_allowed", True),
        lambda items: items[2]["runs"][0]["cases"][0].__setitem__(
            "backward_uses_full_state_replay", True
        ),
        lambda items: items[0]["collective_checks"][4].__setitem__("error", None),
    ),
)
def test_tampered_or_overclaimed_evidence_fails_closed(mutate):
    payloads, _ = _evidence()
    payloads = list(copy.deepcopy(payloads))
    mutate(payloads)
    with pytest.raises(FlagOSWorkloadCapabilityError):
        _build(tuple(payloads))


def test_checked_in_f4_artifact_is_deterministically_regenerated(tmp_path: Path):
    output = tmp_path / "f4.json"
    assert main(["--output", str(output)]) == 0
    assert json.loads(output.read_text()) == json.loads(DEFAULT_OUTPUT.read_text())
