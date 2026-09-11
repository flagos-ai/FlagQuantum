"""Direct Quafu submission does not load a local compiler or calibration SDK."""

import builtins
from unittest.mock import Mock

import pytest

import flagquantum as fq
from flagquantum.remote.qpu.quafu import QuafuProvider


class Transport:
    def __init__(self):
        self.posts = []
        self.gets = []

    def post_json(self, url, payload, headers, timeout):
        self.posts.append((url, payload))
        return {"task_id": f"job-{len(self.posts)}"}

    def get_json(self, url, headers, timeout):
        self.gets.append(url)
        if "/task/status/" in url:
            return {"status": "Finished"}
        if "/task/result/" in url:
            return {
                "status": "Finished",
                "count": {"10": 1024},
                "transpiled": "service circuit",
            }
        raise AssertionError(f"Unexpected discovery/calibration request: {url}")


@pytest.fixture
def transport(monkeypatch):
    transport = Transport()
    provider = QuafuProvider(base_url="https://quafu.test", transport=transport)
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.QuafuProvider", lambda: provider
    )
    monkeypatch.setattr(
        "flagquantum._api.compile",
        Mock(side_effect=AssertionError("local plugin called")),
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"qsteed", "flagquantum_compiler_qsteed", "quark"}:
            raise AssertionError(f"Local compiler dependency loaded: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return transport


def test_direct_counts_with_service_compilation(transport):
    result = fq.run(fq.Circuit(2).x(0), target="quafu:Baihua", shots=1024)
    url, payload = transport.posts[0]
    assert "chip=Baihua" in url
    assert payload["options"] == {"compiler": "quarkcircuit", "target_qubits": []}
    assert "qreg q[2];" in payload["circuit"]
    assert result.counts == [{"10": 1024}]
    assert result.provenance["compiler"] is None
    assert result.provenance["compilation_location"] == "service"
    assert result.provenance["service_compiler"] == "quarkcircuit"
    assert result.provenance["deployment"]["transpiled"] == "service circuit"


def test_direct_expectations_keep_rotations_grouping_and_requested_mapping(transport):
    result = fq.run(
        fq.Circuit(2),
        target="quafu:Baihua",
        shots=1024,
        target_qubits=(17, 18),
        outputs=fq.expectation(fq.X(0) + fq.Z(0)),
    )
    assert len(transport.posts) == 2
    assert all(
        p["options"] == {"compiler": "quarkcircuit", "target_qubits": [17, 18]}
        for _, p in transport.posts
    )
    assert any("h q[0]" in p["circuit"] for _, p in transport.posts)
    assert result.expectation().item() == -2
    assert result.provenance["target_qubits"] == (17, 18)


@pytest.mark.parametrize("mapping", [(1,), (1, 1), (-1, 2), (True, 2), (1.5, 2), "12"])
def test_invalid_mapping_never_submits(transport, mapping):
    with pytest.raises(ValueError, match="target_qubits"):
        fq.run(fq.Circuit(2), target="quafu:Baihua", shots=1024, target_qubits=mapping)
    assert not transport.posts


@pytest.mark.parametrize(
    "options", [{"shots": 12}, {"shots": 1024, "target": "quafu:"}]
)
def test_invalid_service_request_never_submits(transport, options):
    with pytest.raises(ValueError):
        fq.run(fq.Circuit(2), **{"target": "quafu:Baihua", **options})
    assert not transport.posts
