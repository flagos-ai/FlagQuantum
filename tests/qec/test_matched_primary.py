"""Independent scalar count oracles; synthetic arrays are not physical evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.qec import continued_primary as previous
from examples.qec import matched_parity as parity
from examples.qec import matched_primary as primary

pytestmark = pytest.mark.integration


def fixture_rows(shots: int = 64) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    truth = torch.tensor([[i % 2, (i // 3) % 2] for i in range(shots)])
    predictions = {
        name: torch.tensor(
            [
                [
                    int(truth[i, j]) ^ int((i + k * (j + 1)) % (k + 2) == 0)
                    for j in range(2)
                ]
                for i in range(shots)
            ]
        )
        for k, name in enumerate(parity.PATHS)
    }
    return truth, predictions


def scalar_count(
    truth: torch.Tensor, ref: torch.Tensor, cand: torch.Tensor
) -> dict[str, int]:
    pairs = [
        (
            any(a != b for a, b in zip(t, r, strict=True)),
            any(a != b for a, b in zip(t, c, strict=True)),
        )
        for t, r, c in zip(truth.tolist(), ref.tolist(), cand.tolist(), strict=True)
    ]
    return dict(
        shots=len(pairs),
        baseline_failures=sum(r for r, c in pairs),
        hybrid_failures=sum(c for r, c in pairs),
        wins=sum(r and not c for r, c in pairs),
        losses=sum(c and not r for r, c in pairs),
    )


def test_all32_shared_rows_match_independent_scalar_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(primary, "SHOTS", 64)
    truth, predictions = fixture_rows()
    chunks = (
        (truth[a:b], {k: v[a:b] for k, v in predictions.items()})
        for a, b in ((0, 7), (7, 39), (39, 64))
    )
    report = primary._reduce_predictions(chunks)
    expected = {(n, r) for n in parity.PATHS[3:] for r in ("global", "fast")}
    expected.update(
        (n, n.replace("cnn-", "mlp-", 1)) for n in parity.PATHS if n.startswith("cnn-")
    )
    expected.update({("fast", "global"), ("mask-only", "global")})
    assert len(report["comparisons"]) == len(expected) == 32
    assert {(r["candidate"], r["reference"]) for r in report["comparisons"]} == expected
    for row in report["comparisons"]:
        assert row["counts"] == scalar_count(
            truth, predictions[row["reference"]], predictions[row["candidate"]]
        )
    assert report["shots"] == 64
    assert report["comparison_alpha"] == 0.05 / 32
    assert previous.ALPHA == 0.05 / 26
    assert len(previous.COMPARISONS) == 26


def test_registered_constants_and_no_old_horizon_aliases() -> None:
    assert (primary.SEED, primary.SHOTS, primary.CHUNK) == (22000011, 4194304, 4096)
    assert all(
        "step-200-" not in n and not n.startswith("native-")
        for pair in primary.COMPARISONS
        for n in pair
    )


def test_exact_zero_and_all_failure_endpoints_use_family_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(primary, "SHOTS", 64)
    truth = torch.zeros(64, 1, dtype=torch.int8)
    predictions = {n: torch.zeros_like(truth) for n in parity.PATHS}
    predictions["global"] = torch.ones_like(truth)
    report = primary._reduce_predictions([(truth, predictions)])
    row = report["comparisons"][0]
    # Independent closed-form CP endpoints: alpha/2 per marginal, /2 per tail.
    lower = math.exp(math.log(0.05 / 128) / 64)
    upper = -math.expm1(math.log(0.05 / 128) / 64)
    assert row["exact"]["baseline_interval"][0] <= lower
    assert row["exact"]["hybrid_interval"][1] >= upper
    assert math.isclose(row["exact"]["ratio_upper"], upper / lower, rel_tol=1e-12)
    assert row["legacy_hoeffding"]["ratio_upper"] != row["exact"]["ratio_upper"]


@pytest.mark.parametrize(
    "case", ["pass", "one_cnn_seed", "zero_global", "zero_mlp", "zero_fast"]
)
def test_three_seed_conjunctions_and_zero_reference_are_separate(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(primary, "SHOTS", 4096)
    truth = torch.zeros(4096, 1, dtype=torch.int8)
    baseline = (torch.arange(4096) % 2).reshape(-1, 1).to(torch.int8)
    predictions = {
        n: (torch.zeros_like(truth) if n.startswith("cnn-") else baseline.clone())
        for n in parity.PATHS
    }
    if case == "one_cnn_seed":
        predictions["cnn-22-step-2000-calibrated"] = baseline.clone()
    elif case == "zero_global":
        predictions["global"] = torch.zeros_like(truth)
    elif case == "zero_mlp":
        predictions["mlp-33-step-2000-calibrated"] = torch.zeros_like(truth)
    elif case == "zero_fast":
        predictions["fast"] = torch.zeros_like(truth)
    report = primary._reduce_predictions([(truth, predictions)])
    assert (report["native_quality_gate"] == "supported_10_percent_reduction") == (
        case not in ("one_cnn_seed", "zero_global")
    )
    assert (
        report["matched_recipe_incremental_benefit"] == "supported_10_percent_reduction"
    ) == (case in ("pass", "zero_fast"))
    assert report["competitive_with_fast"] == (
        case not in ("one_cnn_seed", "zero_fast")
    )


@pytest.mark.parametrize(
    "case", ["short", "extra", "order", "missing", "nonbinary", "width"]
)
def test_invalid_prediction_stream_cannot_return_summary(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(primary, "SHOTS", 64)
    truth, predictions = fixture_rows()
    if case == "short":
        truth, predictions = truth[:-1], {k: v[:-1] for k, v in predictions.items()}
    elif case == "order":
        predictions = dict(reversed(list(predictions.items())))
    elif case == "missing":
        predictions.pop(parity.PATHS[-1])
    elif case == "nonbinary":
        predictions["mask-only"][0, 0] = 2
    chunks = [(truth, predictions)]
    if case == "extra":
        chunks.append((truth, predictions))
    elif case == "width":
        chunks = [
            (truth[:32], {k: v[:32] for k, v in predictions.items()}),
            (truth[32:, :1], {k: v[32:, :1] for k, v in predictions.items()}),
        ]
    with pytest.raises(ValueError):
        primary._reduce_predictions(chunks)


def test_exact_failure_propagates_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(primary, "SHOTS", 64)

    def refused(*args: object, **kwargs: object) -> None:
        raise ArithmeticError("uncertified exact interval")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("exact failure reached descriptive fallback")

    monkeypatch.setattr(previous.exact_statistics, "summarize_exact", refused)
    monkeypatch.setattr(previous, "summarize", forbidden)
    with pytest.raises(ArithmeticError, match="uncertified"):
        primary._reduce_predictions([fixture_rows()])


# These tests mock the hardware, bound-model and receipt authority boundaries.
# The shared sampler/capture/replay/reducer execute against exact synthetic rows.
@pytest.fixture
def captured_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, ...]:
    import hashlib
    import json
    from types import SimpleNamespace

    from examples.qec import diagnose_predecoder as archive

    monkeypatch.setattr(primary, "SHOTS", 64)
    monkeypatch.setattr(primary, "CHUNK", 32)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(primary.parity, "_precision", lambda: None)

    class Circuit:
        num_detectors = 3
        num_observables = 2

        def __init__(self) -> None:
            self.seeds: list[int] = []
            self.calls: list[int] = []

        def __str__(self) -> str:
            return "explicit software primary fixture"

        def compile_detector_sampler(self, *, seed: int) -> Circuit:
            self.seeds.append(seed)
            return self

        def sample(
            self, shots: int, *, separate_observables: bool
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert separate_observables
            self.calls.append(shots)
            rows = torch.arange(shots)
            return torch.stack((rows % 2, rows % 3 == 0, rows % 5 == 0), dim=1).to(
                torch.int8
            ), torch.zeros(shots, 2, dtype=torch.int8)

    circuit = Circuit()
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    circuit_sha = hashlib.sha256(str(circuit).encode()).hexdigest()
    (anchor / "results.json").write_text(json.dumps({"circuit_sha256": circuit_sha}))
    (anchor / "manifest.json").write_text(json.dumps({"torch": str(torch.__version__)}))
    monkeypatch.setattr(archive, "ARCHIVE", anchor)
    directories = []
    for name in ("initial", "trained", "calibration", "choices", "probes"):
        path = tmp_path / name
        path.mkdir()
        (path / ("choices.json" if name == "choices" else "manifest.json")).write_text(
            name
        )
        directories.append(path)
    initial, trained, calibration, choices, probes = directories
    choices_sha = archive.digest(choices / "choices.json")
    probes_sha = archive.digest(probes / "manifest.json")
    weights = tmp_path / "weights"
    weights.write_bytes(b"invalid software weights, never loaded")
    monkeypatch.setattr(primary.comparison, "FAST_SHA256", archive.digest(weights))
    hardware = dict(
        device="cuda:0",
        name="A800",
        image_id=primary.device.IMAGE,
        tf32=False,
        autocast=False,
        deterministic_algorithms=True,
        nvidia_smi_uuid_name_driver="GPU-11111111-1111-1111-1111-111111111111,A800,fixture",
    )
    monkeypatch.setattr(
        primary.comparison,
        "source_digest",
        lambda p: primary.comparison.ISING_SOURCE_SHA256,
    )
    pipeline = torch.nn.Linear(1, 1).eval()
    models = []
    for seed in (11, 22, 33):
        for arch in ("cnn", "mlp"):
            model = torch.nn.Linear(1, 1).eval()
            model.dem_text = "synthetic DEM identity"
            trainer = SimpleNamespace(
                model=model,
                optimizer=torch.optim.Adam(model.parameters()),
                generator=torch.Generator().manual_seed(seed),
                step=2000,
            )
            models.append((seed, arch, trainer))
    frozen = {
        "inputs": {
            "circuit_sha256": circuit_sha,
            "dem_sha256": hashlib.sha256(
                models[0][2].model.dem_text.encode()
            ).hexdigest(),
        }
    }
    monkeypatch.setattr(
        primary.validation, "restore_choices", lambda *a, **k: (models, frozen)
    )
    monkeypatch.setattr(primary.parity, "verify_two_hosts", lambda *a, **k: None)

    def reference(*a: Any, **k: Any) -> tuple[dict[str, Any], Any]:
        return hardware, SimpleNamespace(circuit=circuit, pipeline=pipeline)

    monkeypatch.setattr(primary.device, "_device_reference", reference)

    def zero(bits: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(bits), 2, dtype=torch.int8)

    monkeypatch.setattr(primary.comparison, "global_decoder", lambda c: zero)

    def pairs(*a: Any, **k: Any) -> dict[str, Any]:
        assert k["device"] == torch.device("cuda:0")

        def make(index: int) -> Any:
            def correct(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                frame = torch.zeros(len(bits), 2, dtype=torch.int8)
                frame[:, index % 2] = bits[:, index % 3]
                return bits, frame

            return correct

        return {n: (make(i), make(i)) for i, n in enumerate(parity.PATHS)}

    monkeypatch.setattr(primary.parity, "path_pairs", pairs)
    receipts = {}
    for host in ("a800-172", "a800-171"):
        path = tmp_path / (host + ".json")
        path.write_text(
            json.dumps(
                dict(
                    torch_version=str(torch.__version__),
                    hardware=hardware,
                    pipeline_state_sha256=archive.state_digest(pipeline.state_dict()),
                )
            )
        )
        receipts[host] = (path, archive.digest(path))
    args = (
        tmp_path,
        initial,
        trained,
        calibration,
        choices,
        probes,
        receipts,
        tmp_path / "primary",
    )
    kwargs = dict(
        host="a800-172",
        choices_sha256=choices_sha,
        probes_sha256=probes_sha,
        ising_weights=weights,
    )
    return args, kwargs, circuit, models, pipeline, hardware


def test_guarded_capture_replay_and_all32_counts_with_explicit_mock_boundaries(
    captured_primary: tuple[Any, ...], tmp_path: Path
) -> None:
    from examples.qec import diagnose_predecoder as archive

    args, kwargs, circuit, models, pipeline, hardware = captured_primary
    primary.run_primary(*args, **kwargs)
    assert circuit.seeds == [22000011] and circuit.calls == [32, 32]
    directory = args[-1]
    digest = archive.digest(directory / "manifest.json")
    chunks = list(primary.replay_primary(directory, expected_manifest_sha256=digest))
    result = tmp_path / "summary.json"
    primary.summarize_primary(directory, result, manifest_sha256=digest)
    import json

    report = json.loads(result.read_text())
    assert (
        report["independent_physical_shots"] == 64
        and report["speed_claim_allowed"] is False
    )
    truth = torch.cat([t for t, p in chunks])
    predictions = {k: torch.cat([p[k] for t, p in chunks]) for k in parity.PATHS}
    for row in report["comparisons"]:
        assert row["counts"] == scalar_count(
            truth, predictions[row["reference"]], predictions[row["candidate"]]
        )
    assert circuit.calls == [32, 32]
    with pytest.raises(FileExistsError):
        primary.run_primary(*args, **kwargs)
    assert circuit.calls == [32, 32]


@pytest.mark.parametrize(
    "damage",
    [
        "host",
        "choices",
        "receipts",
        "weights",
        "uuid",
        "pipeline",
        "autocast",
        "threads",
    ],
)
def test_guard_rejects_before_only_sampler(
    captured_primary: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    args, kwargs, circuit, models, pipeline, hardware = captured_primary

    def refused(*a: Any, **k: Any) -> None:
        raise ValueError("explicit guard refusal")

    if damage == "source":
        monkeypatch.setattr(primary.comparison, "source_digest", lambda p: "a" * 64)
    elif damage == "host":
        kwargs["host"] = "a800-171"
    elif damage == "choices":
        monkeypatch.setattr(primary.validation, "restore_choices", refused)
    elif damage == "receipts":
        monkeypatch.setattr(primary.parity, "verify_two_hosts", refused)
    elif damage == "weights":
        kwargs["ising_weights"].write_bytes(b"changed")
    elif damage == "uuid":
        hardware["nvidia_smi_uuid_name_driver"] = (
            "GPU-22222222-2222-2222-2222-222222222222,A800,fixture"
        )
    elif damage == "pipeline":
        with torch.no_grad():
            pipeline.weight.add_(1)
    elif damage == "autocast":
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a: True)
    elif damage == "threads":
        monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    with pytest.raises(ValueError):
        primary.run_primary(*args, **kwargs)
    assert not circuit.seeds and not args[-1].exists()


@pytest.mark.parametrize(
    "damage",
    [
        "input",
        "receipt",
        "state",
        "pipeline",
        "plan",
        "shard",
        "extra",
        "shared_bits",
        "vendor",
        "decode",
    ],
)
def test_partial_capture_never_writes_success_or_retries(
    captured_primary: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    args, kwargs, circuit, models, pipeline, hardware = captured_primary
    factory = primary.parity.path_pairs

    def pairs(*a: Any, **k: Any) -> dict[str, Any]:
        result = factory(*a, **k)
        original = result[parity.PATHS[-1]][1]

        def mutate(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if damage == "vendor":
                monkeypatch.setattr(
                    primary.comparison, "source_digest", lambda p: "a" * 64
                )
            elif damage == "input":
                (args[1] / "manifest.json").write_text("changed")
            elif damage == "receipt":
                args[6]["a800-172"][0].write_text("changed")
            elif damage == "state":
                models[0][2].step += 1
            elif damage == "pipeline":
                with torch.no_grad():
                    pipeline.weight.add_(1)
            elif damage == "plan":
                (args[-1] / "plan.json").write_text("{}")
            elif damage == "shard" and (args[-1] / "chunk-0000.pt").exists():
                (args[-1] / "chunk-0000.pt").write_bytes(b"changed")
            elif damage == "extra":
                (args[-1] / "extra").write_text("extra")
            elif damage == "shared_bits":
                bits[:, 0] ^= 1
            elif damage == "decode":
                raise ArithmeticError("synthetic decoder failure")
            return original(bits)

        result[parity.PATHS[-1]] = (mutate, mutate)
        return result

    monkeypatch.setattr(primary.parity, "path_pairs", pairs)
    with pytest.raises((ValueError, ArithmeticError)):
        primary.run_primary(*args, **kwargs)
    assert not (args[-1] / "manifest.json").exists()
    calls = list(circuit.calls)
    with pytest.raises(FileExistsError):
        primary.run_primary(*args, **kwargs)
    assert circuit.calls == calls


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "missing",
        "extra",
        "plan",
        "chunk",
        "paths",
        "source",
        "protocol",
        "budget",
        "inventory",
    ],
)
def test_replay_damage_cannot_write_summary(
    captured_primary: tuple[Any, ...], tmp_path: Path, damage: str
) -> None:
    import json

    from examples.qec import diagnose_predecoder as archive

    args, kwargs, circuit, models, pipeline, hardware = captured_primary
    primary.run_primary(*args, **kwargs)
    directory = args[-1]
    manifest = directory / "manifest.json"
    digest = archive.digest(manifest)
    raw = json.loads(manifest.read_text())
    if damage == "hash":
        digest = "a" * 64
    elif damage == "missing":
        (directory / "chunk-0001.pt").unlink()
    elif damage == "extra":
        (directory / "extra").write_text("extra")
    elif damage == "plan":
        (directory / "plan.json").write_text("{}")
    elif damage == "chunk":
        (directory / "chunk-0001.pt").write_bytes(b"bad")
    elif damage == "paths":
        p = directory / "chunk-0001.pt"
        payload = torch.load(p, weights_only=True)
        payload["predictions"].pop(parity.PATHS[-1])
        torch.save(payload, p)
        raw["shards"][1]["file_sha256"] = archive.digest(p)
    elif damage == "source":
        raw["binding"]["source_sha256"] = {}
    elif damage == "protocol":
        raw["binding"]["protocols"] = {}
    elif damage == "budget":
        raw["seed"] = 16000011
    elif damage == "inventory":
        raw["binding"]["input_files"]["choices"]["choices.json"] = "a" * 64
    if damage in ("paths", "source", "protocol", "budget", "inventory"):
        manifest.write_text(json.dumps(raw))
        digest = archive.digest(manifest)
        (directory / "plan.json").write_text(json.dumps({**raw, "shards": []}))
    output = tmp_path / "no-summary.json"
    with pytest.raises((ValueError, KeyError)):
        primary.summarize_primary(directory, output, manifest_sha256=digest)
    assert not output.exists() and circuit.calls == [32, 32]


@pytest.mark.parametrize("damage", ["exact", "shard", "source"])
def test_summary_rechecks_evidence_after_statistics(
    captured_primary: tuple[Any, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    from examples.qec import diagnose_predecoder as archive

    args, kwargs, circuit, models, pipeline, hardware = captured_primary
    primary.run_primary(*args, **kwargs)
    directory = args[-1]
    digest = archive.digest(directory / "manifest.json")
    original = previous._comparison_reports

    def reports(*a: Any, **k: Any) -> Any:
        if damage == "exact":
            raise ArithmeticError("uncertified synthetic exact interval")
        result = original(*a, **k)
        if damage == "shard":
            (directory / "chunk-0000.pt").write_bytes(b"changed after replay")
        else:
            monkeypatch.setattr(primary, "_sources", lambda: {})
        return result

    monkeypatch.setattr(previous, "_comparison_reports", reports)
    output = tmp_path / "no-summary.json"
    with pytest.raises((ValueError, ArithmeticError)):
        primary.summarize_primary(directory, output, manifest_sha256=digest)
    assert not output.exists() and circuit.calls == [32, 32]
