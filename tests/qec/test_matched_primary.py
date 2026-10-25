"""Independent scalar count oracles; synthetic arrays are not physical evidence."""

from __future__ import annotations

import math

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
