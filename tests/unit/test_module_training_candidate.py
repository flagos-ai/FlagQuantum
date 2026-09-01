from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.training as fqt

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "module-training-v1-candidate.json"


def _load() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_module_training_contract_is_frozen_without_whole_api_freeze() -> None:
    candidate = _load()

    assert candidate["status"] == "frozen"
    assert candidate["implementation_authorized"] is True
    assert candidate["root_manifest_change"] is False
    assert candidate["rules"]["candidate_is_frozen_contract"] is True
    assert candidate["rules"]["whole_api_freeze_implied"] is False


def test_module_training_public_signatures_match_candidate() -> None:
    signatures = _load()["public_signatures"]
    objects = {
        "Module.forward": fq.Module.forward,
        "Module.execute": fq.Module.execute,
        "Module.save_checkpoint": fq.Module.save_checkpoint,
        "Module.load_checkpoint": fq.Module.load_checkpoint,
        "train": fq.train,
    }

    assert {
        name: str(inspect.signature(value, eval_str=False))
        for name, value in objects.items()
    } == signatures


def test_training_extension_matches_candidate_exactly() -> None:
    extension = _load()["stable_extension"]

    assert extension["namespace"] == "flagquantum.training"
    assert set(fqt.__all__) == set(extension["additions"])
    assert set(extension["new_namespace_only"]) == {"TrainingCheckpointRestore"}


def test_candidate_records_intentional_training_boundaries() -> None:
    candidate = _load()

    assert candidate["module_semantics"]["fq_run_accepts_module"] is False
    assert candidate["module_semantics"]["module_run_method"] is False
    assert candidate["training_semantics"]["checkpoint_resume_in_train"] is False
    assert candidate["training_semantics"]["early_stopping_in_train"] is False
    assert (
        candidate["training_semantics"]["distributed_training_namespace"]
        == "flagquantum.experimental.distributed"
    )
