from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from flagquantum.ecosystem.kaiwu import KaiwuSampler, bind_qdiffusion_builder

pytestmark = pytest.mark.unit


@dataclass
class _EnergyModel:
    sampler: object


@dataclass
class _Generator:
    energy_model: _EnergyModel
    received: dict[str, Any]


def _sampler() -> KaiwuSampler:
    return cast(KaiwuSampler, object())


def test_binding_overrides_plugin_sampler_factory_hints() -> None:
    sampler = _sampler()

    def builder(*args: object, **kwargs: Any) -> _Generator:
        assert args == ("proposal", "energy")
        return _Generator(_EnergyModel(kwargs["bm_sampler"]), kwargs)

    bound = bind_qdiffusion_builder(builder, sampler)
    generator = bound(
        "proposal",
        "energy",
        bm_sampler_type="sa",
        bm_sampler_kwargs={"untrusted": "ignored"},
    )

    assert generator.energy_model.sampler is sampler
    assert generator.received["bm_sampler_type"] == "flagquantum-kaiwu"
    assert generator.received["bm_sampler_kwargs"] == {}


def test_binding_rejects_another_explicit_sampler() -> None:
    sampler = _sampler()

    def builder(**kwargs: Any) -> _Generator:
        return _Generator(_EnergyModel(kwargs["bm_sampler"]), kwargs)

    bound = bind_qdiffusion_builder(builder, sampler)

    with pytest.raises(ValueError, match="another BM sampler"):
        bound(bm_sampler=object())


def test_binding_fails_if_factory_drops_injected_sampler() -> None:
    sampler = _sampler()

    def builder(**kwargs: Any) -> _Generator:
        return _Generator(_EnergyModel(object()), kwargs)

    bound = bind_qdiffusion_builder(builder, sampler)

    with pytest.raises(RuntimeError, match="did not retain"):
        bound()


def test_binding_reuses_one_sampler_across_all_workflow_builds() -> None:
    sampler = _sampler()
    observed: list[object] = []

    def builder(**kwargs: Any) -> _Generator:
        observed.append(kwargs["bm_sampler"])
        return _Generator(_EnergyModel(kwargs["bm_sampler"]), kwargs)

    bound = bind_qdiffusion_builder(builder, sampler)
    for _ in range(4):
        bound()

    assert observed == [sampler] * 4
